"""
Distogram tiling Triton kernel with forward + backward (SPEC §18 item 3).

Fuses the nested Python tile loops from distogram loss §11.3:
  Z_ij = U_i * V_j   (Hadamard interaction in d_low space)
  logits_ij = W_bin @ Z_ij  (project to bins)
  loss += cross_entropy(logits_ij, target_ij)

Forward: tiled, fuses Hadamard + matmul + cross-entropy in one pass.
Backward: two kernels recomputing Z/logits/softmax per tile from x_true coordinates.
  - dU kernel (i-centric): dU + dW_bin + dbias
  - dV kernel (j-centric): dV (tiled i-loop with static_range)
  Zero O(N²) storage — target bins recomputed from x_true per tile.

num_bins is padded to PAD_BINS (next power of 2) for tl.dot compatibility.
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _pad_bin_params(w_bin, bias, num_bins, pad_bins, d_low, device):
    """Pad W_bin and bias to next power-of-2 for tl.dot. Extra bins get -1e30 bias."""
    w_bin_f = w_bin.contiguous().float()
    bias_f = bias.contiguous().float()
    if pad_bins > num_bins:
        w_bin_padded = torch.zeros(pad_bins, d_low, device=device, dtype=torch.float32)
        w_bin_padded[:num_bins] = w_bin_f
        bias_padded = torch.full((pad_bins,), -1e30, device=device, dtype=torch.float32)
        bias_padded[:num_bins] = bias_f
    else:
        w_bin_padded = w_bin_f
        bias_padded = bias_f
    return w_bin_padded, bias_padded


def _launch_fwd(U, V, w_bin_padded, bias_padded, target_bins, mask,
                N, d_low, num_bins, pad_bins, BLOCK_I, BLOCK_J):
    """Launch forward kernel, return (loss, count)."""
    B = U.shape[0]
    has_mask = mask is not None
    loss_sum = torch.zeros(B, device=U.device, dtype=torch.float32)
    count = torch.zeros(B, device=U.device, dtype=torch.float32)

    grid = (B, triton.cdiv(N, BLOCK_I), triton.cdiv(N, BLOCK_J))
    _distogram_fwd_kernel[grid](
        U, V, w_bin_padded, bias_padded, target_bins,
        mask if has_mask else U,  # dummy, gated by HAS_MASK
        loss_sum, count,
        N, d_low, num_bins, pad_bins,
        U.stride(0), U.stride(1), U.stride(2),
        w_bin_padded.stride(0), w_bin_padded.stride(1),
        target_bins.stride(0), target_bins.stride(1), target_bins.stride(2),
        mask.stride(0) if has_mask else 0,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
        HAS_MASK=has_mask,
    )

    per_sample = loss_sum / count.clamp(min=1)
    return per_sample.mean(), count


# ============================================================================
# Forward kernel
# ============================================================================


@triton.jit
def _distogram_fwd_kernel(
    U_ptr, V_ptr, W_BIN_ptr, BIAS_ptr, TARGET_ptr,
    MASK_ptr,
    LOSS_ptr, COUNT_ptr,
    N: tl.constexpr, D_LOW: tl.constexpr,
    NUM_BINS: tl.constexpr, PAD_BINS: tl.constexpr,
    stride_ub: tl.constexpr, stride_un: tl.constexpr, stride_ud: tl.constexpr,
    stride_wb: tl.constexpr, stride_wd: tl.constexpr,
    stride_tb: tl.constexpr, stride_tn: tl.constexpr, stride_tm: tl.constexpr,
    stride_mb,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    # HAS_MASK gates all MASK_ptr accesses; when False the pointer is a dummy.
    HAS_MASK: tl.constexpr,
):
    """Tiled distogram loss: Hadamard + tl.dot matmul + cross-entropy."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J

    j_idx = j_start + tl.arange(0, BLOCK_J)
    j_mask = j_idx < N

    d_idx = tl.arange(0, D_LOW)
    b_idx = tl.arange(0, PAD_BINS)

    uv_batch_offset = pid_b * stride_ub
    target_batch_offset = pid_b * stride_tb

    v_tile = tl.load(
        V_ptr + uv_batch_offset + j_idx[:, None] * stride_un + d_idx[None, :] * stride_ud,
        mask=j_mask[:, None], other=0.0,
    )

    w_bin_t = tl.load(
        W_BIN_ptr + b_idx[None, :] * stride_wb + d_idx[:, None] * stride_wd,
    )

    bias = tl.load(BIAS_ptr + b_idx)

    if HAS_MASK:
        mj = tl.load(MASK_ptr + pid_b * stride_mb + j_idx, mask=j_mask, other=0.0)

    tile_loss = tl.zeros([], dtype=tl.float32)
    tile_count = tl.zeros([], dtype=tl.float32)

    for i_off in tl.static_range(0, BLOCK_I):
        i = i_start + i_off
        i_valid = i < N

        u_row = tl.load(
            U_ptr + uv_batch_offset + i * stride_un + d_idx * stride_ud,
            mask=i_valid, other=0.0,
        )

        z_row = v_tile * u_row[None, :]
        logits = tl.dot(z_row, w_bin_t) + bias[None, :]

        targets = tl.load(
            TARGET_ptr + target_batch_offset + i * stride_tn + j_idx * stride_tm,
            mask=j_mask & i_valid, other=0,
        )

        max_logit = tl.max(logits, axis=1)
        shifted = logits - max_logit[:, None]
        sum_exp = tl.sum(tl.exp(shifted), axis=1)
        log_sum_exp = max_logit + tl.log(sum_exp + 1e-10)

        target_one_hot = b_idx[None, :] == targets[:, None]
        target_logit = tl.sum(tl.where(target_one_hot, logits, 0.0), axis=1)

        ce = -target_logit + log_sum_exp

        valid = j_mask & i_valid
        if HAS_MASK:
            mi = tl.load(MASK_ptr + pid_b * stride_mb + i, mask=i_valid, other=0.0)
            valid = valid & (mi > 0.0) & (mj > 0.0)

        ce = tl.where(valid, ce, 0.0)
        tile_loss += tl.sum(ce)
        tile_count += tl.sum(valid.to(tl.float32))

    tl.atomic_add(LOSS_ptr + pid_b, tile_loss)
    tl.atomic_add(COUNT_ptr + pid_b, tile_count)


# ============================================================================
# Backward kernel 1: dU + dW_bin + dbias  (i-centric)
# ============================================================================


@triton.jit
def _distogram_bwd_dU_kernel(
    U_ptr, V_ptr, X_ptr,
    W_BIN_ptr, BIAS_ptr, MASK_ptr,
    COUNT_ptr,
    DU_ptr, DW_BIN_ptr, DBIAS_ptr,
    grad_scale,
    N: tl.constexpr, D_LOW: tl.constexpr,
    NUM_BINS: tl.constexpr, PAD_BINS: tl.constexpr,
    dist_min, inv_bin_width,
    stride_ub: tl.constexpr, stride_un: tl.constexpr, stride_ud: tl.constexpr,
    stride_xb, stride_xn,
    stride_wb: tl.constexpr, stride_wd: tl.constexpr,
    stride_mb,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Grid: (B, ceil(N/BLOCK_I)). Recomputes per tile, accumulates dU + dW + dbias."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    i_start = pid_i * BLOCK_I

    uv_offset = pid_b * stride_ub
    x_offset = pid_b * stride_xb
    d_idx = tl.arange(0, D_LOW)
    b_idx = tl.arange(0, PAD_BINS)

    count_b = tl.load(COUNT_ptr + pid_b)
    scale = grad_scale / tl.maximum(count_b, 1.0)

    w_bin_t = tl.load(W_BIN_ptr + b_idx[None, :] * stride_wb + d_idx[:, None] * stride_wd)
    w_bin = tl.load(W_BIN_ptr + b_idx[:, None] * stride_wb + d_idx[None, :] * stride_wd)
    bias = tl.load(BIAS_ptr + b_idx)

    dW_acc = tl.zeros([PAD_BINS, D_LOW], dtype=tl.float32)
    dbias_acc = tl.zeros([PAD_BINS], dtype=tl.float32)

    for i_off in tl.static_range(0, BLOCK_I):
        i = i_start + i_off
        i_valid = i < N

        u_row = tl.load(U_ptr + uv_offset + i * stride_un + d_idx * stride_ud,
                        mask=i_valid, other=0.0)
        xi_x = tl.load(X_ptr + x_offset + i * stride_xn + 0, mask=i_valid, other=0.0)
        xi_y = tl.load(X_ptr + x_offset + i * stride_xn + 1, mask=i_valid, other=0.0)
        xi_z = tl.load(X_ptr + x_offset + i * stride_xn + 2, mask=i_valid, other=0.0)

        if HAS_MASK:
            mi = tl.load(MASK_ptr + pid_b * stride_mb + i, mask=i_valid, other=0.0)

        dU_row = tl.zeros([D_LOW], dtype=tl.float32)

        for j_start in range(0, N, BLOCK_J):
            j_idx = j_start + tl.arange(0, BLOCK_J)
            j_mask = j_idx < N

            v_tile = tl.load(V_ptr + uv_offset + j_idx[:, None] * stride_un + d_idx[None, :] * stride_ud,
                             mask=j_mask[:, None], other=0.0)

            xj_x = tl.load(X_ptr + x_offset + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
            xj_y = tl.load(X_ptr + x_offset + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
            xj_z = tl.load(X_ptr + x_offset + j_idx * stride_xn + 2, mask=j_mask, other=0.0)
            dx = xi_x - xj_x
            dy = xi_y - xj_y
            dz = xi_z - xj_z
            dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
            target = ((dist - dist_min) * inv_bin_width).to(tl.int32)
            target = tl.maximum(tl.minimum(target, NUM_BINS - 1), 0)

            valid = j_mask & i_valid
            if HAS_MASK:
                mj = tl.load(MASK_ptr + pid_b * stride_mb + j_idx, mask=j_mask, other=0.0)
                valid = valid & (mi > 0.0) & (mj > 0.0)

            z_row = v_tile * u_row[None, :]
            logits = tl.dot(z_row, w_bin_t) + bias[None, :]

            max_l = tl.max(logits, axis=1)
            shifted = logits - max_l[:, None]
            exp_s = tl.exp(shifted)
            probs = exp_s / (tl.sum(exp_s, axis=1)[:, None] + 1e-10)

            one_hot = (b_idx[None, :] == target[:, None]).to(tl.float32)
            dlogits = (probs - one_hot) * scale
            dlogits = tl.where(valid[:, None], dlogits, 0.0)

            dz_row = tl.dot(dlogits, w_bin, input_precision="ieee")
            dU_row += tl.sum(dz_row * v_tile, axis=0)

            dW_acc += tl.dot(tl.trans(dlogits), z_row, input_precision="ieee")
            dbias_acc += tl.sum(dlogits, axis=0)

        du_ptrs = DU_ptr + uv_offset + i * stride_un + d_idx * stride_ud
        tl.store(du_ptrs, dU_row, mask=i_valid)

    tl.atomic_add(DW_BIN_ptr + b_idx[:, None] * stride_wb + d_idx[None, :] * stride_wd, dW_acc)
    tl.atomic_add(DBIAS_ptr + b_idx, dbias_acc, mask=b_idx < PAD_BINS)


# ============================================================================
# Backward kernel 2: dV  (j-centric, tiled i-loop)
# ============================================================================


@triton.jit
def _distogram_bwd_dV_kernel(
    U_ptr, V_ptr, X_ptr,
    W_BIN_ptr, BIAS_ptr, MASK_ptr,
    COUNT_ptr,
    DV_ptr,
    grad_scale,
    N: tl.constexpr, D_LOW: tl.constexpr,
    NUM_BINS: tl.constexpr, PAD_BINS: tl.constexpr,
    dist_min, inv_bin_width,
    stride_ub: tl.constexpr, stride_un: tl.constexpr, stride_ud: tl.constexpr,
    stride_xb, stride_xn,
    stride_wb: tl.constexpr, stride_wd: tl.constexpr,
    stride_mb,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Grid: (B, ceil(N/BLOCK_J)). Recomputes per tile, accumulates dV."""
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    j_start = pid_j * BLOCK_J

    uv_offset = pid_b * stride_ub
    x_offset = pid_b * stride_xb
    d_idx = tl.arange(0, D_LOW)
    b_idx = tl.arange(0, PAD_BINS)

    j_idx = j_start + tl.arange(0, BLOCK_J)
    j_mask = j_idx < N

    count_b = tl.load(COUNT_ptr + pid_b)
    scale = grad_scale / tl.maximum(count_b, 1.0)

    w_bin_t = tl.load(W_BIN_ptr + b_idx[None, :] * stride_wb + d_idx[:, None] * stride_wd)
    w_bin = tl.load(W_BIN_ptr + b_idx[:, None] * stride_wb + d_idx[None, :] * stride_wd)
    bias = tl.load(BIAS_ptr + b_idx)

    xj_x = tl.load(X_ptr + x_offset + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
    xj_y = tl.load(X_ptr + x_offset + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
    xj_z = tl.load(X_ptr + x_offset + j_idx * stride_xn + 2, mask=j_mask, other=0.0)

    if HAS_MASK:
        mj = tl.load(MASK_ptr + pid_b * stride_mb + j_idx, mask=j_mask, other=0.0)

    v_tile = tl.load(V_ptr + uv_offset + j_idx[:, None] * stride_un + d_idx[None, :] * stride_ud,
                     mask=j_mask[:, None], other=0.0)

    dV_acc = tl.zeros([BLOCK_J, D_LOW], dtype=tl.float32)

    # Tile over i with static_range for compile-time unrolling
    for i_start in range(0, N, BLOCK_I):
        for i_off in tl.static_range(0, BLOCK_I):
            i = i_start + i_off
            i_valid = i < N

            u_row = tl.load(U_ptr + uv_offset + i * stride_un + d_idx * stride_ud,
                            mask=i_valid, other=0.0)

            xi_x = tl.load(X_ptr + x_offset + i * stride_xn + 0, mask=i_valid, other=0.0)
            xi_y = tl.load(X_ptr + x_offset + i * stride_xn + 1, mask=i_valid, other=0.0)
            xi_z = tl.load(X_ptr + x_offset + i * stride_xn + 2, mask=i_valid, other=0.0)

            dx = xi_x - xj_x
            dy = xi_y - xj_y
            dz = xi_z - xj_z
            dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
            target = ((dist - dist_min) * inv_bin_width).to(tl.int32)
            target = tl.maximum(tl.minimum(target, NUM_BINS - 1), 0)

            valid = j_mask & i_valid
            if HAS_MASK:
                mi = tl.load(MASK_ptr + pid_b * stride_mb + i, mask=i_valid, other=0.0)
                valid = valid & (mi > 0.0) & (mj > 0.0)

            z_row = v_tile * u_row[None, :]
            logits = tl.dot(z_row, w_bin_t) + bias[None, :]

            max_l = tl.max(logits, axis=1)
            shifted = logits - max_l[:, None]
            exp_s = tl.exp(shifted)
            probs = exp_s / (tl.sum(exp_s, axis=1)[:, None] + 1e-10)

            one_hot = (b_idx[None, :] == target[:, None]).to(tl.float32)
            dlogits = (probs - one_hot) * scale
            dlogits = tl.where(valid[:, None], dlogits, 0.0)

            dz_row = tl.dot(dlogits, w_bin, input_precision="ieee")
            dV_acc += dz_row * u_row[None, :]

    dv_ptrs = DV_ptr + uv_offset + j_idx[:, None] * stride_un + d_idx[None, :] * stride_ud
    tl.store(dv_ptrs, dV_acc, mask=j_mask[:, None])


# ============================================================================
# Autograd function
# ============================================================================


class DistogramFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, w_bin, bias, target_bins, x_true, mask,
                dist_min, bin_width, BLOCK_I, BLOCK_J):
        B, N, d_low = U.shape
        num_bins = w_bin.shape[0]
        pad_bins = _next_power_of_2(num_bins)

        U = U.contiguous().float()
        V = V.contiguous().float()
        target_bins = target_bins.contiguous().long()
        w_bin_padded, bias_padded = _pad_bin_params(w_bin, bias, num_bins, pad_bins, d_low, U.device)

        loss, count = _launch_fwd(U, V, w_bin_padded, bias_padded, target_bins, mask,
                                  N, d_low, num_bins, pad_bins, BLOCK_I, BLOCK_J)

        # Save O(N) tensors only — no target_bins (O(N²))
        ctx.save_for_backward(
            U, V, x_true.contiguous().float(), w_bin_padded, bias_padded,
            mask if mask is not None else torch.empty(0, device=U.device),
            count,
        )
        ctx.has_mask = mask is not None
        ctx.dist_min = dist_min
        ctx.bin_width = bin_width
        ctx.BLOCK_I = BLOCK_I
        ctx.BLOCK_J = BLOCK_J
        ctx.num_bins = num_bins
        ctx.pad_bins = pad_bins

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        U, V, x_true, w_bin_padded, bias_padded, mask_or_empty, count = ctx.saved_tensors
        has_mask = ctx.has_mask
        mask = mask_or_empty if has_mask else None
        B, N, d_low = U.shape
        num_bins = ctx.num_bins

        grad_scale = grad_output.item() / B
        inv_bin_width = 1.0 / ctx.bin_width

        dU = torch.zeros_like(U)
        dV = torch.zeros_like(V)
        dW = torch.zeros_like(w_bin_padded)
        dbias = torch.zeros_like(bias_padded)

        mask_ptr = mask if has_mask else U  # dummy, gated by HAS_MASK

        common = dict(
            N=N, D_LOW=d_low, NUM_BINS=num_bins, PAD_BINS=ctx.pad_bins,
            dist_min=ctx.dist_min, inv_bin_width=inv_bin_width,
            stride_ub=U.stride(0), stride_un=U.stride(1), stride_ud=U.stride(2),
            stride_xb=x_true.stride(0), stride_xn=x_true.stride(1),
            stride_wb=w_bin_padded.stride(0), stride_wd=w_bin_padded.stride(1),
            stride_mb=mask.stride(0) if has_mask else 0,
            BLOCK_I=ctx.BLOCK_I, BLOCK_J=ctx.BLOCK_J,
            HAS_MASK=has_mask,
        )

        grid_i = (B, triton.cdiv(N, ctx.BLOCK_I))
        _distogram_bwd_dU_kernel[grid_i](
            U, V, x_true, w_bin_padded, bias_padded, mask_ptr,
            count, dU, dW, dbias, grad_scale, **common,
        )

        grid_j = (B, triton.cdiv(N, ctx.BLOCK_J))
        _distogram_bwd_dV_kernel[grid_j](
            U, V, x_true, w_bin_padded, bias_padded, mask_ptr,
            count, dV, grad_scale, **common,
        )

        return dU, dV, dW[:num_bins], dbias[:num_bins], None, None, None, None, None, None, None


# ============================================================================
# Public API
# ============================================================================


def triton_distogram_loss(
    U: torch.Tensor,
    V: torch.Tensor,
    w_bin: torch.Tensor,
    bias: torch.Tensor,
    target_bins: torch.Tensor,
    x_true: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    dist_min: float = 2.0,
    bin_width: float = 0.5,
    BLOCK_I: int = 16,
    BLOCK_J: int = 16,
) -> torch.Tensor:
    """
    Triton-fused distogram loss with optional autograd support.

    When any input requires grad AND x_true is provided, uses autograd.Function
    with O(N) memory backward. Otherwise uses the forward-only kernel.

    Returns scalar loss (mean cross-entropy over all valid pairs).
    """
    unbatched = U.dim() == 2
    if unbatched:
        U = U.unsqueeze(0)
        V = V.unsqueeze(0)
        target_bins = target_bins.unsqueeze(0)
        if x_true is not None:
            x_true = x_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in [U, V, w_bin, bias]
    )

    if needs_grad and x_true is not None:
        return DistogramFunction.apply(
            U, V, w_bin, bias, target_bins, x_true, mask,
            dist_min, bin_width, BLOCK_I, BLOCK_J,
        )

    # Inference-only forward
    B, N, d_low = U.shape
    num_bins = w_bin.shape[0]
    pad_bins = _next_power_of_2(num_bins)

    U = U.contiguous().float()
    V = V.contiguous().float()
    target_bins = target_bins.contiguous().long()
    w_bin_padded, bias_padded = _pad_bin_params(w_bin, bias, num_bins, pad_bins, d_low, U.device)

    loss, _ = _launch_fwd(U, V, w_bin_padded, bias_padded, target_bins, mask,
                          N, d_low, num_bins, pad_bins, BLOCK_I, BLOCK_J)
    return loss
