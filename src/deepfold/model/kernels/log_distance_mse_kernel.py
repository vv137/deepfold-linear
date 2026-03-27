"""
Triton fused log-distance MSE kernel (fwd + bwd).

L = mean_{i≠j, valid} [log(d_pred_ij + 1) - log(d_true_ij + 1)]^2

Forward:  grid (B, n_tiles_i, n_tiles_j) — fused pairwise distance + log1p + MSE.
Backward: grid (B, n_tiles_i) — each program owns one i-block, loops over j.

Zero O(N²) storage: distances recomputed in backward from coordinates.
"""

import torch
import triton
import triton.language as tl


# ── Forward kernel ──────────────────────────────────────────────────────────

@triton.jit
def _log_dist_mse_fwd_kernel(
    X_PRED, X_TRUE, MASK,
    LOSS_OUT, COUNT_OUT,
    N,
    stride_xb, stride_xn,
    stride_mb,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    i_idx = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_idx = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    i_valid = i_idx < N
    j_valid = j_idx < N

    # Diagonal exclusion + bounds mask
    pair_valid = (i_idx[:, None] != j_idx[None, :]) & i_valid[:, None] & j_valid[None, :]

    if HAS_MASK:
        m_i = tl.load(MASK + pid_b * stride_mb + i_idx, mask=i_valid, other=0.0)
        m_j = tl.load(MASK + pid_b * stride_mb + j_idx, mask=j_valid, other=0.0)
        pair_valid = pair_valid & (m_i[:, None] > 0.0) & (m_j[None, :] > 0.0)

    # Load x_pred (B, N, 3)
    xp_base = X_PRED + pid_b * stride_xb
    xp_i0 = tl.load(xp_base + i_idx * stride_xn + 0, mask=i_valid, other=0.0)
    xp_i1 = tl.load(xp_base + i_idx * stride_xn + 1, mask=i_valid, other=0.0)
    xp_i2 = tl.load(xp_base + i_idx * stride_xn + 2, mask=i_valid, other=0.0)
    xp_j0 = tl.load(xp_base + j_idx * stride_xn + 0, mask=j_valid, other=0.0)
    xp_j1 = tl.load(xp_base + j_idx * stride_xn + 1, mask=j_valid, other=0.0)
    xp_j2 = tl.load(xp_base + j_idx * stride_xn + 2, mask=j_valid, other=0.0)

    dp0 = xp_i0[:, None] - xp_j0[None, :]
    dp1 = xp_i1[:, None] - xp_j1[None, :]
    dp2 = xp_i2[:, None] - xp_j2[None, :]
    d_pred = tl.sqrt(dp0 * dp0 + dp1 * dp1 + dp2 * dp2 + 1e-8)

    # Load x_true (B, N, 3)
    xt_base = X_TRUE + pid_b * stride_xb
    xt_i0 = tl.load(xt_base + i_idx * stride_xn + 0, mask=i_valid, other=0.0)
    xt_i1 = tl.load(xt_base + i_idx * stride_xn + 1, mask=i_valid, other=0.0)
    xt_i2 = tl.load(xt_base + i_idx * stride_xn + 2, mask=i_valid, other=0.0)
    xt_j0 = tl.load(xt_base + j_idx * stride_xn + 0, mask=j_valid, other=0.0)
    xt_j1 = tl.load(xt_base + j_idx * stride_xn + 1, mask=j_valid, other=0.0)
    xt_j2 = tl.load(xt_base + j_idx * stride_xn + 2, mask=j_valid, other=0.0)

    dt0 = xt_i0[:, None] - xt_j0[None, :]
    dt1 = xt_i1[:, None] - xt_j1[None, :]
    dt2 = xt_i2[:, None] - xt_j2[None, :]
    d_true = tl.sqrt(dt0 * dt0 + dt1 * dt1 + dt2 * dt2 + 1e-8)

    # log1p distance MSE
    log_pred = tl.log(d_pred + 1.0)
    log_true = tl.log(d_true + 1.0)
    diff = log_pred - log_true
    sq = diff * diff

    # Masked sum
    sq_masked = tl.where(pair_valid, sq, 0.0)
    tile_loss = tl.sum(sq_masked)
    tile_count = tl.sum(pair_valid.to(tl.float32))

    tl.atomic_add(LOSS_OUT + pid_b, tile_loss)
    tl.atomic_add(COUNT_OUT + pid_b, tile_count)


# ── Backward kernel ─────────────────────────────────────────────────────────

@triton.jit
def _log_dist_mse_bwd_kernel(
    X_PRED, X_TRUE, MASK,
    GRAD_X,
    GRAD_SCALE,  # (B,) = 2 * grad_output / count
    N,
    stride_xb, stride_xn,
    stride_gb, stride_gn,
    stride_mb,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Each program owns BLOCK_I rows of i, loops over all j."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    i_idx = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_valid = i_idx < N

    scale = tl.load(GRAD_SCALE + pid_b)

    # Load x_pred[i], x_true[i] — kept in registers across j-loop
    xp_base = X_PRED + pid_b * stride_xb
    xt_base = X_TRUE + pid_b * stride_xb

    xp_i0 = tl.load(xp_base + i_idx * stride_xn + 0, mask=i_valid, other=0.0)
    xp_i1 = tl.load(xp_base + i_idx * stride_xn + 1, mask=i_valid, other=0.0)
    xp_i2 = tl.load(xp_base + i_idx * stride_xn + 2, mask=i_valid, other=0.0)

    xt_i0 = tl.load(xt_base + i_idx * stride_xn + 0, mask=i_valid, other=0.0)
    xt_i1 = tl.load(xt_base + i_idx * stride_xn + 1, mask=i_valid, other=0.0)
    xt_i2 = tl.load(xt_base + i_idx * stride_xn + 2, mask=i_valid, other=0.0)

    if HAS_MASK:
        m_i = tl.load(MASK + pid_b * stride_mb + i_idx, mask=i_valid, other=0.0)

    acc0 = tl.zeros([BLOCK_I], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_I], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_I], dtype=tl.float32)

    n_j_blocks = tl.cdiv(N, BLOCK_J)
    for j_block in tl.static_range(0, 4096):  # upper bound, break via mask
        if j_block >= n_j_blocks:
            break
        j_idx = j_block * BLOCK_J + tl.arange(0, BLOCK_J)
        j_valid = j_idx < N

        pair_valid = (i_idx[:, None] != j_idx[None, :]) & i_valid[:, None] & j_valid[None, :]

        if HAS_MASK:
            m_j = tl.load(MASK + pid_b * stride_mb + j_idx, mask=j_valid, other=0.0)
            pair_valid = pair_valid & (m_i[:, None] > 0.0) & (m_j[None, :] > 0.0)

        # x_pred[j], x_true[j]
        xp_j0 = tl.load(xp_base + j_idx * stride_xn + 0, mask=j_valid, other=0.0)
        xp_j1 = tl.load(xp_base + j_idx * stride_xn + 1, mask=j_valid, other=0.0)
        xp_j2 = tl.load(xp_base + j_idx * stride_xn + 2, mask=j_valid, other=0.0)

        xt_j0 = tl.load(xt_base + j_idx * stride_xn + 0, mask=j_valid, other=0.0)
        xt_j1 = tl.load(xt_base + j_idx * stride_xn + 1, mask=j_valid, other=0.0)
        xt_j2 = tl.load(xt_base + j_idx * stride_xn + 2, mask=j_valid, other=0.0)

        # Pairwise differences (pred)
        dp0 = xp_i0[:, None] - xp_j0[None, :]
        dp1 = xp_i1[:, None] - xp_j1[None, :]
        dp2 = xp_i2[:, None] - xp_j2[None, :]
        d_pred = tl.sqrt(dp0 * dp0 + dp1 * dp1 + dp2 * dp2 + 1e-8)

        # Pairwise differences (true)
        dt0 = xt_i0[:, None] - xt_j0[None, :]
        dt1 = xt_i1[:, None] - xt_j1[None, :]
        dt2 = xt_i2[:, None] - xt_j2[None, :]
        d_true = tl.sqrt(dt0 * dt0 + dt1 * dt1 + dt2 * dt2 + 1e-8)

        r = tl.log(d_pred + 1.0) - tl.log(d_true + 1.0)

        # coeff = scale * r / ((d_pred + 1) * d_pred)
        # d_pred >= sqrt(1e-8) ≈ 1e-4, so denominator is safe
        coeff = scale * r / ((d_pred + 1.0) * d_pred)
        coeff = tl.where(pair_valid, coeff, 0.0)

        # Accumulate: grad_x[i] += coeff * (x_pred[i] - x_pred[j])
        acc0 += tl.sum(coeff * dp0, axis=1)
        acc1 += tl.sum(coeff * dp1, axis=1)
        acc2 += tl.sum(coeff * dp2, axis=1)

    # Store grad
    g_base = GRAD_X + pid_b * stride_gb
    tl.store(g_base + i_idx * stride_gn + 0, acc0, mask=i_valid)
    tl.store(g_base + i_idx * stride_gn + 1, acc1, mask=i_valid)
    tl.store(g_base + i_idx * stride_gn + 2, acc2, mask=i_valid)


# ── Autograd wrapper ────────────────────────────────────────────────────────

# Tile sizes — 64 is good for N up to ~4096 tokens
_BLOCK_I = 64
_BLOCK_J = 64


class _LogDistMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_pred, x_true, mask):
        """
        Args:
            x_pred: (B, N, 3) float32 predicted coordinates
            x_true: (B, N, 3) float32 ground truth coordinates
            mask:   (B, N) float32 or None — 1=valid, 0=pad
        Returns:
            scalar loss (mean over batch)
        """
        B, N, _ = x_pred.shape
        has_mask = mask is not None

        loss_sum = torch.zeros(B, device=x_pred.device, dtype=torch.float32)
        count = torch.zeros(B, device=x_pred.device, dtype=torch.float32)

        grid = (B, triton.cdiv(N, _BLOCK_I), triton.cdiv(N, _BLOCK_J))
        _log_dist_mse_fwd_kernel[grid](
            x_pred, x_true,
            mask if has_mask else x_pred,  # dummy when no mask
            loss_sum, count,
            N,
            x_pred.stride(0), x_pred.stride(1),
            mask.stride(0) if has_mask else 0,
            BLOCK_I=_BLOCK_I, BLOCK_J=_BLOCK_J,
            HAS_MASK=has_mask,
        )

        # Per-batch mean, then mean over batch
        safe_count = count.clamp(min=1.0)
        per_batch = loss_sum / safe_count  # (B,)
        loss = per_batch.mean()

        # Save for backward
        ctx.save_for_backward(x_pred, x_true, mask, safe_count)
        ctx.B = B
        ctx.N = N
        ctx.has_mask = has_mask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_pred, x_true, mask, safe_count = ctx.saved_tensors
        B, N = ctx.B, ctx.N
        has_mask = ctx.has_mask

        # grad_scale = 2 * grad_output / (B * count_b) for each batch element
        grad_scale = (2.0 * grad_output / B) / safe_count  # (B,)

        grad_x = torch.empty_like(x_pred)

        grid = (B, triton.cdiv(N, _BLOCK_I))
        _log_dist_mse_bwd_kernel[grid](
            x_pred, x_true,
            mask if has_mask else x_pred,
            grad_x,
            grad_scale,
            N,
            x_pred.stride(0), x_pred.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            mask.stride(0) if has_mask else 0,
            BLOCK_I=_BLOCK_I, BLOCK_J=_BLOCK_J,
            HAS_MASK=has_mask,
        )

        return grad_x, None, None


def triton_log_distance_mse(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused log-distance MSE loss via Triton.

    L = mean_{b} mean_{i≠j, valid} [log(d_pred_ij + 1) - log(d_true_ij + 1)]^2

    Args:
        x_pred: (B, N, 3) predicted token coordinates (gradient flows here)
        x_true: (B, N, 3) ground truth token coordinates (no gradient)
        mask:   (B, N) float, 1=valid 0=pad. None means all valid.

    Returns:
        Scalar loss.
    """
    x_pred_f = x_pred.contiguous().float()
    x_true_f = x_true.detach().contiguous().float()
    mask_f = mask.contiguous().float() if mask is not None else None
    return _LogDistMSE.apply(x_pred_f, x_true_f, mask_f)
