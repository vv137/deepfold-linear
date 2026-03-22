"""
Distogram tiling Triton kernel (SPEC §18 item 3).

Fuses the nested Python tile loops from distogram loss §11.3:
  Z_ij = U_i * V_j   (Hadamard interaction in d_low space)
  logits_ij = W_bin @ Z_ij  (project to bins)
  loss += cross_entropy(logits_ij, target_ij)

Uses tl.dot for the Z @ W_bin^T matmul. For each row i in the tile,
we broadcast-multiply U[i] * V_tile to get z_row (BLOCK_J, D_LOW),
then compute logits = z_row @ W_bin^T via tl.dot (BLOCK_J, PAD_BINS).
Cross-entropy is fused in the same kernel with max-subtraction for
numerical stability.

num_bins is padded to PAD_BINS (next multiple of 16) for tl.dot
compatibility. Padded bins get bias = -1e30 so they contribute
negligibly to logsumexp.
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x (Triton requires power-of-2 ranges)."""
    p = 1
    while p < x:
        p *= 2
    return p


@triton.jit
def _distogram_fwd_kernel(
    # U: (N, D_LOW), V: (N, D_LOW)
    U_ptr,
    V_ptr,
    # W_bin_padded: (PAD_BINS, D_LOW), bias_padded: (PAD_BINS,)
    W_BIN_ptr,
    BIAS_ptr,
    # Target bins: (N, N) int
    TARGET_ptr,
    # Output: loss_sum (scalar), count (scalar)
    LOSS_ptr,
    COUNT_ptr,
    # Dims
    N: tl.constexpr,
    D_LOW: tl.constexpr,
    NUM_BINS: tl.constexpr,
    PAD_BINS: tl.constexpr,
    # Strides
    stride_un: tl.constexpr,
    stride_ud: tl.constexpr,
    stride_wb: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tm: tl.constexpr,
    # Tile
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Tiled distogram loss: Hadamard interaction + tl.dot matmul + cross-entropy."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J

    j_idx = j_start + tl.arange(0, BLOCK_J)  # (BLOCK_J,)
    j_mask = j_idx < N

    d_idx = tl.arange(0, D_LOW)  # (D_LOW,)
    b_idx = tl.arange(0, PAD_BINS)  # (PAD_BINS,)

    # Load V_tile: (BLOCK_J, D_LOW)
    v_tile = tl.load(
        V_ptr + j_idx[:, None] * stride_un + d_idx[None, :] * stride_ud,
        mask=j_mask[:, None],
        other=0.0,
    )

    # Load W_bin^T: (D_LOW, PAD_BINS) — transposed for tl.dot
    # W_bin is stored as (PAD_BINS, D_LOW), so W_bin^T[d, b] = W_bin[b, d]
    w_bin_t = tl.load(
        W_BIN_ptr + b_idx[None, :] * stride_wb + d_idx[:, None] * stride_wd,
    )  # (D_LOW, PAD_BINS)

    # Load bias: (PAD_BINS,)
    bias = tl.load(BIAS_ptr + b_idx)  # (PAD_BINS,)

    # Accumulate tile loss and count
    tile_loss = tl.zeros([], dtype=tl.float32)
    tile_count = tl.zeros([], dtype=tl.float32)

    for i_off in tl.static_range(0, BLOCK_I):
        i = i_start + i_off
        i_valid = i < N

        # Load U[i, :]: (D_LOW,)
        u_row = tl.load(
            U_ptr + i * stride_un + d_idx * stride_ud,
            mask=i_valid,
            other=0.0,
        )  # (D_LOW,)

        # z_row = U[i] * V_tile: broadcast (D_LOW,) * (BLOCK_J, D_LOW) → (BLOCK_J, D_LOW)
        z_row = v_tile * u_row[None, :]  # (BLOCK_J, D_LOW)

        # logits = z_row @ W_bin^T: (BLOCK_J, D_LOW) @ (D_LOW, PAD_BINS) → (BLOCK_J, PAD_BINS)
        logits = tl.dot(z_row, w_bin_t)  # (BLOCK_J, PAD_BINS)
        logits = logits + bias[None, :]  # add bias

        # Load targets for row i: (BLOCK_J,)
        targets = tl.load(
            TARGET_ptr + i * stride_tn + j_idx * stride_tm,
            mask=j_mask & i_valid,
            other=0,
        )  # (BLOCK_J,) int

        # --- Numerically stable cross-entropy ---
        # Max logit per (i, j) pair for stability (over PAD_BINS)
        max_logit = tl.max(logits, axis=1)  # (BLOCK_J,)

        # shifted logits: (BLOCK_J, PAD_BINS)
        shifted = logits - max_logit[:, None]

        # logsumexp: (BLOCK_J,)
        sum_exp = tl.sum(tl.exp(shifted), axis=1)  # (BLOCK_J,)
        log_sum_exp = max_logit + tl.log(sum_exp + 1e-10)  # (BLOCK_J,)

        # Gather target logit: logits[j, targets[j]] for each j
        # Create one-hot mask: (BLOCK_J, PAD_BINS) where [j, targets[j]] = 1
        target_one_hot = b_idx[None, :] == targets[:, None]  # (BLOCK_J, PAD_BINS)
        target_logit = tl.sum(
            tl.where(target_one_hot, logits, 0.0), axis=1
        )  # (BLOCK_J,)

        # CE = -target_logit + logsumexp
        ce = -target_logit + log_sum_exp  # (BLOCK_J,)

        # Mask out invalid pairs
        valid = j_mask & i_valid
        ce = tl.where(valid, ce, 0.0)

        tile_loss += tl.sum(ce)
        tile_count += tl.sum(valid.to(tl.float32))

    # Atomic add to global accumulators
    tl.atomic_add(LOSS_ptr, tile_loss)
    tl.atomic_add(COUNT_ptr, tile_count)


def triton_distogram_loss(
    U: torch.Tensor,  # (N, d_low)
    V: torch.Tensor,  # (N, d_low)
    w_bin: torch.Tensor,  # (num_bins, d_low)
    bias: torch.Tensor,  # (num_bins,)
    target_bins: torch.Tensor,  # (N, N) int
    BLOCK_I: int = 16,
    BLOCK_J: int = 16,
) -> torch.Tensor:
    """
    Triton-fused distogram loss.
    Returns scalar loss (mean cross-entropy over all pairs).
    """
    N, d_low = U.shape
    num_bins = w_bin.shape[0]
    pad_bins = _next_power_of_2(num_bins)

    U = U.contiguous().float()
    V = V.contiguous().float()
    w_bin = w_bin.contiguous().float()
    bias = bias.contiguous().float()
    target_bins = target_bins.contiguous().long()

    # Pad W_bin and bias to PAD_BINS (next multiple of 16).
    # Extra bins get bias = -1e30 so exp(logit) ≈ 0 in logsumexp.
    if pad_bins > num_bins:
        w_bin_padded = torch.zeros(
            pad_bins, d_low, device=w_bin.device, dtype=w_bin.dtype
        )
        w_bin_padded[:num_bins] = w_bin
        # Extra rows of W stay zero; with bias=-1e30, logits for
        # padded bins are ≈ -1e30 regardless of input.
        bias_padded = torch.full(
            (pad_bins,), -1e30, device=bias.device, dtype=bias.dtype
        )
        bias_padded[:num_bins] = bias
    else:
        w_bin_padded = w_bin
        bias_padded = bias

    loss_sum = torch.zeros(1, device=U.device, dtype=torch.float32)
    count = torch.zeros(1, device=U.device, dtype=torch.float32)

    grid = (
        (N + BLOCK_I - 1) // BLOCK_I,
        (N + BLOCK_J - 1) // BLOCK_J,
    )
    _distogram_fwd_kernel[grid](
        U,
        V,
        w_bin_padded,
        bias_padded,
        target_bins,
        loss_sum,
        count,
        N,
        d_low,
        num_bins,
        pad_bins,
        U.stride(0),
        U.stride(1),
        w_bin_padded.stride(0),
        w_bin_padded.stride(1),
        target_bins.stride(0),
        target_bins.stride(1),
        BLOCK_I=BLOCK_I,
        BLOCK_J=BLOCK_J,
    )

    return loss_sum / count.clamp(min=1)
