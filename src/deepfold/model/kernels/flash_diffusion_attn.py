"""
Flash Diffusion Attention — Triton tiled kernel with 68-bin position bias.

O(N) memory flash attention for the diffusion transformer's token self-attention.
Position bias is gathered per-tile from (H, 68) weight + (N, N) int32 bins,
never materialized as a full (H, N, N) float tensor.

Architecture:
  Forward:  tiled flash attention with online softmax (Milakov & Gimelshein)
  Backward: recomputation-based (flash attention v2 style)

Grid: (B*H, ceil(N/TILE_Q)) for full parallelism across batch, heads, and query tiles.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Forward kernel
# ============================================================================


@triton.jit
def _flash_diff_attn_fwd_kernel(
    Q_ptr,          # (B, H, N, D)
    K_ptr,          # (B, H, N, D)
    V_ptr,          # (B, H, N, D)
    O_ptr,          # (B, H, N, D) output
    LSE_ptr,        # (B, H, N) log-sum-exp for backward
    POS_W_ptr,      # (H, 68) position bias weights
    POS_BINS_ptr,   # (B, N, N) int32 bin indices
    MASK_ptr,       # (B, N) float32, 1=valid 0=pad
    scale,          # 1/sqrt(d_h)
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_bins_b: tl.constexpr,
    stride_bins_n: tl.constexpr,
    stride_mask_b: tl.constexpr,
    NUM_BINS: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Grid: (B*H, ceil(N/TILE_Q))
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query tile indices
    q_start = pid_q * TILE_Q
    q_idx = q_start + tl.arange(0, TILE_Q)  # (TILE_Q,)
    q_mask = q_idx < N

    # Base pointers for this (batch, head)
    qkv_offset = pid_b * stride_qb + pid_h * stride_qh

    # Load Q tile: (TILE_Q, D)
    q_ptrs = Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Load query-side padding mask
    q_pad_ptrs = MASK_ptr + pid_b * stride_mask_b + q_idx
    q_pad = tl.load(q_pad_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators
    m_i = tl.full([TILE_Q], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([TILE_Q], dtype=tl.float32)                # running sum(exp)
    acc = tl.zeros([TILE_Q, D], dtype=tl.float32)             # running weighted V sum

    # Position bias base pointer for this head: (68,)
    pos_w_base = POS_W_ptr + pid_h * NUM_BINS

    # pos_bins base pointer for this batch: (N, N)
    bins_base = POS_BINS_ptr + pid_b * stride_bins_b

    # Key mask base pointer
    mask_base = MASK_ptr + pid_b * stride_mask_b

    # Iterate over K/V tiles
    n_tiles_k = tl.cdiv(N, TILE_K)
    for t in range(n_tiles_k):
        k_start = t * TILE_K
        k_idx = k_start + tl.arange(0, TILE_K)  # (TILE_K,)
        k_mask = k_idx < N

        # Load K tile: (TILE_K, D)
        k_ptrs = K_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        # Compute scores: (TILE_Q, TILE_K)
        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Gather position bias: pos_w[pos_bins[q_idx, k_idx]]
        # bins_ptrs: (TILE_Q, TILE_K) pointers into pos_bins
        bins_ptrs = bins_base + q_idx[:, None] * stride_bins_n + k_idx[None, :]
        bin_vals = tl.load(bins_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0)
        # Gather from pos_w: (TILE_Q, TILE_K)
        pos_bias = tl.load(pos_w_base + bin_vals, mask=q_mask[:, None] & k_mask[None, :], other=0.0)
        scores = scores + pos_bias

        # Apply key padding mask: -inf for padded keys
        k_pad = tl.load(mask_base + k_idx, mask=k_mask, other=0.0)
        pad_bias = tl.where(k_pad[None, :] > 0.0, 0.0, float("-inf"))
        scores = scores + pad_bias

        # Also mask out-of-range keys
        scores = tl.where(k_mask[None, :], scores, float("-inf"))
        # Mask out-of-range queries (they stay at -inf, but their acc stays 0)
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # Correction factor for previous accumulations
        alpha = tl.exp(m_i - m_new)
        # New exponentials
        p = tl.exp(scores - m_new[:, None])

        # Load V tile: (TILE_K, D)
        v_ptrs = V_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        V_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        # Update accumulators
        acc = acc * alpha[:, None] + tl.dot(p.to(V_tile.dtype), V_tile)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Finalize: normalize by sum of exponentials
    acc = acc / l_i[:, None]

    # Zero out padded query positions
    acc = tl.where(q_pad[:, None] > 0.0, acc, 0.0)

    # Store output: (TILE_Q, D)
    o_ptrs = O_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(o_ptrs, acc.to(tl.float16 if Q_tile.dtype == tl.float16 else tl.bfloat16), mask=q_mask[:, None])

    # Store LSE for backward: log(l_i) + m_i
    lse = m_i + tl.log(l_i + 1e-8)
    lse = tl.where(q_pad > 0.0, lse, float("-inf"))
    lse_ptrs = LSE_ptr + pid_bh * N + q_idx
    tl.store(lse_ptrs, lse, mask=q_mask)


# ============================================================================
# Backward kernel — dV, dK (iterate over Q tiles for each K tile)
# ============================================================================


@triton.jit
def _flash_diff_attn_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    POS_W_ptr, POS_BINS_ptr, MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_bins_b: tl.constexpr, stride_bins_n: tl.constexpr,
    stride_mask_b: tl.constexpr,
    NUM_BINS: tl.constexpr,
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr,
):
    """Accumulate dK and dV by iterating Q tiles for each K tile."""
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    k_start = pid_k * TILE_K
    k_idx = k_start + tl.arange(0, TILE_K)
    k_mask = k_idx < N

    qkv_offset = pid_b * stride_qb + pid_h * stride_qh

    # Load K, V tiles
    k_ptrs = K_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
    v_ptrs = V_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    V_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

    k_pad = tl.load(MASK_ptr + pid_b * stride_mask_b + k_idx, mask=k_mask, other=0.0)

    pos_w_base = POS_W_ptr + pid_h * NUM_BINS
    bins_base = POS_BINS_ptr + pid_b * stride_bins_b
    mask_base = MASK_ptr + pid_b * stride_mask_b

    # Accumulators for dK, dV
    dK_acc = tl.zeros([TILE_K, D], dtype=tl.float32)
    dV_acc = tl.zeros([TILE_K, D], dtype=tl.float32)

    n_tiles_q = tl.cdiv(N, TILE_Q)
    for t in range(n_tiles_q):
        q_start = t * TILE_Q
        q_idx = q_start + tl.arange(0, TILE_Q)
        q_mask = q_idx < N

        # Load Q, O, dO, LSE for this Q tile
        q_ptrs = Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        o_ptrs = O_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        O_tile = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
        do_ptrs = dO_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        dO_tile = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
        lse_ptrs = LSE_ptr + pid_bh * N + q_idx
        lse = tl.load(lse_ptrs, mask=q_mask, other=float("-inf"))

        q_pad = tl.load(mask_base + q_idx, mask=q_mask, other=0.0)

        # Recompute scores: (TILE_Q, TILE_K)
        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Position bias
        bins_ptrs = bins_base + q_idx[:, None] * stride_bins_n + k_idx[None, :]
        bin_vals = tl.load(bins_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0)
        pos_bias = tl.load(pos_w_base + bin_vals, mask=q_mask[:, None] & k_mask[None, :], other=0.0)
        scores = scores + pos_bias

        # Key padding mask
        pad_bias = tl.where(k_pad[None, :] > 0.0, 0.0, float("-inf"))
        scores = scores + pad_bias
        scores = tl.where(k_mask[None, :], scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        # Recompute softmax: P = exp(scores - lse)
        P = tl.exp(scores - lse[:, None])
        P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

        # dV += P^T @ dO
        dV_acc += tl.dot(tl.trans(P.to(dO_tile.dtype)), dO_tile)

        # dP = dO @ V^T
        dP = tl.dot(dO_tile, tl.trans(V_tile))

        # D_i = sum(dO * O, axis=-1)
        D_i = tl.sum(dO_tile * O_tile, axis=1)

        # dS = P * (dP - D_i[:, None])
        dS = P * (dP - D_i[:, None])
        dS = dS * scale

        # dK += dS^T @ Q
        dK_acc += tl.dot(tl.trans(dS.to(Q_tile.dtype)), Q_tile)

    # Store dK, dV
    dk_ptrs = dK_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(dk_ptrs, dK_acc.to(K_tile.dtype), mask=k_mask[:, None])
    dv_ptrs = dV_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(dv_ptrs, dV_acc.to(V_tile.dtype), mask=k_mask[:, None])


# ============================================================================
# Backward kernel — dQ (iterate over K tiles for each Q tile)
# ============================================================================


@triton.jit
def _flash_diff_attn_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dQ_ptr,
    POS_W_ptr, POS_BINS_ptr, MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_bins_b: tl.constexpr, stride_bins_n: tl.constexpr,
    stride_mask_b: tl.constexpr,
    NUM_BINS: tl.constexpr,
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr,
):
    """Accumulate dQ by iterating K tiles for each Q tile."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_start = pid_q * TILE_Q
    q_idx = q_start + tl.arange(0, TILE_Q)
    q_mask = q_idx < N

    qkv_offset = pid_b * stride_qb + pid_h * stride_qh

    # Load Q, O, dO, LSE
    q_ptrs = Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    o_ptrs = O_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    O_tile = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = dO_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    dO_tile = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + pid_bh * N + q_idx, mask=q_mask, other=float("-inf"))

    q_pad = tl.load(MASK_ptr + pid_b * stride_mask_b + q_idx, mask=q_mask, other=0.0)

    pos_w_base = POS_W_ptr + pid_h * NUM_BINS
    bins_base = POS_BINS_ptr + pid_b * stride_bins_b

    # D_i = sum(dO * O, axis=-1)
    D_i = tl.sum(dO_tile * O_tile, axis=1)

    dQ_acc = tl.zeros([TILE_Q, D], dtype=tl.float32)

    n_tiles_k = tl.cdiv(N, TILE_K)
    for t in range(n_tiles_k):
        k_start = t * TILE_K
        k_idx = k_start + tl.arange(0, TILE_K)
        k_mask = k_idx < N

        # Load K, V
        k_ptrs = K_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v_ptrs = V_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        V_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        k_pad = tl.load(MASK_ptr + pid_b * stride_mask_b + k_idx, mask=k_mask, other=0.0)

        # Recompute scores
        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Position bias
        bins_ptrs = bins_base + q_idx[:, None] * stride_bins_n + k_idx[None, :]
        bin_vals = tl.load(bins_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0)
        pos_bias = tl.load(pos_w_base + bin_vals, mask=q_mask[:, None] & k_mask[None, :], other=0.0)
        scores = scores + pos_bias

        pad_bias = tl.where(k_pad[None, :] > 0.0, 0.0, float("-inf"))
        scores = scores + pad_bias
        scores = tl.where(k_mask[None, :], scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        P = tl.exp(scores - lse[:, None])
        P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

        # dP = dO @ V^T
        dP = tl.dot(dO_tile, tl.trans(V_tile))

        # dS = P * (dP - D_i[:, None]) * scale
        dS = P * (dP - D_i[:, None])
        dS = dS * scale

        # dQ += dS @ K
        dQ_acc += tl.dot(dS.to(K_tile.dtype), K_tile)

    # Store dQ
    dq_ptrs = dQ_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(dq_ptrs, dQ_acc.to(Q_tile.dtype), mask=q_mask[:, None])


# ============================================================================
# Backward kernel — dW_pos (gradient to position bias weights)
# ============================================================================


@triton.jit
def _flash_diff_attn_bwd_dw_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dW_ptr,         # (H, 68) output — atomically accumulated
    POS_W_ptr, POS_BINS_ptr, MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_bins_b: tl.constexpr, stride_bins_n: tl.constexpr,
    stride_mask_b: tl.constexpr,
    NUM_BINS: tl.constexpr,
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr,
):
    """Accumulate dW_pos[h, bin] = sum over (b, i, j) of dS[b,h,i,j] where pos_bins[b,i,j]==bin."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_start = pid_q * TILE_Q
    q_idx = q_start + tl.arange(0, TILE_Q)
    q_mask = q_idx < N

    qkv_offset = pid_b * stride_qb + pid_h * stride_qh

    # Load Q, O, dO, LSE
    q_ptrs = Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    o_ptrs = O_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    O_tile = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = dO_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    dO_tile = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + pid_bh * N + q_idx, mask=q_mask, other=float("-inf"))
    q_pad = tl.load(MASK_ptr + pid_b * stride_mask_b + q_idx, mask=q_mask, other=0.0)

    pos_w_base = POS_W_ptr + pid_h * NUM_BINS
    bins_base = POS_BINS_ptr + pid_b * stride_bins_b
    D_i = tl.sum(dO_tile * O_tile, axis=1)

    n_tiles_k = tl.cdiv(N, TILE_K)
    for t in range(n_tiles_k):
        k_start = t * TILE_K
        k_idx = k_start + tl.arange(0, TILE_K)
        k_mask = k_idx < N

        k_ptrs = K_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v_ptrs = V_ptr + qkv_offset + k_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
        V_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        k_pad = tl.load(MASK_ptr + pid_b * stride_mask_b + k_idx, mask=k_mask, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        bins_ptrs = bins_base + q_idx[:, None] * stride_bins_n + k_idx[None, :]
        bin_vals = tl.load(bins_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0)
        pos_bias = tl.load(pos_w_base + bin_vals, mask=q_mask[:, None] & k_mask[None, :], other=0.0)
        scores = scores + pos_bias

        pad_bias = tl.where(k_pad[None, :] > 0.0, 0.0, float("-inf"))
        scores = scores + pad_bias
        scores = tl.where(k_mask[None, :], scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        P = tl.exp(scores - lse[:, None])
        P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

        dP = tl.dot(dO_tile, tl.trans(V_tile))
        dS = P * (dP - D_i[:, None])  # gradient through softmax

        # Scatter-add dS into dW by bin index using tl.atomic_add
        # Process row by row to use atomic_add on 1D slices
        valid_tile = q_mask[:, None] & k_mask[None, :]
        dS_masked = tl.where(valid_tile, dS, 0.0)

        dw_base = dW_ptr + pid_h * NUM_BINS
        # Atomic add each element — Triton supports per-element atomic
        tl.atomic_add(dw_base + bin_vals, dS_masked, mask=valid_tile)


# ============================================================================
# Python reference (for testing)
# ============================================================================


def flash_diff_attn_ref(Q, K, V, pos_weight, pos_bins, mask=None):
    """Pure PyTorch reference implementation for testing.

    Args:
        Q, K, V: (B, H, N, D) float
        pos_weight: (H, 68) float
        pos_bins: (B, N, N) int32
        mask: (B, N) float, 1=valid 0=pad

    Returns:
        O: (B, H, N, D) float
    """
    B, H, N, D = Q.shape
    scale = D ** -0.5
    scores = torch.einsum("bhid,bhjd->bhij", Q, K) * scale  # (B, H, N, N)

    # Position bias: gather from (H, 68) using (B, N, N) bins
    pos_bias = pos_weight[:, pos_bins.long()]  # (H, B, N, N)
    pos_bias = pos_bias.permute(1, 0, 2, 3)   # (B, H, N, N)
    scores = scores + pos_bias

    if mask is not None:
        # Key mask: (B, 1, 1, N) — mask padded keys
        scores = scores + (1 - mask[:, None, None, :]) * (-1e9)
        # Query mask: zero out padded query rows after softmax
        q_pad = mask[:, None, :, None]  # (B, 1, N, 1)

    attn = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn = attn * q_pad
    attn = attn.nan_to_num(0.0)

    out = torch.einsum("bhij,bhjd->bhid", attn, V)
    return out


# ============================================================================
# autograd.Function wrapper
# ============================================================================


TILE_Q = 64
TILE_K = 64


class FlashDiffusionAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, pos_weight, pos_bins, mask):
        """
        Args:
            Q, K, V: (B, H, N, D) contiguous float16/bfloat16
            pos_weight: (H, NUM_BINS) float32
            pos_bins: (B, N, N) int32
            mask: (B, N) float32, 1=valid 0=pad

        Returns:
            O: (B, H, N, D)
        """
        B, H, N, D = Q.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert pos_bins.is_contiguous() and pos_bins.dtype == torch.int32
        NUM_BINS = pos_weight.shape[1]

        O = torch.empty_like(Q)
        LSE = torch.empty(B * H, N, device=Q.device, dtype=torch.float32)

        scale = D ** -0.5
        grid = (B * H, triton.cdiv(N, TILE_Q))

        _flash_diff_attn_fwd_kernel[grid](
            Q, K, V, O, LSE,
            pos_weight, pos_bins, mask,
            scale,
            B, H, N, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            pos_bins.stride(0), pos_bins.stride(1),
            mask.stride(0),
            NUM_BINS,
            TILE_Q, TILE_K,
        )

        ctx.save_for_backward(Q, K, V, O, LSE, pos_weight, pos_bins, mask)
        ctx.scale = scale
        ctx.B = B
        ctx.H = H
        ctx.N = N
        ctx.D = D
        ctx.NUM_BINS = NUM_BINS
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE, pos_weight, pos_bins, mask = ctx.saved_tensors
        B, H, N, D = ctx.B, ctx.H, ctx.N, ctx.D
        scale = ctx.scale
        NUM_BINS = ctx.NUM_BINS

        dO = dO.contiguous()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dW = torch.zeros_like(pos_weight)  # (H, NUM_BINS)

        common_args = dict(
            B=B, H=H, N=N, D=D,
            stride_qb=Q.stride(0), stride_qh=Q.stride(1), stride_qn=Q.stride(2),
            stride_bins_b=pos_bins.stride(0), stride_bins_n=pos_bins.stride(1),
            stride_mask_b=mask.stride(0),
            NUM_BINS=NUM_BINS,
            TILE_Q=TILE_Q, TILE_K=TILE_K,
        )

        grid_k = (B * H, triton.cdiv(N, TILE_K))
        grid_q = (B * H, triton.cdiv(N, TILE_Q))

        # dK, dV
        _flash_diff_attn_bwd_dkdv_kernel[grid_k](
            Q, K, V, O, LSE, dO, dK, dV,
            pos_weight, pos_bins, mask, scale,
            **common_args,
        )

        # dQ
        _flash_diff_attn_bwd_dq_kernel[grid_q](
            Q, K, V, O, LSE, dO, dQ,
            pos_weight, pos_bins, mask, scale,
            **common_args,
        )

        # dW_pos
        _flash_diff_attn_bwd_dw_kernel[grid_q](
            Q, K, V, O, LSE, dO, dW,
            pos_weight, pos_bins, mask, scale,
            **common_args,
        )

        return dQ, dK, dV, dW, None, None  # no grad for pos_bins, mask


def flash_diffusion_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    pos_weight: torch.Tensor,
    pos_bins: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flash diffusion attention with 68-bin position bias.

    Args:
        Q, K, V: (B, H, N, D) or (H, N, D)
        pos_weight: (H, NUM_BINS) learnable position bias weights
        pos_bins: (B, N, N) or (N, N) int32 bin indices
        mask: (B, N) or (N,) float, 1=valid 0=pad. None=all valid.

    Returns:
        O: same shape as Q
    """
    unbatched = Q.dim() == 3
    if unbatched:
        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)
        pos_bins = pos_bins.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, H, N, D = Q.shape
    if mask is None:
        mask = Q.new_ones(B, N, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    pos_bins = pos_bins.contiguous().to(torch.int32)
    pos_weight = pos_weight.contiguous().float()
    mask = mask.contiguous().float()

    O = FlashDiffusionAttnFn.apply(Q, K, V, pos_weight, pos_bins, mask)

    if unbatched:
        O = O.squeeze(0)
    return O
