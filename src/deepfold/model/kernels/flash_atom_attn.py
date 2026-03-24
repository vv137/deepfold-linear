"""
Flash Windowed Atom Attention — Triton kernel (W=32 queries, H=128 keys).

Boltz-1-style windowed attention for atom-level self-attention in the diffusion
encoder/decoder. Each window of W=32 query atoms attends to H=128 key atoms
(centered: the 32 query atoms + 48 neighbors on each side).

Since W×H = 32×128 = 4096 is small, the full score matrix fits in SRAM.
No online softmax needed — direct materialization within each window.

Supports:
  - Intra-window pair bias from p_lm (atom pair features)
  - Atom padding mask
  - Forward and backward passes

Grid: (B*n_heads, K) where K = ceil(M/W) windows.
"""

import torch
import triton
import triton.language as tl

W_QUERY = 32   # queries per window
W_KEY = 128    # keys per window (centered around queries)


# ============================================================================
# Forward kernel
# ============================================================================


@triton.jit
def _flash_atom_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    MASK_ptr,       # (B, M) float, 1=valid 0=pad
    scale,
    B: tl.constexpr,
    n_heads: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_mask_b: tl.constexpr,
    W_Q: tl.constexpr,    # 32
    W_K: tl.constexpr,    # 128
):
    # Grid: (B * n_heads, n_windows)
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    # Query window: [q_start, q_start + W_Q)
    q_start = pid_w * W_Q
    q_idx = q_start + tl.arange(0, W_Q)  # (W_Q,)
    q_mask = q_idx < M

    # Key window: centered around query window
    # key_start = q_start - (W_K - W_Q) // 2
    half_extra = (W_K - W_Q) // 2  # 48
    k_start = q_start - half_extra
    k_idx = k_start + tl.arange(0, W_K)  # (W_K,)
    k_mask = (k_idx >= 0) & (k_idx < M)

    # Base offset for this (batch, head)
    qkv_offset = pid_b * stride_qb + pid_h * stride_qh

    # Load Q: (W_Q, D)
    q_ptrs = Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Load K: (W_K, D)
    # Clamp k_idx to valid range for loading (masked-out positions will be ignored)
    k_idx_safe = tl.where(k_mask, k_idx, 0)
    k_ptrs = K_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + tl.arange(0, D)[None, :]
    K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

    # Load V: (W_K, D)
    v_ptrs = V_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + tl.arange(0, D)[None, :]
    V_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

    # Compute scores: (W_Q, W_K)
    scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale

    # Atom padding mask
    mask_base = MASK_ptr + pid_b * stride_mask_b
    q_pad = tl.load(mask_base + q_idx, mask=q_mask, other=0.0)
    k_pad = tl.load(mask_base + k_idx_safe, mask=k_mask, other=0.0)

    # Mask: -inf for padded keys and out-of-range keys
    scores = tl.where(k_mask[None, :] & (k_pad[None, :] > 0.0), scores, float("-inf"))
    scores = tl.where(q_mask[:, None], scores, float("-inf"))

    # Softmax
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    exp_scores = tl.exp(scores)
    row_sum = tl.sum(exp_scores, axis=1) + 1e-8

    # Output: (W_Q, D)
    out = tl.dot(exp_scores.to(V_tile.dtype), V_tile)
    out = out / row_sum[:, None]

    # Zero out padded queries
    out = tl.where(q_pad[:, None] > 0.0, out, 0.0)

    # Store
    o_ptrs = O_ptr + qkv_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(o_ptrs, out.to(Q_tile.dtype), mask=q_mask[:, None])


# ============================================================================
# Backward kernel — dQ, dK, dV in a single pass per window
# ============================================================================


@triton.jit
def _flash_atom_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    MASK_ptr,
    scale,
    B: tl.constexpr,
    n_heads: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_mask_b: tl.constexpr,
    W_Q: tl.constexpr,
    W_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    q_start = pid_w * W_Q
    q_idx = q_start + tl.arange(0, W_Q)
    q_mask = q_idx < M

    half_extra = (W_K - W_Q) // 2
    k_start = q_start - half_extra
    k_idx = k_start + tl.arange(0, W_K)
    k_mask = (k_idx >= 0) & (k_idx < M)
    k_idx_safe = tl.where(k_mask, k_idx, 0)

    qkv_offset = pid_b * stride_qb + pid_h * stride_qh
    mask_base = MASK_ptr + pid_b * stride_mask_b

    # Load Q, K, V, O, dO
    d_range = tl.arange(0, D)
    Q_tile = tl.load(Q_ptr + qkv_offset + q_idx[:, None] * stride_qn + d_range[None, :],
                      mask=q_mask[:, None], other=0.0)
    K_tile = tl.load(K_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + d_range[None, :],
                      mask=k_mask[:, None], other=0.0)
    V_tile = tl.load(V_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + d_range[None, :],
                      mask=k_mask[:, None], other=0.0)
    O_tile = tl.load(O_ptr + qkv_offset + q_idx[:, None] * stride_qn + d_range[None, :],
                      mask=q_mask[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + qkv_offset + q_idx[:, None] * stride_qn + d_range[None, :],
                       mask=q_mask[:, None], other=0.0)

    q_pad = tl.load(mask_base + q_idx, mask=q_mask, other=0.0)
    k_pad = tl.load(mask_base + k_idx_safe, mask=k_mask, other=0.0)

    # Recompute attention
    scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale
    scores = tl.where(k_mask[None, :] & (k_pad[None, :] > 0.0), scores, float("-inf"))
    scores = tl.where(q_mask[:, None], scores, float("-inf"))

    row_max = tl.max(scores, axis=1)
    exp_scores = tl.exp(scores - row_max[:, None])
    row_sum = tl.sum(exp_scores, axis=1) + 1e-8
    P = exp_scores / row_sum[:, None]
    P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

    # D_i = sum(dO * O, axis=-1)
    D_i = tl.sum(dO_tile * O_tile, axis=1)

    # dP = dO @ V^T: (W_Q, W_K)
    dP = tl.dot(dO_tile, tl.trans(V_tile))

    # dS = P * (dP - D_i) * scale
    dS = P * (dP - D_i[:, None]) * scale

    # dQ = dS @ K: (W_Q, D)
    dQ_tile = tl.dot(dS.to(K_tile.dtype), K_tile)

    # dK = dS^T @ Q: (W_K, D)
    dK_tile = tl.dot(tl.trans(dS.to(Q_tile.dtype)), Q_tile)

    # dV = P^T @ dO: (W_K, D)
    dV_tile = tl.dot(tl.trans(P.to(dO_tile.dtype)), dO_tile)

    # Store dQ (direct write — each query belongs to exactly one window)
    dq_ptrs = dQ_ptr + qkv_offset + q_idx[:, None] * stride_qn + d_range[None, :]
    tl.store(dq_ptrs, dQ_tile.to(Q_tile.dtype), mask=q_mask[:, None])

    # dK and dV: keys may overlap between windows, need atomic_add
    dk_ptrs = dK_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + d_range[None, :]
    tl.atomic_add(dk_ptrs, dK_tile.to(K_tile.dtype), mask=k_mask[:, None])

    dv_ptrs = dV_ptr + qkv_offset + k_idx_safe[:, None] * stride_qn + d_range[None, :]
    tl.atomic_add(dv_ptrs, dV_tile.to(V_tile.dtype), mask=k_mask[:, None])


# ============================================================================
# Python reference
# ============================================================================


def flash_atom_attn_ref(Q, K, V, mask=None, w_query=W_QUERY, w_key=W_KEY):
    """Pure PyTorch windowed attention reference.

    Args:
        Q, K, V: (B, H, M, D)
        mask: (B, M) float, 1=valid 0=pad

    Returns:
        O: (B, H, M, D)
    """
    B, H, M, D = Q.shape
    scale = D ** -0.5
    half_extra = (w_key - w_query) // 2
    O = torch.zeros_like(Q)

    for w in range(0, M, w_query):
        q_start = w
        q_end = min(w + w_query, M)
        k_start = max(0, q_start - half_extra)
        k_end = min(M, q_start + w_query + half_extra)

        Q_w = Q[:, :, q_start:q_end, :]       # (B, H, w_q, D)
        K_w = K[:, :, k_start:k_end, :]       # (B, H, w_k, D)
        V_w = V[:, :, k_start:k_end, :]

        scores = torch.einsum("bhid,bhjd->bhij", Q_w, K_w) * scale

        if mask is not None:
            k_pad = mask[:, k_start:k_end]     # (B, w_k)
            scores = scores + (1 - k_pad[:, None, None, :]) * (-1e9)
            q_pad = mask[:, q_start:q_end]     # (B, w_q)

        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn * q_pad[:, None, :, None]
        attn = attn.nan_to_num(0.0)

        O[:, :, q_start:q_end, :] = torch.einsum("bhij,bhjd->bhid", attn, V_w)

    return O


# ============================================================================
# autograd.Function wrapper
# ============================================================================


class FlashAtomAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, mask):
        B, H, M, D = Q.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        O = torch.empty_like(Q)
        scale = D ** -0.5
        n_windows = (M + W_QUERY - 1) // W_QUERY
        grid = (B * H, n_windows)

        _flash_atom_attn_fwd_kernel[grid](
            Q, K, V, O, mask,
            scale,
            B, H, M, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            mask.stride(0),
            W_QUERY, W_KEY,
        )

        ctx.save_for_backward(Q, K, V, O, mask)
        ctx.scale = scale
        ctx.B = B
        ctx.H = H
        ctx.M = M
        ctx.D = D
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, mask = ctx.saved_tensors
        B, H, M, D = ctx.B, ctx.H, ctx.M, ctx.D

        dO = dO.contiguous()
        dQ = torch.empty_like(Q)
        # dK and dV need atomic_add from overlapping windows → init to zero
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        n_windows = (M + W_QUERY - 1) // W_QUERY
        grid = (B * H, n_windows)

        _flash_atom_attn_bwd_kernel[grid](
            Q, K, V, O, dO,
            dQ, dK, dV,
            mask,
            ctx.scale,
            B, H, M, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            mask.stride(0),
            W_QUERY, W_KEY,
        )

        return dQ, dK, dV, None  # no grad for mask


def flash_atom_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Windowed atom attention (W=32 queries, H=128 keys).

    Args:
        Q, K, V: (B, H, M, D) or (H, M, D)
        mask: (B, M) or (M,) float, 1=valid 0=pad. None=all valid.

    Returns:
        O: same shape as Q
    """
    unbatched = Q.dim() == 3
    if unbatched:
        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, H, M, D = Q.shape
    if mask is None:
        mask = Q.new_ones(B, M, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    mask = mask.contiguous().float()

    O = FlashAtomAttnFn.apply(Q, K, V, mask)

    if unbatched:
        O = O.squeeze(0)
    return O
