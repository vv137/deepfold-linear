"""
Cross-Attention Triton Kernels — atom↔token cross-attention for diffusion encoder/decoder.

Two variants:
  1. AtomToToken (encoder): tokens query, atoms key/value.
     Sparse — each token attends only to its own atoms via token_atom_starts/counts.
     Grid: (B*H, N) — one program per token per head.

  2. TokenToAtom (decoder): atoms query, tokens key/value.
     Dense — each atom attends to all tokens.
     Grid: (B*H, ceil(M/TILE_Q)) — tiled over atom queries.

Both include forward + backward kernels with Python references for testing.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# 1. AtomToToken: sparse cross-attention (encoder)
# ============================================================================

# Max atoms per token for static allocation in Triton
MAX_ATOMS_PER_TOKEN: tl.constexpr = 32


@triton.jit
def _atom_to_token_fwd_kernel(
    Q_ptr,              # (B, H, N, D) token queries
    K_ptr,              # (B, H, M, D) atom keys
    V_ptr,              # (B, H, M, D) atom values
    O_ptr,              # (B, H, N, D) output
    STARTS_ptr,         # (B, N) int32 — atom start index per token
    COUNTS_ptr,         # (B, N) int32 — atom count per token
    TOKEN_MASK_ptr,     # (B, N) float, 1=valid 0=pad
    scale,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_starts_b: tl.constexpr,
    stride_tmask_b: tl.constexpr,
    MAX_A: tl.constexpr,  # max atoms per token (compile-time bound)
):
    """Each program handles one (batch, head, token)."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    if pid_n >= N:
        return

    # Check token validity
    tok_valid = tl.load(TOKEN_MASK_ptr + pid_b * stride_tmask_b + pid_n)
    if tok_valid == 0.0:
        # Zero output for padded tokens
        o_ptrs = O_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn + tl.arange(0, D)
        tl.store(o_ptrs, tl.zeros([D], dtype=tl.bfloat16))
        return

    # Load atom range for this token
    atom_start = tl.load(STARTS_ptr + pid_b * stride_starts_b + pid_n)
    atom_count = tl.load(COUNTS_ptr + pid_b * stride_starts_b + pid_n)

    # Load query: (D,)
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn
    Q_vec = tl.load(q_base + tl.arange(0, D))

    if atom_count == 0:
        o_ptrs = O_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn + tl.arange(0, D)
        tl.store(o_ptrs, tl.zeros([D], dtype=Q_vec.dtype))
        return

    # Load K, V for this token's atoms: (MAX_A, D)
    a_idx = tl.arange(0, MAX_A)
    a_mask = a_idx < atom_count
    atom_global_idx = atom_start + a_idx  # (MAX_A,)

    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    k_ptrs = k_base + atom_global_idx[:, None] * stride_kn + tl.arange(0, D)[None, :]
    K_tile = tl.load(k_ptrs, mask=a_mask[:, None], other=0.0)  # (MAX_A, D)

    v_ptrs = V_ptr + pid_b * stride_kb + pid_h * stride_kh + atom_global_idx[:, None] * stride_kn + tl.arange(0, D)[None, :]
    V_tile = tl.load(v_ptrs, mask=a_mask[:, None], other=0.0)  # (MAX_A, D)

    # Scores: Q_vec @ K^T → (MAX_A,)
    scores = tl.sum(Q_vec[None, :] * K_tile, axis=1) * scale  # (MAX_A,)
    scores = tl.where(a_mask, scores, float("-inf"))

    # Softmax
    row_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - row_max)
    exp_scores = tl.where(a_mask, exp_scores, 0.0)
    row_sum = tl.sum(exp_scores, axis=0) + 1e-8

    # Weighted sum: (D,)
    out = tl.sum(exp_scores[:, None] * V_tile, axis=0) / row_sum

    # Store
    o_ptrs = O_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn + tl.arange(0, D)
    tl.store(o_ptrs, out.to(Q_vec.dtype))


@triton.jit
def _atom_to_token_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    STARTS_ptr, COUNTS_ptr, TOKEN_MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_starts_b: tl.constexpr,
    stride_tmask_b: tl.constexpr,
    MAX_A: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    if pid_n >= N:
        return

    tok_valid = tl.load(TOKEN_MASK_ptr + pid_b * stride_tmask_b + pid_n)
    if tok_valid == 0.0:
        dq_ptrs = dQ_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn + tl.arange(0, D)
        tl.store(dq_ptrs, tl.zeros([D], dtype=tl.bfloat16))
        return

    atom_start = tl.load(STARTS_ptr + pid_b * stride_starts_b + pid_n)
    atom_count = tl.load(COUNTS_ptr + pid_b * stride_starts_b + pid_n)

    d_range = tl.arange(0, D)
    q_base = pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn

    Q_vec = tl.load(Q_ptr + q_base + d_range)
    O_vec = tl.load(O_ptr + q_base + d_range)
    dO_vec = tl.load(dO_ptr + q_base + d_range)

    if atom_count == 0:
        tl.store(dQ_ptr + q_base + d_range, tl.zeros([D], dtype=Q_vec.dtype))
        return

    a_idx = tl.arange(0, MAX_A)
    a_mask = a_idx < atom_count
    atom_global_idx = atom_start + a_idx

    k_base = pid_b * stride_kb + pid_h * stride_kh
    K_tile = tl.load(K_ptr + k_base + atom_global_idx[:, None] * stride_kn + d_range[None, :],
                      mask=a_mask[:, None], other=0.0)
    V_tile = tl.load(V_ptr + k_base + atom_global_idx[:, None] * stride_kn + d_range[None, :],
                      mask=a_mask[:, None], other=0.0)

    # Recompute attention
    scores = tl.sum(Q_vec[None, :] * K_tile, axis=1) * scale
    scores = tl.where(a_mask, scores, float("-inf"))
    row_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - row_max)
    exp_scores = tl.where(a_mask, exp_scores, 0.0)
    row_sum = tl.sum(exp_scores, axis=0) + 1e-8
    P = exp_scores / row_sum  # (MAX_A,)

    # D_i = sum(dO * O)
    D_i = tl.sum(dO_vec * O_vec, axis=0)

    # dP = dO @ V^T → (MAX_A,)
    dP = tl.sum(dO_vec[None, :] * V_tile, axis=1)

    # dS = P * (dP - D_i) * scale
    dS = P * (dP - D_i) * scale  # (MAX_A,)

    # dQ = sum(dS[:, None] * K, axis=0) → (D,)
    dQ_vec = tl.sum(dS[:, None] * K_tile, axis=0)
    tl.store(dQ_ptr + q_base + d_range, dQ_vec.to(Q_vec.dtype))

    # dK = dS[:, None] * Q[None, :] → (MAX_A, D)
    dK_tile = dS[:, None] * Q_vec[None, :]
    # dV = P[:, None] * dO[None, :] → (MAX_A, D)
    dV_tile = P[:, None] * dO_vec[None, :]

    # Atomic add to dK, dV (atoms may be shared across tokens in theory, but
    # each atom belongs to exactly one token so direct store is safe)
    dk_ptrs = dK_ptr + k_base + atom_global_idx[:, None] * stride_kn + d_range[None, :]
    tl.store(dk_ptrs, dK_tile.to(K_tile.dtype), mask=a_mask[:, None])
    dv_ptrs = dV_ptr + k_base + atom_global_idx[:, None] * stride_kn + d_range[None, :]
    tl.store(dv_ptrs, dV_tile.to(V_tile.dtype), mask=a_mask[:, None])


# ============================================================================
# 2. TokenToAtom: dense cross-attention (decoder)
# ============================================================================

TILE_Q_T2A = 64  # tile size for atom queries


@triton.jit
def _token_to_atom_fwd_kernel(
    Q_ptr,              # (B, H, M, D) atom queries
    K_ptr,              # (B, H, N, D) token keys
    V_ptr,              # (B, H, N, D) token values
    O_ptr,              # (B, H, M, D) output
    ATOM_MASK_ptr,      # (B, M) float
    TOKEN_MASK_ptr,     # (B, N) float
    scale,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_amask_b: tl.constexpr,
    stride_tmask_b: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Tiled: atom queries attend to all token keys."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_start = pid_q * TILE_Q
    q_idx = q_start + tl.arange(0, TILE_Q)
    q_mask = q_idx < M

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh

    # Load Q tile
    Q_tile = tl.load(Q_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                      mask=q_mask[:, None], other=0.0)

    q_pad = tl.load(ATOM_MASK_ptr + pid_b * stride_amask_b + q_idx, mask=q_mask, other=0.0)

    # Online softmax over token keys
    m_i = tl.full([TILE_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([TILE_Q], dtype=tl.float32)
    acc = tl.zeros([TILE_Q, D], dtype=tl.float32)

    n_tiles_k = tl.cdiv(N, TILE_K)
    for t in range(n_tiles_k):
        k_start = t * TILE_K
        k_idx = k_start + tl.arange(0, TILE_K)
        k_mask_range = k_idx < N

        K_tile = tl.load(K_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                          mask=k_mask_range[:, None], other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale  # (TILE_Q, TILE_K)

        # Token padding mask
        k_pad = tl.load(TOKEN_MASK_ptr + pid_b * stride_tmask_b + k_idx, mask=k_mask_range, other=0.0)
        scores = tl.where(k_mask_range[None, :] & (k_pad[None, :] > 0.0), scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        V_tile = tl.load(V_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                          mask=k_mask_range[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(V_tile.dtype), V_tile)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / (l_i[:, None] + 1e-8)
    acc = tl.where(q_pad[:, None] > 0.0, acc, 0.0)

    o_ptrs = O_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])


@triton.jit
def _token_to_atom_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    ATOM_MASK_ptr, TOKEN_MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_amask_b: tl.constexpr, stride_tmask_b: tl.constexpr,
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr,
):
    """Accumulate dK, dV for token-to-atom cross-attention."""
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    k_start = pid_k * TILE_K
    k_idx = k_start + tl.arange(0, TILE_K)
    k_mask = k_idx < N

    k_offset = pid_b * stride_kb + pid_h * stride_kh
    q_offset = pid_b * stride_qb + pid_h * stride_qh

    K_tile = tl.load(K_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                      mask=k_mask[:, None], other=0.0)
    V_tile = tl.load(V_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                      mask=k_mask[:, None], other=0.0)
    k_pad = tl.load(TOKEN_MASK_ptr + pid_b * stride_tmask_b + k_idx, mask=k_mask, other=0.0)

    dK_acc = tl.zeros([TILE_K, D], dtype=tl.float32)
    dV_acc = tl.zeros([TILE_K, D], dtype=tl.float32)

    n_tiles_q = tl.cdiv(M, TILE_Q)
    for t in range(n_tiles_q):
        q_start = t * TILE_Q
        q_idx = q_start + tl.arange(0, TILE_Q)
        q_mask = q_idx < M

        Q_tile = tl.load(Q_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                          mask=q_mask[:, None], other=0.0)
        O_tile = tl.load(O_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                          mask=q_mask[:, None], other=0.0)
        dO_tile = tl.load(dO_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                           mask=q_mask[:, None], other=0.0)
        lse = tl.load(LSE_ptr + pid_bh * M + q_idx, mask=q_mask, other=float("-inf"))
        q_pad = tl.load(ATOM_MASK_ptr + pid_b * stride_amask_b + q_idx, mask=q_mask, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        scores = tl.where(k_mask[None, :] & (k_pad[None, :] > 0.0), scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        P = tl.exp(scores - lse[:, None])
        P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

        dV_acc += tl.dot(tl.trans(P.to(dO_tile.dtype)), dO_tile)

        dP = tl.dot(dO_tile, tl.trans(V_tile))
        D_i = tl.sum(dO_tile * O_tile, axis=1)
        dS = P * (dP - D_i[:, None]) * scale

        dK_acc += tl.dot(tl.trans(dS.to(Q_tile.dtype)), Q_tile)

    dk_ptrs = dK_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :]
    tl.store(dk_ptrs, dK_acc.to(K_tile.dtype), mask=k_mask[:, None])
    dv_ptrs = dV_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :]
    tl.store(dv_ptrs, dV_acc.to(V_tile.dtype), mask=k_mask[:, None])


@triton.jit
def _token_to_atom_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dQ_ptr,
    ATOM_MASK_ptr, TOKEN_MASK_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_amask_b: tl.constexpr, stride_tmask_b: tl.constexpr,
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_start = pid_q * TILE_Q
    q_idx = q_start + tl.arange(0, TILE_Q)
    q_mask = q_idx < M

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh

    Q_tile = tl.load(Q_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                      mask=q_mask[:, None], other=0.0)
    O_tile = tl.load(O_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                      mask=q_mask[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :],
                       mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + pid_bh * M + q_idx, mask=q_mask, other=float("-inf"))
    q_pad = tl.load(ATOM_MASK_ptr + pid_b * stride_amask_b + q_idx, mask=q_mask, other=0.0)

    D_i = tl.sum(dO_tile * O_tile, axis=1)
    dQ_acc = tl.zeros([TILE_Q, D], dtype=tl.float32)

    n_tiles_k = tl.cdiv(N, TILE_K)
    for t in range(n_tiles_k):
        k_start = t * TILE_K
        k_idx = k_start + tl.arange(0, TILE_K)
        k_mask = k_idx < N

        K_tile = tl.load(K_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                          mask=k_mask[:, None], other=0.0)
        V_tile = tl.load(V_ptr + k_offset + k_idx[:, None] * stride_kn + tl.arange(0, D)[None, :],
                          mask=k_mask[:, None], other=0.0)
        k_pad = tl.load(TOKEN_MASK_ptr + pid_b * stride_tmask_b + k_idx, mask=k_mask, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        scores = tl.where(k_mask[None, :] & (k_pad[None, :] > 0.0), scores, float("-inf"))
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        P = tl.exp(scores - lse[:, None])
        P = tl.where(q_pad[:, None] > 0.0, P, 0.0)

        dP = tl.dot(dO_tile, tl.trans(V_tile))
        dS = P * (dP - D_i[:, None]) * scale
        dQ_acc += tl.dot(dS.to(K_tile.dtype), K_tile)

    dq_ptrs = dQ_ptr + q_offset + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :]
    tl.store(dq_ptrs, dQ_acc.to(Q_tile.dtype), mask=q_mask[:, None])


# ============================================================================
# Python references
# ============================================================================


def atom_to_token_ref(Q, K, V, token_atom_starts, token_atom_counts, token_mask=None):
    """Sparse cross-attention: tokens query their own atoms.

    Args:
        Q: (B, H, N, D) token queries
        K: (B, H, M, D) atom keys
        V: (B, H, M, D) atom values
        token_atom_starts: (B, N) int — start index of atoms per token
        token_atom_counts: (B, N) int — number of atoms per token
        token_mask: (B, N) float

    Returns:
        O: (B, H, N, D)
    """
    B, H, N, D = Q.shape
    scale = D ** -0.5
    O = torch.zeros_like(Q)

    for b in range(B):
        for n in range(N):
            if token_mask is not None and token_mask[b, n] == 0:
                continue
            start = token_atom_starts[b, n].item()
            count = token_atom_counts[b, n].item()
            if count == 0:
                continue
            q = Q[b, :, n, :]                       # (H, D)
            k = K[b, :, start:start+count, :]       # (H, count, D)
            v = V[b, :, start:start+count, :]
            scores = (q[:, None, :] * k).sum(-1) * scale  # (H, count)
            attn = torch.softmax(scores, dim=-1)           # (H, count)
            O[b, :, n, :] = (attn.unsqueeze(-1) * v).sum(dim=1)

    return O


def token_to_atom_ref(Q, K, V, atom_mask=None, token_mask=None):
    """Dense cross-attention: atoms query all tokens.

    Args:
        Q: (B, H, M, D) atom queries
        K: (B, H, N, D) token keys
        V: (B, H, N, D) token values
        atom_mask: (B, M) float
        token_mask: (B, N) float

    Returns:
        O: (B, H, M, D)
    """
    B, H, M, D = Q.shape
    N = K.shape[2]
    scale = D ** -0.5

    scores = torch.einsum("bhid,bhjd->bhij", Q, K) * scale  # (B, H, M, N)

    if token_mask is not None:
        scores = scores + (1 - token_mask[:, None, None, :]) * (-1e9)

    attn = torch.softmax(scores, dim=-1)
    if atom_mask is not None:
        attn = attn * atom_mask[:, None, :, None]
    attn = attn.nan_to_num(0.0)

    O = torch.einsum("bhij,bhjd->bhid", attn, V)
    return O


# ============================================================================
# autograd.Function wrappers
# ============================================================================


class AtomToTokenAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, token_atom_starts, token_atom_counts, token_mask):
        B, H, N, D = Q.shape
        M = K.shape[2]
        max_atoms = int(token_atom_counts.max().item())
        # Round up to next power of 2 for Triton constexpr, min 1
        max_a_constexpr = max(1, 1 << (max_atoms - 1).bit_length()) if max_atoms > 0 else 1
        max_a_constexpr = min(max_a_constexpr, 32)  # cap at 32

        O = torch.empty_like(Q)
        scale = D ** -0.5
        grid = (B * H, N)

        _atom_to_token_fwd_kernel[grid](
            Q, K, V, O,
            token_atom_starts, token_atom_counts, token_mask,
            scale,
            B, H, N, M, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            token_atom_starts.stride(0),
            token_mask.stride(0),
            max_a_constexpr,
        )

        ctx.save_for_backward(Q, K, V, O, token_atom_starts, token_atom_counts, token_mask)
        ctx.scale = scale
        ctx.B, ctx.H, ctx.N, ctx.M, ctx.D = B, H, N, M, D
        ctx.max_a = max_a_constexpr
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, starts, counts, token_mask = ctx.saved_tensors
        B, H, N, M, D = ctx.B, ctx.H, ctx.N, ctx.M, ctx.D

        dO = dO.contiguous()
        dQ = torch.empty_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        grid = (B * H, N)
        _atom_to_token_bwd_kernel[grid](
            Q, K, V, O, dO,
            dQ, dK, dV,
            starts, counts, token_mask,
            ctx.scale,
            B, H, N, M, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            starts.stride(0),
            token_mask.stride(0),
            ctx.max_a,
        )

        return dQ, dK, dV, None, None, None


class TokenToAtomAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, atom_mask, token_mask):
        B, H, M, D = Q.shape
        N = K.shape[2]

        O = torch.empty_like(Q)
        LSE = torch.empty(B * H, M, device=Q.device, dtype=torch.float32)
        scale = D ** -0.5

        TILE_Q = 64
        # Triton requires power-of-2 tile sizes
        TILE_K = min(64, 1 << (N - 1).bit_length())  # next power of 2

        # Need to store LSE for backward — compute in forward via a modified kernel
        # For now, use the ref-based approach: materialize for small N
        # Actually, let's just use the tiled fwd kernel and compute LSE separately
        # The forward kernel uses online softmax internally but doesn't output LSE.
        # We'll compute LSE from re-running scores in backward (recomputation approach).

        grid = (B * H, triton.cdiv(M, TILE_Q))

        _token_to_atom_fwd_kernel[grid](
            Q, K, V, O,
            atom_mask, token_mask,
            scale,
            B, H, M, N, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            atom_mask.stride(0),
            token_mask.stride(0),
            TILE_Q, TILE_K,
        )

        # Compute LSE for backward by recomputing scores
        # (This is cheap since N is small — typically ≤384)
        with torch.no_grad():
            scores = torch.einsum("bhid,bhjd->bhij",
                                   Q.float(), K.float()) * scale  # (B, H, M, N)
            scores = scores + (1 - token_mask[:, None, None, :].float()) * (-1e9)
            lse = torch.logsumexp(scores, dim=-1)  # (B, H, M)
            LSE = lse.reshape(B * H, M).contiguous()

        ctx.save_for_backward(Q, K, V, O, LSE, atom_mask, token_mask)
        ctx.scale = scale
        ctx.B, ctx.H, ctx.M, ctx.N, ctx.D = B, H, M, N, D
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE, atom_mask, token_mask = ctx.saved_tensors
        B, H, M, N, D = ctx.B, ctx.H, ctx.M, ctx.N, ctx.D

        dO = dO.contiguous()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        TILE_Q = 64
        TILE_K = min(64, 1 << (N - 1).bit_length())

        common = dict(
            B=B, H=H, M=M, N=N, D=D,
            stride_qb=Q.stride(0), stride_qh=Q.stride(1), stride_qn=Q.stride(2),
            stride_kb=K.stride(0), stride_kh=K.stride(1), stride_kn=K.stride(2),
            stride_amask_b=atom_mask.stride(0),
            stride_tmask_b=token_mask.stride(0),
            TILE_Q=TILE_Q, TILE_K=TILE_K,
        )

        grid_k = (B * H, triton.cdiv(N, TILE_K))
        _token_to_atom_bwd_dkdv_kernel[grid_k](
            Q, K, V, O, LSE, dO, dK, dV,
            atom_mask, token_mask, ctx.scale, **common,
        )

        grid_q = (B * H, triton.cdiv(M, TILE_Q))
        _token_to_atom_bwd_dq_kernel[grid_q](
            Q, K, V, O, LSE, dO, dQ,
            atom_mask, token_mask, ctx.scale, **common,
        )

        return dQ, dK, dV, None, None


# ============================================================================
# Public API
# ============================================================================


def atom_to_token_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    token_atom_starts: torch.Tensor,
    token_atom_counts: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse cross-attention: tokens query their own atoms.

    Args:
        Q: (B, H, N, D) token queries
        K: (B, H, M, D) atom keys
        V: (B, H, M, D) atom values
        token_atom_starts: (B, N) int32
        token_atom_counts: (B, N) int32
        token_mask: (B, N) float. None=all valid.

    Returns:
        O: (B, H, N, D)
    """
    B, H, N, D = Q.shape
    if token_mask is None:
        token_mask = Q.new_ones(B, N, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    token_atom_starts = token_atom_starts.contiguous().to(torch.int32)
    token_atom_counts = token_atom_counts.contiguous().to(torch.int32)
    token_mask = token_mask.contiguous().float()

    return AtomToTokenAttnFn.apply(Q, K, V, token_atom_starts, token_atom_counts, token_mask)


def token_to_atom_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense cross-attention: atoms query all tokens.

    Args:
        Q: (B, H, M, D) atom queries
        K: (B, H, N, D) token keys
        V: (B, H, N, D) token values
        atom_mask: (B, M) float. None=all valid.
        token_mask: (B, N) float. None=all valid.

    Returns:
        O: (B, H, M, D)
    """
    B, H, M, D = Q.shape
    N = K.shape[2]
    if atom_mask is None:
        atom_mask = Q.new_ones(B, M, dtype=torch.float32)
    if token_mask is None:
        token_mask = K.new_ones(B, N, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    atom_mask = atom_mask.contiguous().float()
    token_mask = token_mask.contiguous().float()

    return TokenToAtomAttnFn.apply(Q, K, V, atom_mask, token_mask)
