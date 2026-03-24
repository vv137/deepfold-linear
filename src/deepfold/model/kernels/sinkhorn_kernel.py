"""
Flash-Sinkhorn Triton kernel (SPEC §18 item 1).

Optimized two-pass architecture with batch dimension support:
  Forward:
    Sinkhorn iterations via separate row/col update kernels (parallel across B*H×N_tiles)
    Transport output: T_norm @ V and T_norm @ x (EGNN centroid) with online softmax
  Backward (IFT):
    Adjoint iterations with same tiling structure (SPEC §7.5)
    Cost gradient computed on-the-fly

Key optimizations over naive version:
  - tl.dot for QK^T (tensor cores)
  - Grid=(B*H, N_tiles) for full parallelism across batch, heads, and tiles
  - Distances computed on-the-fly per tile (no O(N²) storage)
  - Separate row/col update kernels (avoid serial bottleneck)

All per-batch tensors (Q_ln, K_ln, V, x_res, pos_bins, log_u, log_v, etc.)
have shape (B, H, N, ...) or (B, N, ...). Shared tensors (eps, w_dist) remain (H,).
grad_x_transport and grad_x_cost are (B, N, 3) to avoid cross-batch atomic contention.
"""

import torch
import triton
import triton.language as tl


def _compute_cost_tile_py(
    Q_ln, K_ln, x_res, pos_weight, pos_bins, w_dist_h, r_0, i_slice, j_slice, eps_h
):
    """Python cost tile computation for backward. Returns (C, log_K, dist, diff).

    Args:
        pos_weight: (68,) weight vector for the current head h
        pos_bins:   (N, N) int32 bin indices
    """
    Q_i = Q_ln[i_slice]
    K_j = K_ln[j_slice]
    d_h = Q_i.shape[-1]
    content = -(Q_i @ K_j.T) / (d_h**0.5)
    xi = x_res[i_slice]
    xj = x_res[j_slice]
    diff = xi[:, None, :] - xj[None, :, :]
    dist = (diff**2).sum(-1).sqrt().clamp(min=1e-8)
    geo = w_dist_h * dist / (r_0 + dist)
    bins_tile = pos_bins[i_slice, :][:, j_slice].long()
    pos = pos_weight[bins_tile]  # (ti, tj)
    C_tile = content + pos + geo
    log_K_tile = -C_tile / eps_h
    return C_tile, log_K_tile, dist, diff


# ============================================================================
# Helper: compute cost tile on-the-fly
# ============================================================================


@triton.jit
def _compute_cost_tile(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    w_dist_h,
    r_0,
    i_idx,
    j_idx,
    i_mask,
    j_mask,
    pid_h,
    N: tl.constexpr,
    D_H: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute cost tile C[i, j] = content + pos + geo. Returns (BLOCK_I, BLOCK_J).

    Pointers must already be batch-offset by the caller.
    POS_WEIGHT_ptr: pointer to (H, NUM_BINS) — no batch offset needed.
    POS_BINS_ptr: pointer to (N, N) int32 — already batch-offset by caller.
    """
    # Load Q tile: (BLOCK_I, D_H)
    q_ptrs = (
        Q_ptr
        + pid_h * stride_qh
        + i_idx[:, None] * stride_qn
        + tl.arange(0, D_H)[None, :]
    )
    Q_tile = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)

    # Load K tile: (BLOCK_J, D_H)
    k_ptrs = (
        K_ptr
        + pid_h * stride_qh
        + j_idx[:, None] * stride_qn
        + tl.arange(0, D_H)[None, :]
    )
    K_tile = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)

    # Content: -Q @ K^T / sqrt(d_h)  → (BLOCK_I, BLOCK_J)
    content = -tl.dot(Q_tile, tl.trans(K_tile)) * (1.0 / tl.sqrt(D_H * 1.0))

    # Geometry: w_dist * d/(r_0 + d)
    xi_0 = tl.load(X_ptr + i_idx * stride_xn + 0, mask=i_mask, other=0.0)
    xi_1 = tl.load(X_ptr + i_idx * stride_xn + 1, mask=i_mask, other=0.0)
    xi_2 = tl.load(X_ptr + i_idx * stride_xn + 2, mask=i_mask, other=0.0)
    xj_0 = tl.load(X_ptr + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
    xj_1 = tl.load(X_ptr + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
    xj_2 = tl.load(X_ptr + j_idx * stride_xn + 2, mask=j_mask, other=0.0)

    dx = xi_0[:, None] - xj_0[None, :]
    dy = xi_1[:, None] - xj_1[None, :]
    dz = xi_2[:, None] - xj_2[None, :]
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
    geo = w_dist_h * dist / (r_0 + dist)

    # Position bias via gather: load bin indices, then gather from weight
    bin_ptrs = POS_BINS_ptr + i_idx[:, None] * stride_bins_n + j_idx[None, :]
    bins_tile = tl.load(bin_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0).to(tl.int32)
    pos = tl.load(POS_WEIGHT_ptr + pid_h * NUM_BINS + bins_tile, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

    return content + pos + geo


@triton.jit
def _load_x_components(X_ptr, idx, mask, stride_xn, BLOCK: tl.constexpr):
    """Load x, y, z components for a tile of coordinates."""
    x0 = tl.load(X_ptr + idx * stride_xn + 0, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + idx * stride_xn + 1, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + idx * stride_xn + 2, mask=mask, other=0.0)
    return x0, x1, x2


# ============================================================================
# Backward Step 1: Compute log_Z, D, g_v, grad_V, grad_x_transport
# ============================================================================


@triton.jit
def _compute_log_Z(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    LOG_Z_ptr,  # output (B, H, N)
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,  # stride for log_u/log_v/log_Z batch dim
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """log_Z[b,h,i] = LSE_j(log_u[b,h,i] + log_K[b,h,i,j] + log_v[b,h,j])."""
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30)

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(
            LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30
        )
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        score = (
            log_u_i[:, None]
            + (-C_tile * inv_eps_h)
            + log_v_j[None, :]
            + mask_bias_j[None, :]
        )
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        LOG_Z_ptr + uv_off + pid_h * N + i_idx,
        max_val + tl.log(sum_exp + 1e-30),
        mask=i_mask,
    )


@triton.jit
def _backward_gv_grad_V_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    V_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    LOG_Z_ptr,
    GRAD_O_ptr,
    GRAD_XC_ptr,
    D_ptr,
    G_V_ptr,  # output: g_v (B, H, N)
    GRAD_V_ptr,  # output: grad_V (B, H, N, d_h) — atomic add
    GRAD_X_TRANSPORT_ptr,  # output: grad_x_transport (B, N, 3) — atomic add
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_uvb,
    stride_gxt_b,  # batch stride for grad_x_transport (B, N, 3)
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """Backward through T_norm: compute g_v, accumulate grad_V and grad_x_transport."""
    pid_bh = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    V_b = V_ptr + pid_b * stride_vb
    GRAD_O_b = GRAD_O_ptr + pid_b * stride_vb
    GRAD_XC_b = GRAD_XC_ptr + pid_b * H * N * 3
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N
    log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30)

    # Accumulators
    acc_gv = tl.zeros([BLOCK], dtype=tl.float32)
    d_idx = tl.arange(0, D_H)
    acc_grad_V = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_gx_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_2 = tl.zeros([BLOCK], dtype=tl.float32)

    # Load V[j]: (BLOCK, D_H)
    v_ptrs = V_b + pid_h * stride_vh + j_idx[:, None] * stride_vn + d_idx[None, :]
    V_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)

    # Load x_res[j]: (BLOCK, 3)
    xj_0, xj_1, xj_2 = _load_x_components(X_b, j_idx, j_mask, stride_xn, BLOCK)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N
        log_u_i = tl.load(
            LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30
        )
        log_Z_i = tl.load(
            LOG_Z_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0
        )
        D_i = tl.load(D_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
        mask_bias_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h
        log_score = (
            log_u_i[:, None] + log_K_tile + log_v_j[None, :] + mask_bias_i[:, None]
        )
        T_norm_tile = tl.exp(log_score - log_Z_i[:, None])
        T_norm_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_norm_tile, 0.0)

        # Load grad_O_avg[i]: (BLOCK_I, D_H)
        go_ptrs = (
            GRAD_O_b + pid_h * stride_vh + i_idx[:, None] * stride_vn + d_idx[None, :]
        )
        grad_O_i = tl.load(go_ptrs, mask=i_mask[:, None], other=0.0)

        # Load grad_xc[i]: (BLOCK_I, 3)
        gxc_base = GRAD_XC_b + pid_h * N * 3
        gxc_i0 = tl.load(gxc_base + i_idx * 3 + 0, mask=i_mask, other=0.0)
        gxc_i1 = tl.load(gxc_base + i_idx * 3 + 1, mask=i_mask, other=0.0)
        gxc_i2 = tl.load(gxc_base + i_idx * 3 + 2, mask=i_mask, other=0.0)

        # dT_ij = grad_O_avg[i] · V[j]^T + grad_xc[i] · x_res[j]^T
        dT_attn = tl.dot(grad_O_i, tl.trans(V_j))
        dT_xc = (
            gxc_i0[:, None] * xj_0[None, :]
            + gxc_i1[:, None] * xj_1[None, :]
            + gxc_i2[:, None] * xj_2[None, :]
        )
        dT = dT_attn + dT_xc

        grad_ls = T_norm_tile * (dT - D_i[:, None])
        acc_gv += tl.sum(grad_ls, axis=0)
        acc_grad_V += tl.dot(tl.trans(T_norm_tile), grad_O_i)

        T_t_gxc_0 = tl.sum(T_norm_tile * gxc_i0[:, None], axis=0)
        T_t_gxc_1 = tl.sum(T_norm_tile * gxc_i1[:, None], axis=0)
        T_t_gxc_2 = tl.sum(T_norm_tile * gxc_i2[:, None], axis=0)
        acc_gx_0 += T_t_gxc_0
        acc_gx_1 += T_t_gxc_1
        acc_gx_2 += T_t_gxc_2

    # Store g_v
    tl.store(G_V_ptr + uv_off + pid_h * N + j_idx, acc_gv, mask=j_mask)

    # Atomic add grad_V
    gv_ptrs = (
        GRAD_V_ptr
        + pid_b * stride_vb
        + pid_h * stride_vh
        + j_idx[:, None] * stride_vn
        + d_idx[None, :]
    )
    tl.atomic_add(gv_ptrs, acc_grad_V, mask=j_mask[:, None])

    # Atomic add grad_x_transport — per-batch buffer, sum over heads
    gxt_base = GRAD_X_TRANSPORT_ptr + pid_b * stride_gxt_b
    tl.atomic_add(gxt_base + j_idx * 3 + 0, acc_gx_0, mask=j_mask)
    tl.atomic_add(gxt_base + j_idx * 3 + 1, acc_gx_1, mask=j_mask)
    tl.atomic_add(gxt_base + j_idx * 3 + 2, acc_gx_2, mask=j_mask)


# ============================================================================
# Backward Step 4: Cost gradient propagation
# ============================================================================


@triton.jit
def _cost_gradient_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    V_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    LOG_Z_ptr,
    ROW_LSE_ptr,
    COL_LSE_ptr,
    GRAD_O_ptr,
    GRAD_XC_ptr,
    D_ptr,
    Z_U_ptr,
    Z_V_ptr,
    KAPPA_ptr,
    # Outputs (atomic add)
    GRAD_Q_ptr,
    GRAD_K_ptr,
    GRAD_X_COST_ptr,  # (B, N, 3)
    GRAD_W_DIST_ptr,  # (H,)
    GRAD_POS_WEIGHT_ptr,  # (H, NUM_BINS) — atomic add
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_uvb,
    stride_gxc_b,  # batch stride for grad_x_cost (B, N, 3)
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """Fused cost gradient: direct (through T_norm) + IFT (through z_u, z_v)."""
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)
    inv_sqrt_dh = 1.0 / tl.sqrt(D_H * 1.0)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    V_b = V_ptr + pid_b * stride_vb
    GRAD_O_b = GRAD_O_ptr + pid_b * stride_vb
    GRAD_XC_b = GRAD_XC_ptr + pid_b * H * N * 3
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
    log_Z_i = tl.load(LOG_Z_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
    row_lse_i = tl.load(
        ROW_LSE_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0
    )
    D_i = tl.load(D_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
    z_u_i = tl.load(Z_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)

    # Load grad_O_avg[i]: (BLOCK, D_H)
    d_idx = tl.arange(0, D_H)
    go_ptrs = GRAD_O_b + pid_h * stride_vh + i_idx[:, None] * stride_vn + d_idx[None, :]
    grad_O_i = tl.load(go_ptrs, mask=i_mask[:, None], other=0.0)

    # Load grad_xc[i]: 3 components
    gxc_base = GRAD_XC_b + pid_h * N * 3
    gxc_i0 = tl.load(gxc_base + i_idx * 3 + 0, mask=i_mask, other=0.0)
    gxc_i1 = tl.load(gxc_base + i_idx * 3 + 1, mask=i_mask, other=0.0)
    gxc_i2 = tl.load(gxc_base + i_idx * 3 + 2, mask=i_mask, other=0.0)

    # Load Q_ln[i]: (BLOCK, D_H)
    q_ptrs = Q_b + pid_h * stride_qh + i_idx[:, None] * stride_qn + d_idx[None, :]
    Q_i = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)

    # Accumulators
    acc_grad_Q = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_grad_w_dist = tl.zeros([], dtype=tl.float32)
    acc_gx_i_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_i_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_i_2 = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N

        log_v_j = tl.load(
            LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0
        )
        col_lse_j = tl.load(
            COL_LSE_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0
        )
        z_v_j = tl.load(Z_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0)
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        # Load K_ln[j], V[j], x[j]
        k_ptrs = K_b + pid_h * stride_qh + j_idx[:, None] * stride_qn + d_idx[None, :]
        K_j = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)

        v_ptrs = V_b + pid_h * stride_vh + j_idx[:, None] * stride_vn + d_idx[None, :]
        V_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)

        xj_0, xj_1, xj_2 = _load_x_components(X_b, j_idx, j_mask, stride_xn, BLOCK)

        # Recompute cost tile
        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        # --- Direct gradient through T_norm ---
        log_score = (
            log_u_i[:, None] + log_K_tile + log_v_j[None, :] + mask_bias_j[None, :]
        )
        T_norm_tile = tl.exp(log_score - log_Z_i[:, None])
        T_norm_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_norm_tile, 0.0)

        dT_attn = tl.dot(grad_O_i, tl.trans(V_j))
        dT_xc = (
            gxc_i0[:, None] * xj_0[None, :]
            + gxc_i1[:, None] * xj_1[None, :]
            + gxc_i2[:, None] * xj_2[None, :]
        )
        grad_ls_direct = T_norm_tile * (dT_attn + dT_xc - D_i[:, None])
        grad_C_direct = grad_ls_direct * (-inv_eps_h)

        # --- IFT gradient ---
        log_s_row = log_K_tile + log_v_j[None, :] - row_lse_i[:, None]
        s_row_tile = tl.exp(log_s_row)
        log_s_col = log_K_tile + log_u_i[:, None] - col_lse_j[None, :]
        s_col_tile = tl.exp(log_s_col)
        s_row_tile = tl.where(i_mask[:, None] & j_mask[None, :], s_row_tile, 0.0)
        s_col_tile = tl.where(i_mask[:, None] & j_mask[None, :], s_col_tile, 0.0)

        grad_C_ift = -(kappa_h * inv_eps_h) * (
            z_u_i[:, None] * s_row_tile + z_v_j[None, :] * s_col_tile
        )

        grad_C_total = grad_C_direct + grad_C_ift

        # --- Propagate to Q_ln ---
        acc_grad_Q += tl.dot(grad_C_total, K_j) * (-inv_sqrt_dh)

        # --- Propagate to K_ln (atomic add since j varies) ---
        grad_K_tile = tl.dot(tl.trans(grad_C_total), Q_i) * (-inv_sqrt_dh)
        gk_ptrs = (
            GRAD_K_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + j_idx[:, None] * stride_qn
            + d_idx[None, :]
        )
        tl.atomic_add(gk_ptrs, grad_K_tile, mask=j_mask[:, None])

        # --- Propagate to x_res through geometry ---
        xi_0, xi_1, xi_2 = _load_x_components(X_b, i_idx, i_mask, stride_xn, BLOCK)
        dx = xi_0[:, None] - xj_0[None, :]
        dy = xi_1[:, None] - xj_1[None, :]
        dz = xi_2[:, None] - xj_2[None, :]
        dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)

        geo_grad_coeff = w_dist_h * R_0 / ((R_0 + dist) * (R_0 + dist))
        weighted = grad_C_total * geo_grad_coeff / dist

        acc_gx_i_0 += tl.sum(weighted * dx, axis=1)
        acc_gx_i_1 += tl.sum(weighted * dy, axis=1)
        acc_gx_i_2 += tl.sum(weighted * dz, axis=1)

        # grad_x[j] -= weighted * diff
        gx_j_0 = -tl.sum(weighted * dx, axis=0)
        gx_j_1 = -tl.sum(weighted * dy, axis=0)
        gx_j_2 = -tl.sum(weighted * dz, axis=0)
        gxc_base_j = GRAD_X_COST_ptr + pid_b * stride_gxc_b
        tl.atomic_add(gxc_base_j + j_idx * 3 + 0, gx_j_0, mask=j_mask)
        tl.atomic_add(gxc_base_j + j_idx * 3 + 1, gx_j_1, mask=j_mask)
        tl.atomic_add(gxc_base_j + j_idx * 3 + 2, gx_j_2, mask=j_mask)

        # --- Propagate to w_dist ---
        f_dist_tile = dist / (R_0 + dist)
        acc_grad_w_dist += tl.sum(grad_C_total * f_dist_tile)

        # --- Scatter grad_C_total into grad_pos_weight via atomic_add ---
        bin_ptrs = BINS_b + i_idx[:, None] * stride_bins_n + j_idx[None, :]
        bins_tile = tl.load(bin_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0).to(tl.int32)
        tl.atomic_add(GRAD_POS_WEIGHT_ptr + pid_h * NUM_BINS + bins_tile, grad_C_total, mask=i_mask[:, None] & j_mask[None, :])

    # Store accumulated gradients
    gq_ptrs = (
        GRAD_Q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + i_idx[:, None] * stride_qn
        + d_idx[None, :]
    )
    tl.store(gq_ptrs, acc_grad_Q, mask=i_mask[:, None])

    # Atomic add grad_x_cost for i
    gxc_base_i = GRAD_X_COST_ptr + pid_b * stride_gxc_b
    tl.atomic_add(gxc_base_i + i_idx * 3 + 0, acc_gx_i_0, mask=i_mask)
    tl.atomic_add(gxc_base_i + i_idx * 3 + 1, acc_gx_i_1, mask=i_mask)
    tl.atomic_add(gxc_base_i + i_idx * 3 + 2, acc_gx_i_2, mask=i_mask)

    # Atomic add grad_w_dist (one scalar per head, summed across batch)
    tl.atomic_add(GRAD_W_DIST_ptr + pid_h, acc_grad_w_dist)


# ============================================================================
# Sinkhorn Row Update: log_u_i = kappa * (log_mu_i - LSE_j(log_K_ij + log_v_j))
# ============================================================================


@triton.jit
def _sinkhorn_row_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_MU_ptr,
    LOG_V_ptr,
    LOG_U_ptr,  # output
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    LAM: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    kappa_h = LAM / (LAM + eps_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_mu_i = tl.load(
        LOG_MU_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30
    )

    # Online LSE over j tiles
    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(
            LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30
        )
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        score = log_K_tile + log_v_j[None, :] + mask_bias_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    lse = max_val + tl.log(sum_exp + 1e-30)
    log_u_new = kappa_h * (log_mu_i - lse)
    tl.store(LOG_U_ptr + uv_off + pid_h * N + i_idx, log_u_new, mask=i_mask)


# ============================================================================
# Sinkhorn Col Update: log_v_j = kappa * (log_nu_j - LSE_i(log_K_ij + log_u_i))
# ============================================================================


@triton.jit
def _sinkhorn_col_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_NU_ptr,
    LOG_U_ptr,
    LOG_V_ptr,  # output
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    LAM: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    kappa_h = LAM / (LAM + eps_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N
    log_nu_j = tl.load(
        LOG_NU_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30
    )

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N
        log_u_i = tl.load(
            LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30
        )
        mask_bias_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        score = log_K_tile + log_u_i[:, None] + mask_bias_i[:, None]
        score = tl.where(i_mask[:, None], score, -1e30)
        score_t = tl.trans(score)

        tile_max = tl.max(score_t, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score_t - new_max[:, None]), axis=1
        )
        max_val = new_max

    lse = max_val + tl.log(sum_exp + 1e-30)
    log_v_new = kappa_h * (log_nu_j - lse)
    tl.store(LOG_V_ptr + uv_off + pid_h * N + j_idx, log_v_new, mask=j_mask)


# ============================================================================
# Transport Output: O_avg = T_norm @ V, x_centroid = T_norm @ x_res
# ============================================================================


@triton.jit
def _transport_output_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    V_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    O_ptr,
    X_CENT_ptr,
    ROW_SUM_ptr,
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_xcb,
    stride_xch,
    stride_xcn,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    # Batch-offset pointers
    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    V_b = V_ptr + pid_b * stride_vb
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30)

    # Accumulators (online softmax style)
    row_max = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    row_sum = tl.zeros([BLOCK], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    xc_0 = tl.zeros([BLOCK], dtype=tl.float32)
    xc_1 = tl.zeros([BLOCK], dtype=tl.float32)
    xc_2 = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(
            LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30
        )
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        log_score = (
            log_u_i[:, None] + log_K_tile + log_v_j[None, :] + mask_bias_j[None, :]
        )
        log_score = tl.where(j_mask[None, :], log_score, -1e30)

        # Online max update
        tile_max = tl.max(log_score, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        scale = tl.exp(row_max - new_max)

        T_tile = tl.exp(log_score - new_max[:, None])
        T_tile = tl.where(j_mask[None, :], T_tile, 0.0)
        tile_sum = tl.sum(T_tile, axis=1)

        # Rescale + accumulate
        row_sum = row_sum * scale + tile_sum
        o_acc = o_acc * scale[:, None]
        xc_0 = xc_0 * scale
        xc_1 = xc_1 * scale
        xc_2 = xc_2 * scale

        # T_tile @ V_j
        v_ptrs = (
            V_b
            + pid_h * stride_vh
            + j_idx[:, None] * stride_vn
            + tl.arange(0, D_H)[None, :]
        )
        V_tile = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)
        o_acc += tl.dot(T_tile.to(V_tile.dtype), V_tile)

        # T_tile @ x_j for EGNN centroid
        xj_0 = tl.load(X_b + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
        xj_1 = tl.load(X_b + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
        xj_2 = tl.load(X_b + j_idx * stride_xn + 2, mask=j_mask, other=0.0)
        xc_0 += tl.sum(T_tile * xj_0[None, :], axis=1)
        xc_1 += tl.sum(T_tile * xj_1[None, :], axis=1)
        xc_2 += tl.sum(T_tile * xj_2[None, :], axis=1)

        row_max = new_max

    # Normalize
    inv_sum = 1.0 / (row_sum + 1e-6)
    o_acc = o_acc * inv_sum[:, None]

    # Store O_avg
    o_ptrs = (
        O_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + i_idx[:, None] * stride_on
        + tl.arange(0, D_H)[None, :]
    )
    tl.store(o_ptrs, o_acc, mask=i_mask[:, None])

    # Store x_centroid
    xc_base = X_CENT_ptr + pid_b * stride_xcb + pid_h * stride_xch
    tl.store(xc_base + i_idx * stride_xcn + 0, xc_0 * inv_sum, mask=i_mask)
    tl.store(xc_base + i_idx * stride_xcn + 1, xc_1 * inv_sum, mask=i_mask)
    tl.store(xc_base + i_idx * stride_xcn + 2, xc_2 * inv_sum, mask=i_mask)
    tl.store(ROW_SUM_ptr + uv_off + pid_h * N + i_idx, row_sum, mask=i_mask)


# ============================================================================
# IFT Backward: Compute col_lse / row_lse (needed for softmax weights)
# ============================================================================


@triton.jit
def _compute_row_lse(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_V_ptr,
    ROW_LSE_ptr,  # output (B, H, N)
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """row_lse[b,h,i] = LSE_j(log_K[b,h,i,j] + log_v[b,h,j])."""
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(
            LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30
        )
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        score = -C_tile * inv_eps_h + log_v_j[None, :] + mask_bias_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        ROW_LSE_ptr + uv_off + pid_h * N + i_idx,
        max_val + tl.log(sum_exp + 1e-30),
        mask=i_mask,
    )


@triton.jit
def _compute_col_lse(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    COL_LSE_ptr,  # output (B, H, N)
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """col_lse[b,h,j] = LSE_i(log_K[b,h,i,j] + log_u[b,h,i])."""
    pid_bh = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N
        log_u_i = tl.load(
            LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30
        )
        mask_bias_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        score = -C_tile * inv_eps_h + log_u_i[:, None] + mask_bias_i[:, None]
        score = tl.where(i_mask[:, None], score, -1e30)
        score_t = tl.trans(score)

        tile_max = tl.max(score_t, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score_t - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        COL_LSE_ptr + uv_off + pid_h * N + j_idx,
        max_val + tl.log(sum_exp + 1e-30),
        mask=j_mask,
    )


# ============================================================================
# IFT Adjoint: z_u update and z_v update
# ============================================================================


@triton.jit
def _ift_z_u_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    COL_LSE_ptr,
    G_U_ptr,
    Z_V_ptr,
    Z_U_ptr,  # output
    KAPPA_ptr,
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """z_u[b,h,i] = g_u[b,h,i] + kappa * sum_j S_col[b,h,i,j] * z_v[b,h,j]."""
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)

    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
    g_u_i = tl.load(G_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N

        z_v_j = tl.load(Z_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0)
        col_lse_j = tl.load(
            COL_LSE_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0
        )
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_s = (
            -C_tile * inv_eps_h
            + log_u_i[:, None]
            - col_lse_j[None, :]
            + mask_bias_j[None, :]
        )
        s_col_tile = tl.exp(log_s)
        s_col_tile = tl.where(j_mask[None, :], s_col_tile, 0.0)

        acc += tl.sum(s_col_tile * z_v_j[None, :], axis=1)

    z_u_new = g_u_i + kappa_h * acc
    tl.store(Z_U_ptr + uv_off + pid_h * N + i_idx, z_u_new, mask=i_mask)


@triton.jit
def _ift_z_v_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_WEIGHT_ptr,
    POS_BINS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_V_ptr,
    ROW_LSE_ptr,
    G_V_ptr,
    Z_U_ptr,
    Z_V_ptr,  # output
    KAPPA_ptr,
    MASK_BIAS_ptr,  # (B, N) float, 0 for valid, -1e9 for pad
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    H: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_xb,
    stride_xn,
    stride_bins_b,
    stride_bins_n,
    NUM_BINS: tl.constexpr,
    stride_uvb,
    stride_mb,  # batch stride for mask_bias
    BLOCK: tl.constexpr,
):
    """z_v[b,h,j] = g_v[b,h,j] + kappa * sum_i S_row[b,h,i,j] * z_u[b,h,i]."""
    pid_bh = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)

    Q_b = Q_ptr + pid_b * stride_qb
    K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb
    BINS_b = POS_BINS_ptr + pid_b * stride_bins_b
    uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N

    log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0)
    g_v_j = tl.load(G_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=0.0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N

        z_u_i = tl.load(Z_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
        row_lse_i = tl.load(
            ROW_LSE_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0
        )
        mask_bias_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)

        C_tile = _compute_cost_tile(
            Q_b,
            K_b,
            X_b,
            POS_WEIGHT_ptr,
            BINS_b,
            w_dist_h,
            R_0,
            i_idx,
            j_idx,
            i_mask,
            j_mask,
            pid_h,
            N,
            D_H,
            stride_qh,
            stride_qn,
            stride_xn,
            stride_bins_n,
            NUM_BINS,
            BLOCK,
            BLOCK,
        )
        log_s = (
            -C_tile * inv_eps_h
            + log_v_j[None, :]
            - row_lse_i[:, None]
            + mask_bias_i[:, None]
        )
        s_row_tile = tl.exp(log_s)
        s_row_tile = tl.where(i_mask[:, None], s_row_tile, 0.0)

        acc += tl.sum(s_row_tile * z_u_i[:, None], axis=0)

    z_v_new = g_v_j + kappa_h * acc
    tl.store(Z_V_ptr + uv_off + pid_h * N + j_idx, z_v_new, mask=j_mask)


# ============================================================================
# Python wrapper + autograd Function
# ============================================================================


class FlashSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q_ln,
        K_ln,
        V,
        x_res,
        pos_weight,
        pos_bins,
        eps,
        w_dist,
        log_mu,
        log_nu,
        K_iter,
        lam,
        r_0,
        log_u_init,
        log_v_init,
        BLOCK,
        mask_bias,
    ):
        B, H, N, d_h = Q_ln.shape
        device = Q_ln.device

        # Ensure FP32 contiguous
        Q_ln = Q_ln.contiguous().float()
        K_ln = K_ln.contiguous().float()
        V = V.contiguous().float()
        x_res = x_res.contiguous().float()
        pos_weight = pos_weight.contiguous().float()
        pos_bins = pos_bins.contiguous().to(torch.int32)
        eps = eps.contiguous().float()
        w_dist = w_dist.contiguous().float()
        log_mu = log_mu.contiguous().float()
        log_nu = log_nu.contiguous().float()
        mask_bias = mask_bias.contiguous().float()

        log_u = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        log_v = torch.zeros(B, H, N, device=device, dtype=torch.float32)

        if log_u_init is not None:
            log_u.copy_(log_u_init.float())
        if log_v_init is not None:
            log_v.copy_(log_v_init.float())

        n_tiles = (N + BLOCK - 1) // BLOCK
        grid = (B * H, n_tiles)

        NUM_BINS = 68
        common = (Q_ln, K_ln, x_res, pos_weight, pos_bins, eps, w_dist)
        dim_args = (N, d_h, lam, r_0, H)
        stride_args = (
            Q_ln.stride(0),
            Q_ln.stride(1),
            Q_ln.stride(2),
            x_res.stride(0),
            x_res.stride(1),
            pos_bins.stride(0),
            pos_bins.stride(1),
            NUM_BINS,
            log_mu.stride(0),  # stride_uvb = H * N
            mask_bias.stride(0),  # stride_mb
            BLOCK,
        )

        # Sinkhorn iterations: save all intermediate states for unrolled backward.
        # Memory: K × 2 × (B, H, N) ≈ 1MB at K=20, H=16, N=384. Negligible.
        log_u_history = []
        log_v_history = []
        for _ in range(K_iter):
            log_u_history.append(log_u.clone())
            log_v_history.append(log_v.clone())
            _sinkhorn_row_update[grid](
                *common,
                log_mu,
                log_v,
                log_u,
                mask_bias,
                *dim_args,
                *stride_args,
            )
            _sinkhorn_col_update[grid](
                *common,
                log_nu,
                log_u,
                log_v,
                mask_bias,
                *dim_args,
                *stride_args,
            )

        # Transport output
        O_avg = torch.zeros(B, H, N, d_h, device=device, dtype=torch.float32)
        x_centroid = torch.zeros(B, H, N, 3, device=device, dtype=torch.float32)
        row_sum = torch.zeros(B, H, N, device=device, dtype=torch.float32)

        _transport_output_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_weight,
            pos_bins,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            row_sum,
            mask_bias,
            N,
            d_h,
            r_0,
            H,
            Q_ln.stride(0),
            Q_ln.stride(1),
            Q_ln.stride(2),
            x_res.stride(0),
            x_res.stride(1),
            pos_bins.stride(0),
            pos_bins.stride(1),
            NUM_BINS,
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O_avg.stride(0),
            O_avg.stride(1),
            O_avg.stride(2),
            x_centroid.stride(0),
            x_centroid.stride(1),
            x_centroid.stride(2),
            log_u.stride(0),  # stride_uvb
            mask_bias.stride(0),  # stride_mb
            BLOCK,
        )

        ctx.save_for_backward(
            Q_ln,
            K_ln,
            V,
            x_res,
            pos_weight,
            pos_bins,
            eps,
            w_dist,
            log_u,
            log_v,
            log_mu,
            log_nu,
            row_sum,
            mask_bias,
            *log_u_history,
            *log_v_history,
        )
        ctx.K_iter = K_iter
        ctx.BLOCK = BLOCK
        ctx.lam = lam
        ctx.r_0 = r_0
        ctx.N_saved_base = 14  # number of tensors before history (added pos_bins)

        return O_avg, x_centroid, log_u, log_v

    @staticmethod
    def backward(ctx, grad_O, grad_xc, _grad_lu, _grad_lv):
        saved = ctx.saved_tensors
        N_base = ctx.N_saved_base
        Q_ln, K_ln, V, x_res, pos_weight, pos_bins, eps, w_dist = saved[:8]
        log_u, log_v, log_mu, log_nu, row_sum, mask_bias = saved[8:N_base]
        K_back = ctx.K_iter
        list(saved[N_base : N_base + K_back])
        log_v_hist = list(saved[N_base + K_back : N_base + 2 * K_back])
        BLOCK = ctx.BLOCK
        lam = ctx.lam
        r_0 = ctx.r_0

        B, H, N, d_h = Q_ln.shape
        device = Q_ln.device
        kappa = (lam / (lam + eps)).contiguous()

        # Cap BLOCK at 32 for backward
        BLOCK_BWD = min(BLOCK, 32)
        n_tiles = (N + BLOCK_BWD - 1) // BLOCK_BWD
        grid = (B * H, n_tiles)

        stride_uvb = log_u.stride(0)  # H * N
        stride_mb = mask_bias.stride(0)

        NUM_BINS = 68
        stride_args_bwd = (
            Q_ln.stride(0),
            Q_ln.stride(1),
            Q_ln.stride(2),
            x_res.stride(0),
            x_res.stride(1),
            pos_bins.stride(0),
            pos_bins.stride(1),
            NUM_BINS,
            stride_uvb,
            stride_mb,
            BLOCK_BWD,
        )

        if grad_O is None:
            grad_O = torch.zeros(B, H, N, d_h, device=device, dtype=torch.float32)
        else:
            grad_O = grad_O.contiguous().float()
        if grad_xc is None:
            grad_xc = torch.zeros(B, H, N, 3, device=device, dtype=torch.float32)
        else:
            grad_xc = grad_xc.contiguous().float()

        # ---- Step 1: Compute log_Z ----
        log_Z = torch.empty(B, H, N, device=device, dtype=torch.float32)
        _compute_log_Z[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_weight,
            pos_bins,
            eps,
            w_dist,
            log_u,
            log_v,
            log_Z,
            mask_bias,
            N,
            d_h,
            r_0,
            H,
            *stride_args_bwd,
        )

        # Recompute O_avg and x_centroid for D computation
        O_avg_recomp = torch.zeros(B, H, N, d_h, device=device, dtype=torch.float32)
        x_cent_recomp = torch.zeros(B, H, N, 3, device=device, dtype=torch.float32)
        row_sum_recomp = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        _transport_output_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_weight,
            pos_bins,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            O_avg_recomp,
            x_cent_recomp,
            row_sum_recomp,
            mask_bias,
            N,
            d_h,
            r_0,
            H,
            Q_ln.stride(0),
            Q_ln.stride(1),
            Q_ln.stride(2),
            x_res.stride(0),
            x_res.stride(1),
            pos_bins.stride(0),
            pos_bins.stride(1),
            NUM_BINS,
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O_avg_recomp.stride(0),
            O_avg_recomp.stride(1),
            O_avg_recomp.stride(2),
            x_cent_recomp.stride(0),
            x_cent_recomp.stride(1),
            x_cent_recomp.stride(2),
            stride_uvb,
            stride_mb,
            BLOCK_BWD,
        )

        D = (grad_O * O_avg_recomp).sum(dim=-1) + (grad_xc * x_cent_recomp).sum(
            dim=-1
        )  # (B, H, N)

        # ---- Step 2: Compute grad_V, grad_x_transport, and marginal sums r, c ----
        # Also compute g_v (gradient w.r.t. log_v from transport output)
        g_v = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        grad_V = torch.zeros_like(V)
        grad_x_transport = torch.zeros(B, N, 3, device=device, dtype=torch.float32)

        _backward_gv_grad_V_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_weight,
            pos_bins,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            log_Z,
            grad_O,
            grad_xc,
            D,
            g_v,
            grad_V,
            grad_x_transport,
            mask_bias,
            N,
            d_h,
            r_0,
            H,
            Q_ln.stride(0),
            Q_ln.stride(1),
            Q_ln.stride(2),
            x_res.stride(0),
            x_res.stride(1),
            pos_bins.stride(0),
            pos_bins.stride(1),
            NUM_BINS,
            V.stride(0),
            V.stride(1),
            V.stride(2),
            stride_uvb,
            grad_x_transport.stride(0),
            stride_mb,
            BLOCK,
        )

        # ---- Step 3: Unrolled backward through K Sinkhorn iterations ----
        #
        # Each iteration: log_u = κ(log_a - LSE_j(log_K_ij + log_v_j))
        #                 log_v = κ(log_b - LSE_i(log_K_ij + log_u_i))
        #
        # Backward per iteration (tiled, O(N) memory):
        #   s_ij = softmax_j(-C_ij/ε + log_v_j)     [recomputed per tile]
        #   grad_log_v_j += -κ · Σ_i grad_log_u_i · s_ij
        #   grad_log_mu_i += κ · grad_log_u_i
        #   grad_C_ij += (κ/ε) · grad_log_u_i · s_ij
        #   (similarly for col update)
        #
        # Memory: O(K·N) saved states + O(tile²) working = O(N)

        TILE = BLOCK_BWD
        grad_log_u = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        grad_log_v = g_v.clone()  # from transport output backward
        grad_log_mu = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        grad_log_nu = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        grad_Q_ln = torch.zeros_like(Q_ln)
        grad_K_ln = torch.zeros_like(K_ln)
        grad_x_cost = torch.zeros(B, N, 3, device=device, dtype=torch.float32)
        grad_w_dist = torch.zeros(H, device=device, dtype=torch.float32)
        grad_pos_weight = torch.zeros_like(pos_weight)

        for k in reversed(range(K_back)):
            # Saved states: log_u_hist[k], log_v_hist[k] = states BEFORE iteration k
            log_v_before = log_v_hist[k]  # log_v input to row update
            # After row update: log_u_after = κ(log_mu - LSE_j(-C/ε + log_v_before))
            # After col update: log_v_after = κ(log_nu - LSE_i(-C/ε + log_u_after))
            # log_u_after is recomputed below; log_v_after = log_v_hist[k+1] if k<K-1 else log_v

            # Recompute log_u_after for this iteration (needed for col update backward)
            log_u_after = torch.zeros(B, H, N, device=device, dtype=torch.float32)
            _sinkhorn_row_update[(B * H, n_tiles)](
                Q_ln,
                K_ln,
                x_res,
                pos_weight,
                pos_bins,
                eps,
                w_dist,
                log_mu,
                log_v_before,
                log_u_after,
                mask_bias,
                N,
                d_h,
                lam,
                r_0,
                H,
                *stride_args_bwd[:8],
                stride_uvb,
                stride_mb,
                BLOCK_BWD,
            )

            # -- Col update backward: log_v_after = κ(log_nu - LSE_i(-C/ε + log_u_after)) --
            # grad_log_v_after is the incoming gradient (from next iteration or transport output)
            for b_idx in range(B):
                for h_idx in range(H):
                    eps_h = eps[h_idx].item()
                    kappa_h = kappa[h_idx].item()
                    w_h = w_dist[h_idx].item()
                    for j0 in range(0, N, TILE):
                        je = min(j0 + TILE, N)
                        # s'_ji = softmax_i(-C_ij/ε + log_u_after_i) over i
                        max_val = torch.full((je - j0,), -1e30, device=device)
                        sum_exp = torch.zeros(je - j0, device=device)
                        # First pass: compute LSE for normalization
                        for i0 in range(0, N, TILE):
                            ie = min(i0 + TILE, N)
                            _, log_K_t, _, _ = _compute_cost_tile_py(
                                Q_ln[b_idx, h_idx],
                                K_ln[b_idx, h_idx],
                                x_res[b_idx],
                                pos_weight[h_idx],
                                pos_bins[b_idx],
                                w_h,
                                r_0,
                                slice(i0, ie),
                                slice(j0, je),
                                eps_h,
                            )
                            score = (
                                log_K_t
                                + log_u_after[b_idx, h_idx, i0:ie, None]
                                + mask_bias[b_idx, None, i0:ie]
                            ).T  # (tj, ti)
                            tile_max = score.max(dim=-1).values
                            new_max = torch.maximum(max_val, tile_max)
                            sum_exp = sum_exp * torch.exp(
                                max_val - new_max
                            ) + torch.exp(score - new_max[:, None]).sum(dim=-1)
                            max_val = new_max
                        lse_j = max_val + torch.log(sum_exp + 1e-30)

                        # Second pass: accumulate gradients
                        gl_v = grad_log_v[b_idx, h_idx, j0:je]
                        for i0 in range(0, N, TILE):
                            ie = min(i0 + TILE, N)
                            C_t, log_K_t, dist_t, diff_t = _compute_cost_tile_py(
                                Q_ln[b_idx, h_idx],
                                K_ln[b_idx, h_idx],
                                x_res[b_idx],
                                pos_weight[h_idx],
                                pos_bins[b_idx],
                                w_h,
                                r_0,
                                slice(i0, ie),
                                slice(j0, je),
                                eps_h,
                            )
                            score = (
                                log_K_t
                                + log_u_after[b_idx, h_idx, i0:ie, None]
                                + mask_bias[b_idx, None, i0:ie]
                            ).T
                            s_tile = torch.exp(score - lse_j[:, None])  # (tj, ti)

                            # grad_log_u_after_i += -κ · Σ_j grad_log_v_j · s'_ji
                            grad_log_u[b_idx, h_idx, i0:ie] += -kappa_h * (
                                s_tile.T @ gl_v
                            )
                            # grad_C from col update
                            gc = (
                                (kappa_h / eps_h)
                                * torch.outer(gl_v, torch.ones(ie - i0, device=device))
                                * s_tile
                            )
                            inv_sqrt_dh = 1.0 / (d_h**0.5)
                            grad_Q_ln[b_idx, h_idx, i0:ie] += (-inv_sqrt_dh) * (
                                gc.T @ K_ln[b_idx, h_idx, j0:je]
                            )
                            grad_K_ln[b_idx, h_idx, j0:je] += (-inv_sqrt_dh) * (
                                gc @ Q_ln[b_idx, h_idx, i0:ie]
                            )
                            if dist_t is not None:
                                geo_gc = w_h * r_0 / (r_0 + dist_t) ** 2
                                wt = gc.T * geo_gc / dist_t.clamp(min=1e-8)
                                grad_x_cost[b_idx, i0:ie] += (
                                    wt.unsqueeze(-1) * diff_t
                                ).sum(dim=1)
                                grad_x_cost[b_idx, j0:je] -= (
                                    wt.unsqueeze(-1) * diff_t
                                ).sum(dim=0)
                                grad_w_dist[h_idx] += (
                                    gc.T * dist_t / (r_0 + dist_t)
                                ).sum()
                            bins_t = pos_bins[b_idx, i0:ie, j0:je].long().reshape(-1)
                            grad_pos_weight[h_idx].scatter_add_(0, bins_t, gc.T.reshape(-1))

                        grad_log_nu[b_idx, h_idx, j0:je] += kappa_h * gl_v

            # -- Row update backward: log_u_after = κ(log_mu - LSE_j(-C/ε + log_v_before)) --
            # grad_log_u now has contributions from col update backward above
            grad_log_v_new = torch.zeros(B, H, N, device=device, dtype=torch.float32)
            for b_idx in range(B):
                for h_idx in range(H):
                    eps_h = eps[h_idx].item()
                    kappa_h = kappa[h_idx].item()
                    w_h = w_dist[h_idx].item()
                    for i0 in range(0, N, TILE):
                        ie = min(i0 + TILE, N)
                        max_val = torch.full((ie - i0,), -1e30, device=device)
                        sum_exp = torch.zeros(ie - i0, device=device)
                        for j0 in range(0, N, TILE):
                            je = min(j0 + TILE, N)
                            _, log_K_t, _, _ = _compute_cost_tile_py(
                                Q_ln[b_idx, h_idx],
                                K_ln[b_idx, h_idx],
                                x_res[b_idx],
                                pos_weight[h_idx],
                                pos_bins[b_idx],
                                w_h,
                                r_0,
                                slice(i0, ie),
                                slice(j0, je),
                                eps_h,
                            )
                            score = (
                                log_K_t
                                + log_v_before[b_idx, h_idx, None, j0:je]
                                + mask_bias[b_idx, None, j0:je]
                            )
                            tile_max = score.max(dim=-1).values
                            new_max = torch.maximum(max_val, tile_max)
                            sum_exp = sum_exp * torch.exp(
                                max_val - new_max
                            ) + torch.exp(score - new_max[:, None]).sum(dim=-1)
                            max_val = new_max
                        lse_i = max_val + torch.log(sum_exp + 1e-30)

                        gl_u = grad_log_u[b_idx, h_idx, i0:ie]
                        for j0 in range(0, N, TILE):
                            je = min(j0 + TILE, N)
                            C_t, log_K_t, dist_t, diff_t = _compute_cost_tile_py(
                                Q_ln[b_idx, h_idx],
                                K_ln[b_idx, h_idx],
                                x_res[b_idx],
                                pos_weight[h_idx],
                                pos_bins[b_idx],
                                w_h,
                                r_0,
                                slice(i0, ie),
                                slice(j0, je),
                                eps_h,
                            )
                            score = (
                                log_K_t
                                + log_v_before[b_idx, h_idx, None, j0:je]
                                + mask_bias[b_idx, None, j0:je]
                            )
                            s_tile = torch.exp(score - lse_i[:, None])  # (ti, tj)

                            grad_log_v_new[b_idx, h_idx, j0:je] += -kappa_h * (
                                s_tile.T @ gl_u
                            )
                            gc = (
                                (kappa_h / eps_h)
                                * torch.outer(gl_u, torch.ones(je - j0, device=device))
                                * s_tile
                            )
                            inv_sqrt_dh = 1.0 / (d_h**0.5)
                            grad_Q_ln[b_idx, h_idx, i0:ie] += (-inv_sqrt_dh) * (
                                gc @ K_ln[b_idx, h_idx, j0:je]
                            )
                            grad_K_ln[b_idx, h_idx, j0:je] += (-inv_sqrt_dh) * (
                                gc.T @ Q_ln[b_idx, h_idx, i0:ie]
                            )
                            if dist_t is not None:
                                geo_gc = w_h * r_0 / (r_0 + dist_t) ** 2
                                wt = gc * geo_gc / dist_t.clamp(min=1e-8)
                                grad_x_cost[b_idx, i0:ie] += (
                                    wt.unsqueeze(-1) * diff_t
                                ).sum(dim=1)
                                grad_x_cost[b_idx, j0:je] -= (
                                    wt.unsqueeze(-1) * diff_t
                                ).sum(dim=0)
                                grad_w_dist[h_idx] += (
                                    gc * dist_t / (r_0 + dist_t)
                                ).sum()
                            bins_t = pos_bins[b_idx, i0:ie, j0:je].long().reshape(-1)
                            grad_pos_weight[h_idx].scatter_add_(0, bins_t, gc.reshape(-1))

                        grad_log_mu[b_idx, h_idx, i0:ie] += kappa_h * gl_u

            # Prepare for next iteration (going backward)
            grad_log_v = grad_log_v_new
            grad_log_u = torch.zeros(B, H, N, device=device, dtype=torch.float32)

        # ---- Step 4: Cost gradient from transport output (already in _backward_gv_grad_V_kernel) ----
        # The cost gradient from the transport output was computed in Step 2 via g_v.
        # The unrolled backward above adds the cost gradient from the Sinkhorn iterations.
        # Combine x_res gradients
        grad_x_res = grad_x_transport + grad_x_cost

        # Return gradients for: Q_ln, K_ln, V, x_res, pos_weight, pos_bins, eps, w_dist,
        #   log_mu, log_nu, K_iter, lam, r_0, log_u_init, log_v_init, BLOCK, mask_bias
        return (
            grad_Q_ln,
            grad_K_ln,
            grad_V,
            grad_x_res,
            grad_pos_weight,
            None,  # pos_bins (int, no grad)
            None,  # eps
            grad_w_dist,
            grad_log_mu,
            grad_log_nu,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # mask_bias
        )


def flash_sinkhorn(
    Q_ln: torch.Tensor,
    K_ln: torch.Tensor,
    V: torch.Tensor,
    x_res: torch.Tensor,
    pos_weight: torch.Tensor,
    pos_bins: torch.Tensor,
    eps: torch.Tensor,
    w_dist: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    K_iter: int = 7,
    lam: float = 1.0,
    r_0: float = 10.0,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
    BLOCK_N: int = 64,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash-Sinkhorn with IFT backward.

    Accepts both unbatched (H, N, d_h) and batched (B, H, N, d_h) inputs.

    Args:
        pos_weight: (H, 68) learnable position bias weights
        pos_bins:   (B, N, N) or (N, N) int32 bin indices
        mask: (B, N) or (N,) float tensor, 1=valid, 0=pad. None means all valid.

    Returns:
        O_avg:      (B, H, N, d_h) transport-weighted value output
        x_centroid: (B, H, N, 3)   EGNN centroid
        log_u:      (B, H, N)      converged row duals
        log_v:      (B, H, N)      converged column duals

    When called with unbatched inputs, B=1 is added internally and results
    are returned with B dimension.
    """
    # Handle unbatched inputs: add B=1
    unbatched = Q_ln.dim() == 3
    if unbatched:
        Q_ln = Q_ln.unsqueeze(0)
        K_ln = K_ln.unsqueeze(0)
        V = V.unsqueeze(0)
        x_res = x_res.unsqueeze(0)
        pos_bins = pos_bins.unsqueeze(0)
        log_mu = log_mu.unsqueeze(0)
        log_nu = log_nu.unsqueeze(0)
        if log_u_init is not None:
            log_u_init = log_u_init.unsqueeze(0)
        if log_v_init is not None:
            log_v_init = log_v_init.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, _, N, _ = Q_ln.shape
    if mask is None:
        mask_bias = torch.zeros(B, N, device=Q_ln.device, dtype=torch.float32)
    else:
        mask_bias = (1.0 - mask.float()) * (-1e9)

    result = FlashSinkhornFunction.apply(
        Q_ln,
        K_ln,
        V,
        x_res,
        pos_weight,
        pos_bins,
        eps,
        w_dist,
        log_mu,
        log_nu,
        K_iter,
        lam,
        r_0,
        log_u_init,
        log_v_init,
        BLOCK_N,
        mask_bias,
    )

    if unbatched:
        return tuple(t.squeeze(0) for t in result)
    return result
