"""
Flash-Sinkhorn Triton kernel (SPEC §18 item 1).

Optimized two-pass architecture:
  Forward:
    Sinkhorn iterations via separate row/col update kernels (parallel across H×N_tiles)
    Transport output: T_norm @ V and T_norm @ x (EGNN centroid) with online softmax
  Backward (IFT):
    Adjoint iterations with same tiling structure (SPEC §7.5)
    Cost gradient computed on-the-fly

Key optimizations over naive version:
  - tl.dot for QK^T (tensor cores)
  - Grid=(H, N_tiles) for full parallelism across heads and tiles
  - Distances computed on-the-fly per tile (no O(N²) storage)
  - Separate row/col update kernels (avoid serial bottleneck)
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Helper: compute cost tile on-the-fly
# ============================================================================


@triton.jit
def _compute_cost_tile(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
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
    stride_ph,
    stride_pn,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute cost tile C[i, j] = content + pos + geo. Returns (BLOCK_I, BLOCK_J)."""
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

    # Position bias: (BLOCK_I, BLOCK_J)
    pos_ptrs = (
        POS_BIAS_ptr + pid_h * stride_ph + i_idx[:, None] * stride_pn + j_idx[None, :]
    )
    pos = tl.load(pos_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

    return content + pos + geo


@triton.jit
def _load_x_components(X_ptr, idx, mask, stride_xn, BLOCK: tl.constexpr):
    """Load x, y, z components for a tile of coordinates."""
    x0 = tl.load(X_ptr + idx * stride_xn + 0, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + idx * stride_xn + 1, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + idx * stride_xn + 2, mask=mask, other=0.0)
    return x0, x1, x2


@triton.jit
def _compute_dist_tile(
    X_ptr,
    i_idx,
    j_idx,
    i_mask,
    j_mask,
    stride_xn,
    r_0,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute pairwise distances and displacement components for a tile.

    Returns: dist (BI,BJ), dx (BI,BJ), dy (BI,BJ), dz (BI,BJ), f_dist (BI,BJ).
    """
    xi_0, xi_1, xi_2 = _load_x_components(X_ptr, i_idx, i_mask, stride_xn, BLOCK_I)
    xj_0, xj_1, xj_2 = _load_x_components(X_ptr, j_idx, j_mask, stride_xn, BLOCK_J)
    dx = xi_0[:, None] - xj_0[None, :]
    dy = xi_1[:, None] - xj_1[None, :]
    dz = xi_2[:, None] - xj_2[None, :]
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
    f_dist = dist / (r_0 + dist)
    return dist, dx, dy, dz, f_dist


# ============================================================================
# Backward Step 1: Compute log_Z, D, g_v, grad_V, grad_x_transport
# ============================================================================


@triton.jit
def _compute_log_Z(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    LOG_Z_ptr,  # output (H, N)
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """log_Z[h,i] = LSE_j(log_u[h,i] + log_K[h,i,j] + log_v[h,j])."""
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        score = log_u_i[:, None] + (-C_tile * inv_eps_h) + log_v_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        LOG_Z_ptr + pid_h * N + i_idx, max_val + tl.log(sum_exp + 1e-30), mask=i_mask
    )


@triton.jit
def _backward_gv_grad_V_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    V_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    LOG_Z_ptr,
    GRAD_O_ptr,
    GRAD_XC_ptr,  # upstream: (H, N, d_h), (H, N, 3)
    D_ptr,  # D[h,i] = grad_O·O_avg + grad_xc·xc (precomputed)
    G_V_ptr,  # output: g_v (H, N)
    GRAD_V_ptr,  # output: grad_V (H, N, d_h) — atomic add
    GRAD_X_TRANSPORT_ptr,  # output: grad_x_transport (N, 3) — atomic add
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    stride_vh,
    stride_vn,
    BLOCK: tl.constexpr,
):
    """Backward through T_norm: compute g_v, accumulate grad_V and grad_x_transport.

    For each j-tile, reduces over all i-tiles:
      g_v[j] = sum_i T_norm_ij * (dT_ij - D_i)
      grad_V[j] += sum_i T_norm_ij * grad_O_avg[i]
      grad_x_transport[j] += sum_i T_norm_ij * grad_xc[i]
    """
    pid_h = tl.program_id(0)
    pid_j = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N
    log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

    # Accumulators
    acc_gv = tl.zeros([BLOCK], dtype=tl.float32)
    d_idx = tl.arange(0, D_H)
    acc_grad_V = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_gx_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_2 = tl.zeros([BLOCK], dtype=tl.float32)

    # Load V[j]: (BLOCK, D_H) — needed for dT computation
    v_ptrs = V_ptr + pid_h * stride_vh + j_idx[:, None] * stride_vn + d_idx[None, :]
    V_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)

    # Load x_res[j]: (BLOCK, 3)
    xj_0, xj_1, xj_2 = _load_x_components(X_ptr, j_idx, j_mask, stride_xn, BLOCK)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)
        log_Z_i = tl.load(LOG_Z_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
        D_i = tl.load(D_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h
        log_score = log_u_i[:, None] + log_K_tile + log_v_j[None, :]
        T_norm_tile = tl.exp(log_score - log_Z_i[:, None])
        T_norm_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_norm_tile, 0.0)

        # Load grad_O_avg[i]: (BLOCK_I, D_H)
        go_ptrs = (
            GRAD_O_ptr + pid_h * stride_vh + i_idx[:, None] * stride_vn + d_idx[None, :]
        )
        grad_O_i = tl.load(go_ptrs, mask=i_mask[:, None], other=0.0)

        # Load grad_xc[i]: (BLOCK_I, 3)
        gxc_base = GRAD_XC_ptr + pid_h * N * 3
        gxc_i0 = tl.load(gxc_base + i_idx * 3 + 0, mask=i_mask, other=0.0)
        gxc_i1 = tl.load(gxc_base + i_idx * 3 + 1, mask=i_mask, other=0.0)
        gxc_i2 = tl.load(gxc_base + i_idx * 3 + 2, mask=i_mask, other=0.0)

        # dT_ij = grad_O_avg[i] · V[j]^T + grad_xc[i] · x_res[j]^T
        dT_attn = tl.dot(grad_O_i, tl.trans(V_j))  # (BLOCK_I, BLOCK_J)
        dT_xc = (
            gxc_i0[:, None] * xj_0[None, :]
            + gxc_i1[:, None] * xj_1[None, :]
            + gxc_i2[:, None] * xj_2[None, :]
        )
        dT = dT_attn + dT_xc

        # grad_log_score = T_norm * (dT - D_i)
        grad_ls = T_norm_tile * (dT - D_i[:, None])

        # g_v[j] = sum_i grad_ls[i, j]
        acc_gv += tl.sum(grad_ls, axis=0)

        # grad_V[j] += T_norm^T @ grad_O_avg[i]  (reduce over i)
        acc_grad_V += tl.dot(tl.trans(T_norm_tile), grad_O_i)

        # grad_x_transport[j] += T_norm^T @ grad_xc[i]
        T_t_gxc_0 = tl.sum(T_norm_tile * gxc_i0[:, None], axis=0)
        T_t_gxc_1 = tl.sum(T_norm_tile * gxc_i1[:, None], axis=0)
        T_t_gxc_2 = tl.sum(T_norm_tile * gxc_i2[:, None], axis=0)
        acc_gx_0 += T_t_gxc_0
        acc_gx_1 += T_t_gxc_1
        acc_gx_2 += T_t_gxc_2

    # Store g_v
    tl.store(G_V_ptr + pid_h * N + j_idx, acc_gv, mask=j_mask)

    # Atomic add grad_V
    gv_ptrs = (
        GRAD_V_ptr + pid_h * stride_vh + j_idx[:, None] * stride_vn + d_idx[None, :]
    )
    tl.atomic_add(gv_ptrs, acc_grad_V, mask=j_mask[:, None])

    # Atomic add grad_x_transport (sum over heads, hence atomic)
    tl.atomic_add(GRAD_X_TRANSPORT_ptr + j_idx * 3 + 0, acc_gx_0, mask=j_mask)
    tl.atomic_add(GRAD_X_TRANSPORT_ptr + j_idx * 3 + 1, acc_gx_1, mask=j_mask)
    tl.atomic_add(GRAD_X_TRANSPORT_ptr + j_idx * 3 + 2, acc_gx_2, mask=j_mask)


# ============================================================================
# Backward Step 4: Cost gradient propagation
# ============================================================================


@triton.jit
def _cost_gradient_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
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
    GRAD_X_COST_ptr,  # (N, 3)
    GRAD_W_DIST_ptr,  # (H,)
    GRAD_POS_BIAS_ptr,  # (H, N, N) — direct store
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    stride_vh,
    stride_vn,
    BLOCK: tl.constexpr,
):
    """Fused cost gradient: direct (through T_norm) + IFT (through z_u, z_v).

    Propagates grad_C to Q_ln, K_ln, x_res, w_dist, pos_bias.
    Each program handles one (h, i-tile) and iterates over all j-tiles.
    """
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)
    inv_sqrt_dh = 1.0 / tl.sqrt(D_H * 1.0)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
    log_Z_i = tl.load(LOG_Z_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
    row_lse_i = tl.load(ROW_LSE_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
    D_i = tl.load(D_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
    z_u_i = tl.load(Z_U_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)

    # Load grad_O_avg[i]: (BLOCK, D_H) for dT computation
    d_idx = tl.arange(0, D_H)
    go_ptrs = (
        GRAD_O_ptr + pid_h * stride_vh + i_idx[:, None] * stride_vn + d_idx[None, :]
    )
    grad_O_i = tl.load(go_ptrs, mask=i_mask[:, None], other=0.0)

    # Load grad_xc[i]: 3 components
    gxc_base = GRAD_XC_ptr + pid_h * N * 3
    gxc_i0 = tl.load(gxc_base + i_idx * 3 + 0, mask=i_mask, other=0.0)
    gxc_i1 = tl.load(gxc_base + i_idx * 3 + 1, mask=i_mask, other=0.0)
    gxc_i2 = tl.load(gxc_base + i_idx * 3 + 2, mask=i_mask, other=0.0)

    # Load Q_ln[i]: (BLOCK, D_H)
    q_ptrs = Q_ptr + pid_h * stride_qh + i_idx[:, None] * stride_qn + d_idx[None, :]
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

        log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)
        col_lse_j = tl.load(COL_LSE_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)
        z_v_j = tl.load(Z_V_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)

        # Load K_ln[j], V[j], x[j]
        k_ptrs = K_ptr + pid_h * stride_qh + j_idx[:, None] * stride_qn + d_idx[None, :]
        K_j = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)

        v_ptrs = V_ptr + pid_h * stride_vh + j_idx[:, None] * stride_vn + d_idx[None, :]
        V_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)

        xj_0, xj_1, xj_2 = _load_x_components(X_ptr, j_idx, j_mask, stride_xn, BLOCK)

        # Recompute cost tile
        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        # --- Direct gradient through T_norm ---
        log_score = log_u_i[:, None] + log_K_tile + log_v_j[None, :]
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

        grad_C_total = grad_C_direct + grad_C_ift  # (BLOCK, BLOCK)

        # --- Propagate to Q_ln ---
        # content = -(Q @ K^T) / sqrt(d_h)
        # grad_Q += grad_C @ (-K / sqrt(d_h))
        acc_grad_Q += tl.dot(grad_C_total, K_j) * (-inv_sqrt_dh)

        # --- Propagate to K_ln (atomic add since j varies) ---
        grad_K_tile = tl.dot(tl.trans(grad_C_total), Q_i) * (-inv_sqrt_dh)
        gk_ptrs = (
            GRAD_K_ptr + pid_h * stride_qh + j_idx[:, None] * stride_qn + d_idx[None, :]
        )
        tl.atomic_add(gk_ptrs, grad_K_tile, mask=j_mask[:, None])

        # --- Propagate to x_res through geometry ---
        # geo = w_dist * d/(r_0+d), d(d/(r_0+d))/dd = r_0/(r_0+d)^2
        # dd/dx_i = (x_i-x_j)/d
        xi_0, xi_1, xi_2 = _load_x_components(X_ptr, i_idx, i_mask, stride_xn, BLOCK)
        dx = xi_0[:, None] - xj_0[None, :]
        dy = xi_1[:, None] - xj_1[None, :]
        dz = xi_2[:, None] - xj_2[None, :]
        dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)

        geo_grad_coeff = w_dist_h * R_0 / ((R_0 + dist) * (R_0 + dist))
        weighted = grad_C_total * geo_grad_coeff / dist  # (BLOCK, BLOCK)

        acc_gx_i_0 += tl.sum(weighted * dx, axis=1)
        acc_gx_i_1 += tl.sum(weighted * dy, axis=1)
        acc_gx_i_2 += tl.sum(weighted * dz, axis=1)

        # grad_x[j] -= weighted * diff (negative sign from x_j)
        gx_j_0 = -tl.sum(weighted * dx, axis=0)
        gx_j_1 = -tl.sum(weighted * dy, axis=0)
        gx_j_2 = -tl.sum(weighted * dz, axis=0)
        tl.atomic_add(GRAD_X_COST_ptr + j_idx * 3 + 0, gx_j_0, mask=j_mask)
        tl.atomic_add(GRAD_X_COST_ptr + j_idx * 3 + 1, gx_j_1, mask=j_mask)
        tl.atomic_add(GRAD_X_COST_ptr + j_idx * 3 + 2, gx_j_2, mask=j_mask)

        # --- Propagate to w_dist ---
        f_dist_tile = dist / (R_0 + dist)
        acc_grad_w_dist += tl.sum(grad_C_total * f_dist_tile)

        # --- Store grad_pos_bias directly (unique i,j per program) ---
        gp_ptrs = (
            GRAD_POS_BIAS_ptr
            + pid_h * stride_ph
            + i_idx[:, None] * stride_pn
            + j_idx[None, :]
        )
        tl.store(gp_ptrs, grad_C_total, mask=i_mask[:, None] & j_mask[None, :])

    # Store accumulated gradients
    gq_ptrs = (
        GRAD_Q_ptr + pid_h * stride_qh + i_idx[:, None] * stride_qn + d_idx[None, :]
    )
    tl.store(gq_ptrs, acc_grad_Q, mask=i_mask[:, None])

    # Atomic add grad_x_cost for i
    tl.atomic_add(GRAD_X_COST_ptr + i_idx * 3 + 0, acc_gx_i_0, mask=i_mask)
    tl.atomic_add(GRAD_X_COST_ptr + i_idx * 3 + 1, acc_gx_i_1, mask=i_mask)
    tl.atomic_add(GRAD_X_COST_ptr + i_idx * 3 + 2, acc_gx_i_2, mask=i_mask)

    # Atomic add grad_w_dist (one scalar per head)
    tl.atomic_add(GRAD_W_DIST_ptr + pid_h, acc_grad_w_dist)


# ============================================================================
# Sinkhorn Row Update: log_u_i = kappa * (log_mu_i - LSE_j(log_K_ij + log_v_j))
# ============================================================================


@triton.jit
def _sinkhorn_row_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_MU_ptr,
    LOG_V_ptr,
    LOG_U_ptr,  # output
    N: tl.constexpr,
    D_H: tl.constexpr,
    LAM: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    kappa_h = LAM / (LAM + eps_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_mu_i = tl.load(LOG_MU_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)

    # Online LSE over j tiles
    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N

        log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        score = log_K_tile + log_v_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    lse = max_val + tl.log(sum_exp + 1e-30)
    log_u_new = kappa_h * (log_mu_i - lse)
    tl.store(LOG_U_ptr + pid_h * N + i_idx, log_u_new, mask=i_mask)


# ============================================================================
# Sinkhorn Col Update: log_v_j = kappa * (log_nu_j - LSE_i(log_K_ij + log_u_i))
# ============================================================================


@triton.jit
def _sinkhorn_col_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_NU_ptr,
    LOG_U_ptr,
    LOG_V_ptr,  # output
    N: tl.constexpr,
    D_H: tl.constexpr,
    LAM: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_j = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    kappa_h = LAM / (LAM + eps_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N
    log_nu_j = tl.load(LOG_NU_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N

        log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        # score[i, j] = log_K[i,j] + log_u[i], reduce over i for each j
        score = log_K_tile + log_u_i[:, None]
        score = tl.where(i_mask[:, None], score, -1e30)

        # Transpose to get (BLOCK_J, BLOCK_I) for per-j reduction
        score_t = tl.trans(score)
        tile_max = tl.max(score_t, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score_t - new_max[:, None]), axis=1
        )
        max_val = new_max

    lse = max_val + tl.log(sum_exp + 1e-30)
    log_v_new = kappa_h * (log_nu_j - lse)
    tl.store(LOG_V_ptr + pid_h * N + j_idx, log_v_new, mask=j_mask)


# ============================================================================
# Transport Output: O_avg = T_norm @ V, x_centroid = T_norm @ x_res
# ============================================================================


@triton.jit
def _transport_output_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    V_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    LOG_V_ptr,
    O_ptr,
    X_CENT_ptr,
    ROW_SUM_ptr,
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    stride_vh,
    stride_vn,
    stride_oh,
    stride_on,
    stride_xch,
    stride_xcn,
    BLOCK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)

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

        log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        log_K_tile = -C_tile * inv_eps_h

        log_score = log_u_i[:, None] + log_K_tile + log_v_j[None, :]
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

        # T_tile @ V_j: (BLOCK_I, BLOCK_J) @ (BLOCK_J, D_H) → (BLOCK_I, D_H)
        v_ptrs = (
            V_ptr
            + pid_h * stride_vh
            + j_idx[:, None] * stride_vn
            + tl.arange(0, D_H)[None, :]
        )
        V_tile = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)
        o_acc += tl.dot(T_tile.to(V_tile.dtype), V_tile)

        # T_tile @ x_j for EGNN centroid
        xj_0 = tl.load(X_ptr + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
        xj_1 = tl.load(X_ptr + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
        xj_2 = tl.load(X_ptr + j_idx * stride_xn + 2, mask=j_mask, other=0.0)
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
        + pid_h * stride_oh
        + i_idx[:, None] * stride_on
        + tl.arange(0, D_H)[None, :]
    )
    tl.store(o_ptrs, o_acc, mask=i_mask[:, None])

    # Store x_centroid
    tl.store(
        X_CENT_ptr + pid_h * stride_xch + i_idx * stride_xcn + 0,
        xc_0 * inv_sum,
        mask=i_mask,
    )
    tl.store(
        X_CENT_ptr + pid_h * stride_xch + i_idx * stride_xcn + 1,
        xc_1 * inv_sum,
        mask=i_mask,
    )
    tl.store(
        X_CENT_ptr + pid_h * stride_xch + i_idx * stride_xcn + 2,
        xc_2 * inv_sum,
        mask=i_mask,
    )
    tl.store(ROW_SUM_ptr + pid_h * N + i_idx, row_sum, mask=i_mask)


# ============================================================================
# IFT Backward: Compute col_lse / row_lse (needed for softmax weights)
# ============================================================================


@triton.jit
def _compute_row_lse(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_V_ptr,
    ROW_LSE_ptr,  # output (H, N)
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """row_lse[h,i] = LSE_j(log_K[h,i,j] + log_v[h,j])."""
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        score = -C_tile * inv_eps_h + log_v_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)

        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        ROW_LSE_ptr + pid_h * N + i_idx, max_val + tl.log(sum_exp + 1e-30), mask=i_mask
    )


@triton.jit
def _compute_col_lse(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    COL_LSE_ptr,  # output (H, N)
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """col_lse[h,j] = LSE_i(log_K[h,i,j] + log_u[h,i])."""
    pid_h = tl.program_id(0)
    pid_j = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N

    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=-1e30)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        score = -C_tile * inv_eps_h + log_u_i[:, None]
        score = tl.where(i_mask[:, None], score, -1e30)
        score_t = tl.trans(score)

        tile_max = tl.max(score_t, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(score_t - new_max[:, None]), axis=1
        )
        max_val = new_max

    tl.store(
        COL_LSE_ptr + pid_h * N + j_idx, max_val + tl.log(sum_exp + 1e-30), mask=j_mask
    )


# ============================================================================
# IFT Adjoint: z_u update and z_v update
# ============================================================================


@triton.jit
def _ift_z_u_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_U_ptr,
    COL_LSE_ptr,
    G_U_ptr,
    Z_V_ptr,
    Z_U_ptr,  # g_u, z_v (input), z_u (output)
    KAPPA_ptr,
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """z_u[i] = g_u[i] + kappa * sum_j S_col[i,j] * z_v[j]."""
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    i_mask = i_idx < N

    log_u_i = tl.load(LOG_U_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
    g_u_i = tl.load(G_U_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK)
        j_mask = j_idx < N

        z_v_j = tl.load(Z_V_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)
        col_lse_j = tl.load(COL_LSE_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        # S_col[i,j] = exp(log_K[i,j] + log_u[i] - col_lse[j])
        log_s = -C_tile * inv_eps_h + log_u_i[:, None] - col_lse_j[None, :]
        s_col_tile = tl.exp(log_s)
        s_col_tile = tl.where(j_mask[None, :], s_col_tile, 0.0)

        acc += tl.sum(s_col_tile * z_v_j[None, :], axis=1)

    z_u_new = g_u_i + kappa_h * acc
    tl.store(Z_U_ptr + pid_h * N + i_idx, z_u_new, mask=i_mask)


@triton.jit
def _ift_z_v_update(
    Q_ptr,
    K_ptr,
    X_ptr,
    POS_BIAS_ptr,
    EPS_ptr,
    W_DIST_ptr,
    LOG_V_ptr,
    ROW_LSE_ptr,
    G_V_ptr,
    Z_U_ptr,
    Z_V_ptr,  # g_v, z_u (input), z_v (output)
    KAPPA_ptr,
    N: tl.constexpr,
    D_H: tl.constexpr,
    R_0: tl.constexpr,
    stride_qh,
    stride_qn,
    stride_xn,
    stride_ph,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """z_v[j] = g_v[j] + kappa * sum_i S_row[i,j] * z_u[i]."""
    pid_h = tl.program_id(0)
    pid_j = tl.program_id(1)

    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_dist_h = tl.load(W_DIST_ptr + pid_h)
    kappa_h = tl.load(KAPPA_ptr + pid_h)

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_idx < N

    log_v_j = tl.load(LOG_V_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)
    g_v_j = tl.load(G_V_ptr + pid_h * N + j_idx, mask=j_mask, other=0.0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK)
        i_mask = i_idx < N

        z_u_i = tl.load(Z_U_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)
        row_lse_i = tl.load(ROW_LSE_ptr + pid_h * N + i_idx, mask=i_mask, other=0.0)

        C_tile = _compute_cost_tile(
            Q_ptr,
            K_ptr,
            X_ptr,
            POS_BIAS_ptr,
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
            stride_ph,
            stride_pn,
            BLOCK,
            BLOCK,
        )
        # S_row[i,j] = exp(log_K[i,j] + log_v[j] - row_lse[i])
        log_s = -C_tile * inv_eps_h + log_v_j[None, :] - row_lse_i[:, None]
        s_row_tile = tl.exp(log_s)
        s_row_tile = tl.where(i_mask[:, None], s_row_tile, 0.0)

        # sum over i: (BLOCK_J,)
        acc += tl.sum(s_row_tile * z_u_i[:, None], axis=0)

    z_v_new = g_v_j + kappa_h * acc
    tl.store(Z_V_ptr + pid_h * N + j_idx, z_v_new, mask=j_mask)


# ============================================================================
# Python wrapper + autograd Function
# ============================================================================


def _launch_common_args(Q_ln, K_ln, x_res, pos_bias, eps, w_dist, N, D_H):
    return (
        Q_ln,
        K_ln,
        x_res,
        pos_bias,
        eps,
        w_dist,
        N,
        D_H,
        1.0,  # LAM
        10.0,  # R_0
        Q_ln.stride(0),
        Q_ln.stride(1),
        x_res.stride(0),
        pos_bias.stride(0),
        pos_bias.stride(1),
    )


class FlashSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q_ln,
        K_ln,
        V,
        x_res,
        pos_bias,
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
    ):
        H, N, d_h = Q_ln.shape
        device = Q_ln.device

        # Ensure FP32 contiguous
        Q_ln = Q_ln.contiguous().float()
        K_ln = K_ln.contiguous().float()
        V = V.contiguous().float()
        x_res = x_res.contiguous().float()
        pos_bias = pos_bias.contiguous().float()
        eps = eps.contiguous().float()
        w_dist = w_dist.contiguous().float()
        log_mu = log_mu.contiguous().float()
        log_nu = log_nu.contiguous().float()

        log_u = torch.zeros(H, N, device=device, dtype=torch.float32)
        log_v = torch.zeros(H, N, device=device, dtype=torch.float32)

        if log_u_init is not None:
            log_u.copy_(log_u_init.float())
        if log_v_init is not None:
            log_v.copy_(log_v_init.float())

        n_tiles = (N + BLOCK - 1) // BLOCK
        grid = (H, n_tiles)

        common = (
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            eps,
            w_dist,
        )
        dim_args = (N, d_h, lam, r_0)
        stride_args = (
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            BLOCK,
        )

        # Sinkhorn iterations: 2K kernel launches, each fully parallel
        for _ in range(K_iter):
            _sinkhorn_row_update[grid](
                *common,
                log_mu,
                log_v,
                log_u,
                *dim_args,
                *stride_args,
            )
            _sinkhorn_col_update[grid](
                *common,
                log_nu,
                log_u,
                log_v,
                *dim_args,
                *stride_args,
            )

        # Transport output
        O_avg = torch.zeros(H, N, d_h, device=device, dtype=torch.float32)
        x_centroid = torch.zeros(H, N, 3, device=device, dtype=torch.float32)
        row_sum = torch.zeros(H, N, device=device, dtype=torch.float32)

        _transport_output_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            row_sum,
            N,
            d_h,
            r_0,
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            V.stride(0),
            V.stride(1),
            O_avg.stride(0),
            O_avg.stride(1),
            x_centroid.stride(0),
            x_centroid.stride(1),
            BLOCK,
        )

        ctx.save_for_backward(
            Q_ln,
            K_ln,
            V,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_u,
            log_v,
            log_mu,
            log_nu,
            row_sum,
        )
        ctx.K_iter = K_iter
        ctx.BLOCK = BLOCK
        ctx.lam = lam
        ctx.r_0 = r_0

        return O_avg, x_centroid, log_u, log_v

    @staticmethod
    def backward(ctx, grad_O, grad_xc, grad_lu, grad_lv):
        (
            Q_ln,
            K_ln,
            V,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_u,
            log_v,
            log_mu,
            log_nu,
            row_sum,
        ) = ctx.saved_tensors
        K_back = ctx.K_iter
        BLOCK = ctx.BLOCK
        lam = ctx.lam
        r_0 = ctx.r_0

        H, N, d_h = Q_ln.shape
        device = Q_ln.device
        kappa = (lam / (lam + eps)).contiguous()

        # Cap BLOCK at 32 for backward — SRAM-heavy kernels need smaller tiles
        BLOCK_BWD = min(BLOCK, 32)
        n_tiles = (N + BLOCK_BWD - 1) // BLOCK_BWD
        grid = (H, n_tiles)

        stride_args = (
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            BLOCK_BWD,
        )

        if grad_O is None:
            grad_O = torch.zeros(H, N, d_h, device=device, dtype=torch.float32)
        else:
            grad_O = grad_O.contiguous().float()
        if grad_xc is None:
            grad_xc = torch.zeros(H, N, 3, device=device, dtype=torch.float32)
        else:
            grad_xc = grad_xc.contiguous().float()

        # ---- Step 1: Compute log_Z and D ----
        log_Z = torch.empty(H, N, device=device, dtype=torch.float32)
        _compute_log_Z[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_u,
            log_v,
            log_Z,
            N,
            d_h,
            r_0,
            *stride_args,
        )

        # D[h,i] = sum_d grad_O[h,i,d]*O_avg[h,i,d] + sum_c grad_xc[h,i,c]*xc[h,i,c]
        # Recompute O_avg, x_centroid from row_sum is not easily available.
        # Instead, recompute from forward saved tensors.
        # O_avg = T_norm @ V was not saved, but we saved row_sum.
        # Actually we need to recompute O_avg and x_centroid — or save them.
        # The forward already ran the transport output kernel which computed them.
        # But they weren't saved. We need to recompute via another tiled pass.
        # More efficient: save O_avg and x_centroid in forward (small: O(N*d)).
        # For now, recompute D in the _backward_gv_grad_V_kernel alongside g_v.
        # Wait — D = grad_O · O_avg + grad_xc · xc requires O_avg and xc.
        # Let's recompute them with a second transport output kernel call.
        O_avg_recomp = torch.zeros(H, N, d_h, device=device, dtype=torch.float32)
        x_cent_recomp = torch.zeros(H, N, 3, device=device, dtype=torch.float32)
        row_sum_recomp = torch.zeros(H, N, device=device, dtype=torch.float32)
        _transport_output_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            O_avg_recomp,
            x_cent_recomp,
            row_sum_recomp,
            N,
            d_h,
            r_0,
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            V.stride(0),
            V.stride(1),
            O_avg_recomp.stride(0),
            O_avg_recomp.stride(1),
            x_cent_recomp.stride(0),
            x_cent_recomp.stride(1),
            BLOCK_BWD,
        )

        D = (grad_O * O_avg_recomp).sum(dim=-1) + (grad_xc * x_cent_recomp).sum(
            dim=-1
        )  # (H, N)

        # ---- Step 2: Compute g_v, grad_V, grad_x_transport ----
        g_v = torch.zeros(H, N, device=device, dtype=torch.float32)
        grad_V = torch.zeros_like(V)
        grad_x_transport = torch.zeros(N, 3, device=device, dtype=torch.float32)

        _backward_gv_grad_V_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
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
            N,
            d_h,
            r_0,
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            V.stride(0),
            V.stride(1),
            BLOCK,
        )

        # g_u = 0 (row-normalization cancels log_u effect)
        g_u = torch.zeros(H, N, device=device, dtype=torch.float32)

        # ---- Step 3: Precompute row_lse, col_lse for IFT ----
        row_lse = torch.empty(H, N, device=device, dtype=torch.float32)
        col_lse = torch.empty(H, N, device=device, dtype=torch.float32)

        _compute_row_lse[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_v,
            row_lse,
            N,
            d_h,
            r_0,
            *stride_args,
        )
        _compute_col_lse[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_u,
            col_lse,
            N,
            d_h,
            r_0,
            *stride_args,
        )

        # ---- Step 4: IFT adjoint iterations ----
        z_u = g_u.clone()
        z_v = g_v.clone()
        z_u_new = torch.empty_like(z_u)
        z_v_new = torch.empty_like(z_v)

        common = (Q_ln, K_ln, x_res, pos_bias, eps, w_dist)

        for _ in range(K_back):
            _ift_z_u_update[grid](
                *common,
                log_u,
                col_lse,
                g_u,
                z_v,
                z_u_new,
                kappa,
                N,
                d_h,
                r_0,
                *stride_args,
            )
            _ift_z_v_update[grid](
                *common,
                log_v,
                row_lse,
                g_v,
                z_u_new,
                z_v_new,
                kappa,
                N,
                d_h,
                r_0,
                *stride_args,
            )
            z_u, z_u_new = z_u_new, z_u
            z_v, z_v_new = z_v_new, z_v

        # ---- Step 5: Cost gradient propagation ----
        grad_Q_ln = torch.zeros_like(Q_ln)
        grad_K_ln = torch.zeros_like(K_ln)
        grad_x_cost = torch.zeros(N, 3, device=device, dtype=torch.float32)
        grad_w_dist = torch.zeros(H, device=device, dtype=torch.float32)
        grad_pos_bias = torch.zeros_like(pos_bias)

        _cost_gradient_kernel[grid](
            Q_ln,
            K_ln,
            x_res,
            pos_bias,
            V,
            eps,
            w_dist,
            log_u,
            log_v,
            log_Z,
            row_lse,
            col_lse,
            grad_O,
            grad_xc,
            D,
            z_u,
            z_v,
            kappa,
            grad_Q_ln,
            grad_K_ln,
            grad_x_cost,
            grad_w_dist,
            grad_pos_bias,
            N,
            d_h,
            r_0,
            Q_ln.stride(0),
            Q_ln.stride(1),
            x_res.stride(0),
            pos_bias.stride(0),
            pos_bias.stride(1),
            V.stride(0),
            V.stride(1),
            BLOCK_BWD,
        )

        # Combine x_res gradients
        grad_x_res = grad_x_transport + grad_x_cost

        # Marginal gradients from IFT
        grad_log_mu = kappa[:, None] * z_u
        grad_log_nu = kappa[:, None] * z_v

        # Return gradients for: Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist,
        #   log_mu, log_nu, K_iter, lam, r_0, log_u_init, log_v_init, BLOCK
        return (
            grad_Q_ln,
            grad_K_ln,
            grad_V,
            grad_x_res,
            grad_pos_bias,
            None,
            grad_w_dist,
            grad_log_mu,
            grad_log_nu,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_sinkhorn(
    Q_ln: torch.Tensor,
    K_ln: torch.Tensor,
    V: torch.Tensor,
    x_res: torch.Tensor,
    pos_bias: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash-Sinkhorn with IFT backward.

    Returns:
        O_avg:      (H, N, d_h) transport-weighted value output
        x_centroid: (H, N, 3)   EGNN centroid
        log_u:      (H, N)      converged row duals
        log_v:      (H, N)      converged column duals
    """
    return FlashSinkhornFunction.apply(
        Q_ln,
        K_ln,
        V,
        x_res,
        pos_bias,
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
    )
