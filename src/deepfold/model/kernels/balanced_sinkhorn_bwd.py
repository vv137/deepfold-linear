"""
Triton backward kernels for Balanced Sinkhorn Transport (SPEC §7, v6.0).

O(N) memory — cost tiles recomputed on-the-fly via _compute_balanced_cost_tile.
Unrolled backward through K Sinkhorn iterations for exact gradients.

Kernels:
  1. _centroid_bwd_D_kernel  — D[i] = Σ_j T_norm_ij * (grad_xc_i · x_j)
  2. _centroid_bwd_kernel    — g_u, g_v, grad_x_transport + direct cost gradient
  3. _balanced_row_bwd_kernel — backward through row update (κ=1)
  4. _balanced_col_bwd_kernel — backward through col update (κ=1)
"""

import triton
import triton.language as tl

from deepfold.model.kernels.flash_sinkhorn_transport import _compute_balanced_cost_tile


# ---- helpers ----

@triton.jit
def _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h):
    eps_h = tl.load(EPS_ptr + pid_h)
    inv_eps_h = 1.0 / eps_h
    w_h = tl.sigmoid(tl.load(ALPHA_ptr + pid_h))
    r_h = tl.maximum(tl.abs(tl.load(R_H_ptr + pid_h)), 0.1)
    return eps_h, inv_eps_h, w_h, r_h


# ============================================================================
# Centroid backward pass 1: D[i]
# ============================================================================

@triton.jit
def _centroid_bwd_D_kernel(
    Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_U_ptr, LOG_V_ptr, LOG_Z_ptr,
    GRAD_XC_ptr, D_ptr, MASK_BIAS_ptr,
    N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn,
    stride_xb, stride_xn,
    stride_xcb, stride_xch, stride_xcn,
    stride_uvb, BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    _, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b * stride_qb; K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb; uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * N

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK); i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30)
    log_Z_i = tl.load(LOG_Z_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)

    gxc_base = GRAD_XC_ptr + pid_b * stride_xcb + pid_h * stride_xch
    gxc_0 = tl.load(gxc_base + i_idx * stride_xcn + 0, mask=i_mask, other=0.0)
    gxc_1 = tl.load(gxc_base + i_idx * stride_xcn + 1, mask=i_mask, other=0.0)
    gxc_2 = tl.load(gxc_base + i_idx * stride_xcn + 2, mask=i_mask, other=0.0)

    D_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK); j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30)
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        log_score = log_u_i[:, None] + (-C_tile * inv_eps_h) + log_v_j[None, :] + mask_bias_j[None, :]
        T_tile = tl.exp(log_score - log_Z_i[:, None])
        T_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_tile, 0.0)

        xj_0 = tl.load(X_b + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
        xj_1 = tl.load(X_b + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
        xj_2 = tl.load(X_b + j_idx * stride_xn + 2, mask=j_mask, other=0.0)
        dT = gxc_0[:, None]*xj_0[None, :] + gxc_1[:, None]*xj_1[None, :] + gxc_2[:, None]*xj_2[None, :]
        D_acc += tl.sum(T_tile * dT, axis=1)

    tl.store(D_ptr + uv_off + pid_h * N + i_idx, D_acc, mask=i_mask)


# ============================================================================
# Centroid backward pass 2: g_u, g_v, grad_x_transport + direct cost gradient
# ============================================================================

@triton.jit
def _centroid_bwd_kernel(
    Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_U_ptr, LOG_V_ptr, LOG_Z_ptr,
    GRAD_XC_ptr, D_ptr,
    G_U_ptr, G_V_ptr, GRAD_X_TR_ptr,
    GRAD_Q_ptr, GRAD_K_ptr, GRAD_X_COST_ptr, GRAD_ALPHA_ptr, GRAD_R_H_ptr,
    MASK_BIAS_ptr,
    N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
    stride_xcb, stride_xch, stride_xcn,
    stride_uvb, stride_gxt_b, stride_gxc_b,
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_j = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    _, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b * stride_qb; K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb; uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * N

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK); j_mask = j_idx < N
    log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h * N + j_idx, mask=j_mask, other=-1e30)
    mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

    xj_0 = tl.load(X_b + j_idx * stride_xn + 0, mask=j_mask, other=0.0)
    xj_1 = tl.load(X_b + j_idx * stride_xn + 1, mask=j_mask, other=0.0)
    xj_2 = tl.load(X_b + j_idx * stride_xn + 2, mask=j_mask, other=0.0)

    d_idx = tl.arange(0, D_H)
    k_ptrs = K_b + pid_h * stride_qh + j_idx[:, None] * stride_qn + d_idx[None, :]
    K_j = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)

    acc_gv = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_grad_K = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_gxc_j_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gxc_j_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gxc_j_2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_grad_alpha = tl.zeros([], dtype=tl.float32)
    acc_grad_r_h = tl.zeros([], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK); i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=-1e30)
        log_Z_i = tl.load(LOG_Z_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)
        D_i = tl.load(D_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)

        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        log_score = log_u_i[:, None] + (-C_tile * inv_eps_h) + log_v_j[None, :] + mask_bias_j[None, :]
        T_tile = tl.exp(log_score - log_Z_i[:, None])
        T_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_tile, 0.0)

        gxc_base = GRAD_XC_ptr + pid_b * stride_xcb + pid_h * stride_xch
        gxc_0 = tl.load(gxc_base + i_idx * stride_xcn + 0, mask=i_mask, other=0.0)
        gxc_1 = tl.load(gxc_base + i_idx * stride_xcn + 1, mask=i_mask, other=0.0)
        gxc_2 = tl.load(gxc_base + i_idx * stride_xcn + 2, mask=i_mask, other=0.0)

        dT = gxc_0[:, None]*xj_0[None, :] + gxc_1[:, None]*xj_1[None, :] + gxc_2[:, None]*xj_2[None, :]
        grad_ls = T_tile * (dT - D_i[:, None])

        acc_gv += tl.sum(grad_ls, axis=0)
        tl.atomic_add(G_U_ptr + uv_off + pid_h * N + i_idx, tl.sum(grad_ls, axis=1), mask=i_mask)

        acc_gx_0 += tl.sum(T_tile * gxc_0[:, None], axis=0)
        acc_gx_1 += tl.sum(T_tile * gxc_1[:, None], axis=0)
        acc_gx_2 += tl.sum(T_tile * gxc_2[:, None], axis=0)

        # ---- Direct cost gradient: grad_C = -(1/ε) * grad_ls ----
        grad_C = -inv_eps_h * grad_ls
        q_ptrs = Q_b + pid_h * stride_qh + i_idx[:, None] * stride_qn + d_idx[None, :]
        Q_i = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)

        gq_ptrs = GRAD_Q_ptr + pid_b * stride_qb + pid_h * stride_qh + i_idx[:, None] * stride_qn + d_idx[None, :]
        tl.atomic_add(gq_ptrs, tl.dot(grad_C * (-(1.0 - w_h)), K_j), mask=i_mask[:, None])
        acc_grad_K += tl.dot(tl.trans(grad_C * (-(1.0 - w_h))), Q_i)

        xi_0 = tl.load(X_b + i_idx * stride_xn + 0, mask=i_mask, other=0.0)
        xi_1 = tl.load(X_b + i_idx * stride_xn + 1, mask=i_mask, other=0.0)
        xi_2 = tl.load(X_b + i_idx * stride_xn + 2, mask=i_mask, other=0.0)
        dx = xi_0[:, None] - xj_0[None, :]; dy = xi_1[:, None] - xj_1[None, :]
        dz = xi_2[:, None] - xj_2[None, :]
        dist = tl.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
        rpd = r_h + dist
        weighted = grad_C * w_h * r_h / (rpd * rpd) / dist

        gxc_cost = GRAD_X_COST_ptr + pid_b * stride_gxc_b
        tl.atomic_add(gxc_cost + i_idx*3+0, tl.sum(weighted*dx, axis=1), mask=i_mask)
        tl.atomic_add(gxc_cost + i_idx*3+1, tl.sum(weighted*dy, axis=1), mask=i_mask)
        tl.atomic_add(gxc_cost + i_idx*3+2, tl.sum(weighted*dz, axis=1), mask=i_mask)
        acc_gxc_j_0 += -tl.sum(weighted*dx, axis=0)
        acc_gxc_j_1 += -tl.sum(weighted*dy, axis=0)
        acc_gxc_j_2 += -tl.sum(weighted*dz, axis=0)

        C_feat_t = 1.0 - tl.dot(Q_i, tl.trans(K_j)) if D_H >= 16 else 1.0 - tl.sum(Q_i[:, None, :] * K_j[None, :, :], axis=2)
        C_geom_t = dist / rpd
        acc_grad_alpha += tl.sum(grad_C * (C_geom_t - C_feat_t) * w_h * (1.0 - w_h))
        acc_grad_r_h += tl.sum(grad_C * w_h * (-dist / (rpd * rpd)))

    tl.store(G_V_ptr + uv_off + pid_h * N + j_idx, acc_gv, mask=j_mask)
    gxt = GRAD_X_TR_ptr + pid_b * stride_gxt_b
    tl.atomic_add(gxt + j_idx*3+0, acc_gx_0, mask=j_mask)
    tl.atomic_add(gxt + j_idx*3+1, acc_gx_1, mask=j_mask)
    tl.atomic_add(gxt + j_idx*3+2, acc_gx_2, mask=j_mask)
    gk_ptrs = GRAD_K_ptr + pid_b * stride_qb + pid_h * stride_qh + j_idx[:, None] * stride_qn + d_idx[None, :]
    tl.atomic_add(gk_ptrs, acc_grad_K, mask=j_mask[:, None])
    gxc_cost = GRAD_X_COST_ptr + pid_b * stride_gxc_b
    tl.atomic_add(gxc_cost + j_idx*3+0, acc_gxc_j_0, mask=j_mask)
    tl.atomic_add(gxc_cost + j_idx*3+1, acc_gxc_j_1, mask=j_mask)
    tl.atomic_add(gxc_cost + j_idx*3+2, acc_gxc_j_2, mask=j_mask)
    tl.atomic_add(GRAD_ALPHA_ptr + pid_h, acc_grad_alpha)
    tl.atomic_add(GRAD_R_H_ptr + pid_h, acc_grad_r_h)


# ============================================================================
# Unrolled row backward (κ=1): log_u = log_marg - LSE_j(log_K + log_v)
# ============================================================================

@triton.jit
def _balanced_row_bwd_kernel(
    Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_V_BEFORE_ptr, GRAD_LOG_U_ptr, MASK_BIAS_ptr,
    GRAD_LOG_V_ptr, GRAD_Q_ptr, GRAD_K_ptr, GRAD_X_COST_ptr, GRAD_ALPHA_ptr, GRAD_R_H_ptr,
    N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
    stride_uvb, stride_gxc_b, stride_mb, BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    _, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b * stride_qb; K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb; uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK); i_mask = i_idx < N
    gl_u = tl.load(GRAD_LOG_U_ptr + uv_off + pid_h * N + i_idx, mask=i_mask, other=0.0)

    d_idx = tl.arange(0, D_H)
    Q_i = tl.load(Q_b + pid_h*stride_qh + i_idx[:, None]*stride_qn + d_idx[None, :], mask=i_mask[:, None], other=0.0)
    xi_0 = tl.load(X_b + i_idx*stride_xn+0, mask=i_mask, other=0.0)
    xi_1 = tl.load(X_b + i_idx*stride_xn+1, mask=i_mask, other=0.0)
    xi_2 = tl.load(X_b + i_idx*stride_xn+2, mask=i_mask, other=0.0)

    # Pass 1: LSE
    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)
    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK); j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_BEFORE_ptr + uv_off + pid_h*N + j_idx, mask=j_mask, other=-1e30)
        mb_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)
        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        score = -C_tile * inv_eps_h + log_v_j[None, :] + mb_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)
        tile_max = tl.max(score, axis=1)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(score - new_max[:, None]), axis=1)
        max_val = new_max
    lse_i = max_val + tl.log(sum_exp + 1e-30)

    # Pass 2: gradient
    acc_grad_Q = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_gx_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_ga = tl.zeros([], dtype=tl.float32); acc_gr = tl.zeros([], dtype=tl.float32)

    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK); j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_BEFORE_ptr + uv_off + pid_h*N + j_idx, mask=j_mask, other=-1e30)
        mb_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)
        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        score = -C_tile * inv_eps_h + log_v_j[None, :] + mb_j[None, :]
        score = tl.where(j_mask[None, :], score, -1e30)
        s_tile = tl.exp(score - lse_i[:, None])
        s_tile = tl.where(i_mask[:, None] & j_mask[None, :], s_tile, 0.0)

        grad_C = inv_eps_h * gl_u[:, None] * s_tile  # κ=1
        tl.atomic_add(GRAD_LOG_V_ptr + uv_off + pid_h*N + j_idx,
                       -tl.sum(s_tile * gl_u[:, None], axis=0), mask=j_mask)

        K_j = tl.load(K_b + pid_h*stride_qh + j_idx[:, None]*stride_qn + d_idx[None, :], mask=j_mask[:, None], other=0.0)
        acc_grad_Q += tl.dot(grad_C * (-(1.0-w_h)), K_j)
        gk_ptrs = GRAD_K_ptr + pid_b*stride_qb + pid_h*stride_qh + j_idx[:, None]*stride_qn + d_idx[None, :]
        tl.atomic_add(gk_ptrs, tl.dot(tl.trans(grad_C * (-(1.0-w_h))), Q_i), mask=j_mask[:, None])

        xj_0 = tl.load(X_b + j_idx*stride_xn+0, mask=j_mask, other=0.0)
        xj_1 = tl.load(X_b + j_idx*stride_xn+1, mask=j_mask, other=0.0)
        xj_2 = tl.load(X_b + j_idx*stride_xn+2, mask=j_mask, other=0.0)
        dx = xi_0[:, None]-xj_0[None, :]; dy = xi_1[:, None]-xj_1[None, :]; dz = xi_2[:, None]-xj_2[None, :]
        dist = tl.sqrt(dx*dx+dy*dy+dz*dz+1e-8); rpd = r_h+dist
        weighted = grad_C * w_h * r_h / (rpd*rpd) / dist
        acc_gx_0 += tl.sum(weighted*dx, axis=1); acc_gx_1 += tl.sum(weighted*dy, axis=1); acc_gx_2 += tl.sum(weighted*dz, axis=1)
        gxc = GRAD_X_COST_ptr + pid_b * stride_gxc_b
        tl.atomic_add(gxc+j_idx*3+0, -tl.sum(weighted*dx, axis=0), mask=j_mask)
        tl.atomic_add(gxc+j_idx*3+1, -tl.sum(weighted*dy, axis=0), mask=j_mask)
        tl.atomic_add(gxc+j_idx*3+2, -tl.sum(weighted*dz, axis=0), mask=j_mask)

        C_feat_t = 1.0 - tl.dot(Q_i, tl.trans(K_j)) if D_H >= 16 else 1.0 - tl.sum(Q_i[:, None, :]*K_j[None, :, :], axis=2)
        acc_ga += tl.sum(grad_C * (dist/rpd - C_feat_t) * w_h * (1.0-w_h))
        acc_gr += tl.sum(grad_C * w_h * (-dist/(rpd*rpd)))

    gq_ptrs = GRAD_Q_ptr + pid_b*stride_qb + pid_h*stride_qh + i_idx[:, None]*stride_qn + d_idx[None, :]
    tl.atomic_add(gq_ptrs, acc_grad_Q, mask=i_mask[:, None])
    gxc = GRAD_X_COST_ptr + pid_b * stride_gxc_b
    tl.atomic_add(gxc+i_idx*3+0, acc_gx_0, mask=i_mask)
    tl.atomic_add(gxc+i_idx*3+1, acc_gx_1, mask=i_mask)
    tl.atomic_add(gxc+i_idx*3+2, acc_gx_2, mask=i_mask)
    tl.atomic_add(GRAD_ALPHA_ptr + pid_h, acc_ga)
    tl.atomic_add(GRAD_R_H_ptr + pid_h, acc_gr)


# ============================================================================
# Unrolled col backward (κ=1): log_v = log_marg - LSE_i(log_K + log_u)
# ============================================================================

@triton.jit
def _balanced_col_bwd_kernel(
    Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_U_AFTER_ptr, GRAD_LOG_V_ptr, MASK_BIAS_ptr,
    GRAD_LOG_U_ptr, GRAD_Q_ptr, GRAD_K_ptr, GRAD_X_COST_ptr, GRAD_ALPHA_ptr, GRAD_R_H_ptr,
    N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
    stride_uvb, stride_gxc_b, stride_mb, BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_j = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    _, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b * stride_qb; K_b = K_ptr + pid_b * stride_qb
    X_b = X_ptr + pid_b * stride_xb; uv_off = pid_b * stride_uvb
    MB_b = MASK_BIAS_ptr + pid_b * stride_mb

    j_idx = pid_j * BLOCK + tl.arange(0, BLOCK); j_mask = j_idx < N
    gl_v = tl.load(GRAD_LOG_V_ptr + uv_off + pid_h*N + j_idx, mask=j_mask, other=0.0)

    d_idx = tl.arange(0, D_H)
    K_j = tl.load(K_b + pid_h*stride_qh + j_idx[:, None]*stride_qn + d_idx[None, :], mask=j_mask[:, None], other=0.0)
    xj_0 = tl.load(X_b + j_idx*stride_xn+0, mask=j_mask, other=0.0)
    xj_1 = tl.load(X_b + j_idx*stride_xn+1, mask=j_mask, other=0.0)
    xj_2 = tl.load(X_b + j_idx*stride_xn+2, mask=j_mask, other=0.0)

    # Pass 1: col LSE
    max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)
    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK); i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_AFTER_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=-1e30)
        mb_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)
        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        score_ij = -C_tile * inv_eps_h + log_u_i[:, None] + mb_i[:, None]
        score_ij = tl.where(i_mask[:, None], score_ij, -1e30)
        tile_max = tl.max(score_ij, axis=0)
        new_max = tl.maximum(max_val, tile_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(score_ij - new_max[None, :]), axis=0)
        max_val = new_max
    lse_j = max_val + tl.log(sum_exp + 1e-30)

    # Pass 2: gradient
    acc_grad_K = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_gx_j_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_j_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx_j_2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_ga = tl.zeros([], dtype=tl.float32); acc_gr = tl.zeros([], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK); i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_AFTER_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=-1e30)
        mb_i = tl.load(MB_b + i_idx, mask=i_mask, other=-1e9)
        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        score_ij = -C_tile * inv_eps_h + log_u_i[:, None] + mb_i[:, None]
        score_ij = tl.where(i_mask[:, None], score_ij, -1e30)
        s_tile = tl.exp(score_ij - lse_j[None, :])
        s_tile = tl.where(i_mask[:, None] & j_mask[None, :], s_tile, 0.0)

        grad_C = inv_eps_h * gl_v[None, :] * s_tile
        tl.atomic_add(GRAD_LOG_U_ptr + uv_off + pid_h*N + i_idx,
                       -tl.sum(s_tile * gl_v[None, :], axis=1), mask=i_mask)

        Q_i = tl.load(Q_b + pid_h*stride_qh + i_idx[:, None]*stride_qn + d_idx[None, :], mask=i_mask[:, None], other=0.0)
        acc_grad_K += tl.dot(tl.trans(grad_C * (-(1.0-w_h))), Q_i)
        gq_ptrs = GRAD_Q_ptr + pid_b*stride_qb + pid_h*stride_qh + i_idx[:, None]*stride_qn + d_idx[None, :]
        tl.atomic_add(gq_ptrs, tl.dot(grad_C * (-(1.0-w_h)), K_j), mask=i_mask[:, None])

        xi_0 = tl.load(X_b + i_idx*stride_xn+0, mask=i_mask, other=0.0)
        xi_1 = tl.load(X_b + i_idx*stride_xn+1, mask=i_mask, other=0.0)
        xi_2 = tl.load(X_b + i_idx*stride_xn+2, mask=i_mask, other=0.0)
        dx = xi_0[:, None]-xj_0[None, :]; dy = xi_1[:, None]-xj_1[None, :]; dz = xi_2[:, None]-xj_2[None, :]
        dist = tl.sqrt(dx*dx+dy*dy+dz*dz+1e-8); rpd = r_h+dist
        weighted = grad_C * w_h * r_h / (rpd*rpd) / dist
        gxc = GRAD_X_COST_ptr + pid_b * stride_gxc_b
        tl.atomic_add(gxc+i_idx*3+0, tl.sum(weighted*dx, axis=1), mask=i_mask)
        tl.atomic_add(gxc+i_idx*3+1, tl.sum(weighted*dy, axis=1), mask=i_mask)
        tl.atomic_add(gxc+i_idx*3+2, tl.sum(weighted*dz, axis=1), mask=i_mask)
        acc_gx_j_0 += -tl.sum(weighted*dx, axis=0); acc_gx_j_1 += -tl.sum(weighted*dy, axis=0)
        acc_gx_j_2 += -tl.sum(weighted*dz, axis=0)

        C_feat_t = 1.0 - tl.dot(Q_i, tl.trans(K_j)) if D_H >= 16 else 1.0 - tl.sum(Q_i[:, None, :]*K_j[None, :, :], axis=2)
        acc_ga += tl.sum(grad_C * (dist/rpd - C_feat_t) * w_h * (1.0-w_h))
        acc_gr += tl.sum(grad_C * w_h * (-dist/(rpd*rpd)))

    gk_ptrs = GRAD_K_ptr + pid_b*stride_qb + pid_h*stride_qh + j_idx[:, None]*stride_qn + d_idx[None, :]
    tl.atomic_add(gk_ptrs, acc_grad_K, mask=j_mask[:, None])
    gxc = GRAD_X_COST_ptr + pid_b * stride_gxc_b
    tl.atomic_add(gxc+j_idx*3+0, acc_gx_j_0, mask=j_mask)
    tl.atomic_add(gxc+j_idx*3+1, acc_gx_j_1, mask=j_mask)
    tl.atomic_add(gxc+j_idx*3+2, acc_gx_j_2, mask=j_mask)
    tl.atomic_add(GRAD_ALPHA_ptr + pid_h, acc_ga)
    tl.atomic_add(GRAD_R_H_ptr + pid_h, acc_gr)
