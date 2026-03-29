"""
Triton backward kernels for transport-weighted feature aggregation.

Mirrors balanced_sinkhorn_bwd.py but for D_V-dimensional features
instead of 3D coordinates. Used by BalancedSinkhornDualFn.

Kernels:
  1. _feature_bwd_D_kernel  — D_h[i] = Σ_j T_norm_ij * (grad_hc_i · V_h_j)
  2. _feature_bwd_kernel    — g_u, g_v for V_h path + grad_V + cost gradients
"""

import triton
import triton.language as tl

from deepfold.model.kernels.flash_sinkhorn_transport import _compute_balanced_cost_tile
from deepfold.model.kernels.balanced_sinkhorn_bwd import _load_head_params


# ============================================================================
# Feature centroid backward pass 1: D_h[i]
# ============================================================================

@triton.jit
def _feature_bwd_D_kernel(
    Q_ptr, K_ptr, X_ptr, V_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_U_ptr, LOG_V_ptr, LOG_Z_ptr,
    GRAD_HC_ptr, D_H_ptr, MASK_BIAS_ptr,
    N: tl.constexpr, D_H: tl.constexpr, D_V: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn,
    stride_xb, stride_xn,
    stride_vb, stride_vh, stride_vn,
    stride_hcb, stride_hch, stride_hcn,
    stride_uvb, BLOCK: tl.constexpr,
):
    """D_h[i] = Σ_j T_norm[i,j] * dot(grad_hc[i], V_h[j]) over D_V dims."""
    pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    _, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b*stride_qb; K_b = K_ptr + pid_b*stride_qb
    X_b = X_ptr + pid_b*stride_xb; uv_off = pid_b*stride_uvb
    V_b = V_ptr + pid_b*stride_vb + pid_h*stride_vh
    MB_b = MASK_BIAS_ptr + pid_b*N

    i_idx = pid_i*BLOCK + tl.arange(0, BLOCK); i_mask = i_idx < N
    log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=-1e30)
    log_Z_i = tl.load(LOG_Z_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=0.0)

    # Load grad_hc[i]: (BLOCK, D_V)
    d_idx = tl.arange(0, D_V)
    ghc_base = GRAD_HC_ptr + pid_b*stride_hcb + pid_h*stride_hch
    ghc_i = tl.load(ghc_base + i_idx[:, None]*stride_hcn + d_idx[None, :],
                     mask=i_mask[:, None], other=0.0)  # (BLOCK, D_V)

    D_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j0 in range(0, N, BLOCK):
        j_idx = j0 + tl.arange(0, BLOCK); j_mask = j_idx < N
        log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h*N + j_idx, mask=j_mask, other=-1e30)
        mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        log_score = log_u_i[:, None] + (-C_tile*inv_eps_h) + log_v_j[None, :] + mask_bias_j[None, :]
        T_tile = tl.exp(log_score - log_Z_i[:, None])
        T_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_tile, 0.0)

        # Load V_h[j]: (BLOCK, D_V)
        V_j = tl.load(V_b + j_idx[:, None]*stride_vn + d_idx[None, :],
                       mask=j_mask[:, None], other=0.0)

        # dot(grad_hc[i], V_h[j]) = (BLOCK_I, D_V) @ (D_V, BLOCK_J) → (BLOCK_I, BLOCK_J)
        # But we need sum over D_V for each (i,j) pair
        dT = tl.dot(ghc_i.to(tl.float32), tl.trans(V_j.to(tl.float32)))  # (BLOCK, BLOCK)
        D_acc += tl.sum(T_tile * dT, axis=1)

    tl.store(D_H_ptr + uv_off + pid_h*N + i_idx, D_acc, mask=i_mask)


# ============================================================================
# Feature centroid backward pass 2: g_u_h, g_v_h, grad_V, cost gradients
# ============================================================================

@triton.jit
def _feature_bwd_kernel(
    Q_ptr, K_ptr, X_ptr, V_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
    LOG_U_ptr, LOG_V_ptr, LOG_Z_ptr,
    GRAD_HC_ptr, D_H_ptr,
    G_U_ptr, G_V_ptr, GRAD_V_ptr,
    GRAD_Q_ptr, GRAD_K_ptr, GRAD_X_COST_ptr, GRAD_ALPHA_ptr, GRAD_R_H_ptr,
    MASK_BIAS_ptr,
    N: tl.constexpr, D_H: tl.constexpr, D_V: tl.constexpr, H: tl.constexpr,
    stride_qb, stride_qh, stride_qn,
    stride_xb, stride_xn,
    stride_vb, stride_vh, stride_vn,
    stride_hcb, stride_hch, stride_hcn,
    stride_uvb, stride_gxc_b,
    BLOCK: tl.constexpr,
):
    """Compute g_u_h, g_v_h (for Sinkhorn bwd), grad_V, and cost gradients from V_h path."""
    pid_bh = tl.program_id(0); pid_j = tl.program_id(1)
    pid_b = pid_bh // H; pid_h = pid_bh % H
    eps_h, inv_eps_h, w_h, r_h = _load_head_params(EPS_ptr, ALPHA_ptr, R_H_ptr, pid_h)

    Q_b = Q_ptr + pid_b*stride_qb; K_b = K_ptr + pid_b*stride_qb
    X_b = X_ptr + pid_b*stride_xb; uv_off = pid_b*stride_uvb
    V_b = V_ptr + pid_b*stride_vb + pid_h*stride_vh
    MB_b = MASK_BIAS_ptr + pid_b*N
    d_idx = tl.arange(0, D_V)

    j_idx = pid_j*BLOCK + tl.arange(0, BLOCK); j_mask = j_idx < N
    log_v_j = tl.load(LOG_V_ptr + uv_off + pid_h*N + j_idx, mask=j_mask, other=-1e30)
    mask_bias_j = tl.load(MB_b + j_idx, mask=j_mask, other=-1e9)

    # Load V_h[j] and grad_hc accumulators
    V_j = tl.load(V_b + j_idx[:, None]*stride_vn + d_idx[None, :],
                   mask=j_mask[:, None], other=0.0)  # (BLOCK, D_V)

    # Accumulators
    acc_gv = tl.zeros([BLOCK], dtype=tl.float32)          # g_v[j]
    acc_grad_V = tl.zeros([BLOCK, D_V], dtype=tl.float32)  # grad_V[j]
    acc_grad_alpha = tl.zeros([1], dtype=tl.float32)
    acc_grad_r = tl.zeros([1], dtype=tl.float32)

    # Load x[j] and K[j] for cost gradient
    xj_0 = tl.load(X_b + j_idx*stride_xn + 0, mask=j_mask, other=0.0)
    xj_1 = tl.load(X_b + j_idx*stride_xn + 1, mask=j_mask, other=0.0)
    xj_2 = tl.load(X_b + j_idx*stride_xn + 2, mask=j_mask, other=0.0)
    K_j_base = K_b + pid_h*stride_qh
    d_feat = tl.arange(0, D_H)
    K_j = tl.load(K_j_base + j_idx[:, None]*stride_qn + d_feat[None, :],
                   mask=j_mask[:, None], other=0.0)

    acc_grad_K = tl.zeros([BLOCK, D_H], dtype=tl.float32)
    acc_grad_xj_0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_grad_xj_1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc_grad_xj_2 = tl.zeros([BLOCK], dtype=tl.float32)

    for i0 in range(0, N, BLOCK):
        i_idx = i0 + tl.arange(0, BLOCK); i_mask = i_idx < N
        log_u_i = tl.load(LOG_U_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=-1e30)
        log_Z_i = tl.load(LOG_Z_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=0.0)
        D_h_i = tl.load(D_H_ptr + uv_off + pid_h*N + i_idx, mask=i_mask, other=0.0)

        C_tile = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h,
            i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
        log_score = log_u_i[:, None] + (-C_tile*inv_eps_h) + log_v_j[None, :] + mask_bias_j[None, :]
        T_tile = tl.exp(log_score - log_Z_i[:, None])
        T_tile = tl.where(i_mask[:, None] & j_mask[None, :], T_tile, 0.0)

        # Load grad_hc[i]: (BLOCK_I, D_V)
        ghc_base = GRAD_HC_ptr + pid_b*stride_hcb + pid_h*stride_hch
        ghc_i = tl.load(ghc_base + i_idx[:, None]*stride_hcn + d_idx[None, :],
                         mask=i_mask[:, None], other=0.0)

        # dT[i,j] = dot(grad_hc[i], V_h[j]) — (BLOCK_I, BLOCK_J)
        dT = tl.dot(ghc_i.to(tl.float32), tl.trans(V_j.to(tl.float32)))
        grad_ls = T_tile * (dT - D_h_i[:, None])  # (BLOCK_I, BLOCK_J)

        # g_v[j] += sum_i grad_ls[i,j]
        acc_gv += tl.sum(grad_ls, axis=0)
        # g_u[i] += sum_j grad_ls[i,j] — atomic add
        gu_i = tl.sum(grad_ls, axis=1)
        tl.atomic_add(G_U_ptr + uv_off + pid_h*N + i_idx, gu_i, mask=i_mask)

        # grad_V[j] += sum_i T_tile[i,j] * grad_hc[i]
        # = T_tile^T @ ghc_i: (BLOCK_J, BLOCK_I) @ (BLOCK_I, D_V) → (BLOCK_J, D_V)
        acc_grad_V += tl.dot(tl.trans(T_tile.to(tl.float32)), ghc_i.to(tl.float32))

        # ---- Cost gradients (same pattern as centroid_bwd_kernel) ----
        grad_C = -inv_eps_h * grad_ls  # (BLOCK_I, BLOCK_J)

        # Feature cost gradient
        Q_i_base = Q_b + pid_h*stride_qh
        Q_i = tl.load(Q_i_base + i_idx[:, None]*stride_qn + d_feat[None, :],
                       mask=i_mask[:, None], other=0.0)

        # grad_Q[i] += grad_C[i,:] @ (-(1-w_h)*K[j]) — atomic add
        neg_1mw = -(1.0 - w_h)
        gQ_i = tl.dot(grad_C.to(tl.float32), (neg_1mw * K_j).to(tl.float32))
        gQ_off = GRAD_Q_ptr + pid_b*stride_qb + pid_h*stride_qh
        tl.atomic_add(gQ_off + i_idx[:, None]*stride_qn + d_feat[None, :],
                       gQ_i, mask=i_mask[:, None])
        # grad_K[j] += grad_C^T[j,:] @ (-(1-w_h)*Q[i])
        acc_grad_K += tl.dot(tl.trans(grad_C.to(tl.float32)), (neg_1mw * Q_i).to(tl.float32))

        # Geometric cost gradients
        xi_0 = tl.load(X_b + i_idx*stride_xn + 0, mask=i_mask, other=0.0)
        xi_1 = tl.load(X_b + i_idx*stride_xn + 1, mask=i_mask, other=0.0)
        xi_2 = tl.load(X_b + i_idx*stride_xn + 2, mask=i_mask, other=0.0)
        dx_0 = xi_0[:, None] - xj_0[None, :]
        dx_1 = xi_1[:, None] - xj_1[None, :]
        dx_2 = xi_2[:, None] - xj_2[None, :]
        dist = tl.sqrt(dx_0*dx_0 + dx_1*dx_1 + dx_2*dx_2 + 1e-8)
        geo_factor = grad_C * w_h * r_h / ((r_h + dist)*(r_h + dist)) / dist

        # grad_x (geometric part) — atomic for i, accumulate for j
        gxi_0 = tl.sum(geo_factor * dx_0, axis=1)
        gxi_1 = tl.sum(geo_factor * dx_1, axis=1)
        gxi_2 = tl.sum(geo_factor * dx_2, axis=1)
        tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + i_idx*3 + 0, gxi_0, mask=i_mask)
        tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + i_idx*3 + 1, gxi_1, mask=i_mask)
        tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + i_idx*3 + 2, gxi_2, mask=i_mask)
        acc_grad_xj_0 -= tl.sum(geo_factor * dx_0, axis=0)
        acc_grad_xj_1 -= tl.sum(geo_factor * dx_1, axis=0)
        acc_grad_xj_2 -= tl.sum(geo_factor * dx_2, axis=0)

        # Parameter gradients
        cos_sim = tl.dot(Q_i.to(tl.float32), tl.trans(K_j.to(tl.float32)))
        C_feat_tile = 1.0 - cos_sim
        C_geom_tile = dist / (r_h + dist)
        dC_dalpha = (C_geom_tile - C_feat_tile) * w_h * (1.0 - w_h)
        acc_grad_alpha += tl.sum(grad_C * dC_dalpha)
        acc_grad_r += tl.sum(grad_C * w_h * (-dist / ((r_h + dist)*(r_h + dist))))

    # Store g_v[j]
    tl.store(G_V_ptr + uv_off + pid_h*N + j_idx, acc_gv, mask=j_mask)

    # Store grad_V[j]
    grad_V_base = GRAD_V_ptr + pid_b*stride_vb + pid_h*stride_vh
    tl.store(grad_V_base + j_idx[:, None]*stride_vn + d_idx[None, :],
             acc_grad_V, mask=j_mask[:, None])

    # Atomic add accumulated K, x_cost, alpha, r gradients
    tl.atomic_add(GRAD_K_ptr + pid_b*stride_qb + pid_h*stride_qh + j_idx[:, None]*stride_qn + d_feat[None, :],
                  acc_grad_K, mask=j_mask[:, None])
    tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + j_idx*3 + 0, acc_grad_xj_0, mask=j_mask)
    tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + j_idx*3 + 1, acc_grad_xj_1, mask=j_mask)
    tl.atomic_add(GRAD_X_COST_ptr + pid_b*stride_gxc_b + j_idx*3 + 2, acc_grad_xj_2, mask=j_mask)
    tl.atomic_add(GRAD_ALPHA_ptr + pid_h, tl.sum(acc_grad_alpha))
    tl.atomic_add(GRAD_R_H_ptr + pid_h, tl.sum(acc_grad_r))
