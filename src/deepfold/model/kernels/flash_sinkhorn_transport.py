"""
Balanced Sinkhorn Transport — coordinate centroid only (SPEC §7, v6.0).

CUDA: autograd.Function with Triton O(N) forward + unrolled Triton backward.
CPU: materialized O(N²) with gradient checkpointing.

All gradients exact (unrolled through K Sinkhorn iterations).
Iteration history: K × 2 × (B,H,N) ≈ 1-5 MB saved for backward.
"""

import torch
from torch.utils.checkpoint import checkpoint

BLOCK = 64

# ============================================================================
# Triton forward kernels
# ============================================================================

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True

    @triton.jit
    def _compute_balanced_cost_tile(
        Q_ptr, K_ptr, X_ptr, w_h, r_h,
        i_idx, j_idx, i_mask, j_mask, pid_h,
        N: tl.constexpr, D_H: tl.constexpr,
        stride_qh, stride_qn, stride_xn,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        q_ptrs = Q_ptr + pid_h*stride_qh + i_idx[:, None]*stride_qn + tl.arange(0, D_H)[None, :]
        Q_tile = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)
        k_ptrs = K_ptr + pid_h*stride_qh + j_idx[:, None]*stride_qn + tl.arange(0, D_H)[None, :]
        K_tile = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)

        if D_H >= 16:
            cos_sim = tl.dot(Q_tile, tl.trans(K_tile))
        else:
            cos_sim = tl.sum(Q_tile[:, None, :] * K_tile[None, :, :], axis=2)
        C_feat = 1.0 - cos_sim

        xi_0 = tl.load(X_ptr + i_idx*stride_xn+0, mask=i_mask, other=0.0)
        xi_1 = tl.load(X_ptr + i_idx*stride_xn+1, mask=i_mask, other=0.0)
        xi_2 = tl.load(X_ptr + i_idx*stride_xn+2, mask=i_mask, other=0.0)
        xj_0 = tl.load(X_ptr + j_idx*stride_xn+0, mask=j_mask, other=0.0)
        xj_1 = tl.load(X_ptr + j_idx*stride_xn+1, mask=j_mask, other=0.0)
        xj_2 = tl.load(X_ptr + j_idx*stride_xn+2, mask=j_mask, other=0.0)
        dx = xi_0[:, None]-xj_0[None, :]; dy = xi_1[:, None]-xj_1[None, :]; dz = xi_2[:, None]-xj_2[None, :]
        dist = tl.sqrt(dx*dx+dy*dy+dz*dz+1e-8)
        C_geom = dist / (r_h + dist)
        return (1.0-w_h)*C_feat + w_h*C_geom

    @triton.jit
    def _balanced_row_update(
        Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
        LOG_V_ptr, LOG_U_ptr, MASK_BIAS_ptr,
        N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr, LOG_MARGINAL,
        stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
        stride_uvb, stride_mb, BLOCK: tl.constexpr,
    ):
        pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
        pid_b = pid_bh // H; pid_h = pid_bh % H
        inv_eps = 1.0 / tl.load(EPS_ptr + pid_h)
        w_h = tl.sigmoid(tl.load(ALPHA_ptr + pid_h))
        r_h = tl.maximum(tl.abs(tl.load(R_H_ptr + pid_h)), 0.1)
        Q_b = Q_ptr+pid_b*stride_qb; K_b = K_ptr+pid_b*stride_qb
        X_b = X_ptr+pid_b*stride_xb; uv = pid_b*stride_uvb; MB = MASK_BIAS_ptr+pid_b*stride_mb
        i_idx = pid_i*BLOCK+tl.arange(0, BLOCK); i_mask = i_idx < N
        max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
        sum_exp = tl.zeros([BLOCK], dtype=tl.float32)
        for j0 in range(0, N, BLOCK):
            j_idx = j0+tl.arange(0, BLOCK); j_mask = j_idx < N
            lv = tl.load(LOG_V_ptr+uv+pid_h*N+j_idx, mask=j_mask, other=-1e30)
            mb = tl.load(MB+j_idx, mask=j_mask, other=-1e9)
            C = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h, i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
            score = -C*inv_eps + lv[None, :] + mb[None, :]
            score = tl.where(j_mask[None, :], score, -1e30)
            tm = tl.max(score, axis=1); nm = tl.maximum(max_val, tm)
            sum_exp = sum_exp*tl.exp(max_val-nm) + tl.sum(tl.exp(score-nm[:, None]), axis=1)
            max_val = nm
        tl.store(LOG_U_ptr+uv+pid_h*N+i_idx, LOG_MARGINAL-(max_val+tl.log(sum_exp+1e-30)), mask=i_mask)

    @triton.jit
    def _balanced_col_update(
        Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
        LOG_U_ptr, LOG_V_ptr, MASK_BIAS_ptr,
        N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr, LOG_MARGINAL,
        stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
        stride_uvb, stride_mb, BLOCK: tl.constexpr,
    ):
        pid_bh = tl.program_id(0); pid_j = tl.program_id(1)
        pid_b = pid_bh // H; pid_h = pid_bh % H
        inv_eps = 1.0 / tl.load(EPS_ptr + pid_h)
        w_h = tl.sigmoid(tl.load(ALPHA_ptr + pid_h))
        r_h = tl.maximum(tl.abs(tl.load(R_H_ptr + pid_h)), 0.1)
        Q_b = Q_ptr+pid_b*stride_qb; K_b = K_ptr+pid_b*stride_qb
        X_b = X_ptr+pid_b*stride_xb; uv = pid_b*stride_uvb; MB = MASK_BIAS_ptr+pid_b*stride_mb
        j_idx = pid_j*BLOCK+tl.arange(0, BLOCK); j_mask = j_idx < N
        max_val = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
        sum_exp = tl.zeros([BLOCK], dtype=tl.float32)
        for i0 in range(0, N, BLOCK):
            i_idx = i0+tl.arange(0, BLOCK); i_mask = i_idx < N
            lu = tl.load(LOG_U_ptr+uv+pid_h*N+i_idx, mask=i_mask, other=-1e30)
            mb = tl.load(MB+i_idx, mask=i_mask, other=-1e9)
            C = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h, i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
            score = -C*inv_eps + lu[:, None] + mb[:, None]
            score = tl.where(i_mask[:, None], score, -1e30)
            tm = tl.max(score, axis=0); nm = tl.maximum(max_val, tm)
            sum_exp = sum_exp*tl.exp(max_val-nm) + tl.sum(tl.exp(score-nm[None, :]), axis=0)
            max_val = nm
        tl.store(LOG_V_ptr+uv+pid_h*N+j_idx, LOG_MARGINAL-(max_val+tl.log(sum_exp+1e-30)), mask=j_mask)

    @triton.jit
    def _transport_centroid_kernel(
        Q_ptr, K_ptr, X_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
        LOG_U_ptr, LOG_V_ptr, X_CENT_ptr, LOG_Z_ptr, MASK_BIAS_ptr,
        N: tl.constexpr, D_H: tl.constexpr, H: tl.constexpr,
        stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
        stride_xcb, stride_xch, stride_xcn, stride_uvb, stride_mb,
        BLOCK: tl.constexpr,
    ):
        pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
        pid_b = pid_bh // H; pid_h = pid_bh % H
        inv_eps = 1.0 / tl.load(EPS_ptr + pid_h)
        w_h = tl.sigmoid(tl.load(ALPHA_ptr + pid_h))
        r_h = tl.maximum(tl.abs(tl.load(R_H_ptr + pid_h)), 0.1)
        Q_b = Q_ptr+pid_b*stride_qb; K_b = K_ptr+pid_b*stride_qb
        X_b = X_ptr+pid_b*stride_xb; uv = pid_b*stride_uvb; MB = MASK_BIAS_ptr+pid_b*stride_mb
        i_idx = pid_i*BLOCK+tl.arange(0, BLOCK); i_mask = i_idx < N
        lu_i = tl.load(LOG_U_ptr+uv+pid_h*N+i_idx, mask=i_mask, other=-1e30)

        row_max = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
        row_sum = tl.zeros([BLOCK], dtype=tl.float32)
        xc_0 = tl.zeros([BLOCK], dtype=tl.float32)
        xc_1 = tl.zeros([BLOCK], dtype=tl.float32)
        xc_2 = tl.zeros([BLOCK], dtype=tl.float32)

        for j0 in range(0, N, BLOCK):
            j_idx = j0+tl.arange(0, BLOCK); j_mask = j_idx < N
            lv = tl.load(LOG_V_ptr+uv+pid_h*N+j_idx, mask=j_mask, other=-1e30)
            mb = tl.load(MB+j_idx, mask=j_mask, other=-1e9)
            C = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h, i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
            ls = lu_i[:, None] + (-C*inv_eps) + lv[None, :] + mb[None, :]
            ls = tl.where(j_mask[None, :], ls, -1e30)
            tm = tl.max(ls, axis=1); nm = tl.maximum(row_max, tm)
            scale = tl.exp(row_max-nm)
            T_t = tl.exp(ls-nm[:, None]); T_t = tl.where(j_mask[None, :], T_t, 0.0)
            row_sum = row_sum*scale + tl.sum(T_t, axis=1)
            xc_0 = xc_0*scale; xc_1 = xc_1*scale; xc_2 = xc_2*scale
            xj_0 = tl.load(X_b+j_idx*stride_xn+0, mask=j_mask, other=0.0)
            xj_1 = tl.load(X_b+j_idx*stride_xn+1, mask=j_mask, other=0.0)
            xj_2 = tl.load(X_b+j_idx*stride_xn+2, mask=j_mask, other=0.0)
            xc_0 += tl.sum(T_t*xj_0[None, :], axis=1)
            xc_1 += tl.sum(T_t*xj_1[None, :], axis=1)
            xc_2 += tl.sum(T_t*xj_2[None, :], axis=1)
            row_max = nm

        # Row-normalized centroid (softmax pattern). Row-last iteration order
        # ensures row sums = 1/N exactly → this is dividing by a constant.
        inv_s = 1.0 / (row_sum + 1e-8)
        xc_base = X_CENT_ptr + pid_b*stride_xcb + pid_h*stride_xch
        tl.store(xc_base+i_idx*stride_xcn+0, xc_0*inv_s, mask=i_mask)
        tl.store(xc_base+i_idx*stride_xcn+1, xc_1*inv_s, mask=i_mask)
        tl.store(xc_base+i_idx*stride_xcn+2, xc_2*inv_s, mask=i_mask)
        tl.store(LOG_Z_ptr+uv+pid_h*N+i_idx, row_max+tl.log(row_sum+1e-30), mask=i_mask)

    @triton.jit
    def _transport_centroid_dual_kernel(
        Q_ptr, K_ptr, X_ptr, V_ptr, EPS_ptr, ALPHA_ptr, R_H_ptr,
        LOG_U_ptr, LOG_V_ptr, X_CENT_ptr, H_CENT_ptr, LOG_Z_ptr, MASK_BIAS_ptr,
        N: tl.constexpr, D_H: tl.constexpr, D_V: tl.constexpr, H: tl.constexpr,
        stride_qb, stride_qh, stride_qn, stride_xb, stride_xn,
        stride_vb, stride_vh, stride_vn,
        stride_xcb, stride_xch, stride_xcn,
        stride_hcb, stride_hch, stride_hcn,
        stride_uvb, stride_mb,
        BLOCK: tl.constexpr,
    ):
        """Fused centroid kernel: computes both T@x (3D) and T@V_h (D_V dim) in one pass."""
        pid_bh = tl.program_id(0); pid_i = tl.program_id(1)
        pid_b = pid_bh // H; pid_h = pid_bh % H
        inv_eps = 1.0 / tl.load(EPS_ptr + pid_h)
        w_h = tl.sigmoid(tl.load(ALPHA_ptr + pid_h))
        r_h = tl.maximum(tl.abs(tl.load(R_H_ptr + pid_h)), 0.1)
        Q_b = Q_ptr+pid_b*stride_qb; K_b = K_ptr+pid_b*stride_qb
        X_b = X_ptr+pid_b*stride_xb
        V_b = V_ptr+pid_b*stride_vb+pid_h*stride_vh
        uv = pid_b*stride_uvb; MB = MASK_BIAS_ptr+pid_b*stride_mb
        i_idx = pid_i*BLOCK+tl.arange(0, BLOCK); i_mask = i_idx < N
        lu_i = tl.load(LOG_U_ptr+uv+pid_h*N+i_idx, mask=i_mask, other=-1e30)

        row_max = tl.full([BLOCK], value=-1e30, dtype=tl.float32)
        row_sum = tl.zeros([BLOCK], dtype=tl.float32)
        # Coordinate accumulators (3D)
        xc_0 = tl.zeros([BLOCK], dtype=tl.float32)
        xc_1 = tl.zeros([BLOCK], dtype=tl.float32)
        xc_2 = tl.zeros([BLOCK], dtype=tl.float32)
        # Feature accumulators (D_V dim) — use tl.dot for efficiency
        hc = tl.zeros([BLOCK, D_V], dtype=tl.float32)
        d_idx = tl.arange(0, D_V)

        for j0 in range(0, N, BLOCK):
            j_idx = j0+tl.arange(0, BLOCK); j_mask = j_idx < N
            lv = tl.load(LOG_V_ptr+uv+pid_h*N+j_idx, mask=j_mask, other=-1e30)
            mb = tl.load(MB+j_idx, mask=j_mask, other=-1e9)
            C = _compute_balanced_cost_tile(Q_b, K_b, X_b, w_h, r_h, i_idx, j_idx, i_mask, j_mask, pid_h, N, D_H, stride_qh, stride_qn, stride_xn, BLOCK, BLOCK)
            ls = lu_i[:, None] + (-C*inv_eps) + lv[None, :] + mb[None, :]
            ls = tl.where(j_mask[None, :], ls, -1e30)
            tm = tl.max(ls, axis=1); nm = tl.maximum(row_max, tm)
            scale = tl.exp(row_max-nm)
            T_t = tl.exp(ls-nm[:, None]); T_t = tl.where(j_mask[None, :], T_t, 0.0)
            row_sum = row_sum*scale + tl.sum(T_t, axis=1)
            # Rescale coordinate accumulators
            xc_0 = xc_0*scale; xc_1 = xc_1*scale; xc_2 = xc_2*scale
            xj_0 = tl.load(X_b+j_idx*stride_xn+0, mask=j_mask, other=0.0)
            xj_1 = tl.load(X_b+j_idx*stride_xn+1, mask=j_mask, other=0.0)
            xj_2 = tl.load(X_b+j_idx*stride_xn+2, mask=j_mask, other=0.0)
            xc_0 += tl.sum(T_t*xj_0[None, :], axis=1)
            xc_1 += tl.sum(T_t*xj_1[None, :], axis=1)
            xc_2 += tl.sum(T_t*xj_2[None, :], axis=1)
            # Rescale and accumulate feature: hc += T_t @ V_tile
            hc = hc * scale[:, None]
            V_tile = tl.load(V_b + j_idx[:, None]*stride_vn + d_idx[None, :],
                              mask=j_mask[:, None], other=0.0)  # (BLOCK, D_V)
            hc += tl.dot(T_t.to(tl.float32), V_tile.to(tl.float32))
            row_max = nm

        inv_s = 1.0 / (row_sum + 1e-8)
        # Store x centroid
        xc_base = X_CENT_ptr + pid_b*stride_xcb + pid_h*stride_xch
        tl.store(xc_base+i_idx*stride_xcn+0, xc_0*inv_s, mask=i_mask)
        tl.store(xc_base+i_idx*stride_xcn+1, xc_1*inv_s, mask=i_mask)
        tl.store(xc_base+i_idx*stride_xcn+2, xc_2*inv_s, mask=i_mask)
        # Store h centroid
        hc_base = H_CENT_ptr + pid_b*stride_hcb + pid_h*stride_hch
        hc_norm = hc * inv_s[:, None]
        tl.store(hc_base + i_idx[:, None]*stride_hcn + d_idx[None, :],
                 hc_norm, mask=i_mask[:, None])
        # Store log_Z (shared by both outputs)
        tl.store(LOG_Z_ptr+uv+pid_h*N+i_idx, row_max+tl.log(row_sum+1e-30), mask=i_mask)

except ImportError:
    HAS_TRITON = False


# ============================================================================
# CPU fallback
# ============================================================================

def _sinkhorn_cpu(Q_s, K_s, x_f, eps, alpha_h, r_h, mask_f, N, K_iter, V_h=None):
    B, H = Q_s.shape[:2]; device = Q_s.device; cdtype = Q_s.dtype
    C_feat = 1.0 - torch.einsum("bhid,bhjd->bhij", Q_s, K_s)
    diff = x_f.unsqueeze(2) - x_f.unsqueeze(1)
    dist = diff.norm(dim=-1).clamp(min=1e-8)
    r_pos = r_h.to(cdtype).abs().clamp(min=0.1).view(1,H,1,1)
    C_geom = dist.unsqueeze(1) / (r_pos + dist.unsqueeze(1))
    w_h = torch.sigmoid(alpha_h.to(cdtype)).view(1,H,1,1)
    log_K = -(((1-w_h)*C_feat + w_h*C_geom) / eps.to(cdtype).view(1,H,1,1))
    if mask_f is not None:
        lm = torch.log(mask_f.clamp(min=1e-8))
        log_K = log_K + lm.unsqueeze(1).unsqueeze(-1) + lm.unsqueeze(1).unsqueeze(-2)
    log_marg = -torch.log(mask_f.sum(-1,keepdim=True).clamp(min=1)).unsqueeze(1).unsqueeze(-1) if mask_f is not None else -torch.log(torch.tensor(N, device=device, dtype=cdtype))
    lu = torch.zeros(B,H,N,1, device=device, dtype=cdtype)
    lv = torch.zeros(B,H,1,N, device=device, dtype=cdtype)
    for _ in range(K_iter):
        lv = log_marg - torch.logsumexp(log_K+lu, dim=-2, keepdim=True)  # col first
        lu = log_marg - torch.logsumexp(log_K+lv, dim=-1, keepdim=True)  # row last → row sums exact
    T = (lu+log_K+lv).exp()
    if mask_f is not None: T = T * mask_f.view(B,1,N,1) * mask_f.view(B,1,1,N)
    # Row sums = 1/N exactly (row-last iteration), divide to get weighted average
    T_rsum = T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    T_norm = T / T_rsum
    x_cent = torch.einsum("bhij,bjc->bhic", T_norm, x_f)
    if V_h is not None:
        h_cent = torch.einsum("bhij,bhjd->bhid", T_norm, V_h.to(cdtype))
        return x_cent, h_cent
    return x_cent


# ============================================================================
# Autograd Function: Triton forward + unrolled Triton backward
# ============================================================================

class BalancedSinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_s, K_s, x_res, eps, alpha_h, r_h, K_iter, mask):
        B, H, N, d_h = Q_s.shape; device = Q_s.device
        Q_f = Q_s.contiguous().float(); K_f = K_s.contiguous().float()
        x_f = x_res.contiguous().float()
        mask_bias = torch.where(mask.float()>0.5, 0.0, -1e9).contiguous() if mask is not None else torch.zeros(B,N, device=device)
        log_marg = (-torch.log(mask.float().sum(-1,keepdim=True).clamp(min=1)).mean().item() if mask is not None
                     else -torch.log(torch.tensor(N, dtype=torch.float32)).item())
        log_u = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        log_v = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        s = lambda t: t.stride(); sq = Q_f.stride; sx = x_f.stride
        stride_uvb = H*N; N_tiles = (N+BLOCK-1)//BLOCK; grid = (B*H, N_tiles)

        # col first → row last: row sums exact at convergence (centroid = T @ x stable)
        log_u_hist = []; log_v_hist = []
        for _ in range(K_iter):
            log_u_hist.append(log_u.clone())  # save log_u BEFORE col update
            _balanced_col_update[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,mask_bias,
                N,d_h,H,log_marg, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N,BLOCK)
            log_v_hist.append(log_v.clone())  # save log_v AFTER col (= BEFORE row)
            _balanced_row_update[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_v,log_u,mask_bias,
                N,d_h,H,log_marg, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N,BLOCK)

        x_cent = torch.zeros(B,H,N,3, device=device, dtype=torch.float32)
        log_Z = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        _transport_centroid_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,x_cent,log_Z,mask_bias,
            N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),
            x_cent.stride(0),x_cent.stride(1),x_cent.stride(2),stride_uvb,N,BLOCK)

        ctx.save_for_backward(Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,mask_bias,*log_u_hist,*log_v_hist)
        ctx.K_iter=K_iter; ctx.N_base=10; ctx.input_dtype=Q_s.dtype
        return x_cent.to(Q_s.dtype)

    @staticmethod
    def backward(ctx, grad_xc):
        ts = ctx.saved_tensors; K = ctx.K_iter
        Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,mask_bias = ts[:10]
        lu_hist = list(ts[10:10+K]); lv_hist = list(ts[10+K:10+2*K])
        B,H,N,d_h = Q_f.shape; device = Q_f.device
        grad_xc = grad_xc.contiguous().float()
        sq = Q_f.stride; sx = x_f.stride; sgx = grad_xc.stride
        stride_uvb=H*N; N_tiles=(N+BLOCK-1)//BLOCK; grid=(B*H,N_tiles)
        grad_Q=torch.zeros_like(Q_f); grad_K=torch.zeros_like(K_f)
        grad_x=torch.zeros(B,N,3,device=device,dtype=torch.float32)
        grad_alpha=torch.zeros(H,device=device,dtype=torch.float32)
        grad_r=torch.zeros(H,device=device,dtype=torch.float32)

        from deepfold.model.kernels.balanced_sinkhorn_bwd import (
            _centroid_bwd_D_kernel, _centroid_bwd_kernel,
            _balanced_row_bwd_kernel, _balanced_col_bwd_kernel)

        D=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _centroid_bwd_D_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,grad_xc,D,mask_bias,
            N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),sgx(0),sgx(1),sgx(2),stride_uvb,BLOCK)

        g_u=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        g_v=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _centroid_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,grad_xc,D,
            g_u,g_v,grad_x, grad_Q,grad_K,grad_x,grad_alpha,grad_r, mask_bias,
            N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),sgx(0),sgx(1),sgx(2),stride_uvb,N*3,N*3,BLOCK)

        # Backward in reverse of forward: fwd = col→row, bwd = row_bwd→col_bwd
        grad_lu=g_u.clone(); grad_lv=g_v.clone()
        for k in reversed(range(K)):
            # Backward through row update (last in forward): log_u = f(log_v)
            grad_lv_new=torch.zeros(B,H,N,device=device,dtype=torch.float32)
            _balanced_row_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,
                lv_hist[k],grad_lu,mask_bias,
                grad_lv_new,grad_Q,grad_K,grad_x,grad_alpha,grad_r,
                N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N*3,N,BLOCK)
            grad_lv = grad_lv + grad_lv_new

            # Backward through col update (first in forward): log_v = g(log_u)
            grad_lu_new=torch.zeros(B,H,N,device=device,dtype=torch.float32)
            _balanced_col_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,
                lu_hist[k],grad_lv,mask_bias,
                grad_lu_new,grad_Q,grad_K,grad_x,grad_alpha,grad_r,
                N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N*3,N,BLOCK)
            grad_lu=grad_lu_new; grad_lv=torch.zeros(B,H,N,device=device,dtype=torch.float32)

        dt = ctx.input_dtype
        return grad_Q.to(dt),grad_K.to(dt),grad_x.to(dt),None,grad_alpha,grad_r,None,None


# ============================================================================
# Public API
# ============================================================================

def balanced_sinkhorn_transport(
    Q_s: torch.Tensor, K_s: torch.Tensor, x_res: torch.Tensor,
    eps: torch.Tensor, alpha_h: torch.Tensor, r_h: torch.Tensor,
    K_iter: int = 20, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced Sinkhorn transport — returns centroid.

    CUDA: Triton O(N) forward + unrolled backward, exact gradients.
    CPU: materialized O(N²) with gradient checkpointing.
    """
    B,H,N,d_h = Q_s.shape
    cdtype = Q_s.dtype if Q_s.dtype == torch.float64 else torch.float32
    Q_s=Q_s.to(cdtype); K_s=K_s.to(cdtype); x_f=x_res.to(cdtype)
    mask_f = mask.to(cdtype) if mask is not None else None

    if HAS_TRITON and Q_s.is_cuda:
        return BalancedSinkhornFn.apply(Q_s, K_s, x_res, eps, alpha_h, r_h, K_iter, mask)

    if torch.is_grad_enabled():
        return checkpoint(_sinkhorn_cpu, Q_s, K_s, x_f, eps, alpha_h, r_h, mask_f, N, K_iter, use_reentrant=False).to(x_res.dtype)
    return _sinkhorn_cpu(Q_s, K_s, x_f, eps, alpha_h, r_h, mask_f, N, K_iter).to(x_res.dtype)


# ============================================================================
# Dual transport: x_centroid + h_centroid (coupled update)
# ============================================================================

class BalancedSinkhornDualFn(torch.autograd.Function):
    """Forward: fused T@x + T@V_h in one pass. Backward: exact unrolled."""

    @staticmethod
    def forward(ctx, Q_s, K_s, x_res, V_h, eps, alpha_h, r_h, K_iter, mask):
        B, H, N, d_h = Q_s.shape; D_V = V_h.shape[-1]; device = Q_s.device
        Q_f = Q_s.contiguous().float(); K_f = K_s.contiguous().float()
        x_f = x_res.contiguous().float(); V_f = V_h.contiguous().float()
        mask_bias = torch.where(mask.float()>0.5, 0.0, -1e9).contiguous() if mask is not None else torch.zeros(B,N, device=device)
        log_marg = (-torch.log(mask.float().sum(-1,keepdim=True).clamp(min=1)).mean().item() if mask is not None
                     else -torch.log(torch.tensor(N, dtype=torch.float32)).item())
        log_u = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        log_v = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        sq = Q_f.stride; sx = x_f.stride; sv = V_f.stride
        stride_uvb = H*N; N_tiles = (N+BLOCK-1)//BLOCK; grid = (B*H, N_tiles)

        log_u_hist = []; log_v_hist = []
        for _ in range(K_iter):
            log_u_hist.append(log_u.clone())
            _balanced_col_update[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,mask_bias,
                N,d_h,H,log_marg, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N,BLOCK)
            log_v_hist.append(log_v.clone())
            _balanced_row_update[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_v,log_u,mask_bias,
                N,d_h,H,log_marg, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N,BLOCK)

        # Fused dual centroid kernel
        x_cent = torch.zeros(B,H,N,3, device=device, dtype=torch.float32)
        h_cent = torch.zeros(B,H,N,D_V, device=device, dtype=torch.float32)
        log_Z = torch.zeros(B,H,N, device=device, dtype=torch.float32)
        _transport_centroid_dual_kernel[grid](
            Q_f,K_f,x_f,V_f, eps,alpha_h,r_h, log_u,log_v, x_cent,h_cent,log_Z,mask_bias,
            N,d_h,D_V,H, sq(0),sq(1),sq(2),sx(0),sx(1),
            sv(0),sv(1),sv(2),
            x_cent.stride(0),x_cent.stride(1),x_cent.stride(2),
            h_cent.stride(0),h_cent.stride(1),h_cent.stride(2),
            stride_uvb,N,BLOCK)

        ctx.save_for_backward(Q_f,K_f,x_f,V_f,eps,alpha_h,r_h,
                               log_u,log_v,log_Z,mask_bias,
                               *log_u_hist,*log_v_hist)
        ctx.K_iter=K_iter; ctx.N_base=11; ctx.input_dtype=Q_s.dtype; ctx.D_V=D_V
        return x_cent.to(Q_s.dtype), h_cent.to(Q_s.dtype)

    @staticmethod
    def backward(ctx, grad_xc, grad_hc):
        ts = ctx.saved_tensors; K = ctx.K_iter; D_V = ctx.D_V
        Q_f,K_f,x_f,V_f,eps,alpha_h,r_h,log_u,log_v,log_Z,mask_bias = ts[:11]
        lu_hist = list(ts[11:11+K]); lv_hist = list(ts[11+K:11+2*K])
        B,H,N,d_h = Q_f.shape; device = Q_f.device
        grad_xc = grad_xc.contiguous().float()
        grad_hc = grad_hc.contiguous().float()
        sq = Q_f.stride; sx = x_f.stride; sgx = grad_xc.stride; sv = V_f.stride
        sgh = grad_hc.stride
        stride_uvb=H*N; N_tiles=(N+BLOCK-1)//BLOCK; grid=(B*H,N_tiles)

        grad_Q=torch.zeros_like(Q_f); grad_K=torch.zeros_like(K_f)
        grad_x=torch.zeros(B,N,3,device=device,dtype=torch.float32)
        grad_V=torch.zeros_like(V_f)
        grad_alpha=torch.zeros(H,device=device,dtype=torch.float32)
        grad_r=torch.zeros(H,device=device,dtype=torch.float32)

        from deepfold.model.kernels.balanced_sinkhorn_bwd import (
            _centroid_bwd_D_kernel, _centroid_bwd_kernel,
            _balanced_row_bwd_kernel, _balanced_col_bwd_kernel)
        from deepfold.model.kernels.feature_transport_bwd import (
            _feature_bwd_D_kernel, _feature_bwd_kernel)

        # ---- Phase 1a: x centroid backward ----
        D=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _centroid_bwd_D_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,grad_xc,D,mask_bias,
            N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),sgx(0),sgx(1),sgx(2),stride_uvb,BLOCK)

        g_u_x=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        g_v_x=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _centroid_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,log_u,log_v,log_Z,grad_xc,D,
            g_u_x,g_v_x,grad_x, grad_Q,grad_K,grad_x,grad_alpha,grad_r, mask_bias,
            N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),sgx(0),sgx(1),sgx(2),stride_uvb,N*3,N*3,BLOCK)

        # ---- Phase 1b: V_h feature backward ----
        D_h=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _feature_bwd_D_kernel[grid](Q_f,K_f,x_f,V_f,eps,alpha_h,r_h,log_u,log_v,log_Z,
            grad_hc,D_h,mask_bias,
            N,d_h,D_V,H, sq(0),sq(1),sq(2),sx(0),sx(1),
            sv(0),sv(1),sv(2), sgh(0),sgh(1),sgh(2),stride_uvb,BLOCK)

        g_u_h=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        g_v_h=torch.zeros(B,H,N,device=device,dtype=torch.float32)
        _feature_bwd_kernel[grid](Q_f,K_f,x_f,V_f,eps,alpha_h,r_h,log_u,log_v,log_Z,
            grad_hc,D_h, g_u_h,g_v_h,grad_V,
            grad_Q,grad_K,grad_x,grad_alpha,grad_r, mask_bias,
            N,d_h,D_V,H, sq(0),sq(1),sq(2),sx(0),sx(1),
            sv(0),sv(1),sv(2), sgh(0),sgh(1),sgh(2),stride_uvb,N*3,BLOCK)

        # ---- Phase 2: Combine g_u/g_v, then shared Sinkhorn backward ----
        grad_lu = g_u_x + g_u_h
        grad_lv = g_v_x + g_v_h
        for k in reversed(range(K)):
            grad_lv_new=torch.zeros(B,H,N,device=device,dtype=torch.float32)
            _balanced_row_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,
                lv_hist[k],grad_lu,mask_bias,
                grad_lv_new,grad_Q,grad_K,grad_x,grad_alpha,grad_r,
                N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N*3,N,BLOCK)
            grad_lv = grad_lv + grad_lv_new
            grad_lu_new=torch.zeros(B,H,N,device=device,dtype=torch.float32)
            _balanced_col_bwd_kernel[grid](Q_f,K_f,x_f,eps,alpha_h,r_h,
                lu_hist[k],grad_lv,mask_bias,
                grad_lu_new,grad_Q,grad_K,grad_x,grad_alpha,grad_r,
                N,d_h,H, sq(0),sq(1),sq(2),sx(0),sx(1),stride_uvb,N*3,N,BLOCK)
            grad_lu=grad_lu_new; grad_lv=torch.zeros(B,H,N,device=device,dtype=torch.float32)

        dt = ctx.input_dtype
        return grad_Q.to(dt),grad_K.to(dt),grad_x.to(dt),grad_V.to(dt),None,grad_alpha,grad_r,None,None


def balanced_sinkhorn_transport_dual(
    Q_s: torch.Tensor, K_s: torch.Tensor, x_res: torch.Tensor,
    V_h: torch.Tensor,
    eps: torch.Tensor, alpha_h: torch.Tensor, r_h: torch.Tensor,
    K_iter: int = 20, mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced Sinkhorn transport — returns (x_centroid, h_centroid).

    Fused forward: single pass computes both T@x and T@V_h.
    Shared backward: Sinkhorn unrolled backward used by both output paths.

    Returns:
        x_centroid: (B, H, N, 3) transport-weighted coordinate centroid
        h_centroid: (B, H, N, D_V) transport-weighted feature aggregation
    """
    B,H,N,d_h = Q_s.shape
    cdtype = Q_s.dtype if Q_s.dtype == torch.float64 else torch.float32
    Q_s=Q_s.to(cdtype); K_s=K_s.to(cdtype); x_f=x_res.to(cdtype)
    V_f = V_h.to(cdtype)
    mask_f = mask.to(cdtype) if mask is not None else None

    if HAS_TRITON and Q_s.is_cuda:
        return BalancedSinkhornDualFn.apply(Q_s, K_s, x_res, V_f, eps, alpha_h, r_h, K_iter, mask)

    # CPU fallback with gradient checkpointing
    if torch.is_grad_enabled():
        return checkpoint(_sinkhorn_cpu, Q_s, K_s, x_f, eps, alpha_h, r_h, mask_f, N, K_iter, V_f,
                          use_reentrant=False)
    return _sinkhorn_cpu(Q_s, K_s, x_f, eps, alpha_h, r_h, mask_f, N, K_iter, V_f)
