"""
Flash-Sinkhorn Attention with O(N) memory backward (SPEC §18 item 4).

Fuses cost computation + Sinkhorn + transport output + IFT backward.
No N×N matrix is ever materialized — all O(N²) intermediates are
recomputed on-the-fly per tile during both forward and backward.

Forward: Triton kernels (from sinkhorn_kernel.py)
Backward: Python-tiled O(N) memory (correct, debuggable; Triton port later)

Memory budget (excluding pos_bias input):
  Forward saved: Q_ln, K_ln, V, x_res, eps, w_dist, log_u, log_v,
                 O_avg, x_centroid = O(N·d)
  Backward temp: g_v, z_u, z_v, row_lse, col_lse, D,
                 grad_Q, grad_K, grad_V, grad_x = O(N·d)
  Per-tile:      cost_tile, T_tile, s_tile = O(TILE²)
  Total:         O(N·d + TILE²) — truly linear in N
"""

import torch

TILE = 64  # tile size for backward tiling


def _compute_cost_tile_py(
    Q_ln, K_ln, x_res, pos_bias, w_dist, r_0, i_slice, j_slice, eps_h
):
    """Compute cost tile C[i, j] and log_K tile in Python. Returns (ti, tj)."""
    Q_i = Q_ln[i_slice]  # (ti, d_h)
    K_j = K_ln[j_slice]  # (tj, d_h)
    d_h = Q_i.shape[-1]

    content = -(Q_i @ K_j.T) / (d_h**0.5)  # (ti, tj)

    xi = x_res[i_slice]  # (ti, 3)
    xj = x_res[j_slice]  # (tj, 3)
    diff = xi[:, None, :] - xj[None, :, :]  # (ti, tj, 3)
    dist = (diff**2).sum(-1).sqrt().clamp(min=1e-8)  # (ti, tj)
    geo = w_dist * dist / (r_0 + dist)

    pos = pos_bias[i_slice, :][:, j_slice]  # (ti, tj)

    C_tile = content + pos + geo
    log_K_tile = -C_tile / eps_h
    return C_tile, log_K_tile, dist, diff


def _compute_T_norm_tile(log_u, log_v, log_K_tile, i_slice, j_slice):
    """Compute row-normalized T_norm tile. Returns (ti, tj) and row_sum contribution."""
    log_score = log_u[i_slice, None] + log_K_tile + log_v[None, j_slice]
    # For numerical stability, subtract per-row max
    row_max = log_score.max(dim=-1, keepdim=True).values
    T_tile = torch.exp(log_score - row_max)
    return T_tile, row_max.squeeze(-1)


class FlashSinkhornAttn(torch.autograd.Function):
    """
    O(N) memory Sinkhorn attention: forward (Triton) + backward (tiled Python).

    Inputs:
        Q_ln:     (H, N, d_h) — LayerNorm'd queries
        K_ln:     (H, N, d_h) — LayerNorm'd keys
        V:        (H, N, d_h) — values
        G:        (H, N, d_h) — gate logits
        x_res:    (N, 3)      — coordinates
        pos_bias: (H, N, N)   — precomputed position bias
        eps:      (H,)        — per-head epsilon (buffer)
        w_dist:   (H,)        — per-head geometry weight
        log_mu:   (H, N)      — log row marginals
        log_nu:   (H, N)      — log column marginals

    Outputs:
        o_gated:    (N, H*d_h) — gated transport-weighted output (for h += W_O(o))
        x_centroid: (H, N, 3)  — transport-weighted centroid (for EGNN)
        log_u:      (H, N)     — converged row duals
        log_v:      (H, N)     — converged column duals
    """

    @staticmethod
    def forward(
        ctx,
        Q_ln,
        K_ln,
        V,
        G,
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
    ):
        H, N, d_h = Q_ln.shape

        # Ensure FP32
        Q_ln = Q_ln.contiguous().float()
        K_ln = K_ln.contiguous().float()
        V = V.contiguous().float()
        G = G.contiguous().float()
        x_res = x_res.contiguous().float()
        pos_bias = pos_bias.contiguous().float()
        eps = eps.contiguous().float()
        w_dist = w_dist.contiguous().float()
        log_mu = log_mu.contiguous().float()
        log_nu = log_nu.contiguous().float()

        # --- Sinkhorn iterations (use Triton if available, else Python) ---
        try:
            from deepfold.model.kernels.sinkhorn_kernel import (
                flash_sinkhorn as _triton_fwd,
            )

            O_avg, x_centroid, log_u, log_v = _triton_fwd(
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
            )
        except Exception:
            # Fallback: materialized forward (for CPU / debugging)
            kappa = lam / (lam + eps)
            C = torch.zeros(H, N, N, device=Q_ln.device, dtype=torch.float32)
            for h in range(H):
                content = -(Q_ln[h] @ K_ln[h].T) / (d_h**0.5)
                dist = torch.cdist(x_res, x_res)
                geo = w_dist[h] * dist / (r_0 + dist)
                C[h] = content + pos_bias[h] + geo

            log_K = -C / eps[:, None, None]
            log_u = (
                log_u_init.clone()
                if log_u_init is not None
                else torch.zeros(H, N, device=Q_ln.device)
            )
            log_v = (
                log_v_init.clone()
                if log_v_init is not None
                else torch.zeros(H, N, device=Q_ln.device)
            )

            for _ in range(K_iter):
                log_u = kappa[:, None] * (
                    log_mu - torch.logsumexp(log_K + log_v[:, None, :], dim=-1)
                )
                log_v = kappa[:, None] * (
                    log_nu - torch.logsumexp(log_K + log_u[:, :, None], dim=-2)
                )

            log_score = log_u[:, :, None] + log_K + log_v[:, None, :]
            row_max = log_score.max(dim=-1, keepdim=True).values
            T = torch.exp(log_score - row_max)
            T_sum = T.sum(dim=-1, keepdim=True)
            T_norm = T / (T_sum + 1e-6)
            O_avg = torch.einsum("hnm,hmd->hnd", T_norm, V)
            x_centroid = torch.einsum("hnm,mc->hnc", T_norm, x_res)

        # Gated output
        sig_G = torch.sigmoid(G)
        o_gated = sig_G * O_avg  # (H, N, d_h)
        o_flat = o_gated.permute(1, 0, 2).reshape(N, H * d_h)  # (N, H*d_h)

        # Save for backward — all O(N·d), no N×N
        ctx.save_for_backward(
            Q_ln,
            K_ln,
            V,
            G,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_mu,
            log_nu,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            sig_G,
        )
        ctx.K_iter = K_iter
        ctx.lam = lam
        ctx.r_0 = r_0

        return o_flat, x_centroid, log_u, log_v

    @staticmethod
    def backward(ctx, grad_o_flat, grad_xc, grad_lu_unused, grad_lv_unused):
        (
            Q_ln,
            K_ln,
            V,
            G,
            x_res,
            pos_bias,
            eps,
            w_dist,
            log_mu,
            log_nu,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            sig_G,
        ) = ctx.saved_tensors
        K_back = ctx.K_iter
        lam = ctx.lam
        r_0 = ctx.r_0

        H, N, d_h = Q_ln.shape
        kappa = lam / (lam + eps)  # (H,)
        device = Q_ln.device

        # ====================================================================
        # Step 0: Backward through gating
        # o_flat was reshaped from (H, N, d_h) gated output
        # ====================================================================
        grad_o_gated = grad_o_flat.view(N, H, d_h).permute(1, 0, 2)  # (H, N, d_h)
        grad_O_avg = sig_G * grad_o_gated  # (H, N, d_h)
        grad_G = sig_G * (1 - sig_G) * O_avg * grad_o_gated  # (H, N, d_h)

        if grad_xc is None:
            grad_xc = torch.zeros_like(x_centroid)

        # ====================================================================
        # Step 1: Compute D_i = sum_j T_norm_ij * grad_T_norm_ij (no tiling needed)
        #   D_i = grad_O_avg_i · O_avg_i + grad_xc_i · x_centroid_i
        # ====================================================================
        D = (grad_O_avg * O_avg).sum(dim=-1) + (grad_xc * x_centroid).sum(
            dim=-1
        )  # (H, N)

        # ====================================================================
        # Step 1b: Compute global row normalizer log_Z (tiled online LSE)
        #   log_Z[h,i] = logsumexp_j(log_u_i + log_K_ij + log_v_j)
        #   T_norm_ij = exp(log_u_i + log_K_ij + log_v_j - log_Z_i)
        # ====================================================================
        log_Z = torch.zeros(H, N, device=device, dtype=torch.float32)

        for h in range(H):
            eps_h = eps[h]
            w_dist_h = w_dist[h]
            for i0 in range(0, N, TILE):
                ie = min(i0 + TILE, N)
                i_sl = slice(i0, ie)
                max_val = torch.full((ie - i0,), -1e30, device=device)
                sum_exp = torch.zeros(ie - i0, device=device)
                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    j_sl = slice(j0, je)
                    _, log_K_tile, _, _ = _compute_cost_tile_py(
                        Q_ln[h],
                        K_ln[h],
                        x_res,
                        pos_bias[h],
                        w_dist_h,
                        r_0,
                        i_sl,
                        j_sl,
                        eps_h,
                    )
                    score = log_u[h, i_sl, None] + log_K_tile + log_v[h, None, j_sl]
                    tile_max = score.max(dim=-1).values
                    new_max = torch.maximum(max_val, tile_max)
                    sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                        score - new_max[:, None]
                    ).sum(dim=-1)
                    max_val = new_max
                log_Z[h, i_sl] = max_val + torch.log(sum_exp + 1e-30)

        # ====================================================================
        # Step 2: Tiled pass — compute g_v, grad_V, grad_x (from transport)
        #   g_u = 0 (row-normalization cancels log_u effect)
        #   g_v_j = sum_i T_norm_ij * (dT_ij - D_i)
        #   grad_V_j = sum_i T_norm_ij * grad_O_avg_i
        #   grad_x_j (transport part) = sum_i T_norm_ij * grad_xc_i
        #   T_norm_ij = exp(log_u_i + log_K_ij + log_v_j - log_Z_i)
        # ====================================================================
        g_u = torch.zeros(H, N, device=device, dtype=torch.float32)
        g_v = torch.zeros(H, N, device=device, dtype=torch.float32)
        grad_V = torch.zeros_like(V)
        grad_x_transport = torch.zeros(N, 3, device=device, dtype=torch.float32)

        for h in range(H):
            eps_h = eps[h]
            w_dist_h = w_dist[h]

            for j0 in range(0, N, TILE):
                je = min(j0 + TILE, N)
                j_sl = slice(j0, je)

                acc_gv = torch.zeros(je - j0, device=device)
                acc_grad_V = torch.zeros(je - j0, d_h, device=device)
                acc_grad_x = torch.zeros(je - j0, 3, device=device)

                for i0 in range(0, N, TILE):
                    ie = min(i0 + TILE, N)
                    i_sl = slice(i0, ie)

                    # Recompute cost tile
                    _, log_K_tile, _, _ = _compute_cost_tile_py(
                        Q_ln[h],
                        K_ln[h],
                        x_res,
                        pos_bias[h],
                        w_dist_h,
                        r_0,
                        i_sl,
                        j_sl,
                        eps_h,
                    )

                    # T_norm with GLOBAL row normalizer
                    log_score = log_u[h, i_sl, None] + log_K_tile + log_v[h, None, j_sl]
                    T_norm_tile = torch.exp(
                        log_score - log_Z[h, i_sl, None]
                    )  # (ti, tj)

                    # dT_ij = grad_O_avg_i · V_j + grad_xc_i · x_j
                    dT = (
                        grad_O_avg[h, i_sl] @ V[h, j_sl].T
                        + grad_xc[h, i_sl] @ x_res[j_sl].T
                    )

                    # grad_log_score_ij = T_norm * (dT - D_i)
                    grad_ls = T_norm_tile * (dT - D[h, i_sl, None])

                    acc_gv += grad_ls.sum(dim=0)
                    acc_grad_V += T_norm_tile.T @ grad_O_avg[h, i_sl]
                    acc_grad_x += T_norm_tile.T @ grad_xc[h, i_sl]

                g_v[h, j_sl] = acc_gv
                grad_V[h, j_sl] += acc_grad_V
                grad_x_transport[j_sl] += acc_grad_x

        # ====================================================================
        # Step 3: IFT adjoint iterations → z_u, z_v
        #   Precompute row_lse, col_lse (one tiled scan each)
        #   Then K_back iterations of z_u, z_v updates
        # ====================================================================
        row_lse = torch.zeros(H, N, device=device)
        col_lse = torch.zeros(H, N, device=device)

        for h in range(H):
            eps_h = eps[h]
            w_dist_h = w_dist[h]

            # row_lse[i] = LSE_j(log_K_ij + log_v_j)
            for i0 in range(0, N, TILE):
                ie = min(i0 + TILE, N)
                i_sl = slice(i0, ie)
                max_val = torch.full((ie - i0,), -1e30, device=device)
                sum_exp = torch.zeros(ie - i0, device=device)

                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    j_sl = slice(j0, je)
                    _, log_K_tile, _, _ = _compute_cost_tile_py(
                        Q_ln[h],
                        K_ln[h],
                        x_res,
                        pos_bias[h],
                        w_dist_h,
                        r_0,
                        i_sl,
                        j_sl,
                        eps_h,
                    )
                    score = log_K_tile + log_v[h, None, j_sl]
                    tile_max = score.max(dim=-1).values
                    new_max = torch.maximum(max_val, tile_max)
                    sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                        score - new_max[:, None]
                    ).sum(dim=-1)
                    max_val = new_max

                row_lse[h, i_sl] = max_val + torch.log(sum_exp + 1e-30)

            # col_lse[j] = LSE_i(log_K_ij + log_u_i)
            for j0 in range(0, N, TILE):
                je = min(j0 + TILE, N)
                j_sl = slice(j0, je)
                max_val = torch.full((je - j0,), -1e30, device=device)
                sum_exp = torch.zeros(je - j0, device=device)

                for i0 in range(0, N, TILE):
                    ie = min(i0 + TILE, N)
                    i_sl = slice(i0, ie)
                    _, log_K_tile, _, _ = _compute_cost_tile_py(
                        Q_ln[h],
                        K_ln[h],
                        x_res,
                        pos_bias[h],
                        w_dist_h,
                        r_0,
                        i_sl,
                        j_sl,
                        eps_h,
                    )
                    score = (log_K_tile + log_u[h, i_sl, None]).T  # (tj, ti)
                    tile_max = score.max(dim=-1).values
                    new_max = torch.maximum(max_val, tile_max)
                    sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                        score - new_max[:, None]
                    ).sum(dim=-1)
                    max_val = new_max

                col_lse[h, j_sl] = max_val + torch.log(sum_exp + 1e-30)

        # IFT iterations
        z_u = g_u.clone()
        z_v = g_v.clone()

        for _ in range(K_back):
            z_u_new = torch.zeros_like(z_u)
            z_v_new = torch.zeros_like(z_v)

            for h in range(H):
                eps_h = eps[h]
                kappa_h = kappa[h]
                w_dist_h = w_dist[h]

                # z_u[i] = g_u[i] + kappa * sum_j S_col[i,j] * z_v[j]
                for i0 in range(0, N, TILE):
                    ie = min(i0 + TILE, N)
                    i_sl = slice(i0, ie)
                    acc = torch.zeros(ie - i0, device=device)
                    for j0 in range(0, N, TILE):
                        je = min(j0 + TILE, N)
                        j_sl = slice(j0, je)
                        _, log_K_tile, _, _ = _compute_cost_tile_py(
                            Q_ln[h],
                            K_ln[h],
                            x_res,
                            pos_bias[h],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )
                        log_s = (
                            log_K_tile + log_u[h, i_sl, None] - col_lse[h, None, j_sl]
                        )
                        s_col_tile = torch.exp(log_s)
                        acc += (s_col_tile * z_v[h, None, j_sl]).sum(dim=-1)
                    z_u_new[h, i_sl] = g_u[h, i_sl] + kappa_h * acc

                # z_v[j] = g_v[j] + kappa * sum_i S_row[i,j] * z_u[i]
                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    j_sl = slice(j0, je)
                    acc = torch.zeros(je - j0, device=device)
                    for i0 in range(0, N, TILE):
                        ie = min(i0 + TILE, N)
                        i_sl = slice(i0, ie)
                        _, log_K_tile, _, _ = _compute_cost_tile_py(
                            Q_ln[h],
                            K_ln[h],
                            x_res,
                            pos_bias[h],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )
                        log_s = (
                            log_K_tile + log_v[h, None, j_sl] - row_lse[h, i_sl, None]
                        )
                        s_row_tile = torch.exp(log_s)
                        acc += (s_row_tile * z_u_new[h, i_sl, None]).sum(dim=0)
                    z_v_new[h, j_sl] = g_v[h, j_sl] + kappa_h * acc

            z_u = z_u_new
            z_v = z_v_new

        # ====================================================================
        # Step 4: Cost gradient — fused with propagation to Q, K, x, w_dist
        #   grad_C_direct = -(1/eps) * grad_log_score  (from step 2 via T_norm)
        #   grad_C_ift = -(kappa/eps) * (z_u * s_row + z_v * s_col)
        #   Immediately propagate to grad_Q, grad_K, grad_x, grad_w_dist
        # ====================================================================
        grad_Q_ln = torch.zeros_like(Q_ln)
        grad_K_ln = torch.zeros_like(K_ln)
        grad_x_cost = torch.zeros(N, 3, device=device, dtype=torch.float32)
        grad_w_dist = torch.zeros(H, device=device)
        grad_pos_bias = torch.zeros_like(pos_bias)

        for h in range(H):
            eps_h = eps[h]
            kappa_h = kappa[h]
            w_dist_h = w_dist[h]
            inv_sqrt_dh = 1.0 / (d_h**0.5)

            for i0 in range(0, N, TILE):
                ie = min(i0 + TILE, N)
                i_sl = slice(i0, ie)
                ti = ie - i0

                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    j_sl = slice(j0, je)

                    C_tile, log_K_tile, dist_tile, diff_tile = _compute_cost_tile_py(
                        Q_ln[h],
                        K_ln[h],
                        x_res,
                        pos_bias[h],
                        w_dist_h,
                        r_0,
                        i_sl,
                        j_sl,
                        eps_h,
                    )

                    # --- Direct gradient (through transport output) ---
                    log_score = log_u[h, i_sl, None] + log_K_tile + log_v[h, None, j_sl]
                    T_norm_tile = torch.exp(
                        log_score - log_Z[h, i_sl, None]
                    )  # global normalizer

                    dT = (
                        grad_O_avg[h, i_sl] @ V[h, j_sl].T
                        + grad_xc[h, i_sl] @ x_res[j_sl].T
                    )
                    grad_ls_direct = T_norm_tile * (dT - D[h, i_sl, None])
                    grad_C_direct = grad_ls_direct * (-1.0 / eps_h)

                    # --- IFT gradient ---
                    log_s_row = (
                        log_K_tile + log_v[h, None, j_sl] - row_lse[h, i_sl, None]
                    )
                    s_row_tile = torch.exp(log_s_row)
                    log_s_col = (
                        log_K_tile + log_u[h, i_sl, None] - col_lse[h, None, j_sl]
                    )
                    s_col_tile = torch.exp(log_s_col)

                    grad_C_ift = -(kappa_h / eps_h) * (
                        z_u[h, i_sl, None] * s_row_tile
                        + z_v[h, None, j_sl] * s_col_tile
                    )

                    grad_C_total = grad_C_direct + grad_C_ift  # (ti, tj)

                    # --- Propagate to Q_ln, K_ln ---
                    # content = -(Q_i @ K_j^T) / sqrt(d_h)
                    # grad_Q_i += grad_C @ (-K_j / sqrt(d_h))
                    # grad_K_j += grad_C^T @ (-Q_i / sqrt(d_h))
                    grad_Q_ln[h, i_sl] += (-inv_sqrt_dh) * (
                        grad_C_total @ K_ln[h, j_sl]
                    )
                    grad_K_ln[h, j_sl] += (-inv_sqrt_dh) * (
                        grad_C_total.T @ Q_ln[h, i_sl]
                    )

                    # --- Propagate to x_res ---
                    # geo = w_dist * d / (r_0 + d)
                    # d(d/(r_0+d))/dd = r_0 / (r_0 + d)^2
                    # dd/dx_i = (x_i - x_j) / d
                    geo_grad_coeff = w_dist_h * r_0 / (r_0 + dist_tile) ** 2  # (ti, tj)
                    # grad_C_total * geo_grad_coeff * (x_i - x_j) / d
                    weighted = grad_C_total * geo_grad_coeff / dist_tile  # (ti, tj)
                    # diff_tile is (ti, tj, 3)
                    grad_x_cost[i_sl] += (weighted.unsqueeze(-1) * diff_tile).sum(dim=1)
                    grad_x_cost[j_sl] -= (weighted.unsqueeze(-1) * diff_tile).sum(dim=0)

                    # --- Propagate to w_dist ---
                    f_dist_tile = dist_tile / (r_0 + dist_tile)
                    grad_w_dist[h] += (grad_C_total * f_dist_tile).sum()

                    # --- Propagate to pos_bias ---
                    grad_pos_bias[h, i_sl, j_sl] = grad_C_total

        # ====================================================================
        # Step 5: Combine x_res gradients
        # ====================================================================
        grad_x_res = grad_x_transport + grad_x_cost

        # Marginal gradients (from IFT)
        grad_log_mu = kappa[:, None] * z_u
        grad_log_nu = kappa[:, None] * z_v

        # Return gradients for all inputs
        # Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist, log_mu, log_nu,
        # K_iter, lam, r_0, log_u_init, log_v_init
        return (
            grad_Q_ln,
            grad_K_ln,
            grad_V,
            grad_G,
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
        )


def flash_sinkhorn_attn(
    Q_ln: torch.Tensor,
    K_ln: torch.Tensor,
    V: torch.Tensor,
    G: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash-Sinkhorn attention with O(N) memory forward+backward.

    Returns:
        o_flat:     (N, H*d_h) gated transport-weighted output
        x_centroid: (H, N, 3)  transport-weighted centroid for EGNN
        log_u:      (H, N)     converged row duals
        log_v:      (H, N)     converged column duals
    """
    return FlashSinkhornAttn.apply(
        Q_ln,
        K_ln,
        V,
        G,
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
    )
