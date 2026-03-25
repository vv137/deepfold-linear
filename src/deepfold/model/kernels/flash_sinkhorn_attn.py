"""
Flash-Sinkhorn Attention with O(N) memory backward (SPEC §18 item 4).

Fuses cost computation + Sinkhorn + transport output + IFT backward.
No N×N matrix is ever materialized — all O(N²) intermediates are
recomputed on-the-fly per tile during both forward and backward.

Supports both unbatched (H, N, d_h) and batched (B, H, N, d_h) inputs.

Forward: Triton kernels (from sinkhorn_kernel.py)
Backward: Python-tiled O(N) memory (correct, debuggable; Triton port later)

Memory budget (excluding pos_weight/pos_bins input):
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
    Q_ln, K_ln, x_res, pos_weight, pos_bins, w_dist, r_0, i_slice, j_slice, eps_h
):
    """Compute cost tile C[i, j] and log_K tile in Python. Returns (ti, tj).

    Args:
        pos_weight: (68,) weight vector for the current head h
        pos_bins:   (N, N) int32 bin indices
    """
    Q_i = Q_ln[i_slice]  # (ti, d_h)
    K_j = K_ln[j_slice]  # (tj, d_h)
    d_h = Q_i.shape[-1]

    content = -(Q_i @ K_j.T) / (d_h**0.5)  # (ti, tj)

    xi = x_res[i_slice]  # (ti, 3)
    xj = x_res[j_slice]  # (tj, 3)
    diff = xi[:, None, :] - xj[None, :, :]  # (ti, tj, 3)
    dist = (diff**2).sum(-1).sqrt().clamp(min=1e-8)  # (ti, tj)
    geo = w_dist * dist / (r_0 + dist)

    bins_tile = pos_bins[i_slice, :][:, j_slice].long()
    pos = pos_weight[bins_tile]  # (ti, tj)

    C_tile = content + pos + geo
    log_K_tile = -C_tile / eps_h
    return C_tile, log_K_tile, dist, diff


class FlashSinkhornAttn(torch.autograd.Function):
    """
    O(N) memory Sinkhorn attention: forward (Triton) + backward (tiled Python).

    Always expects batched inputs (unbatched handling is in flash_sinkhorn_attn wrapper):
        Q_ln:       (B, H, N, d_h)
        x_res:      (B, N, 3)
        pos_weight: (H, 68)
        pos_bins:   (B, N, N) int32
        eps:        (H,)
        w_dist:     (H,)
        log_mu:     (B, H, N)
        log_nu:     (B, H, N)

    Outputs:
        o_gated:    (B, N, H*d_h) — gated transport-weighted output
        x_centroid: (B, H, N, 3)  — transport-weighted centroid
        log_u:      (B, H, N)     — converged row duals
        log_v:      (B, H, N)     — converged column duals
    """

    @staticmethod
    def forward(
        ctx,
        Q_ln,
        K_ln,
        V,
        G,
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
        mask,  # (B, N) float, 1=real 0=pad. None means all valid.
    ):
        B, H, N, d_h = Q_ln.shape

        # Ensure FP32
        Q_ln = Q_ln.contiguous().float()
        K_ln = K_ln.contiguous().float()
        V = V.contiguous().float()
        G = G.contiguous().float()
        x_res = x_res.contiguous().float()
        pos_weight = pos_weight.contiguous().float()
        pos_bins = pos_bins.contiguous().to(torch.int32)
        eps = eps.contiguous().float()
        w_dist = w_dist.contiguous().float()
        log_mu = log_mu.contiguous().float()
        log_nu = log_nu.contiguous().float()

        # Mask: (B, N) → column bias for Sinkhorn (-1e9 for padded)
        if mask is None:
            mask = Q_ln.new_ones(B, N)
        mask = mask.contiguous().float()
        # col_mask_bias: (B, 1, N) — broadcast over H, added to log_K columns
        col_mask_bias = (1.0 - mask).unsqueeze(1) * (-1e9)  # (B, 1, N)

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
                mask=mask,
            )
        except Exception:
            # Fallback: materialized forward per sample (for CPU / debugging)
            kappa = lam / (lam + eps)
            O_avg_list = []
            xc_list = []
            lu_list = []
            lv_list = []

            for b in range(B):
                C = torch.zeros(H, N, N, device=Q_ln.device, dtype=torch.float32)
                for h in range(H):
                    content = -(Q_ln[b, h] @ K_ln[b, h].T) / (d_h**0.5)
                    dist = torch.cdist(x_res[b], x_res[b])
                    geo = w_dist[h] * dist / (r_0 + dist)
                    pb_h = pos_weight[h][pos_bins[b].long()]  # (N, N)
                    C[h] = content + pb_h + geo

                log_K = -C / eps[:, None, None]
                lu = (
                    log_u_init[b].clone()
                    if log_u_init is not None
                    else torch.zeros(H, N, device=Q_ln.device)
                )
                lv = (
                    log_v_init[b].clone()
                    if log_v_init is not None
                    else torch.zeros(H, N, device=Q_ln.device)
                )

                # Mask bias: -1e9 for padded positions, 0 for valid
                # (1, N) for column masking in row update (logsumexp over j)
                # (N, 1) for row masking in col update (logsumexp over i)
                mask_vec = col_mask_bias[b].squeeze(0)  # (N,)

                for _ in range(K_iter):
                    # Row update: mask columns (padded j gets -inf)
                    lu = kappa[:, None] * (
                        log_mu[b]
                        - torch.logsumexp(
                            log_K + lv[:, None, :] + mask_vec[None, :], dim=-1
                        )
                    )
                    # Col update: mask rows (padded i gets -inf)
                    lv = kappa[:, None] * (
                        log_nu[b]
                        - torch.logsumexp(
                            log_K + lu[:, :, None] + mask_vec[:, None], dim=-2
                        )
                    )

                log_score = lu[:, :, None] + log_K + lv[:, None, :] + mask_vec[None, :]
                row_max = log_score.max(dim=-1, keepdim=True).values
                T = torch.exp(log_score - row_max)
                T_sum = T.sum(dim=-1, keepdim=True)
                T_norm = T / (T_sum + 1e-6)
                O_avg_list.append(torch.einsum("hnm,hmd->hnd", T_norm, V[b]))
                xc_list.append(torch.einsum("hnm,mc->hnc", T_norm, x_res[b]))
                lu_list.append(lu)
                lv_list.append(lv)

            O_avg = torch.stack(O_avg_list)  # (B, H, N, d_h)
            x_centroid = torch.stack(xc_list)  # (B, H, N, 3)
            log_u = torch.stack(lu_list)  # (B, H, N)
            log_v = torch.stack(lv_list)  # (B, H, N)

        # Gated output
        sig_G = torch.sigmoid(G)
        o_gated = sig_G * O_avg  # (B, H, N, d_h)
        o_flat = o_gated.permute(0, 2, 1, 3).reshape(B, N, H * d_h)  # (B, N, H*d_h)

        # Zero outputs at padded positions
        mask_hn = mask.unsqueeze(1)  # (B, 1, N)
        log_u = log_u * mask_hn
        log_v = log_v * mask_hn
        o_flat = o_flat * mask.unsqueeze(-1)  # (B, N, 1)

        # Save for backward — all O(N·d), no N×N
        ctx.save_for_backward(
            Q_ln,
            K_ln,
            V,
            G,
            x_res,
            pos_weight,
            pos_bins,
            eps,
            w_dist,
            log_mu,
            log_nu,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            sig_G,
            mask,
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
            pos_weight,
            pos_bins,
            eps,
            w_dist,
            log_mu,
            log_nu,
            log_u,
            log_v,
            O_avg,
            x_centroid,
            sig_G,
            mask,
        ) = ctx.saved_tensors
        K_back = ctx.K_iter
        lam = ctx.lam
        r_0 = ctx.r_0

        B, H, N, d_h = Q_ln.shape
        kappa = lam / (lam + eps)  # (H,)
        device = Q_ln.device

        # ====================================================================
        # Step 0: Backward through gating
        # ====================================================================
        grad_o_gated = grad_o_flat.view(B, N, H, d_h).permute(
            0, 2, 1, 3
        )  # (B, H, N, d_h)
        grad_O_avg = sig_G * grad_o_gated  # (B, H, N, d_h)
        grad_G = sig_G * (1 - sig_G) * O_avg * grad_o_gated  # (B, H, N, d_h)

        if grad_xc is None:
            grad_xc = torch.zeros_like(x_centroid)

        # ====================================================================
        # Step 1: Compute D
        # ====================================================================
        D = (grad_O_avg * O_avg).sum(dim=-1) + (grad_xc * x_centroid).sum(
            dim=-1
        )  # (B, H, N)

        # ====================================================================
        # Step 1b: Compute global row normalizer log_Z (tiled online LSE)
        # ====================================================================
        log_Z = torch.zeros(B, H, N, device=device, dtype=torch.float32)

        for b in range(B):
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
                            Q_ln[b, h],
                            K_ln[b, h],
                            x_res[b],
                            pos_weight[h],
                            pos_bins[b],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )
                        score = (
                            log_u[b, h, i_sl, None]
                            + log_K_tile
                            + log_v[b, h, None, j_sl]
                        )
                        tile_max = score.max(dim=-1).values
                        new_max = torch.maximum(max_val, tile_max)
                        sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                            score - new_max[:, None]
                        ).sum(dim=-1)
                        max_val = new_max
                    log_Z[b, h, i_sl] = max_val + torch.log(sum_exp + 1e-30)

        # ====================================================================
        # Step 2: Tiled pass — compute g_v, grad_V, grad_x (from transport)
        # ====================================================================
        g_u = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        g_v = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        grad_V = torch.zeros_like(V)
        grad_x_transport = torch.zeros(B, N, 3, device=device, dtype=torch.float32)

        for b in range(B):
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

                        _, log_K_tile, _, _ = _compute_cost_tile_py(
                            Q_ln[b, h],
                            K_ln[b, h],
                            x_res[b],
                            pos_weight[h],
                            pos_bins[b],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )

                        log_score = (
                            log_u[b, h, i_sl, None]
                            + log_K_tile
                            + log_v[b, h, None, j_sl]
                        )
                        T_norm_tile = torch.exp(log_score - log_Z[b, h, i_sl, None])

                        dT = (
                            grad_O_avg[b, h, i_sl] @ V[b, h, j_sl].T
                            + grad_xc[b, h, i_sl] @ x_res[b, j_sl].T
                        )

                        grad_ls = T_norm_tile * (dT - D[b, h, i_sl, None])

                        acc_gv += grad_ls.sum(dim=0)
                        acc_grad_V += T_norm_tile.T @ grad_O_avg[b, h, i_sl]
                        acc_grad_x += T_norm_tile.T @ grad_xc[b, h, i_sl]

                    g_v[b, h, j_sl] = acc_gv
                    grad_V[b, h, j_sl] += acc_grad_V
                    grad_x_transport[b, j_sl] += acc_grad_x

        # ====================================================================
        # Step 3: IFT adjoint iterations → z_u, z_v
        # ====================================================================
        row_lse = torch.zeros(B, H, N, device=device)
        col_lse = torch.zeros(B, H, N, device=device)

        for b in range(B):
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
                            Q_ln[b, h],
                            K_ln[b, h],
                            x_res[b],
                            pos_weight[h],
                            pos_bins[b],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )
                        score = log_K_tile + log_v[b, h, None, j_sl]
                        tile_max = score.max(dim=-1).values
                        new_max = torch.maximum(max_val, tile_max)
                        sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                            score - new_max[:, None]
                        ).sum(dim=-1)
                        max_val = new_max

                    row_lse[b, h, i_sl] = max_val + torch.log(sum_exp + 1e-30)

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
                            Q_ln[b, h],
                            K_ln[b, h],
                            x_res[b],
                            pos_weight[h],
                            pos_bins[b],
                            w_dist_h,
                            r_0,
                            i_sl,
                            j_sl,
                            eps_h,
                        )
                        score = (log_K_tile + log_u[b, h, i_sl, None]).T
                        tile_max = score.max(dim=-1).values
                        new_max = torch.maximum(max_val, tile_max)
                        sum_exp = sum_exp * torch.exp(max_val - new_max) + torch.exp(
                            score - new_max[:, None]
                        ).sum(dim=-1)
                        max_val = new_max

                    col_lse[b, h, j_sl] = max_val + torch.log(sum_exp + 1e-30)

        # IFT iterations
        z_u = g_u.clone()
        z_v = g_v.clone()

        for _ in range(K_back):
            z_u_new = torch.zeros_like(z_u)
            z_v_new = torch.zeros_like(z_v)

            for b in range(B):
                for h in range(H):
                    eps_h = eps[h]
                    kappa_h = kappa[h]
                    w_dist_h = w_dist[h]

                    for i0 in range(0, N, TILE):
                        ie = min(i0 + TILE, N)
                        i_sl = slice(i0, ie)
                        acc = torch.zeros(ie - i0, device=device)
                        for j0 in range(0, N, TILE):
                            je = min(j0 + TILE, N)
                            j_sl = slice(j0, je)
                            _, log_K_tile, _, _ = _compute_cost_tile_py(
                                Q_ln[b, h],
                                K_ln[b, h],
                                x_res[b],
                                pos_weight[h],
                                pos_bins[b],
                                w_dist_h,
                                r_0,
                                i_sl,
                                j_sl,
                                eps_h,
                            )
                            log_s = (
                                log_K_tile
                                + log_u[b, h, i_sl, None]
                                - col_lse[b, h, None, j_sl]
                            )
                            s_col_tile = torch.exp(log_s)
                            acc += (s_col_tile * z_v[b, h, None, j_sl]).sum(dim=-1)
                        z_u_new[b, h, i_sl] = g_u[b, h, i_sl] + kappa_h * acc

                    for j0 in range(0, N, TILE):
                        je = min(j0 + TILE, N)
                        j_sl = slice(j0, je)
                        acc = torch.zeros(je - j0, device=device)
                        for i0 in range(0, N, TILE):
                            ie = min(i0 + TILE, N)
                            i_sl = slice(i0, ie)
                            _, log_K_tile, _, _ = _compute_cost_tile_py(
                                Q_ln[b, h],
                                K_ln[b, h],
                                x_res[b],
                                pos_weight[h],
                                pos_bins[b],
                                w_dist_h,
                                r_0,
                                i_sl,
                                j_sl,
                                eps_h,
                            )
                            log_s = (
                                log_K_tile
                                + log_v[b, h, None, j_sl]
                                - row_lse[b, h, i_sl, None]
                            )
                            s_row_tile = torch.exp(log_s)
                            acc += (s_row_tile * z_u_new[b, h, i_sl, None]).sum(dim=0)
                        z_v_new[b, h, j_sl] = g_v[b, h, j_sl] + kappa_h * acc

            z_u = z_u_new
            z_v = z_v_new

        # ====================================================================
        # Step 4: Cost gradient — fused with propagation to Q, K, x, w_dist
        # ====================================================================
        grad_Q_ln = torch.zeros_like(Q_ln)
        grad_K_ln = torch.zeros_like(K_ln)
        grad_x_cost = torch.zeros(B, N, 3, device=device, dtype=torch.float32)
        grad_w_dist = torch.zeros(H, device=device)
        grad_pos_weight = torch.zeros_like(pos_weight)

        for b in range(B):
            for h in range(H):
                eps_h = eps[h]
                kappa_h = kappa[h]
                w_dist_h = w_dist[h]
                inv_sqrt_dh = 1.0 / (d_h**0.5)

                for i0 in range(0, N, TILE):
                    ie = min(i0 + TILE, N)
                    i_sl = slice(i0, ie)

                    for j0 in range(0, N, TILE):
                        je = min(j0 + TILE, N)
                        j_sl = slice(j0, je)

                        C_tile, log_K_tile, dist_tile, diff_tile = (
                            _compute_cost_tile_py(
                                Q_ln[b, h],
                                K_ln[b, h],
                                x_res[b],
                                pos_weight[h],
                                pos_bins[b],
                                w_dist_h,
                                r_0,
                                i_sl,
                                j_sl,
                                eps_h,
                            )
                        )

                        log_score = (
                            log_u[b, h, i_sl, None]
                            + log_K_tile
                            + log_v[b, h, None, j_sl]
                        )
                        T_norm_tile = torch.exp(log_score - log_Z[b, h, i_sl, None])

                        dT = (
                            grad_O_avg[b, h, i_sl] @ V[b, h, j_sl].T
                            + grad_xc[b, h, i_sl] @ x_res[b, j_sl].T
                        )
                        grad_ls_direct = T_norm_tile * (dT - D[b, h, i_sl, None])
                        grad_C_direct = grad_ls_direct * (-1.0 / eps_h)

                        log_s_row = (
                            log_K_tile
                            + log_v[b, h, None, j_sl]
                            - row_lse[b, h, i_sl, None]
                        )
                        s_row_tile = torch.exp(log_s_row)
                        log_s_col = (
                            log_K_tile
                            + log_u[b, h, i_sl, None]
                            - col_lse[b, h, None, j_sl]
                        )
                        s_col_tile = torch.exp(log_s_col)

                        grad_C_ift = -(kappa_h / eps_h) * (
                            z_u[b, h, i_sl, None] * s_row_tile
                            + z_v[b, h, None, j_sl] * s_col_tile
                        )

                        grad_C_total = grad_C_direct + grad_C_ift

                        grad_Q_ln[b, h, i_sl] += (-inv_sqrt_dh) * (
                            grad_C_total @ K_ln[b, h, j_sl]
                        )
                        grad_K_ln[b, h, j_sl] += (-inv_sqrt_dh) * (
                            grad_C_total.T @ Q_ln[b, h, i_sl]
                        )

                        geo_grad_coeff = w_dist_h * r_0 / (r_0 + dist_tile) ** 2
                        weighted = grad_C_total * geo_grad_coeff / dist_tile
                        grad_x_cost[b, i_sl] += (
                            weighted.unsqueeze(-1) * diff_tile
                        ).sum(dim=1)
                        grad_x_cost[b, j_sl] -= (
                            weighted.unsqueeze(-1) * diff_tile
                        ).sum(dim=0)

                        f_dist_tile = dist_tile / (r_0 + dist_tile)
                        grad_w_dist[h] += (grad_C_total * f_dist_tile).sum()

                        bins_tile = pos_bins[b, i_sl, j_sl].long().reshape(-1)
                        grad_pos_weight[h].scatter_add_(0, bins_tile, grad_C_total.reshape(-1))

        # ====================================================================
        # Step 5: Combine x_res gradients
        # ====================================================================
        grad_x_res = grad_x_transport + grad_x_cost

        # Marginal gradients (from IFT)
        grad_log_mu = kappa[None, :, None] * z_u  # (B, H, N)
        grad_log_nu = kappa[None, :, None] * z_v

        # Return gradients for all inputs (always batched)
        return (
            grad_Q_ln,
            grad_K_ln,
            grad_V,
            grad_G,
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
            None,  # mask
        )


def flash_sinkhorn_attn(
    Q_ln: torch.Tensor,
    K_ln: torch.Tensor,
    V: torch.Tensor,
    G: torch.Tensor,
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
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash-Sinkhorn attention with O(N) memory forward+backward.

    Uses Triton kernels for both forward AND backward when available.
    Falls back to FlashSinkhornAttn (Python-tiled backward) on CPU.

    Accepts both unbatched and batched inputs:
    Unbatched:
        Q_ln: (H, N, d_h), x_res: (N, 3), pos_weight: (H, 68), pos_bins: (N, N), etc.
        Returns: o_flat (N, H*d_h), x_centroid (H, N, 3), log_u (H, N), log_v (H, N)
    Batched:
        Q_ln: (B, H, N, d_h), x_res: (B, N, 3), pos_weight: (H, 68), pos_bins: (B, N, N), etc.
        Returns: o_flat (B, N, H*d_h), x_centroid (B, H, N, 3), log_u (B, H, N), log_v (B, H, N)

    mask: (B, N) or (N,) float, 1=real 0=pad. None means all valid.
    """
    unbatched = Q_ln.dim() == 3
    if unbatched:
        Q_ln = Q_ln.unsqueeze(0)
        K_ln = K_ln.unsqueeze(0)
        V = V.unsqueeze(0)
        G = G.unsqueeze(0)
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

    B, H, N, d_h = Q_ln.shape

    if mask is None:
        mask = Q_ln.new_ones(B, N)

    try:
        # Triton path: FlashSinkhornFunction has Triton forward + Triton backward
        from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn as _triton_fn

        # flash_sinkhorn returns (O_avg, x_centroid, log_u, log_v)
        # with Triton backward — no Python-tiled IFT needed
        O_avg, x_centroid, log_u, log_v = _triton_fn(
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
            mask=mask,
        )
    except Exception:
        # CPU fallback: use FlashSinkhornAttn (Python-tiled backward)
        result = FlashSinkhornAttn.apply(
            Q_ln,
            K_ln,
            V,
            G,
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
            mask,
        )
        if unbatched:
            return tuple(t.squeeze(0) for t in result)
        return result

    # Apply gating: sigmoid(G) * O_avg → gated output
    sig_G = torch.sigmoid(G)
    o_gated = sig_G * O_avg  # (B, H, N, d_h)
    o_flat = o_gated.permute(0, 2, 1, 3).reshape(B, N, H * d_h)  # (B, N, H*d_h)

    # Zero padded positions
    mask_expand = mask.unsqueeze(-1)  # (B, N, 1)
    o_flat = o_flat * mask_expand
    mask_hn = mask.unsqueeze(1)  # (B, 1, N)
    log_u = log_u * mask_hn
    log_v = log_v * mask_hn

    if unbatched:
        return (
            o_flat.squeeze(0),
            x_centroid.squeeze(0),
            log_u.squeeze(0),
            log_v.squeeze(0),
        )
    return o_flat, x_centroid, log_u, log_v
