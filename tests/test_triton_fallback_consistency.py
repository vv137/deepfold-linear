"""Test consistency between Triton kernel and Python fallback paths.

1. TokenUOTBlock train (dense) vs eval (flash) consistency
2. Direct Triton forward vs Python fallback forward at flash_sinkhorn_attn level
"""

import pytest
import torch

from deepfold.model.position_encoding import PositionBias, compute_bins
from deepfold.model.trunk_block import TokenUOTBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonFallbackConsistency:
    """Compare training (dense Sinkhorn) vs inference (flash Sinkhorn) outputs."""

    def _make_block(self, d_model=64, n_heads=4):
        block = TokenUOTBlock(d_model=d_model, n_heads=n_heads, block_idx=0)
        block.eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        return block

    def test_train_vs_eval_consistency(self):
        """h, x_res, log_u, log_v should be close between dense and flash paths."""
        torch.manual_seed(42)
        N, d_model, n_heads = 16, 64, 4

        block = self._make_block(d_model, n_heads).cuda()

        h = torch.randn(1, N, d_model, device="cuda")
        x_res = torch.randn(1, N, 3, device="cuda")
        mu = torch.softmax(torch.randn(1, n_heads, N, device="cuda"), dim=-1)
        nu = torch.softmax(torch.randn(1, n_heads, N, device="cuda"), dim=-1)
        log_u = torch.zeros(1, n_heads, N, device="cuda")
        log_v = torch.zeros(1, n_heads, N, device="cuda")

        chain_id = torch.zeros(1, N, dtype=torch.long, device="cuda")
        global_idx = torch.arange(N, device="cuda").unsqueeze(0)
        bond_matrix = torch.zeros(1, N, N, dtype=torch.bool, device="cuda")
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)

        pb = PositionBias(n_heads, 68).cuda()
        # Give it nonzero weights for a meaningful test
        pb.weight.data = torch.randn_like(pb.weight) * 0.1
        pos_weight = pb.weight

        # Training path (dense Sinkhorn)
        block.train()
        with torch.no_grad():
            h_train, x_train, lu_train, lv_train = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins,
            )

        # Inference path (flash Sinkhorn / Triton)
        block.eval()
        with torch.no_grad():
            h_eval, x_eval, lu_eval, lv_eval = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins,
            )

        # They won't be bit-exact (different computation order, FP32 accumulation),
        # but should be close since both solve the same Sinkhorn problem.
        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(
            h_train, h_eval, atol=atol, rtol=rtol,
            msg="h mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            x_train, x_eval, atol=atol, rtol=rtol,
            msg="x_res mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            lu_train, lu_eval, atol=atol, rtol=rtol,
            msg="log_u mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            lv_train, lv_eval, atol=atol, rtol=rtol,
            msg="log_v mismatch between train (dense) and eval (flash)"
        )

    def test_pos_bias_gather_equivalence(self):
        """Verify that weight[:, bins] == kernel gather for position bias."""
        torch.manual_seed(123)
        H, N = 4, 32

        weight = torch.randn(H, 68, device="cuda")
        bins = torch.randint(0, 68, (N, N), dtype=torch.int32, device="cuda")

        # Materialized
        pos_bias = weight[:, bins.long()]  # (H, N, N)

        # Gather (same as what kernel does)
        for h in range(H):
            for i in range(N):
                for j in range(N):
                    expected = weight[h, bins[i, j].item()]
                    actual = pos_bias[h, i, j]
                    assert expected == actual, f"Mismatch at h={h}, i={i}, j={j}"

    def test_batched_train_vs_eval(self):
        """Same as above but with B=2 and padding."""
        torch.manual_seed(7)
        B, N, d_model, n_heads = 2, 12, 64, 4

        block = self._make_block(d_model, n_heads).cuda()

        h = torch.randn(B, N, d_model, device="cuda")
        x_res = torch.randn(B, N, 3, device="cuda")
        mu = torch.softmax(torch.randn(B, n_heads, N, device="cuda"), dim=-1)
        nu = torch.softmax(torch.randn(B, n_heads, N, device="cuda"), dim=-1)
        log_u = torch.zeros(B, n_heads, N, device="cuda")
        log_v = torch.zeros(B, n_heads, N, device="cuda")
        pos_bins = torch.randint(0, 68, (B, N, N), dtype=torch.int32, device="cuda")

        pb = PositionBias(n_heads, 68).cuda()
        pb.weight.data = torch.randn_like(pb.weight) * 0.1
        pos_weight = pb.weight

        mask = torch.ones(B, N, device="cuda")
        mask[1, 8:] = 0  # pad last 4 positions in sample 1

        block.train()
        with torch.no_grad():
            h_train, x_train, _, _ = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins, mask=mask,
            )

        block.eval()
        with torch.no_grad():
            h_eval, x_eval, _, _ = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins, mask=mask,
            )

        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(h_train, h_eval, atol=atol, rtol=rtol)
        torch.testing.assert_close(x_train, x_eval, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonVsPythonFallback:
    """Directly compare Triton kernel forward vs Python fallback forward."""

    def _make_inputs(self, B, H, N, d_h, device="cuda"):
        torch.manual_seed(42)
        Q_ln = torch.randn(B, H, N, d_h, device=device)
        K_ln = torch.randn(B, H, N, d_h, device=device)
        V = torch.randn(B, H, N, d_h, device=device)
        G = torch.randn(B, H, N, d_h, device=device)
        x_res = torch.randn(B, N, 3, device=device)
        pos_weight = torch.randn(H, 68, device=device) * 0.1
        pos_bins = torch.randint(0, 68, (B, N, N), dtype=torch.int32, device=device)
        eps = torch.tensor([0.5, 1.0, 2.0, 4.0][:H], device=device, dtype=torch.float32)
        w_dist = torch.randn(H, device=device) * 0.1
        log_mu = torch.log(torch.softmax(torch.randn(B, H, N, device=device), dim=-1).clamp(min=1e-8))
        log_nu = torch.log(torch.softmax(torch.randn(B, H, N, device=device), dim=-1).clamp(min=1e-8))
        return Q_ln, K_ln, V, G, x_res, pos_weight, pos_bins, eps, w_dist, log_mu, log_nu

    def _run_python_fallback(self, Q_ln, K_ln, V, G, x_res, pos_weight, pos_bins,
                              eps, w_dist, log_mu, log_nu, K_iter, lam, r_0, mask):
        """Run the Python fallback path from FlashSinkhornAttn.forward."""
        B, H, N, d_h = Q_ln.shape
        kappa = lam / (lam + eps)

        Q_ln = Q_ln.float()
        K_ln = K_ln.float()
        V = V.float()
        G = G.float()
        x_res = x_res.float()
        pos_weight = pos_weight.float()

        if mask is None:
            mask = Q_ln.new_ones(B, N)
        col_mask_bias = (1.0 - mask).unsqueeze(1) * (-1e9)

        O_avg_list, xc_list, lu_list, lv_list = [], [], [], []
        for b in range(B):
            C = torch.zeros(H, N, N, device=Q_ln.device, dtype=torch.float32)
            for h in range(H):
                content = -(Q_ln[b, h] @ K_ln[b, h].T) / (d_h ** 0.5)
                dist = torch.cdist(x_res[b], x_res[b])
                geo = w_dist[h] * dist / (r_0 + dist)
                pb_h = pos_weight[h][pos_bins[b].long()]
                C[h] = content + pb_h + geo

            log_K = -C / eps[:, None, None]
            lu = torch.zeros(H, N, device=Q_ln.device)
            lv = torch.zeros(H, N, device=Q_ln.device)
            mask_vec = col_mask_bias[b].squeeze(0)  # (N,)

            for _ in range(K_iter):
                lu = kappa[:, None] * (
                    log_mu[b] - torch.logsumexp(log_K + lv[:, None, :] + mask_vec[None, :], dim=-1)
                )
                lv = kappa[:, None] * (
                    log_nu[b] - torch.logsumexp(
                        log_K + lu[:, :, None] + mask_vec[:, None], dim=-2
                    )
                )

            # Transport output
            log_T = lu[:, :, None] + log_K + lv[:, None, :] + mask_vec[None, :]
            row_max = log_T.max(dim=-1, keepdim=True).values
            T = torch.exp(log_T - row_max)
            T_sum = T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            T_norm = T / T_sum

            o_h = torch.bmm(T_norm, V[b])  # (H, N, d_h)
            g_h = torch.sigmoid(G[b])
            o_gated = o_h * g_h
            o_flat = o_gated.permute(1, 0, 2).reshape(N, H * d_h)

            xc_h = torch.bmm(T_norm, x_res[b].unsqueeze(0).expand(H, -1, -1))

            O_avg_list.append(o_flat)
            xc_list.append(xc_h)
            lu_list.append(lu)
            lv_list.append(lv)

        return (
            torch.stack(O_avg_list),
            torch.stack(xc_list),
            torch.stack(lu_list),
            torch.stack(lv_list),
        )

    def _run_triton(self, Q_ln, K_ln, V, G, x_res, pos_weight, pos_bins,
                     eps, w_dist, log_mu, log_nu, K_iter, lam, r_0, mask):
        """Run the Triton kernel path."""
        from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

        B, H, N, d_h = Q_ln.shape
        if mask is None:
            mask = Q_ln.new_ones(B, N)

        O_avg, x_centroid, log_u, log_v = flash_sinkhorn(
            Q_ln, K_ln, V, x_res, pos_weight, pos_bins,
            eps, w_dist, log_mu, log_nu,
            K_iter=K_iter, lam=lam, r_0=r_0, mask=mask,
        )

        # flash_sinkhorn returns O_avg (B, H, N, d_h) — need to gate and flatten
        G_sig = torch.sigmoid(G.float())
        o_gated = O_avg * G_sig
        o_flat = o_gated.permute(0, 2, 1, 3).reshape(B, N, H * d_h)

        return o_flat, x_centroid, log_u, log_v

    def test_forward_consistency_unbatched(self):
        """Single sample: Triton vs Python fallback should match."""
        B, H, N, d_h = 1, 4, 16, 16
        inputs = self._make_inputs(B, H, N, d_h)
        K_iter, lam, r_0 = 7, 1.0, 10.0

        with torch.no_grad():
            o_py, xc_py, lu_py, lv_py = self._run_python_fallback(
                *inputs, K_iter, lam, r_0, mask=None
            )
            o_tr, xc_tr, lu_tr, lv_tr = self._run_triton(
                *inputs, K_iter, lam, r_0, mask=None
            )

        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(lu_py, lu_tr, atol=atol, rtol=rtol, msg="log_u mismatch")
        torch.testing.assert_close(lv_py, lv_tr, atol=atol, rtol=rtol, msg="log_v mismatch")
        torch.testing.assert_close(o_py, o_tr, atol=atol, rtol=rtol, msg="output mismatch")
        torch.testing.assert_close(xc_py, xc_tr, atol=atol, rtol=rtol, msg="centroid mismatch")

    def test_forward_consistency_batched_no_pad(self):
        """B=2 without padding: Triton vs Python fallback should match."""
        B, H, N, d_h = 2, 4, 16, 16
        inputs = self._make_inputs(B, H, N, d_h)
        K_iter, lam, r_0 = 7, 1.0, 10.0

        with torch.no_grad():
            o_py, xc_py, lu_py, lv_py = self._run_python_fallback(
                *inputs, K_iter, lam, r_0, mask=None
            )
            o_tr, xc_tr, lu_tr, lv_tr = self._run_triton(
                *inputs, K_iter, lam, r_0, mask=None
            )

        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(lu_py, lu_tr, atol=atol, rtol=rtol, msg="log_u mismatch")
        torch.testing.assert_close(lv_py, lv_tr, atol=atol, rtol=rtol, msg="log_v mismatch")
        torch.testing.assert_close(o_py, o_tr, atol=atol, rtol=rtol, msg="output mismatch")
        torch.testing.assert_close(xc_py, xc_tr, atol=atol, rtol=rtol, msg="centroid mismatch")

    def test_forward_consistency_batched_masked(self):
        """B=2 with padding: Triton vs Python fallback."""
        B, H, N, d_h = 2, 4, 16, 16
        inputs = self._make_inputs(B, H, N, d_h)
        K_iter, lam, r_0 = 7, 1.0, 10.0

        mask = torch.ones(B, N, device="cuda")
        mask[1, 12:] = 0  # pad last 4 in sample 1

        with torch.no_grad():
            o_py, xc_py, lu_py, lv_py = self._run_python_fallback(
                *inputs, K_iter, lam, r_0, mask=mask
            )
            o_tr, xc_tr, lu_tr, lv_tr = self._run_triton(
                *inputs, K_iter, lam, r_0, mask=mask
            )

        atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(lu_py, lu_tr, atol=atol, rtol=rtol, msg="log_u mismatch")
        torch.testing.assert_close(lv_py, lv_tr, atol=atol, rtol=rtol, msg="log_v mismatch")
        torch.testing.assert_close(o_py, o_tr, atol=atol, rtol=rtol, msg="output mismatch")
        torch.testing.assert_close(xc_py, xc_tr, atol=atol, rtol=rtol, msg="centroid mismatch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonBackward:
    """Verify Triton unrolled backward gradients against PyTorch autograd (dense)."""

    def _dense_forward(self, Q_ln, K_ln, V, x_res, pos_weight, pos_bins,
                       eps, w_dist, log_mu, log_nu, K_iter, lam, r_0, mask):
        """Dense Python Sinkhorn with autograd for reference gradients."""
        B, H, N, d_h = Q_ln.shape
        kappa = lam / (lam + eps)

        if mask is None:
            mask_bias = torch.zeros(B, N, device=Q_ln.device, dtype=torch.float32)
        else:
            mask_bias = (1.0 - mask.float()) * (-1e9)

        C = torch.zeros(B, H, N, N, device=Q_ln.device, dtype=torch.float32)
        for b in range(B):
            for h in range(H):
                content = -(Q_ln[b, h] @ K_ln[b, h].T) / (d_h ** 0.5)
                diff = x_res[b, :, None, :] - x_res[b, None, :, :]
                dist = ((diff ** 2).sum(-1) + 1e-16).sqrt()
                geo = w_dist[h] * dist / (r_0 + dist)
                pb_h = pos_weight[h][pos_bins[b].long()]
                C[b, h] = content + pb_h + geo

        log_K = -C / eps[None, :, None, None]
        log_u = torch.zeros(B, H, N, device=Q_ln.device, dtype=torch.float32)
        log_v = torch.zeros(B, H, N, device=Q_ln.device, dtype=torch.float32)

        for _ in range(K_iter):
            log_u = kappa[None, :, None] * (
                log_mu - torch.logsumexp(log_K + log_v[:, :, None, :] + mask_bias[:, None, None, :], dim=-1)
            )
            log_v = kappa[None, :, None] * (
                log_nu - torch.logsumexp(log_K + log_u[:, :, :, None] + mask_bias[:, None, :, None], dim=-2)
            )

        log_T = log_u[:, :, :, None] + log_K + log_v[:, :, None, :] + mask_bias[:, None, None, :]
        log_Z = torch.logsumexp(log_T, dim=-1, keepdim=True)
        T_norm = torch.exp(log_T - log_Z)

        O_avg = torch.einsum("bhij,bhjd->bhid", T_norm, V)
        x_centroid = torch.einsum("bhij,bjc->bhic", T_norm, x_res)
        return O_avg, x_centroid

    def _make_inputs(self, B, H, N, d_h, seed=42):
        torch.manual_seed(seed)
        Q_ln = torch.randn(B, H, N, d_h, device="cuda", dtype=torch.float32)
        K_ln = torch.randn(B, H, N, d_h, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, N, d_h, device="cuda", dtype=torch.float32)
        x_res = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        pos_weight = torch.randn(H, 68, device="cuda", dtype=torch.float32) * 0.1
        pos_bins = torch.randint(0, 68, (B, N, N), dtype=torch.int32, device="cuda")
        eps = torch.tensor([0.5, 1.0][:H], device="cuda", dtype=torch.float32)
        w_dist = torch.randn(H, device="cuda", dtype=torch.float32).sigmoid()
        log_mu = torch.log(torch.softmax(torch.randn(B, H, N, device="cuda"), dim=-1).clamp(min=1e-8))
        log_nu = torch.log(torch.softmax(torch.randn(B, H, N, device="cuda"), dim=-1).clamp(min=1e-8))
        return Q_ln, K_ln, V, x_res, pos_weight, pos_bins, eps, w_dist, log_mu, log_nu

    def _compare_grads(self, B, H, N, d_h, K_iter, lam, r_0, mask, seed=42):
        from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

        inputs = self._make_inputs(B, H, N, d_h, seed)
        Q_ln, K_ln, V, x_res, pos_weight, pos_bins, eps, w_dist, log_mu, log_nu = inputs

        grad_O = torch.randn(B, H, N, d_h, device="cuda", dtype=torch.float32)
        grad_xc = torch.randn(B, H, N, 3, device="cuda", dtype=torch.float32)

        # Dense autograd
        Q_d = Q_ln.clone().requires_grad_(True)
        K_d = K_ln.clone().requires_grad_(True)
        V_d = V.clone().requires_grad_(True)
        x_d = x_res.clone().requires_grad_(True)
        pw_d = pos_weight.clone().requires_grad_(True)
        wd_d = w_dist.clone().requires_grad_(True)

        O_dense, xc_dense = self._dense_forward(
            Q_d, K_d, V_d, x_d, pw_d, pos_bins, eps, wd_d,
            log_mu, log_nu, K_iter, lam, r_0, mask,
        )
        loss = (O_dense * grad_O).sum() + (xc_dense * grad_xc).sum()
        loss.backward()

        # Triton flash backward
        Q_t = Q_ln.clone().requires_grad_(True)
        K_t = K_ln.clone().requires_grad_(True)
        V_t = V.clone().requires_grad_(True)
        x_t = x_res.clone().requires_grad_(True)
        pw_t = pos_weight.clone().requires_grad_(True)
        wd_t = w_dist.clone().requires_grad_(True)

        O_flash, xc_flash, _, _ = flash_sinkhorn(
            Q_t, K_t, V_t, x_t, pw_t, pos_bins, eps, wd_t,
            log_mu, log_nu, K_iter=K_iter, lam=lam, r_0=r_0, mask=mask,
        )
        loss_f = (O_flash * grad_O).sum() + (xc_flash * grad_xc).sum()
        loss_f.backward()

        return [
            ("grad_Q_ln", Q_d.grad, Q_t.grad),
            ("grad_K_ln", K_d.grad, K_t.grad),
            ("grad_V", V_d.grad, V_t.grad),
            ("grad_x_res", x_d.grad, x_t.grad),
            ("grad_pos_weight", pw_d.grad, pw_t.grad),
            ("grad_w_dist", wd_d.grad, wd_t.grad),
        ]

    def test_backward_unbatched(self):
        grads = self._compare_grads(B=1, H=2, N=16, d_h=32, K_iter=3, lam=1.0, r_0=10.0, mask=None)
        for name, gd, gt in grads:
            torch.testing.assert_close(gd, gt, atol=5e-3, rtol=5e-3, msg=f"{name} mismatch")

    def test_backward_batched_masked(self):
        mask = torch.ones(2, 16, device="cuda")
        mask[1, 12:] = 0
        grads = self._compare_grads(B=2, H=2, N=16, d_h=32, K_iter=3, lam=1.0, r_0=10.0, mask=mask, seed=99)
        for name, gd, gt in grads:
            torch.testing.assert_close(gd, gt, atol=5e-3, rtol=5e-3, msg=f"{name} mismatch (masked)")
