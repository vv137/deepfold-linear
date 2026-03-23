"""Tests that batched flash_sinkhorn_attn matches per-sample calls."""

import pytest
import torch


def _make_flash_inputs(B, H, N, d_h, seed=42):
    """Create random inputs for flash_sinkhorn_attn."""
    g = torch.Generator().manual_seed(seed)
    Q_ln = torch.randn(B, H, N, d_h, generator=g)
    K_ln = torch.randn(B, H, N, d_h, generator=g)
    V = torch.randn(B, H, N, d_h, generator=g)
    G = torch.randn(B, H, N, d_h, generator=g)
    x_res = torch.randn(B, N, 3, generator=g)
    pos_bias = torch.randn(B, H, N, N, generator=g)
    eps = torch.tensor(
        [0.5] * (H // 4) + [1.0] * (H // 4) + [2.0] * (H // 4) + [4.0] * (H // 4),
        dtype=torch.float32,
    )
    w_dist = torch.rand(H, generator=g) * 0.5
    log_mu = torch.log_softmax(torch.randn(B, H, N, generator=g), dim=-1)
    log_nu = torch.log_softmax(torch.randn(B, H, N, generator=g), dim=-1)
    return Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist, log_mu, log_nu


class TestFlashSinkhornBatch:
    """Verify that batched flash_sinkhorn_attn matches per-sample calls."""

    def test_batched_matches_per_sample(self):
        B, H, N, d_h = 2, 4, 16, 8
        K_iter = 3

        Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist, log_mu, log_nu = (
            _make_flash_inputs(B, H, N, d_h)
        )

        from deepfold.model.kernels.flash_sinkhorn_attn import flash_sinkhorn_attn

        # Batched call
        o_batch, xc_batch, lu_batch, lv_batch = flash_sinkhorn_attn(
            Q_ln, K_ln, V, G, x_res, pos_bias,
            eps, w_dist, log_mu, log_nu,
            K_iter=K_iter, lam=1.0, r_0=10.0,
        )

        assert o_batch.shape == (B, N, H * d_h)
        assert xc_batch.shape == (B, H, N, 3)
        assert lu_batch.shape == (B, H, N)
        assert lv_batch.shape == (B, H, N)

        # Per-sample calls (unbatched interface)
        for b in range(B):
            o_b, xc_b, lu_b, lv_b = flash_sinkhorn_attn(
                Q_ln[b], K_ln[b], V[b], G[b],
                x_res[b], pos_bias[b],
                eps, w_dist, log_mu[b], log_nu[b],
                K_iter=K_iter, lam=1.0, r_0=10.0,
            )
            torch.testing.assert_close(
                o_batch[b], o_b, atol=1e-4, rtol=1e-4,
                msg=f"o_flat mismatch at batch {b}",
            )
            torch.testing.assert_close(
                xc_batch[b], xc_b, atol=1e-4, rtol=1e-4,
                msg=f"x_centroid mismatch at batch {b}",
            )
            torch.testing.assert_close(
                lu_batch[b], lu_b, atol=1e-4, rtol=1e-4,
                msg=f"log_u mismatch at batch {b}",
            )
            torch.testing.assert_close(
                lv_batch[b], lv_b, atol=1e-4, rtol=1e-4,
                msg=f"log_v mismatch at batch {b}",
            )

    def test_unbatched_backward_compat(self):
        """Ensure the unbatched path still works (B=1 squeeze)."""
        H, N, d_h = 4, 8, 8
        K_iter = 2

        g = torch.Generator().manual_seed(99)
        Q_ln = torch.randn(H, N, d_h, generator=g)
        K_ln = torch.randn(H, N, d_h, generator=g)
        V = torch.randn(H, N, d_h, generator=g)
        G = torch.randn(H, N, d_h, generator=g)
        x_res = torch.randn(N, 3, generator=g)
        pos_bias = torch.randn(H, N, N, generator=g)
        eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        w_dist = torch.rand(H, generator=g) * 0.5
        log_mu = torch.log_softmax(torch.randn(H, N, generator=g), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N, generator=g), dim=-1)

        from deepfold.model.kernels.flash_sinkhorn_attn import flash_sinkhorn_attn

        o, xc, lu, lv = flash_sinkhorn_attn(
            Q_ln, K_ln, V, G, x_res, pos_bias,
            eps, w_dist, log_mu, log_nu,
            K_iter=K_iter,
        )

        assert o.shape == (N, H * d_h)
        assert xc.shape == (H, N, 3)
        assert lu.shape == (H, N)
        assert lv.shape == (H, N)

    def test_b1_matches_unbatched(self):
        """B=1 batched call should match unbatched call exactly."""
        H, N, d_h = 4, 8, 8
        K_iter = 3

        g = torch.Generator().manual_seed(77)
        Q_ln = torch.randn(H, N, d_h, generator=g)
        K_ln = torch.randn(H, N, d_h, generator=g)
        V = torch.randn(H, N, d_h, generator=g)
        G = torch.randn(H, N, d_h, generator=g)
        x_res = torch.randn(N, 3, generator=g)
        pos_bias = torch.randn(H, N, N, generator=g)
        eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        w_dist = torch.rand(H) * 0.5
        log_mu = torch.log_softmax(torch.randn(H, N), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N), dim=-1)

        from deepfold.model.kernels.flash_sinkhorn_attn import flash_sinkhorn_attn

        # Unbatched
        o_ub, xc_ub, lu_ub, lv_ub = flash_sinkhorn_attn(
            Q_ln, K_ln, V, G, x_res, pos_bias,
            eps, w_dist, log_mu, log_nu,
            K_iter=K_iter,
        )

        # Batched B=1
        o_b, xc_b, lu_b, lv_b = flash_sinkhorn_attn(
            Q_ln.unsqueeze(0), K_ln.unsqueeze(0), V.unsqueeze(0), G.unsqueeze(0),
            x_res.unsqueeze(0), pos_bias.unsqueeze(0),
            eps, w_dist,
            log_mu.unsqueeze(0), log_nu.unsqueeze(0),
            K_iter=K_iter,
        )

        torch.testing.assert_close(o_ub, o_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(xc_ub, xc_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lu_ub, lu_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lv_ub, lv_b.squeeze(0), atol=1e-5, rtol=1e-5)
