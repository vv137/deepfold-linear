"""Tests for batched UOT-Sinkhorn solver with masked logsumexp."""

import torch

from deepfold.model.sinkhorn import sinkhorn_solve, compute_transport_output


def _make_inputs(H, N, d_h, seed=42):
    """Create random Sinkhorn inputs for a single sample."""
    g = torch.Generator().manual_seed(seed)
    C = torch.rand(H, N, N, generator=g)
    C = C + C.transpose(-1, -2)  # symmetric cost
    log_mu = torch.log_softmax(torch.randn(H, N, generator=g), dim=-1)
    log_nu = torch.log_softmax(torch.randn(H, N, generator=g), dim=-1)
    eps = torch.full((H,), 0.1)
    V = torch.randn(H, N, d_h, generator=g)
    G = torch.randn(H, N, d_h, generator=g)
    x_res = torch.randn(N, 3, generator=g)
    return C, log_mu, log_nu, eps, V, G, x_res


class TestSinkhornUnbatchedCompat:
    """Verify that batched path with B=1 matches original unbatched output."""

    def test_sinkhorn_unbatched_compat(self):
        H, N, d_h = 4, 8, 16
        C, log_mu, log_nu, eps, V, G, x_res = _make_inputs(H, N, d_h)

        # Unbatched
        log_u_ub, log_v_ub = sinkhorn_solve(C, log_mu, log_nu, eps, K=10)

        # Batched B=1, no mask
        log_u_b, log_v_b = sinkhorn_solve(
            C.unsqueeze(0), log_mu.unsqueeze(0), log_nu.unsqueeze(0), eps, K=10
        )

        assert log_u_b.shape == (1, H, N)
        assert log_v_b.shape == (1, H, N)
        torch.testing.assert_close(log_u_ub, log_u_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(log_v_ub, log_v_b.squeeze(0), atol=1e-5, rtol=1e-5)


class TestSinkhornBatchedMasked:
    """B=2 where sample 1 has N=8, sample 2 has N=5 (padded to 8)."""

    def test_sinkhorn_batched_masked(self):
        H, N_max, N2, d_h = 4, 8, 5, 16
        K_iters = 10

        # Sample 1: full N=8
        C1, log_mu1, log_nu1, eps, _, _, _ = _make_inputs(H, N_max, d_h, seed=1)

        # Sample 2: N=5, will pad to 8
        C2_raw, log_mu2_raw, log_nu2_raw, _, _, _, _ = _make_inputs(H, N2, d_h, seed=2)

        # Pad sample 2 to N_max
        C2 = torch.zeros(H, N_max, N_max)
        C2[:, :N2, :N2] = C2_raw
        log_mu2 = torch.full((H, N_max), -1e9)
        log_mu2[:, :N2] = log_mu2_raw
        log_nu2 = torch.full((H, N_max), -1e9)
        log_nu2[:, :N2] = log_nu2_raw

        # Mask
        mask = torch.zeros(2, N_max)
        mask[0, :] = 1.0
        mask[1, :N2] = 1.0

        # Batched solve
        C_batch = torch.stack([C1, C2])
        log_mu_batch = torch.stack([log_mu1, log_mu2])
        log_nu_batch = torch.stack([log_nu1, log_nu2])

        log_u_b, log_v_b = sinkhorn_solve(
            C_batch, log_mu_batch, log_nu_batch, eps, K=K_iters, mask=mask
        )

        # Unbatched sample 1 (full, no mask needed)
        log_u1, log_v1 = sinkhorn_solve(C1, log_mu1, log_nu1, eps, K=K_iters)

        # Unbatched sample 2 (only N2 tokens)
        log_u2, log_v2 = sinkhorn_solve(
            C2_raw, log_mu2_raw, log_nu2_raw, eps, K=K_iters
        )

        # Sample 1 should match exactly
        torch.testing.assert_close(log_u_b[0], log_u1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(log_v_b[0], log_v1, atol=1e-5, rtol=1e-5)

        # Sample 2: real tokens should match
        torch.testing.assert_close(log_u_b[1, :, :N2], log_u2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(log_v_b[1, :, :N2], log_v2, atol=1e-5, rtol=1e-5)

    def test_mask_excludes_padded(self):
        """Verify padded positions have zero log_u/log_v output."""
        H, N_max, N_real = 4, 10, 6

        C, log_mu, log_nu, eps, _, _, _ = _make_inputs(H, N_max, 16, seed=3)
        mask = torch.zeros(1, N_max)
        mask[0, :N_real] = 1.0

        log_u, log_v = sinkhorn_solve(
            C.unsqueeze(0),
            log_mu.unsqueeze(0),
            log_nu.unsqueeze(0),
            eps,
            K=10,
            mask=mask,
        )

        # Padded positions must be zero
        assert (log_u[0, :, N_real:] == 0).all()
        assert (log_v[0, :, N_real:] == 0).all()


class TestTransportOutputBatched:
    """Test compute_transport_output with batched + masked inputs."""

    def test_transport_output_unbatched_compat(self):
        H, N, d_h = 4, 8, 16
        C, log_mu, log_nu, eps, V, G, x_res = _make_inputs(H, N, d_h)
        log_u, log_v = sinkhorn_solve(C, log_mu, log_nu, eps, K=10)

        # Unbatched
        o_ub, T_ub, xc_ub = compute_transport_output(V, G, log_u, log_v, C, eps, x_res)

        # Batched B=1
        o_b, T_b, xc_b = compute_transport_output(
            V.unsqueeze(0),
            G.unsqueeze(0),
            log_u.unsqueeze(0),
            log_v.unsqueeze(0),
            C.unsqueeze(0),
            eps,
            x_res.unsqueeze(0),
        )

        assert o_b.shape == (1, N, H * d_h)
        torch.testing.assert_close(o_ub, o_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(T_ub, T_b.squeeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(xc_ub, xc_b.squeeze(0), atol=1e-5, rtol=1e-5)

    def test_transport_output_batched_masked(self):
        H, N_max, N2, d_h = 4, 8, 5, 16
        K_iters = 10

        # Sample 1
        C1, log_mu1, log_nu1, eps, V1, G1, x1 = _make_inputs(H, N_max, d_h, seed=10)
        log_u1, log_v1 = sinkhorn_solve(C1, log_mu1, log_nu1, eps, K=K_iters)
        o1, T1, xc1 = compute_transport_output(V1, G1, log_u1, log_v1, C1, eps, x1)

        # Sample 2 (N=5)
        C2r, lm2r, ln2r, _, V2r, G2r, x2r = _make_inputs(H, N2, d_h, seed=20)
        lu2r, lv2r = sinkhorn_solve(C2r, lm2r, ln2r, eps, K=K_iters)
        o2r, T2r, xc2r = compute_transport_output(V2r, G2r, lu2r, lv2r, C2r, eps, x2r)

        # Pad sample 2 to N_max
        C2 = torch.zeros(H, N_max, N_max)
        C2[:, :N2, :N2] = C2r
        V2 = torch.zeros(H, N_max, d_h)
        V2[:, :N2, :] = V2r
        G2 = torch.zeros(H, N_max, d_h)
        G2[:, :N2, :] = G2r
        x2 = torch.zeros(N_max, 3)
        x2[:N2, :] = x2r
        log_mu2 = torch.full((H, N_max), -1e9)
        log_mu2[:, :N2] = lm2r
        log_nu2 = torch.full((H, N_max), -1e9)
        log_nu2[:, :N2] = ln2r

        mask = torch.zeros(2, N_max)
        mask[0, :] = 1.0
        mask[1, :N2] = 1.0

        # Batched sinkhorn
        C_b = torch.stack([C1, C2])
        lm_b = torch.stack([log_mu1, log_mu2])
        ln_b = torch.stack([log_nu1, log_nu2])
        lu_b, lv_b = sinkhorn_solve(C_b, lm_b, ln_b, eps, K=K_iters, mask=mask)

        # Batched transport
        V_b = torch.stack([V1, V2])
        G_b = torch.stack([G1, G2])
        x_b = torch.stack([x1, x2])
        o_b, T_b, xc_b = compute_transport_output(
            V_b, G_b, lu_b, lv_b, C_b, eps, x_b, mask=mask
        )

        # Sample 1 matches
        torch.testing.assert_close(o_b[0], o1, atol=1e-5, rtol=1e-5)

        # Sample 2: real tokens match
        torch.testing.assert_close(o_b[1, :N2, :], o2r, atol=1e-4, rtol=1e-4)

        # Padded positions in sample 2 output are zero
        assert (o_b[1, N2:, :] == 0).all()
