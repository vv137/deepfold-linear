"""Tests for CG solvers and Sinkhorn convergence check."""

import torch

from deepfold.utils.cg import conjugate_gradient, steihaug_cg


class TestConjugateGradient:
    def test_identity_system(self):
        """A = I, b = [1,2,3] → x = [1,2,3]."""
        b = torch.tensor([1.0, 2.0, 3.0])
        x, info = conjugate_gradient(lambda v: v, b, max_iter=10)
        assert info.converged
        assert info.iters <= 1
        assert torch.allclose(x, b, atol=1e-5)

    def test_spd_system(self):
        """Random SPD system."""
        torch.manual_seed(42)
        n = 20
        A = torch.randn(n, n)
        A = A.T @ A + 0.1 * torch.eye(n)  # SPD
        b = torch.randn(n)
        x_true = torch.linalg.solve(A, b)

        x, info = conjugate_gradient(lambda v: A @ v, b, max_iter=50)
        assert info.converged
        assert torch.allclose(x, x_true, atol=1e-4)

    def test_warm_start(self):
        """Warm-start should converge faster."""
        torch.manual_seed(42)
        n = 20
        A = torch.randn(n, n)
        A = A.T @ A + 0.1 * torch.eye(n)
        b = torch.randn(n)
        x_true = torch.linalg.solve(A, b)

        # Cold start
        _, info_cold = conjugate_gradient(lambda v: A @ v, b, max_iter=50)
        # Warm start near solution
        x0 = x_true + 0.01 * torch.randn(n)
        _, info_warm = conjugate_gradient(lambda v: A @ v, b, x0=x0, max_iter=50)
        assert info_warm.iters <= info_cold.iters

    def test_preconditioner(self):
        """Diagonal preconditioner."""
        torch.manual_seed(42)
        n = 20
        A = torch.diag(torch.linspace(0.1, 10.0, n))
        b = torch.randn(n)
        def precond(v):
            return v / torch.diag(A)

        _, info_no = conjugate_gradient(lambda v: A @ v, b, max_iter=50)
        _, info_yes = conjugate_gradient(
            lambda v: A @ v, b, max_iter=50, preconditioner=precond
        )
        assert info_yes.iters <= info_no.iters


class TestSteihaugCG:
    def test_small_gradient(self):
        """Zero gradient → zero step."""
        g = torch.zeros(5)
        p, info = steihaug_cg(lambda v: v, g, delta=1.0)
        assert info.converged
        assert info.iters == 0

    def test_spd_within_trust_region(self):
        """SPD Hessian, solution within trust region."""
        torch.manual_seed(42)
        n = 10
        H = torch.randn(n, n)
        H = H.T @ H + 0.5 * torch.eye(n)
        g = torch.randn(n)

        p, info = steihaug_cg(lambda v: H @ v, g, delta=100.0, max_iter=50)
        assert info.converged
        assert not info.hit_boundary
        # Check p ≈ -H^{-1} g
        p_true = -torch.linalg.solve(H, g)
        assert torch.allclose(p, p_true, atol=1e-3)

    def test_negative_curvature(self):
        """Indefinite Hessian → should hit boundary via negative curvature."""
        H = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        g = torch.tensor([0.0, 1.0, 0.0])  # gradient along negative curvature direction

        p, info = steihaug_cg(lambda v: H @ v, g, delta=1.0, max_iter=50)
        assert info.hit_boundary or info.negative_curvature_detected

    def test_trust_region_boundary(self):
        """Large gradient relative to trust region → hit boundary."""
        g = torch.tensor([10.0, 0.0])
        H = torch.eye(2)
        p, info = steihaug_cg(lambda v: H @ v, g, delta=1.0, max_iter=50)
        assert info.hit_boundary
        assert abs(torch.linalg.norm(p).item() - 1.0) < 0.02


class TestSinkhornConvergence:
    def test_early_stopping(self):
        """Sinkhorn should converge before K iterations with loose threshold."""
        from deepfold.model.sinkhorn import sinkhorn_solve

        H, N = 2, 8
        C = torch.randn(H, N, N).abs()  # positive cost
        log_mu = torch.log_softmax(torch.randn(H, N), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N), dim=-1)
        eps = torch.tensor([1.0, 2.0])

        # With loose threshold, should stop early
        log_u1, log_v1 = sinkhorn_solve(
            C, log_mu, log_nu, eps, K=100, threshold=1e-3, check_every=2
        )
        # Without threshold, runs all K iterations
        log_u2, log_v2 = sinkhorn_solve(C, log_mu, log_nu, eps, K=100)

        # Both should give similar results (converged either way with K=100)
        assert torch.allclose(log_u1, log_u2, atol=0.01)

    def test_threshold_none_runs_all(self):
        """threshold=None should run all K iterations."""
        from deepfold.model.sinkhorn import sinkhorn_solve

        H, N = 2, 4
        C = torch.randn(H, N, N).abs()
        log_mu = torch.log_softmax(torch.randn(H, N), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N), dim=-1)
        eps = torch.tensor([1.0, 2.0])

        r1 = sinkhorn_solve(C, log_mu, log_nu, eps, K=3)
        r2 = sinkhorn_solve(C, log_mu, log_nu, eps, K=3, threshold=None)
        assert torch.allclose(r1[0], r2[0])
        assert torch.allclose(r1[1], r2[1])
