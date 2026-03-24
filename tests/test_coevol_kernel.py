"""Tests for batched co-evolution Triton kernel."""

import pytest
import torch


def _python_coevol_reference(U, V, h_coevol, w_weight, b_weight):
    """Pure-Python reference for co-evolution (single sample, no batch dim).

    Args:
        U: (S, N, R)
        V: (S, N, R)
        h_coevol: (N, D)
        w_weight: (R,)  — linear weight
        b_weight: (1,)  — linear bias

    Returns:
        h_agg: (N, D)
        c_bar: (N, R)
    """
    S, N, R = U.shape
    h_coevol.shape[1]

    # c[i,j,r] = (1/S) * sum_s U[s,i,r] * V[s,j,r]
    c = torch.einsum("sir,sjr->ijr", U.float(), V.float()) / S  # (N, N, R)

    # w_score[i,j] = sum_r c[i,j,r] * w_weight[r] + b_weight
    w_score = (c * w_weight[None, None, :]).sum(-1) + b_weight  # (N, N)
    w = torch.sigmoid(w_score)  # (N, N)

    # h_agg[i] = sum_j w[i,j] * h_coevol[j]
    h_agg = w @ h_coevol.float()  # (N, D)

    # c_bar[i,r] = sum_j c[i,j,r]
    c_bar = c.sum(dim=1)  # (N, R)

    return h_agg, c_bar


# Skip all tests in this module if no CUDA or no triton
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

try:
    from deepfold.model.kernels.coevol_kernel import triton_coevol

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

skip_no_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not available")


@skip_no_triton
def test_triton_coevol_b1_matches_reference():
    """triton_coevol with B=1 matches pure-Python reference."""
    torch.manual_seed(42)
    S, N, R, D = 32, 64, 16, 128
    U = torch.randn(S, N, R, device="cuda")
    V = torch.randn(S, N, R, device="cuda")
    h = torch.randn(N, D, device="cuda")
    w = torch.randn(R, device="cuda")
    b = torch.randn(1, device="cuda")

    # Reference (unbatched)
    h_ref, c_ref = _python_coevol_reference(U, V, h, w, b)

    # Triton (unbatched — should auto-promote to B=1)
    h_tri, c_tri = triton_coevol(U, V, h, w, b)

    assert h_tri.shape == (N, D)
    assert c_tri.shape == (N, R)
    torch.testing.assert_close(h_tri.cpu(), h_ref.cpu(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(c_tri.cpu(), c_ref.cpu(), atol=1e-2, rtol=1e-2)


@skip_no_triton
def test_triton_coevol_b2_matches_per_sample():
    """triton_coevol with B=2 matches per-sample Python computation."""
    torch.manual_seed(123)
    B, S, N, R, D = 2, 32, 48, 16, 64
    U = torch.randn(B, S, N, R, device="cuda")
    V = torch.randn(B, S, N, R, device="cuda")
    h = torch.randn(B, N, D, device="cuda")
    w = torch.randn(R, device="cuda")
    b = torch.randn(1, device="cuda")

    # Triton batched
    h_tri, c_tri = triton_coevol(U, V, h, w, b)
    assert h_tri.shape == (B, N, D)
    assert c_tri.shape == (B, N, R)

    # Per-sample reference
    for i in range(B):
        h_ref, c_ref = _python_coevol_reference(U[i], V[i], h[i], w, b)
        torch.testing.assert_close(h_tri[i].cpu(), h_ref.cpu(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(c_tri[i].cpu(), c_ref.cpu(), atol=1e-2, rtol=1e-2)


@skip_no_triton
def test_triton_coevol_batched_unbatched_consistent():
    """B=1 batched path gives same result as unbatched path."""
    torch.manual_seed(7)
    S, N, R, D = 32, 32, 16, 64
    U = torch.randn(S, N, R, device="cuda")
    V = torch.randn(S, N, R, device="cuda")
    h = torch.randn(N, D, device="cuda")
    w = torch.randn(R, device="cuda")
    b = torch.randn(1, device="cuda")

    # Unbatched
    h1, c1 = triton_coevol(U, V, h, w, b)

    # Batched with B=1
    h2, c2 = triton_coevol(U.unsqueeze(0), V.unsqueeze(0), h.unsqueeze(0), w, b)

    assert h1.shape == (N, D)
    assert h2.shape == (1, N, D)
    torch.testing.assert_close(h1.cpu(), h2.squeeze(0).cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c1.cpu(), c2.squeeze(0).cpu(), atol=1e-5, rtol=1e-5)
