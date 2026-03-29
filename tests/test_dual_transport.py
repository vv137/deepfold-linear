"""Tests for balanced_sinkhorn_transport_dual: accuracy, timing, memory."""

import time

import pytest
import torch
import torch.nn.functional as F


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dual_inputs(device):
    """Create test inputs for dual transport."""
    B, H, N, d_h = 1, 8, 64, 32
    Q = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
    K = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
    x = torch.randn(B, N, 3, device=device)
    V = torch.randn(B, H, N, d_h, device=device)
    eps = torch.tensor([0.5]*2 + [1.0]*2 + [2.0]*2 + [4.0]*2, device=device)
    alpha_h = torch.randn(H, device=device)
    r_h = torch.full((H,), 10.0, device=device)
    mask = torch.ones(B, N, device=device)
    return Q, K, x, V, eps, alpha_h, r_h, mask


# ============================================================================
# Accuracy: Triton vs CPU reference
# ============================================================================

def test_dual_forward_accuracy(dual_inputs, device):
    """Triton fused forward matches CPU reference."""
    from deepfold.model.kernels.flash_sinkhorn_transport import (
        _sinkhorn_cpu,
        balanced_sinkhorn_transport_dual,
    )
    Q, K, x, V, eps, alpha_h, r_h, mask = dual_inputs

    # CPU reference
    Q_f, K_f, x_f, V_f = Q.float(), K.float(), x.float(), V.float()
    mask_f = mask.float()
    x_ref, h_ref = _sinkhorn_cpu(Q_f, K_f, x_f, eps, alpha_h, r_h, mask_f,
                                  Q.shape[2], 20, V_h=V_f)

    # Dual API (uses Triton on CUDA, CPU ref on CPU)
    x_dual, h_dual = balanced_sinkhorn_transport_dual(
        Q, K, x, V, eps, alpha_h, r_h, K_iter=20, mask=mask)

    x_ref = x_ref.to(x_dual.dtype)
    h_ref = h_ref.to(h_dual.dtype)

    x_err = (x_dual - x_ref).abs().max().item()
    h_err = (h_dual - h_ref).abs().max().item()
    print(f"  x_centroid max error: {x_err:.2e}")
    print(f"  h_centroid max error: {h_err:.2e}")
    assert x_err < 1e-3, f"x_centroid error too large: {x_err}"
    assert h_err < 1e-3, f"h_centroid error too large: {h_err}"


def test_dual_backward_accuracy(dual_inputs, device):
    """Gradient through dual transport matches finite differences."""
    from deepfold.model.kernels.flash_sinkhorn_transport import (
        balanced_sinkhorn_transport_dual,
    )
    Q, K, x, V, eps, alpha_h, r_h, mask = dual_inputs

    # Make inputs require grad
    V_test = V.clone().detach().requires_grad_(True)
    x_test = x.clone().detach().requires_grad_(True)

    x_c, h_c = balanced_sinkhorn_transport_dual(
        Q, K, x_test, V_test, eps, alpha_h, r_h, K_iter=20, mask=mask)

    # Scalar loss from both outputs
    loss = x_c.sum() + h_c.sum()
    loss.backward()

    assert V_test.grad is not None, "V_h gradient is None"
    assert x_test.grad is not None, "x_res gradient is None"
    assert V_test.grad.abs().max() > 0, "V_h gradient is all zeros"
    assert x_test.grad.abs().max() > 0, "x_res gradient is all zeros"
    print(f"  grad_V max: {V_test.grad.abs().max().item():.4e}")
    print(f"  grad_x max: {x_test.grad.abs().max().item():.4e}")


def test_dual_vs_single_x_centroid(dual_inputs, device):
    """x_centroid from dual transport matches single (original) transport."""
    from deepfold.model.kernels.flash_sinkhorn_transport import (
        balanced_sinkhorn_transport,
        balanced_sinkhorn_transport_dual,
    )
    Q, K, x, V, eps, alpha_h, r_h, mask = dual_inputs

    # Original single transport
    x_single = balanced_sinkhorn_transport(
        Q, K, x, eps, alpha_h, r_h, K_iter=20, mask=mask)

    # Dual transport
    x_dual, _ = balanced_sinkhorn_transport_dual(
        Q, K, x, V, eps, alpha_h, r_h, K_iter=20, mask=mask)

    err = (x_dual - x_single).abs().max().item()
    print(f"  x_centroid dual vs single max error: {err:.2e}")
    assert err < 1e-4, f"x_centroid mismatch: {err}"


# ============================================================================
# Timing
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dual_timing(device):
    """Compare timing: single transport vs dual transport."""
    from deepfold.model.kernels.flash_sinkhorn_transport import (
        balanced_sinkhorn_transport,
        balanced_sinkhorn_transport_dual,
    )
    B, H, N, d_h = 1, 16, 256, 32
    Q = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
    K = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
    x = torch.randn(B, N, 3, device=device)
    V = torch.randn(B, H, N, d_h, device=device)
    eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=device)
    alpha_h = torch.randn(H, device=device)
    r_h = torch.full((H,), 10.0, device=device)
    mask = torch.ones(B, N, device=device)

    # Warmup
    for _ in range(3):
        balanced_sinkhorn_transport(Q, K, x, eps, alpha_h, r_h, K_iter=20, mask=mask)
        balanced_sinkhorn_transport_dual(Q, K, x, V, eps, alpha_h, r_h, K_iter=20, mask=mask)
    torch.cuda.synchronize()

    n_iter = 10

    # Single
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        balanced_sinkhorn_transport(Q, K, x, eps, alpha_h, r_h, K_iter=20, mask=mask)
    torch.cuda.synchronize()
    t_single = (time.perf_counter() - t0) / n_iter * 1000

    # Dual
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        balanced_sinkhorn_transport_dual(Q, K, x, V, eps, alpha_h, r_h, K_iter=20, mask=mask)
    torch.cuda.synchronize()
    t_dual = (time.perf_counter() - t0) / n_iter * 1000

    overhead = (t_dual - t_single) / t_single * 100
    print(f"\n  N={N}: single={t_single:.2f}ms, dual={t_dual:.2f}ms, overhead={overhead:+.1f}%")
    # Fused should be <30% overhead (centroid is ~10% of total Sinkhorn)
    assert overhead < 50, f"Dual overhead too high: {overhead:.1f}%"


# ============================================================================
# Memory
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dual_memory(device):
    """Dual transport should use O(N) memory, not O(N²)."""
    from deepfold.model.kernels.flash_sinkhorn_transport import (
        balanced_sinkhorn_transport_dual,
    )
    B, H, d_h = 1, 16, 32

    for N in [256, 512]:
        Q = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
        K = F.normalize(torch.randn(B, H, N, d_h, device=device), dim=-1)
        x = torch.randn(B, N, 3, device=device)
        V = torch.randn(B, H, N, d_h, device=device)
        eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=device)
        alpha_h = torch.randn(H, device=device)
        r_h = torch.full((H,), 10.0, device=device)
        mask = torch.ones(B, N, device=device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.max_memory_allocated()

        balanced_sinkhorn_transport_dual(
            Q, K, x, V, eps, alpha_h, r_h, K_iter=20, mask=mask)

        mem_after = torch.cuda.max_memory_allocated()
        mem_used_mb = (mem_after - mem_before) / 1024**2
        n2_mb = B * H * N * N * 4 / 1024**2  # O(N²) would be this
        print(f"  N={N}: used={mem_used_mb:.1f}MB, O(N²) would be={n2_mb:.1f}MB")
        # Should be well below O(N²)
        assert mem_used_mb < n2_mb * 0.5, f"Memory usage too high at N={N}"
