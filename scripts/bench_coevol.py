#!/usr/bin/env python3
"""Benchmark: Triton coevol kernel vs PyTorch reference (forward + backward).

Tests correctness, timing, and peak memory for various problem sizes,
including edge cases (S < 16, small N).

Usage:
    uv run python scripts/bench_coevol.py [--device cuda:0]
"""

import argparse
import torch

from deepfold.model.kernels.coevol_kernel import triton_coevol


# ============================================================================
# PyTorch reference (matches MSA block tiled loop exactly)
# ============================================================================


def pytorch_coevol(U, V, h_coevol, w_weight, b_weight, mask=None):
    """Pure PyTorch reference — no tiling, just full matmul."""
    if U.dim() == 3:
        U = U.unsqueeze(0)
        V = V.unsqueeze(0)
        h_coevol = h_coevol.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, S, N, R = U.shape
    D = h_coevol.shape[2]

    # c_tile[b, i, j, r] = (1/S) * sum_s U[b,s,i,r] * V[b,s,j,r]
    c_tile = torch.einsum("bsir,bsjr->bijr", U.float(), V.float()) / S

    # w_score[b, i, j] = sum_r c_tile[b,i,j,r] * w_weight[r] + b_weight
    w_score = (c_tile * w_weight[None, None, None, :]).sum(-1) + b_weight

    # w_tile = sigmoid(w_score) * mask_j
    w_tile = torch.sigmoid(w_score)  # (B, N, N)
    if mask is not None:
        w_tile = w_tile * mask[:, None, :]  # mask j dimension

    # h_agg[b, i] = sum_j w_tile[b,i,j] * h_coevol[b,j]
    h_agg = torch.bmm(w_tile, h_coevol.float())  # (B, N, D)

    # c_bar[b, i, r] = sum_j c_tile[b,i,j,r]
    c_bar = c_tile.sum(dim=2)  # (B, N, R)
    if mask is not None:
        c_bar = c_bar * mask[:, :, None]  # mask i dimension

    return h_agg, c_bar


# ============================================================================
# Test correctness
# ============================================================================


def test_correctness(B, S, N, R, D, device, mask_ratio=0.0):
    """Compare forward + backward between Triton and PyTorch."""
    torch.manual_seed(42)

    U = torch.randn(B, S, N, R, device=device, requires_grad=True)
    V = torch.randn(B, S, N, R, device=device, requires_grad=True)
    h = torch.randn(B, N, D, device=device, requires_grad=True)
    w = torch.randn(R, device=device, requires_grad=True)
    b = torch.randn(1, device=device, requires_grad=True)

    mask = None
    if mask_ratio > 0:
        mask = (torch.rand(B, N, device=device) > mask_ratio).float()

    # PyTorch reference
    U_ref, V_ref, h_ref = U.detach().clone().requires_grad_(), V.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
    w_ref, b_ref = w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    h_agg_ref, c_bar_ref = pytorch_coevol(U_ref, V_ref, h_ref, w_ref, b_ref, mask)
    loss_ref = h_agg_ref.sum() + c_bar_ref.sum()
    loss_ref.backward()

    # Triton
    U_tri, V_tri, h_tri = U.detach().clone().requires_grad_(), V.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
    w_tri, b_tri = w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    h_agg_tri, c_bar_tri = triton_coevol(U_tri, V_tri, h_tri, w_tri, b_tri, mask)
    loss_tri = h_agg_tri.sum() + c_bar_tri.sum()
    loss_tri.backward()

    # Check forward — use relative error (accumulation order differs between
    # Triton tiled and PyTorch einsum, but both are IEEE FP32)
    def rel_err(ref, tri):
        diff = (ref - tri.float()).abs()
        scale = ref.abs().amax().clamp(min=1e-6)
        return (diff / scale).amax().item()

    fwd_h_err = rel_err(h_agg_ref, h_agg_tri)
    fwd_c_err = rel_err(c_bar_ref, c_bar_tri)
    rel_tol = 1e-4  # 0.01% relative — tight for IEEE FP32
    fwd_ok = fwd_h_err < rel_tol and fwd_c_err < rel_tol

    # Check backward
    bwd_ok = True
    grad_names = ["dU", "dV", "dh_coevol", "dw_weight", "db_weight"]
    ref_grads = [U_ref.grad, V_ref.grad, h_ref.grad, w_ref.grad, b_ref.grad]
    tri_grads = [U_tri.grad, V_tri.grad, h_tri.grad, w_tri.grad, b_tri.grad]
    grad_errors = {}
    for name, g_ref, g_tri in zip(grad_names, ref_grads, tri_grads):
        if g_ref is None or g_tri is None:
            grad_errors[name] = "None"
            bwd_ok = False
            continue
        r = rel_err(g_ref, g_tri)
        grad_errors[name] = f"{r:.2e}"
        if r >= rel_tol:
            bwd_ok = False

    tag = f"B={B} S={S} N={N} R={R} D={D} mask={mask_ratio:.1f}"
    status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
    fwd_detail = f"h={fwd_h_err:.2e} c={fwd_c_err:.2e}"
    print(f"  [{status}] {tag}  fwd=({fwd_detail})  bwd={grad_errors}")
    return fwd_ok and bwd_ok


# ============================================================================
# Benchmark timing + memory
# ============================================================================


def bench_timing(B, S, N, R, D, device, warmup=10, iters=50):
    """Time forward+backward for both paths."""
    torch.manual_seed(0)

    def make_inputs(requires_grad=True):
        U = torch.randn(B, S, N, R, device=device, requires_grad=requires_grad)
        V = torch.randn(B, S, N, R, device=device, requires_grad=requires_grad)
        h = torch.randn(B, N, D, device=device, requires_grad=requires_grad)
        w = torch.randn(R, device=device, requires_grad=requires_grad)
        b = torch.randn(1, device=device, requires_grad=requires_grad)
        return U, V, h, w, b

    def run_pytorch(inputs):
        h_agg, c_bar = pytorch_coevol(*inputs)
        loss = h_agg.sum() + c_bar.sum()
        loss.backward()

    def run_triton(inputs):
        h_agg, c_bar = triton_coevol(*inputs)
        loss = h_agg.sum() + c_bar.sum()
        loss.backward()

    # Warmup
    for _ in range(warmup):
        run_triton(make_inputs())
        run_pytorch(make_inputs())
    torch.cuda.synchronize(device)

    # Time PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_pytorch(make_inputs())
    end.record()
    torch.cuda.synchronize(device)
    pt_ms = start.elapsed_time(end) / iters

    # Time Triton
    start.record()
    for _ in range(iters):
        run_triton(make_inputs())
    end.record()
    torch.cuda.synchronize(device)
    tri_ms = start.elapsed_time(end) / iters

    # Memory
    torch.cuda.reset_peak_memory_stats(device)
    run_pytorch(make_inputs())
    torch.cuda.synchronize(device)
    pt_mem = torch.cuda.max_memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    run_triton(make_inputs())
    torch.cuda.synchronize(device)
    tri_mem = torch.cuda.max_memory_allocated(device)

    print(
        f"  B={B} S={S:>3} N={N:>4} R={R:>2} D={D:>3} | "
        f"PyTorch {pt_ms:7.2f}ms {pt_mem/1e6:7.1f}MB | "
        f"Triton {tri_ms:7.2f}ms {tri_mem/1e6:7.1f}MB | "
        f"speedup {pt_ms/max(tri_ms, 0.01):.2f}x  mem {pt_mem/max(tri_mem, 1):.2f}x"
    )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)

    all_ok = True

    # Standard sizes
    for B, S, N, R, D in [
        (1, 32, 64, 16, 512),
        (2, 16, 128, 16, 256),
        (1, 32, 256, 16, 512),
    ]:
        all_ok &= test_correctness(B, S, N, R, D, device)

    # With mask
    for B, S, N, R, D in [
        (2, 32, 64, 16, 256),
        (1, 16, 128, 16, 512),
    ]:
        all_ok &= test_correctness(B, S, N, R, D, device, mask_ratio=0.3)

    # Edge cases: small S (< 16, triggers S_CHUNK padding)
    for S in [1, 4, 8, 12, 15]:
        all_ok &= test_correctness(1, S, 32, 16, 64, device)

    # Edge cases: small N (< BLOCK_I=32)
    for N in [1, 15, 17, 31]:
        all_ok &= test_correctness(1, 16, N, 16, 64, device)

    # Edge case: small D
    all_ok &= test_correctness(1, 16, 32, 16, 16, device)
    all_ok &= test_correctness(1, 16, 32, 16, 32, device)

    print(f"\n{'ALL PASSED' if all_ok else 'SOME TESTS FAILED'}\n")

    print("=" * 80)
    print("TIMING + MEMORY BENCHMARKS (fwd + bwd)")
    print("=" * 80)

    for B, S, N, R, D in [
        (1, 32, 64, 16, 512),
        (1, 32, 128, 16, 512),
        (1, 32, 256, 16, 512),
        (2, 32, 128, 16, 512),
        (1, 8, 128, 16, 512),    # small S
        (1, 32, 512, 16, 512),   # large N
    ]:
        bench_timing(B, S, N, R, D, device)


if __name__ == "__main__":
    main()
