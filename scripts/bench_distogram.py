#!/usr/bin/env python3
"""Benchmark: Triton distogram loss vs PyTorch reference (forward + backward).

Usage:
    uv run python scripts/bench_distogram.py [--device cuda:0]
"""

import argparse
import torch
import torch.nn.functional as F

from deepfold.model.kernels.distogram_kernel import triton_distogram_loss


def pytorch_distogram(U, V, w_bin, bias, x_true, mask=None,
                      dist_min=2.0, bin_width=0.5, num_bins=39):
    """Pure PyTorch reference — full materialization."""
    B, N, d_low = U.shape
    d_true = torch.cdist(x_true.float(), x_true.float())
    target_bins = ((d_true - dist_min) / bin_width).long().clamp(0, num_bins - 1)

    Z = U[:, :, None, :] * V[:, None, :, :]  # (B, N, N, d_low)
    logits = F.linear(Z, w_bin, bias)  # (B, N, N, num_bins)

    if mask is not None:
        valid = mask[:, :, None] * mask[:, None, :]  # (B, N, N)
    else:
        valid = torch.ones(B, N, N, device=U.device)

    total = torch.zeros(1, device=U.device)
    cnt = 0
    for b in range(B):
        m = valid[b] > 0
        if m.any():
            total = total + F.cross_entropy(
                logits[b][m].float(), target_bins[b][m], reduction="mean"
            )
            cnt += 1
    return (total / max(cnt, 1)).squeeze()


def test_correctness(B, N, d_low, num_bins, device, mask_ratio=0.0):
    torch.manual_seed(42)
    dist_min, bin_width = 2.0, 0.5

    U = torch.randn(B, N, d_low, device=device, requires_grad=True)
    V = torch.randn(B, N, d_low, device=device, requires_grad=True)
    w = torch.randn(num_bins, d_low, device=device, requires_grad=True)
    b = torch.randn(num_bins, device=device, requires_grad=True)
    x = torch.randn(B, N, 3, device=device) * 10  # coordinates

    mask = None
    if mask_ratio > 0:
        mask = (torch.rand(B, N, device=device) > mask_ratio).float()

    # Target bins (shared)
    d_true = torch.cdist(x.float(), x.float())
    target_bins = ((d_true - dist_min) / bin_width).long().clamp(0, num_bins - 1)

    # PyTorch reference
    U_r, V_r = U.detach().clone().requires_grad_(), V.detach().clone().requires_grad_()
    w_r, b_r = w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    loss_r = pytorch_distogram(U_r, V_r, w_r, b_r, x, mask, dist_min, bin_width, num_bins)
    loss_r.backward()

    # Triton
    U_t, V_t = U.detach().clone().requires_grad_(), V.detach().clone().requires_grad_()
    w_t, b_t = w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    loss_t = triton_distogram_loss(U_t, V_t, w_t, b_t, target_bins, x_true=x, mask=mask,
                                    dist_min=dist_min, bin_width=bin_width)
    loss_t.backward()

    def rel_err(ref, tri):
        diff = (ref.float() - tri.float()).abs()
        scale = ref.float().abs().amax().clamp(min=1e-6)
        return (diff / scale).amax().item()

    rel_tol = 2e-3  # TF32 in forward dot, IEEE in backward
    fwd_err = rel_err(loss_r, loss_t)
    fwd_ok = fwd_err < rel_tol

    grad_names = ["dU", "dV", "dW", "dbias"]
    ref_grads = [U_r.grad, V_r.grad, w_r.grad, b_r.grad]
    tri_grads = [U_t.grad, V_t.grad, w_t.grad, b_t.grad]
    bwd_ok = True
    grad_errs = {}
    for name, g_r, g_t in zip(grad_names, ref_grads, tri_grads):
        if g_r is None or g_t is None:
            grad_errs[name] = "None"
            bwd_ok = False
            continue
        r = rel_err(g_r, g_t)
        grad_errs[name] = f"{r:.2e}"
        if r >= rel_tol:
            bwd_ok = False

    tag = f"B={B} N={N} d={d_low} bins={num_bins} mask={mask_ratio:.1f}"
    status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
    print(f"  [{status}] {tag}  fwd={fwd_err:.2e}  bwd={grad_errs}")
    return fwd_ok and bwd_ok


def bench_timing(B, N, d_low, num_bins, device, warmup=10, iters=50):
    torch.manual_seed(0)
    dist_min, bin_width = 2.0, 0.5

    def make():
        U = torch.randn(B, N, d_low, device=device, requires_grad=True)
        V = torch.randn(B, N, d_low, device=device, requires_grad=True)
        w = torch.randn(num_bins, d_low, device=device, requires_grad=True)
        b = torch.randn(num_bins, device=device, requires_grad=True)
        x = torch.randn(B, N, 3, device=device) * 10
        d_true = torch.cdist(x.float(), x.float())
        tgt = ((d_true - dist_min) / bin_width).long().clamp(0, num_bins - 1)
        return U, V, w, b, x, tgt

    def run_pt(U, V, w, b, x, tgt):
        loss = pytorch_distogram(U, V, w, b, x, dist_min=dist_min, bin_width=bin_width, num_bins=num_bins)
        loss.backward()

    def run_tri(U, V, w, b, x, tgt):
        loss = triton_distogram_loss(U, V, w, b, tgt, x_true=x, dist_min=dist_min, bin_width=bin_width)
        loss.backward()

    for _ in range(warmup):
        run_tri(*make())
        run_pt(*make())
    torch.cuda.synchronize(device)

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run_pt(*make())
    e.record()
    torch.cuda.synchronize(device)
    pt_ms = s.elapsed_time(e) / iters

    s.record()
    for _ in range(iters):
        run_tri(*make())
    e.record()
    torch.cuda.synchronize(device)
    tri_ms = s.elapsed_time(e) / iters

    torch.cuda.reset_peak_memory_stats(device)
    run_pt(*make())
    torch.cuda.synchronize(device)
    pt_mem = torch.cuda.max_memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    run_tri(*make())
    torch.cuda.synchronize(device)
    tri_mem = torch.cuda.max_memory_allocated(device)

    print(
        f"  B={B} N={N:>4} d={d_low} bins={num_bins} | "
        f"PyTorch {pt_ms:7.2f}ms {pt_mem/1e6:7.1f}MB | "
        f"Triton {tri_ms:7.2f}ms {tri_mem/1e6:7.1f}MB | "
        f"speedup {pt_ms/max(tri_ms, 0.01):.2f}x  mem {pt_mem/max(tri_mem, 1):.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)

    all_ok = True
    for B, N, d, bins in [
        (1, 64, 64, 39),
        (2, 128, 64, 39),
        (1, 256, 64, 39),
    ]:
        all_ok &= test_correctness(B, N, d, bins, device)

    # With mask
    for B, N, d, bins in [
        (2, 64, 64, 39),
        (1, 128, 64, 39),
    ]:
        all_ok &= test_correctness(B, N, d, bins, device, mask_ratio=0.3)

    # Edge cases
    all_ok &= test_correctness(1, 1, 64, 39, device)
    all_ok &= test_correctness(1, 17, 64, 39, device)

    print(f"\n{'ALL PASSED' if all_ok else 'SOME TESTS FAILED'}\n")

    print("=" * 80)
    print("TIMING + MEMORY BENCHMARKS (fwd + bwd)")
    print("=" * 80)
    for B, N, d, bins in [
        (1, 64, 64, 39),
        (1, 128, 64, 39),
        (1, 256, 64, 39),
        (1, 512, 64, 39),
        (2, 128, 64, 39),
    ]:
        bench_timing(B, N, d, bins, device)


if __name__ == "__main__":
    main()
