"""Benchmark and tolerance analysis for flash diffusion attention kernel.

Reports: latency, memory, numerical tolerance vs Python reference.
"""

import torch
import time
import gc

from deepfold.model.kernels.flash_diffusion_attn import (
    flash_diffusion_attn,
    flash_diff_attn_ref,
)

DEVICE = "cuda"
NUM_BINS = 68


def _make_inputs(B, H, N, D, dtype=torch.bfloat16, pad_frac=0.0):
    Q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    pw = torch.randn(H, NUM_BINS, device=DEVICE, dtype=torch.float32, requires_grad=True)
    bins = torch.randint(0, NUM_BINS, (B, N, N), device=DEVICE, dtype=torch.int32)
    mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)
    if pad_frac > 0:
        pad_start = int(N * (1 - pad_frac))
        mask[:, pad_start:] = 0.0
    return Q, K, V, pw, bins, mask


def bench_latency(B, H, N, D, n_warmup=5, n_iter=20):
    """Measure forward + backward latency."""
    Q, K, V, pw, bins, mask = _make_inputs(B, H, N, D)

    # Warmup
    for _ in range(n_warmup):
        out = flash_diffusion_attn(Q, K, V, pw, bins, mask)
        out.float().sum().backward()
        Q.grad = K.grad = V.grad = pw.grad = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        out = flash_diffusion_attn(Q, K, V, pw, bins, mask)
        out.float().sum().backward()
        Q.grad = K.grad = V.grad = pw.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter

    return elapsed


def bench_memory(B, H, N, D):
    """Measure peak GPU memory for forward + backward."""
    Q, K, V, pw, bins, mask = _make_inputs(B, H, N, D)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated()

    out = flash_diffusion_attn(Q, K, V, pw, bins, mask)
    mem_fwd = torch.cuda.max_memory_allocated() - mem_before

    out.float().sum().backward()
    mem_fwd_bwd = torch.cuda.max_memory_allocated() - mem_before

    # Compare: naive PyTorch would materialize (B, H, N, N) scores
    naive_scores_mem = B * H * N * N * 4  # float32
    # Plus (H, N, N) pos bias
    naive_pos_mem = H * N * N * 4

    Q.grad = K.grad = V.grad = pw.grad = None
    del out
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fwd_peak_MB": mem_fwd / 1e6,
        "fwd_bwd_peak_MB": mem_fwd_bwd / 1e6,
        "naive_scores_MB": naive_scores_mem / 1e6,
        "naive_pos_MB": naive_pos_mem / 1e6,
        "savings_ratio": (naive_scores_mem + naive_pos_mem) / max(mem_fwd_bwd, 1),
    }


def bench_tolerance(B, H, N, D, dtype=torch.bfloat16):
    """Measure numerical tolerance: Triton vs Python reference."""
    Q, K, V, pw, bins, mask = _make_inputs(B, H, N, D, dtype=dtype)

    # Forward tolerance
    with torch.no_grad():
        out_tri = flash_diffusion_attn(Q, K, V, pw, bins, mask)
        out_ref = flash_diff_attn_ref(Q.float(), K.float(), V.float(), pw, bins, mask)

    fwd_diff = (out_tri.float() - out_ref.float()).abs()
    fwd_max = fwd_diff.max().item()
    fwd_mean = fwd_diff.mean().item()
    fwd_rel = (fwd_diff / (out_ref.float().abs() + 1e-8)).mean().item()

    # Backward tolerance
    Q_ref = Q.detach().clone().float().requires_grad_(True)
    K_ref = K.detach().clone().float().requires_grad_(True)
    V_ref = V.detach().clone().float().requires_grad_(True)
    pw_ref = pw.detach().clone().requires_grad_(True)

    out_ref2 = flash_diff_attn_ref(Q_ref, K_ref, V_ref, pw_ref, bins, mask)
    out_ref2.sum().backward()

    Q_tri = Q.detach().clone().to(dtype).requires_grad_(True)
    K_tri = K.detach().clone().to(dtype).requires_grad_(True)
    V_tri = V.detach().clone().to(dtype).requires_grad_(True)
    pw_tri = pw.detach().clone().requires_grad_(True)

    out_tri2 = flash_diffusion_attn(Q_tri, K_tri, V_tri, pw_tri, bins, mask)
    out_tri2.float().sum().backward()

    grad_diffs = {}
    for name, g_tri, g_ref in [
        ("dQ", Q_tri.grad, Q_ref.grad),
        ("dK", K_tri.grad, K_ref.grad),
        ("dV", V_tri.grad, V_ref.grad),
        ("dW", pw_tri.grad, pw_ref.grad),
    ]:
        diff = (g_tri.float() - g_ref.float()).abs()
        grad_diffs[name] = {
            "max": diff.max().item(),
            "mean": diff.mean().item(),
            "rel_mean": (diff / (g_ref.float().abs() + 1e-8)).mean().item(),
        }

    return {
        "fwd_max_err": fwd_max,
        "fwd_mean_err": fwd_mean,
        "fwd_rel_err": fwd_rel,
        "grad": grad_diffs,
    }


if __name__ == "__main__":
    configs = [
        # (B, H, N, D) — typical training configs
        (1, 16, 128, 32),
        (2, 16, 256, 32),
        (2, 16, 384, 32),   # crop=384
        (1, 16, 512, 32),
        (1, 16, 768, 32),
        (1, 16, 1024, 32),  # large inference
    ]

    print("=" * 90)
    print(f"{'B':>3} {'H':>3} {'N':>5} {'D':>3} | {'fwd+bwd ms':>10} | "
          f"{'peak MB':>8} {'naive MB':>9} {'save':>5} | "
          f"{'fwd_max':>8} {'fwd_rel':>8}")
    print("=" * 90)

    for B, H, N, D in configs:
        try:
            lat = bench_latency(B, H, N, D) * 1000  # ms
            mem = bench_memory(B, H, N, D)
            tol = bench_tolerance(B, H, N, D)

            naive_mb = mem["naive_scores_MB"] + mem["naive_pos_MB"]
            print(
                f"{B:>3} {H:>3} {N:>5} {D:>3} | "
                f"{lat:>8.2f}ms | "
                f"{mem['fwd_bwd_peak_MB']:>7.1f} {naive_mb:>8.1f} "
                f"{mem['savings_ratio']:>4.1f}x | "
                f"{tol['fwd_max_err']:>8.5f} {tol['fwd_rel_err']:>8.5f}"
            )
        except Exception as e:
            print(f"{B:>3} {H:>3} {N:>5} {D:>3} | ERROR: {e}")

    # Detailed tolerance report for crop=384
    print("\n" + "=" * 60)
    print("Detailed tolerance report: B=2, H=16, N=384, D=32")
    print("=" * 60)
    tol = bench_tolerance(2, 16, 384, 32)
    print(f"Forward:  max={tol['fwd_max_err']:.6f}  mean={tol['fwd_mean_err']:.6f}  rel={tol['fwd_rel_err']:.6f}")
    for name, g in tol["grad"].items():
        print(f"  {name:>3}:    max={g['max']:.6f}  mean={g['mean']:.6f}  rel={g['rel_mean']:.6f}")
