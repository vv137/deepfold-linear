"""Benchmark: O(N²) PyTorch vs O(N) Flash-Sinkhorn — speed, memory, accuracy.

Compares forward+backward for:
  1. PyTorch reference: materialized N×N cost, transport, softmax weights
  2. Flash-Sinkhorn: tiled O(N) memory, on-the-fly recomputation
"""

import time
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")


def pytorch_reference_fwd_bwd(Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist,
                               log_mu, log_nu, K_iter, lam=1.0, r_0=10.0):
    """Full PyTorch forward+backward with N×N materialization. Returns grads dict."""
    H, N, d_h = Q_ln.shape

    Q_ln = Q_ln.clone().requires_grad_(True)
    K_ln = K_ln.clone().requires_grad_(True)
    V = V.clone().requires_grad_(True)
    G = G.clone().requires_grad_(True)
    x_res = x_res.clone().requires_grad_(True)
    w_dist = w_dist.clone().requires_grad_(True)

    # Forward: materialized
    content = -torch.einsum("hid,hjd->hij", Q_ln, K_ln) / (d_h ** 0.5)
    dist = torch.cdist(x_res, x_res)
    geo = w_dist[:, None, None] * dist / (r_0 + dist)
    C = content + pos_bias + geo

    # Sinkhorn (IFT via custom function)
    from deepfold.model.sinkhorn import sinkhorn_solve, compute_transport_output
    log_u, log_v = sinkhorn_solve(C, log_mu, log_nu, eps, lam=lam, K=K_iter, use_ift=True)
    o, T_norm, x_centroid = compute_transport_output(V, G, log_u, log_v, C, eps, x_res)

    # Loss (combine both outputs)
    loss = o.sum() + x_centroid.sum()
    loss.backward()

    return {
        "o": o.detach(), "xc": x_centroid.detach(),
        "log_u": log_u.detach(), "log_v": log_v.detach(),
        "grad_Q": Q_ln.grad.detach(), "grad_K": K_ln.grad.detach(),
        "grad_V": V.grad.detach(), "grad_G": G.grad.detach(),
        "grad_x": x_res.grad.detach(), "grad_w_dist": w_dist.grad.detach(),
    }


def flash_fwd_bwd(Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist,
                   log_mu, log_nu, K_iter, lam=1.0, r_0=10.0):
    """Flash-Sinkhorn forward+backward with O(N) memory. Returns grads dict."""
    from deepfold.model.kernels.flash_sinkhorn_attn import flash_sinkhorn_attn

    Q_ln = Q_ln.clone().requires_grad_(True)
    K_ln = K_ln.clone().requires_grad_(True)
    V = V.clone().requires_grad_(True)
    G = G.clone().requires_grad_(True)
    x_res = x_res.clone().requires_grad_(True)
    w_dist = w_dist.clone().requires_grad_(True)

    o_flat, x_centroid, log_u, log_v = flash_sinkhorn_attn(
        Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist,
        log_mu, log_nu, K_iter, lam, r_0,
    )

    loss = o_flat.sum() + x_centroid.sum()
    loss.backward()

    return {
        "o": o_flat.detach(), "xc": x_centroid.detach(),
        "log_u": log_u.detach(), "log_v": log_v.detach(),
        "grad_Q": Q_ln.grad.detach(), "grad_K": K_ln.grad.detach(),
        "grad_V": V.grad.detach(), "grad_G": G.grad.detach(),
        "grad_x": x_res.grad.detach(), "grad_w_dist": w_dist.grad.detach(),
    }


def report(name, ref, flash):
    diff = (ref - flash).float()
    atol = diff.abs().max().item()
    mean = diff.abs().mean().item()
    ref_norm = ref.float().norm().item()
    flash_norm = flash.float().norm().item()
    cos = F.cosine_similarity(ref.float().reshape(1, -1), flash.float().reshape(1, -1)).item()
    print(f"  {name:12s}: atol={atol:.3e}  mean={mean:.3e}  "
          f"norms=({ref_norm:.2f},{flash_norm:.2f})  cos={cos:.6f}")


def measure_peak_memory(fn, *args, n_warmup=2, **kwargs):
    """Measure peak GPU memory of fn(*args)."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6  # MB


def measure_time(fn, *args, n_warmup=3, n_iter=10, **kwargs):
    """Measure execution time."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def run_benchmark(N, H=16, d_h=32, K_iter=7):
    print(f"\n{'='*90}")
    print(f"N={N}, H={H}, d_h={d_h}, K_iter={K_iter}")
    print(f"{'='*90}")

    torch.manual_seed(42)
    Q_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    K_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    V = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    G = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    x_res = torch.randn(N, 3, device=DEVICE, dtype=torch.float32) * 10
    pos_bias = torch.randn(H, N, N, device=DEVICE, dtype=torch.float32) * 0.1
    eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=DEVICE)
    w_dist = torch.randn(H, device=DEVICE) * 0.1
    log_mu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)
    log_nu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)

    common = (Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter)

    # --- Correctness ---
    print("\n[Accuracy: Forward outputs]")
    ref = pytorch_reference_fwd_bwd(*common)
    flash = flash_fwd_bwd(*common)

    report("log_u", ref["log_u"], flash["log_u"])
    report("log_v", ref["log_v"], flash["log_v"])

    # Note: o has different shape (ref: N,H*d_h from permuted gated; flash: N,H*d_h from same)
    # xc shapes match
    report("x_centroid", ref["xc"], flash["xc"])

    print("\n[Accuracy: Backward gradients]")
    report("grad_Q", ref["grad_Q"], flash["grad_Q"])
    report("grad_K", ref["grad_K"], flash["grad_K"])
    report("grad_V", ref["grad_V"], flash["grad_V"])
    report("grad_G", ref["grad_G"], flash["grad_G"])
    report("grad_x", ref["grad_x"], flash["grad_x"])
    report("grad_w_dist", ref["grad_w_dist"], flash["grad_w_dist"])

    # --- Memory ---
    print("\n[Peak GPU Memory (fwd+bwd)]")
    torch.cuda.empty_cache()
    mem_ref = measure_peak_memory(pytorch_reference_fwd_bwd, *common)
    torch.cuda.empty_cache()
    mem_flash = measure_peak_memory(flash_fwd_bwd, *common)
    print(f"  PyTorch (N²): {mem_ref:.1f} MB")
    print(f"  Flash (N):    {mem_flash:.1f} MB")
    print(f"  Savings:      {mem_ref - mem_flash:.1f} MB ({mem_ref/max(mem_flash,1):.1f}x)")

    # --- Speed ---
    print("\n[Speed (fwd+bwd)]")
    t_ref = measure_time(pytorch_reference_fwd_bwd, *common)
    t_flash = measure_time(flash_fwd_bwd, *common)
    print(f"  PyTorch (N²): {t_ref:.2f} ms")
    print(f"  Flash (N):    {t_flash:.2f} ms")
    print(f"  Ratio:        {t_flash/t_ref:.2f}x {'(slower)' if t_flash > t_ref else '(faster)'}")

    return {
        "N": N, "mem_ref": mem_ref, "mem_flash": mem_flash,
        "time_ref": t_ref, "time_flash": t_flash,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Benchmark: O(N²) PyTorch vs O(N) Flash-Sinkhorn (fwd+bwd)")

    results = []
    for N in [64, 128, 256, 512]:
        results.append(run_benchmark(N))

    print(f"\n\n{'='*90}")
    print("Summary Table")
    print(f"{'='*90}")
    print(f"{'N':>6} | {'Mem PT MB':>10} | {'Mem Flash MB':>12} | {'Mem ratio':>10} | {'Time PT ms':>10} | {'Time Flash ms':>13} | {'Time ratio':>10}")
    print("-" * 85)
    for r in results:
        print(f"{r['N']:>6} | {r['mem_ref']:>10.1f} | {r['mem_flash']:>12.1f} | {r['mem_ref']/max(r['mem_flash'],1):>9.1f}x | {r['time_ref']:>10.2f} | {r['time_flash']:>13.2f} | {r['time_flash']/r['time_ref']:>9.2f}x")


if __name__ == "__main__":
    main()
