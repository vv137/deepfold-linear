"""Benchmark Triton kernels vs Python reference: time, memory, numerical accuracy."""

import time
import torch
import torch.nn.functional as F


def _fmt(val, unit=""):
    if abs(val) < 1e-6:
        return f"{val:.2e}{unit}"
    return f"{val:.6f}{unit}"


def bench(name, fn, warmup=3, trials=10):
    """Benchmark a CUDA function: returns median time in ms and peak memory in MB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / (1024**2)
    times.sort()
    median = times[len(times) // 2]
    return median, peak_mem


def compare(name, ref, tri, rtol=1e-3, atol=1e-4):
    """Compare two tensors and print abs/rel error."""
    if ref.shape != tri.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} tri={tri.shape}")
        return
    abs_err = (ref - tri).abs().max().item()
    rel_err = ((ref - tri).abs() / (ref.abs().clamp(min=1e-8))).max().item()
    ok = abs_err < atol or rel_err < rtol
    status = "OK" if ok else "FAIL"
    print(f"  {name}: abs={_fmt(abs_err)} rel={_fmt(rel_err)} [{status}]")


# ============================================================================
# Kernel 1: Flash-Sinkhorn Forward
# ============================================================================


def bench_flash_sinkhorn_forward():
    from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

    print("\n" + "=" * 70)
    print("KERNEL 1: Flash-Sinkhorn Forward")
    print("=" * 70)

    for N in [128, 256, 512]:
        H, d_h = 16, 32
        device = "cuda"

        Q_ln = torch.randn(H, N, d_h, device=device)
        K_ln = torch.randn(H, N, d_h, device=device)
        V = torch.randn(H, N, d_h, device=device)
        x_res = torch.randn(N, 3, device=device) * 16.0
        eps = torch.tensor(
            [0.5] * 4 + [1.0] * 4 + [2.0] * 4 + [4.0] * 4,
            device=device,
            dtype=torch.float32,
        )
        w_dist = torch.randn(H, device=device) * 0.1
        log_mu = torch.log(torch.ones(H, N, device=device) / N)
        log_nu = torch.log(torch.ones(H, N, device=device) / N)
        K_iter = 7

        # Precompute pos_bias for reference
        pos_bias = torch.randn(H, N, N, device=device) * 0.1

        # --- Reference: materialized forward ---
        def ref_fn():
            kappa = 1.0 / (1.0 + eps)
            C = torch.zeros(H, N, N, device=device)
            for h in range(H):
                content = -(Q_ln[h] @ K_ln[h].T) / (d_h**0.5)
                dist = torch.cdist(x_res, x_res)
                geo = w_dist[h] * dist / (10.0 + dist)
                C[h] = content + pos_bias[h] + geo
            log_K = -C / eps[:, None, None]
            lu = torch.zeros(H, N, device=device)
            lv = torch.zeros(H, N, device=device)
            for _ in range(K_iter):
                lu = kappa[:, None] * (
                    log_mu - torch.logsumexp(log_K + lv[:, None, :], dim=-1)
                )
                lv = kappa[:, None] * (
                    log_nu - torch.logsumexp(log_K + lu[:, :, None], dim=-2)
                )
            log_score = lu[:, :, None] + log_K + lv[:, None, :]
            row_max = log_score.max(dim=-1, keepdim=True).values
            T = torch.exp(log_score - row_max)
            T_sum = T.sum(dim=-1, keepdim=True)
            T_norm = T / (T_sum + 1e-6)
            O_avg = torch.einsum("hnm,hmd->hnd", T_norm, V)
            x_cent = torch.einsum("hnm,mc->hnc", T_norm, x_res)
            return O_avg, x_cent, lu, lv

        # --- Triton ---
        def tri_fn():
            return flash_sinkhorn(
                Q_ln.float(),
                K_ln.float(),
                V.float(),
                x_res.float(),
                pos_bias.float(),
                eps.float(),
                w_dist.float(),
                log_mu.float(),
                log_nu.float(),
                K_iter=K_iter,
                lam=1.0,
                r_0=10.0,
            )

        ref_O, ref_xc, ref_lu, ref_lv = ref_fn()
        tri_O, tri_xc, tri_lu, tri_lv = tri_fn()

        print(f"\n  N={N}, H={H}, d_h={d_h}")
        compare("O_avg", ref_O, tri_O)
        compare("x_centroid", ref_xc, tri_xc)
        compare("log_u", ref_lu, tri_lu)
        compare("log_v", ref_lv, tri_lv)

        t_ref, m_ref = bench(f"ref N={N}", ref_fn)
        t_tri, m_tri = bench(f"tri N={N}", tri_fn)
        print(
            f"  Time:   ref={t_ref:.2f}ms  triton={t_tri:.2f}ms  speedup={t_ref / t_tri:.1f}x"
        )
        print(f"  Memory: ref={m_ref:.1f}MB  triton={m_tri:.1f}MB")


# ============================================================================
# Kernel 2: Co-evolution Tiling
# ============================================================================


def bench_coevol():
    from deepfold.model.kernels.coevol_kernel import triton_coevol

    print("\n" + "=" * 70)
    print("KERNEL 2: Co-evolution Tiling")
    print("=" * 70)

    for N in [128, 256, 512]:
        S, R, D = 128, 16, 512
        device = "cuda"

        U = torch.randn(S, N, R, device=device)
        V = torch.randn(S, N, R, device=device)
        h_coevol = torch.randn(N, D, device=device)
        w_weight = torch.randn(R, device=device) * 0.1
        b_weight = torch.randn(1, device=device) * 0.1

        # --- Reference: Python tiling ---
        def ref_fn():
            TILE = 64
            h_agg = torch.zeros(N, D, device=device)
            c_bar = torch.zeros(N, R, device=device)
            for i0 in range(0, N, TILE):
                ie = min(i0 + TILE, N)
                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    c_tile = torch.einsum("sir,sjr->ijr", U[:, i0:ie], V[:, j0:je]) / S
                    w_tile = torch.sigmoid((c_tile * w_weight).sum(-1) + b_weight)
                    h_agg[i0:ie] += w_tile @ h_coevol[j0:je]
                    c_bar[i0:ie] += c_tile.sum(dim=1)
            return h_agg, c_bar

        def tri_fn():
            return triton_coevol(U, V, h_coevol, w_weight, b_weight)

        ref_hagg, ref_cbar = ref_fn()
        tri_hagg, tri_cbar = tri_fn()

        print(f"\n  N={N}, S={S}, R={R}, D={D}")
        compare("h_agg", ref_hagg.float(), tri_hagg)
        compare("c_bar", ref_cbar.float(), tri_cbar)

        t_ref, m_ref = bench(f"ref N={N}", ref_fn)
        t_tri, m_tri = bench(f"tri N={N}", tri_fn)
        print(
            f"  Time:   ref={t_ref:.2f}ms  triton={t_tri:.2f}ms  speedup={t_ref / t_tri:.1f}x"
        )
        print(f"  Memory: ref={m_ref:.1f}MB  triton={m_tri:.1f}MB")


# ============================================================================
# Kernel 3: Distogram Tiling
# ============================================================================


def bench_distogram():
    from deepfold.model.kernels.distogram_kernel import triton_distogram_loss

    print("\n" + "=" * 70)
    print("KERNEL 3: Distogram Tiling")
    print("=" * 70)

    for N in [128, 256, 512]:
        d_low, num_bins = 64, 39
        device = "cuda"

        U = torch.randn(N, d_low, device=device)
        V = torch.randn(N, d_low, device=device)
        w_bin = torch.randn(num_bins, d_low, device=device) * 0.1
        bias = torch.randn(num_bins, device=device) * 0.1
        target_bins = torch.randint(0, num_bins, (N, N), device=device)

        # --- Reference: Python tiling ---
        def ref_fn():
            TILE = 64
            total_loss = torch.tensor(0.0, device=device)
            count = 0
            for i0 in range(0, N, TILE):
                ie = min(i0 + TILE, N)
                for j0 in range(0, N, TILE):
                    je = min(j0 + TILE, N)
                    Z = U[i0:ie, None, :] * V[None, j0:je, :]
                    logits = (Z @ w_bin.T) + bias
                    targets = target_bins[i0:ie, j0:je]
                    total_loss += F.cross_entropy(
                        logits.reshape(-1, num_bins),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    count += (ie - i0) * (je - j0)
            return total_loss / count

        def tri_fn():
            return triton_distogram_loss(U, V, w_bin, bias, target_bins)

        ref_loss = ref_fn()
        tri_loss = tri_fn()

        print(f"\n  N={N}, d_low={d_low}, num_bins={num_bins}")
        compare("loss", ref_loss.float(), tri_loss.squeeze().float())

        t_ref, m_ref = bench(f"ref N={N}", ref_fn)
        t_tri, m_tri = bench(f"tri N={N}", tri_fn)
        print(
            f"  Time:   ref={t_ref:.2f}ms  triton={t_tri:.2f}ms  speedup={t_ref / t_tri:.1f}x"
        )
        print(f"  Memory: ref={m_ref:.1f}MB  triton={m_tri:.1f}MB")


# ============================================================================
# Kernel 1 Backward: Flash-Sinkhorn Backward (Triton vs Python reference)
# ============================================================================


def bench_flash_sinkhorn_backward():
    from deepfold.model.kernels.flash_sinkhorn_attn import FlashSinkhornAttn
    from deepfold.model.kernels.sinkhorn_kernel import FlashSinkhornFunction

    print("\n" + "=" * 70)
    print("KERNEL 1 BACKWARD: Flash-Sinkhorn IFT (Triton vs Python)")
    print("  Note: Both use IFT backward. Compared against each other,")
    print("  not unrolled autograd (IFT ≠ unrolled when Sinkhorn not converged).")
    print("=" * 70)

    for N in [64, 128, 256]:
        H, d_h = 16, 32
        device = "cuda"

        Q_data = torch.randn(H, N, d_h, device=device)
        K_data = torch.randn(H, N, d_h, device=device)
        V_data = torch.randn(H, N, d_h, device=device)
        G_data = torch.full(
            (H, N, d_h), 100.0, device=device
        )  # sigmoid≈1 to disable gating
        x_data = torch.randn(N, 3, device=device) * 16.0
        pb_data = torch.randn(H, N, N, device=device) * 0.1
        eps = torch.tensor(
            [0.5] * 4 + [1.0] * 4 + [2.0] * 4 + [4.0] * 4,
            device=device,
            dtype=torch.float32,
        )
        wd_data = torch.randn(H, device=device) * 0.1
        log_mu = torch.log(torch.ones(H, N, device=device) / N)
        log_nu = torch.log(torch.ones(H, N, device=device) / N)

        # Python IFT backward (flash_sinkhorn_attn.py)
        def ref_backward():
            q, k, v, g, x, pb, wd = [
                t.detach().float().requires_grad_(True)
                for t in [Q_data, K_data, V_data, G_data, x_data, pb_data, wd_data]
            ]
            o, xc, _, _ = FlashSinkhornAttn.apply(
                q, k, v, g, x, pb, eps, wd, log_mu, log_nu, 7, 1.0, 10.0, None, None
            )
            (o.sum() + xc.sum()).backward()
            return q.grad, k.grad, v.grad, x.grad, wd.grad, pb.grad

        # Triton IFT backward
        def tri_backward():
            q, k, v, x, pb, wd = [
                t.detach().float().requires_grad_(True)
                for t in [Q_data, K_data, V_data, x_data, pb_data, wd_data]
            ]
            O, xc, _, _ = FlashSinkhornFunction.apply(
                q, k, v, x, pb, eps, wd, log_mu, log_nu, 7, 1.0, 10.0, None, None, 32
            )
            (O.sum() + xc.sum()).backward()
            return q.grad, k.grad, v.grad, x.grad, wd.grad, pb.grad

        print(f"\n  N={N}, H={H}, d_h={d_h}")

        try:
            ref_grads = ref_backward()
            tri_grads = tri_backward()

            names = [
                "grad_Q",
                "grad_K",
                "grad_V",
                "grad_x",
                "grad_w_dist",
                "grad_pos_bias",
            ]
            for nm, rg, tg in zip(names, ref_grads, tri_grads):
                if rg is not None and tg is not None:
                    abs_err = (rg - tg).abs().max().item()
                    rel = abs_err / (max(rg.norm().item(), tg.norm().item()) + 1e-8)
                    ok = rel < 0.02
                    print(
                        f"  {nm}: abs={abs_err:.4f} rel_norm={rel:.6f} [{'OK' if ok else 'FAIL'}]"
                    )

            t_ref, m_ref = bench(f"ref bwd N={N}", ref_backward)
            t_tri, m_tri = bench(f"tri bwd N={N}", tri_backward)
            print(
                f"  Time:   ref={t_ref:.2f}ms  triton={t_tri:.2f}ms  speedup={t_ref / t_tri:.1f}x"
            )
            print(f"  Memory: ref={m_ref:.1f}MB  triton={m_tri:.1f}MB")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    torch.manual_seed(42)
    bench_flash_sinkhorn_forward()
    bench_coevol()
    bench_distogram()
    bench_flash_sinkhorn_backward()
