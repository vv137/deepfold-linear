"""Benchmark: convergence check overhead + CG solver vs torch.linalg.solve."""

import time
import torch


def bench(name, fn, warmup=3, trials=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ============================================================================
# 1. Sinkhorn: fixed K vs early-stop convergence check
# ============================================================================


def bench_sinkhorn_convergence():
    from deepfold.model.sinkhorn import sinkhorn_solve

    print("=" * 70)
    print("SINKHORN: Fixed K vs Early-Stop Convergence")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for N in [128, 256, 512]:
        H = 16
        torch.manual_seed(42)
        C = torch.randn(H, N, N, device=device).abs()
        log_mu = torch.log_softmax(torch.randn(H, N, device=device), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N, device=device), dim=-1)
        eps = torch.tensor(
            [0.5] * 4 + [1.0] * 4 + [2.0] * 4 + [4.0] * 4,
            device=device,
            dtype=torch.float32,
        )

        # Reference: fixed K=7 (our default cold-start)
        def fixed_7():
            return sinkhorn_solve(C, log_mu, log_nu, eps, K=7)

        # Early stop: K=20 max, threshold=1e-4
        def early_stop():
            return sinkhorn_solve(
                C, log_mu, log_nu, eps, K=20, threshold=1e-4, check_every=2
            )

        # Fixed K=20 (to compare accuracy)
        def fixed_20():
            return sinkhorn_solve(C, log_mu, log_nu, eps, K=20)

        ref_u, ref_v = fixed_20()
        f7_u, f7_v = fixed_7()
        es_u, es_v = early_stop()

        abs_f7 = max(
            (f7_u - ref_u).abs().max().item(), (f7_v - ref_v).abs().max().item()
        )
        abs_es = max(
            (es_u - ref_u).abs().max().item(), (es_v - ref_v).abs().max().item()
        )

        t_f7 = bench(f"fixed7 N={N}", fixed_7)
        t_es = bench(f"early  N={N}", early_stop)
        t_f20 = bench(f"fixed20 N={N}", fixed_20)

        # Count actual iterations for early stop by checking threshold
        log_K = -C / eps[:, None, None]
        kappa = 1.0 / (1.0 + eps)
        lu = torch.zeros_like(log_mu)
        lv = torch.zeros_like(log_nu)
        actual_iters = 0
        for i in range(20):
            lu_prev, lv_prev = lu, lv
            lu = kappa[:, None] * (
                log_mu - torch.logsumexp(log_K + lv[:, None, :], dim=-1)
            )
            lv = kappa[:, None] * (
                log_nu - torch.logsumexp(log_K + lu[:, :, None], dim=-2)
            )
            actual_iters = i + 1
            if (i + 1) % 2 == 0:
                uc = (lu - lu_prev).abs().max().item()
                vc = (lv - lv_prev).abs().max().item()
                if max(uc, vc) < 1e-4:
                    break

        print(f"\n  N={N}, H={H}")
        print(f"  Fixed K=7:   {t_f7:.2f}ms  abs_vs_K20={abs_f7:.6f}")
        print(
            f"  Early stop:  {t_es:.2f}ms  abs_vs_K20={abs_es:.6f}  actual_iters={actual_iters}"
        )
        print(f"  Fixed K=20:  {t_f20:.2f}ms  (reference)")


# ============================================================================
# 2. CG solver vs torch.linalg.solve
# ============================================================================


def bench_cg():
    from deepfold.utils.cg import conjugate_gradient

    print("\n" + "=" * 70)
    print("CG SOLVER vs torch.linalg.solve")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for n in [64, 256, 1024, 4096]:
        torch.manual_seed(42)
        A = torch.randn(n, n, device=device)
        A = A.T @ A + 0.1 * torch.eye(n, device=device)  # SPD
        b = torch.randn(n, device=device)

        x_true = torch.linalg.solve(A, b)

        def solve_direct():
            return torch.linalg.solve(A, b)

        def solve_cg():
            return conjugate_gradient(lambda v: A @ v, b, max_iter=min(n, 300))

        x_cg, info = solve_cg()
        abs_err = (x_cg - x_true).abs().max().item()
        rel_err = abs_err / (x_true.abs().max().item() + 1e-8)

        t_direct = bench(f"direct n={n}", solve_direct)
        t_cg = bench(f"cg n={n}", solve_cg)

        print(f"\n  n={n}")
        print(f"  Direct:  {t_direct:.2f}ms")
        print(
            f"  CG:      {t_cg:.2f}ms  iters={info.iters}  converged={info.converged}"
        )
        print(f"  Accuracy: abs={abs_err:.2e}  rel={rel_err:.2e}")

    # CG with matvec (no matrix materialization) — the real use case
    print("\n  --- CG with implicit matvec (no matrix stored) ---")
    for n in [1024, 4096, 16384]:
        torch.manual_seed(42)
        # Diagonal + low-rank: A = diag(d) + U @ U^T
        d = torch.rand(n, device=device) + 0.1
        rank = 8
        U = torch.randn(n, rank, device=device) * 0.1

        def matvec(v):
            return d * v + U @ (U.T @ v)

        b = torch.randn(n, device=device)

        def solve_cg_implicit():
            return conjugate_gradient(matvec, b, max_iter=100)

        x_cg, info = solve_cg_implicit()

        # Verify: A @ x ≈ b
        residual = (matvec(x_cg) - b).norm().item() / b.norm().item()

        t_cg = bench(f"cg_implicit n={n}", solve_cg_implicit)

        mem_full = n * n * 4 / (1024**2)  # MB if we stored A
        mem_implicit = (n + n * rank) * 4 / (1024**2)  # MB for d + U

        print(f"\n  n={n}")
        print(
            f"  CG implicit: {t_cg:.2f}ms  iters={info.iters}  converged={info.converged}"
        )
        print(f"  Relative residual: {residual:.2e}")
        print(f"  Memory: {mem_implicit:.1f}MB implicit vs {mem_full:.1f}MB dense")


# ============================================================================
# 3. CG warm-start benefit
# ============================================================================


def bench_cg_warmstart():
    from deepfold.utils.cg import conjugate_gradient

    print("\n" + "=" * 70)
    print("CG WARM-START BENEFIT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 1024
    torch.manual_seed(42)

    A = torch.randn(n, n, device=device)
    A = A.T @ A + 0.5 * torch.eye(n, device=device)

    # Simulate solving similar systems (e.g., consecutive Newton steps)
    b1 = torch.randn(n, device=device)
    b2 = b1 + 0.05 * torch.randn(n, device=device)  # slightly perturbed

    x1, info1 = conjugate_gradient(lambda v: A @ v, b1, max_iter=200)

    # Cold start for b2
    x2_cold, info_cold = conjugate_gradient(lambda v: A @ v, b2, max_iter=200)

    # Warm start for b2 (from x1)
    x2_warm, info_warm = conjugate_gradient(lambda v: A @ v, b2, x0=x1, max_iter=200)

    print(f"\n  n={n}")
    print(f"  Solve b1:         iters={info1.iters}  residual={info1.residual:.2e}")
    print(
        f"  Solve b2 (cold):  iters={info_cold.iters}  residual={info_cold.residual:.2e}"
    )
    print(
        f"  Solve b2 (warm):  iters={info_warm.iters}  residual={info_warm.residual:.2e}"
    )
    print(
        f"  Warm-start saves: {info_cold.iters - info_warm.iters} iterations "
        f"({100 * (info_cold.iters - info_warm.iters) / info_cold.iters:.0f}%)"
    )


# ============================================================================
# 4. Steihaug-CG
# ============================================================================


def bench_steihaug():
    from deepfold.utils.cg import steihaug_cg

    print("\n" + "=" * 70)
    print("STEIHAUG-CG TRUST REGION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SPD case — should converge inside TR
    torch.manual_seed(42)
    n = 256
    H = torch.randn(n, n, device=device)
    H = H.T @ H + torch.eye(n, device=device)
    g = torch.randn(n, device=device)

    p_spd, info_spd = steihaug_cg(lambda v: H @ v, g, delta=100.0, max_iter=100)
    p_true = -torch.linalg.solve(H, g)
    err_spd = (p_spd - p_true).norm().item() / p_true.norm().item()

    print(f"\n  SPD case (n={n}, delta=100):")
    print(f"    iters={info_spd.iters}  reason={info_spd.termination_reason}")
    print(f"    rel_error_vs_Newton={err_spd:.2e}")

    # Indefinite case — should detect negative curvature
    H_indef = torch.diag(torch.linspace(-1.0, 2.0, n, device=device))
    g2 = torch.randn(n, device=device)

    p_indef, info_indef = steihaug_cg(
        lambda v: H_indef @ v, g2, delta=1.0, max_iter=100
    )

    print(f"\n  Indefinite case (n={n}, delta=1.0):")
    print(f"    iters={info_indef.iters}  reason={info_indef.termination_reason}")
    print(
        f"    hit_boundary={info_indef.hit_boundary}  neg_curv={info_indef.negative_curvature_detected}"
    )
    print(
        f"    ||p||={torch.linalg.norm(p_indef).item():.4f}  pred_red={info_indef.predicted_reduction:.4f}"
    )


if __name__ == "__main__":
    bench_sinkhorn_convergence()
    bench_cg()
    bench_cg_warmstart()
    bench_steihaug()
