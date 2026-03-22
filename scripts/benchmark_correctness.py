"""Detailed correctness comparison: PyTorch reference vs Triton Flash-Sinkhorn.

Reports per-output atol, rtol, mean/max errors, and gradient comparison.
"""

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")


def pytorch_sinkhorn_full(Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist,
                          log_mu, log_nu, K_iter, lam=1.0, r_0=10.0):
    """Reference PyTorch: full N×N materialized."""
    H, N, d_h = Q_ln.shape

    content = -torch.einsum("hid,hjd->hij", Q_ln, K_ln) / (d_h ** 0.5)
    dist = torch.cdist(x_res, x_res)
    geo = w_dist[:, None, None] * dist / (r_0 + dist)
    C = content + pos_bias + geo

    kappa = lam / (lam + eps)
    log_K = -C / eps[:, None, None]
    log_u = torch.zeros(H, N, device=C.device, dtype=torch.float32)
    log_v = torch.zeros(H, N, device=C.device, dtype=torch.float32)

    for _ in range(K_iter):
        log_u = kappa[:, None] * (log_mu - torch.logsumexp(log_K + log_v[:, None, :], dim=-1))
        log_v = kappa[:, None] * (log_nu - torch.logsumexp(log_K + log_u[:, :, None], dim=-2))

    log_score = log_u[:, :, None] + log_K + log_v[:, None, :]
    row_max = log_score.max(dim=-1, keepdim=True).values
    T = torch.exp(log_score - row_max)
    T_sum = T.sum(dim=-1, keepdim=True)
    T_norm = T / (T_sum + 1e-6)

    O_avg = torch.einsum("hnm,hmd->hnd", T_norm, V)
    x_centroid = torch.einsum("hnm,mc->hnc", T_norm, x_res)

    # Gated output (same as model)
    o = torch.sigmoid(G) * O_avg
    o_flat = o.permute(1, 0, 2).reshape(N, H * d_h)

    # EGNN delta
    delta = x_res.unsqueeze(0) - x_centroid

    return log_u, log_v, O_avg, x_centroid, T_norm, o_flat, delta


def report_error(name, ref, tri):
    """Report detailed error metrics."""
    diff = (ref - tri).float()
    abs_err = diff.abs()
    ref_abs = ref.float().abs().clamp(min=1e-8)
    rel_err = abs_err / ref_abs

    print(f"  {name:20s}  shape={list(ref.shape)}")
    print(f"    atol: max={abs_err.max():.6e}  mean={abs_err.mean():.6e}  "
          f"median={abs_err.median():.6e}  p99={abs_err.quantile(0.99):.6e}")
    print(f"    rtol: max={rel_err.max():.6e}  mean={rel_err.mean():.6e}  "
          f"median={rel_err.median():.6e}  p99={rel_err.quantile(0.99):.6e}")
    print(f"    ref range: [{ref.min():.4f}, {ref.max():.4f}]  "
          f"tri range: [{tri.min():.4f}, {tri.max():.4f}]")
    # Check standard torch.allclose thresholds
    for atol, rtol in [(1e-5, 1e-5), (1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2)]:
        ok = torch.allclose(ref, tri, atol=atol, rtol=rtol)
        if ok:
            print(f"    torch.allclose(atol={atol}, rtol={rtol}): PASS")
            break
        else:
            n_fail = (~torch.isclose(ref, tri, atol=atol, rtol=rtol)).sum().item()
            total = ref.numel()
            print(f"    torch.allclose(atol={atol}, rtol={rtol}): FAIL ({n_fail}/{total} elements)")
    print()


def run_correctness(N, H=16, d_h=32, K_iter=7):
    from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

    print(f"\n{'='*80}")
    print(f"N={N}, H={H}, d_h={d_h}, K_iter={K_iter}")
    print(f"{'='*80}")

    torch.manual_seed(42)
    Q_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    K_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    V = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    G = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    x_res = torch.randn(N, 3, device=DEVICE, dtype=torch.float32) * 10
    pos_bias = torch.randn(H, N, N, device=DEVICE, dtype=torch.float32) * 0.1
    eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=DEVICE, dtype=torch.float32)
    w_dist = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    log_mu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)
    log_nu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)

    # --- Forward comparison ---
    print("\n[Forward Pass]")
    lu_ref, lv_ref, O_ref, xc_ref, T_ref, o_ref, delta_ref = pytorch_sinkhorn_full(
        Q_ln, K_ln, V, G, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter
    )
    O_tri, xc_tri, lu_tri, lv_tri = flash_sinkhorn(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter
    )

    report_error("log_u", lu_ref, lu_tri)
    report_error("log_v", lv_ref, lv_tri)
    report_error("O_avg", O_ref, O_tri)
    report_error("x_centroid", xc_ref, xc_tri)

    # Per-head error breakdown for log_u
    print("  [Per-head log_u max atol]")
    for h in range(H):
        err = (lu_ref[h] - lu_tri[h]).abs().max().item()
        eps_h = eps[h].item()
        print(f"    head {h:2d} (eps={eps_h:.1f}): {err:.6e}", end="")
        if h % 4 == 3:
            print()

    # Per-head error for O_avg
    print("\n  [Per-head O_avg max atol]")
    for h in range(H):
        err = (O_ref[h] - O_tri[h]).abs().max().item()
        eps_h = eps[h].item()
        print(f"    head {h:2d} (eps={eps_h:.1f}): {err:.6e}", end="")
        if h % 4 == 3:
            print()

    # --- Gradient comparison (IFT backward) ---
    print(f"\n\n[Backward Pass (IFT)]")

    # PyTorch reference backward
    from deepfold.model.sinkhorn import sinkhorn_solve, compute_transport_output

    Q_ln_pt = Q_ln.clone().requires_grad_(True)
    K_ln_pt = K_ln.clone().requires_grad_(True)
    x_res_pt = x_res.clone().requires_grad_(True)
    V_pt = V.clone().requires_grad_(True)

    content_pt = -torch.einsum("hid,hjd->hij", Q_ln_pt, K_ln_pt) / (d_h ** 0.5)
    dist_pt = torch.cdist(x_res_pt, x_res_pt)
    geo_pt = w_dist[:, None, None] * dist_pt / (10.0 + dist_pt)
    C_pt = content_pt + pos_bias + geo_pt

    lu_pt, lv_pt = sinkhorn_solve(C_pt, log_mu, log_nu, eps, K=K_iter, use_ift=True)
    o_pt, _, xc_pt = compute_transport_output(V_pt, G, lu_pt, lv_pt, C_pt, eps, x_res_pt)

    # Create a combined scalar loss
    loss_pt = o_pt.sum() + xc_pt.sum()
    loss_pt.backward()

    print(f"  PyTorch IFT backward completed.")
    print(f"  grad_Q norm: {Q_ln_pt.grad.norm():.6e}")
    print(f"  grad_K norm: {K_ln_pt.grad.norm():.6e}")
    print(f"  grad_V norm: {V_pt.grad.norm():.6e}")
    print(f"  grad_x norm: {x_res_pt.grad.norm():.6e}")

    # Triton backward
    Q_ln_tr = Q_ln.clone().requires_grad_(True)
    K_ln_tr = K_ln.clone().requires_grad_(True)
    x_res_tr = x_res.clone().requires_grad_(True)

    # Note: flash_sinkhorn backward currently returns None for input grads
    # (the cost gradient pass is stubbed). We test that IFT kernels run without error.
    O_tr, xc_tr, lu_tr, lv_tr = flash_sinkhorn(
        Q_ln_tr, K_ln_tr, V, x_res_tr, pos_bias, eps, w_dist, log_mu, log_nu, K_iter
    )
    loss_tr = (O_tr.sum() + xc_tr.sum())
    try:
        loss_tr.backward()
        print(f"\n  Triton IFT backward completed.")
        if Q_ln_tr.grad is not None:
            print(f"  grad_Q norm: {Q_ln_tr.grad.norm():.6e}")
            report_error("grad_Q", Q_ln_pt.grad, Q_ln_tr.grad)
        else:
            print(f"  grad_Q: None (cost gradient pass stubbed — IFT adjoint runs, but ∂C→∂Q not yet fused)")
        if x_res_tr.grad is not None:
            print(f"  grad_x norm: {x_res_tr.grad.norm():.6e}")
            report_error("grad_x", x_res_pt.grad, x_res_tr.grad)
        else:
            print(f"  grad_x: None (cost gradient pass stubbed)")
    except Exception as e:
        print(f"  Triton backward error: {e}")


def main():
    print("Detailed Correctness: PyTorch Reference vs Triton Flash-Sinkhorn")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for N in [64, 128, 256, 512, 1024]:
        run_correctness(N)

    # Also test with warm start
    print(f"\n{'='*80}")
    print("Warm-start test (K=4, init from cold K=7)")
    print(f"{'='*80}")
    from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

    N, H, d_h = 256, 16, 32
    torch.manual_seed(0)
    Q_ln = torch.randn(H, N, d_h, device=DEVICE)
    K_ln = torch.randn(H, N, d_h, device=DEVICE)
    V = torch.randn(H, N, d_h, device=DEVICE)
    x_res = torch.randn(N, 3, device=DEVICE) * 10
    pos_bias = torch.randn(H, N, N, device=DEVICE) * 0.1
    eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=DEVICE)
    w_dist = torch.randn(H, device=DEVICE) * 0.1
    log_mu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)
    log_nu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)

    # Cold start
    _, _, lu_cold, lv_cold = flash_sinkhorn(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter=7
    )

    # Warm start from cold
    O_warm, xc_warm, lu_warm, lv_warm = flash_sinkhorn(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu,
        K_iter=4, log_u_init=lu_cold, log_v_init=lv_cold,
    )

    # Compare: warm K=4 from cold K=7 vs fresh cold K=11
    _, _, lu_11, lv_11 = flash_sinkhorn(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter=11
    )

    print("\n  Warm(7+4) vs Cold(11) — should be very close:")
    report_error("log_u", lu_11, lu_warm)
    report_error("log_v", lv_11, lv_warm)


if __name__ == "__main__":
    main()
