"""Benchmark: Naive PyTorch vs optimized implementations.

Tests correctness and measures throughput for:
1. Flash-Sinkhorn: Triton fused (cost + iterations + transport + EGNN centroid)
2. Co-evolution aggregation: torch.compile vs Python tiling
3. Distogram loss: torch.compile vs Python tiling
"""

import time
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")


# ============================================================================
# 1. Sinkhorn: PyTorch (materialized N×N) vs Triton (tiled, no N×N storage)
# ============================================================================

def pytorch_sinkhorn_full(Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter, lam=1.0, r_0=10.0):
    """Reference PyTorch: full N×N cost matrix + Sinkhorn + transport output."""
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

    return O_avg, x_centroid, log_u, log_v


def bench_sinkhorn(N, H=16, d_h=32, K_iter=7, n_warmup=5, n_iter=20):
    from deepfold.model.kernels.sinkhorn_kernel import flash_sinkhorn

    Q_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    K_ln = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    V = torch.randn(H, N, d_h, device=DEVICE, dtype=torch.float32)
    x_res = torch.randn(N, 3, device=DEVICE, dtype=torch.float32) * 10
    pos_bias = torch.randn(H, N, N, device=DEVICE, dtype=torch.float32) * 0.1
    eps = torch.tensor([0.5]*4 + [1.0]*4 + [2.0]*4 + [4.0]*4, device=DEVICE, dtype=torch.float32)
    w_dist = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    log_mu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)
    log_nu = torch.log_softmax(torch.randn(H, N, device=DEVICE), dim=-1)

    # Correctness
    O_ref, xc_ref, lu_ref, lv_ref = pytorch_sinkhorn_full(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter
    )
    O_tri, xc_tri, lu_tri, lv_tri = flash_sinkhorn(
        Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter
    )

    lu_err = (lu_ref - lu_tri).abs().max().item()
    o_err = (O_ref - O_tri).abs().max().item()
    xc_err = (xc_ref - xc_tri).abs().max().item()

    # PyTorch timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        pytorch_sinkhorn_full(Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_sinkhorn_full(Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter)
    torch.cuda.synchronize()
    t_pytorch = (time.perf_counter() - t0) / n_iter * 1000

    # Triton timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        flash_sinkhorn(Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        flash_sinkhorn(Q_ln, K_ln, V, x_res, pos_bias, eps, w_dist, log_mu, log_nu, K_iter)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / n_iter * 1000

    mem_pytorch_MB = H * N * N * 4 / 1e6  # N×N cost matrix
    mem_triton_MB = H * N * 4 * 2 / 1e6   # only log_u, log_v

    return {
        "N": N, "pytorch_ms": t_pytorch, "triton_ms": t_triton,
        "speedup": t_pytorch / max(t_triton, 1e-6),
        "log_u_err": lu_err, "O_err": o_err, "xc_err": xc_err,
        "mem_pytorch_MB": mem_pytorch_MB, "mem_triton_MB": mem_triton_MB,
    }


# ============================================================================
# 2. Co-evolution: Python tiling vs torch.compile
# ============================================================================

def pytorch_coevol_tiled(U, V, h_coevol, w_weight, b_weight, tile_size=64):
    """Reference: Python-level tiling (current implementation)."""
    S, N, R = U.shape
    D = h_coevol.shape[1]
    h_agg = torch.zeros(N, D, device=U.device, dtype=U.dtype)
    c_bar = torch.zeros(N, R, device=U.device, dtype=U.dtype)

    for i0 in range(0, N, tile_size):
        ie = min(i0 + tile_size, N)
        U_i = U[:, i0:ie, :]
        for j0 in range(0, N, tile_size):
            je = min(j0 + tile_size, N)
            V_j = V[:, j0:je, :]
            c_tile = torch.einsum("sir,sjr->ijr", U_i, V_j) / S
            w_tile = torch.sigmoid(F.linear(c_tile, w_weight.unsqueeze(0), b_weight).squeeze(-1))
            h_agg[i0:ie] += w_tile @ h_coevol[j0:je]
            c_bar[i0:ie] += c_tile.sum(dim=1)

    return h_agg, c_bar


def pytorch_coevol_full(U, V, h_coevol, w_weight, b_weight):
    """Vectorized (no tiling) — uses O(N²·R) memory but faster for moderate N."""
    S, N, R = U.shape
    c = torch.einsum("sir,sjr->ijr", U, V) / S          # (N, N, R)
    w = torch.sigmoid(F.linear(c, w_weight.unsqueeze(0), b_weight).squeeze(-1))  # (N, N)
    h_agg = w @ h_coevol                                  # (N, D)
    c_bar = c.sum(dim=1)                                   # (N, R)
    return h_agg, c_bar


_compiled_coevol = None
def compiled_coevol(U, V, h_coevol, w_weight, b_weight):
    global _compiled_coevol
    if _compiled_coevol is None:
        _compiled_coevol = torch.compile(pytorch_coevol_full, mode="reduce-overhead")
    return _compiled_coevol(U, V, h_coevol, w_weight, b_weight)


def bench_coevol(N, S=128, R=16, D=512, n_warmup=10, n_iter=20):
    U = torch.randn(S, N, R, device=DEVICE, dtype=torch.float32)
    V = torch.randn(S, N, R, device=DEVICE, dtype=torch.float32)
    h_coevol = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    w_weight = torch.randn(R, device=DEVICE, dtype=torch.float32)
    b_weight = torch.randn(1, device=DEVICE, dtype=torch.float32)

    # Correctness
    h_ref, c_ref = pytorch_coevol_tiled(U, V, h_coevol, w_weight, b_weight)
    h_full, c_full = pytorch_coevol_full(U, V, h_coevol, w_weight, b_weight)
    h_comp, c_comp = compiled_coevol(U, V, h_coevol, w_weight, b_weight)

    h_err_full = (h_ref - h_full).abs().max().item()
    h_err_comp = (h_ref - h_comp).abs().max().item()

    # Tiled timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        pytorch_coevol_tiled(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_coevol_tiled(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t_tiled = (time.perf_counter() - t0) / n_iter * 1000

    # Vectorized timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        pytorch_coevol_full(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_coevol_full(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / n_iter * 1000

    # Compiled timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        compiled_coevol(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compiled_coevol(U, V, h_coevol, w_weight, b_weight)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / n_iter * 1000

    return {
        "N": N,
        "tiled_ms": t_tiled, "vectorized_ms": t_full, "compiled_ms": t_compiled,
        "speedup_vec": t_tiled / max(t_full, 1e-6),
        "speedup_comp": t_tiled / max(t_compiled, 1e-6),
        "err_vec": h_err_full, "err_comp": h_err_comp,
    }


# ============================================================================
# 3. Distogram: Python tiling vs torch.compile
# ============================================================================

def pytorch_distogram_tiled(U, V, w_bin, bias, target_bins, tile_size=64):
    """Reference: Python-level tiling."""
    N, d_low = U.shape
    num_bins = w_bin.shape[0]
    total_loss = torch.tensor(0.0, device=U.device, dtype=U.dtype)
    count = 0
    for i0 in range(0, N, tile_size):
        ie = min(i0 + tile_size, N)
        for j0 in range(0, N, tile_size):
            je = min(j0 + tile_size, N)
            Z = U[i0:ie, None, :] * V[None, j0:je, :]
            logits = F.linear(Z, w_bin, bias)
            targets = target_bins[i0:ie, j0:je]
            total_loss = total_loss + F.cross_entropy(
                logits.reshape(-1, num_bins), targets.reshape(-1), reduction="sum"
            )
            count += (ie - i0) * (je - j0)
    return total_loss / max(count, 1)


def pytorch_distogram_full(U, V, w_bin, bias, target_bins):
    """Vectorized: single pass, O(N²·d_low) memory."""
    N = U.shape[0]
    num_bins = w_bin.shape[0]
    Z = U[:, None, :] * V[None, :, :]             # (N, N, d_low)
    logits = F.linear(Z, w_bin, bias)               # (N, N, num_bins)
    return F.cross_entropy(logits.reshape(-1, num_bins), target_bins.reshape(-1))


_compiled_disto = None
def compiled_distogram(U, V, w_bin, bias, target_bins):
    global _compiled_disto
    if _compiled_disto is None:
        _compiled_disto = torch.compile(pytorch_distogram_full, mode="reduce-overhead")
    return _compiled_disto(U, V, w_bin, bias, target_bins)


def bench_distogram(N, d_low=64, num_bins=39, n_warmup=10, n_iter=20):
    U = torch.randn(N, d_low, device=DEVICE, dtype=torch.float32)
    V = torch.randn(N, d_low, device=DEVICE, dtype=torch.float32)
    w_bin = torch.randn(num_bins, d_low, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(num_bins, device=DEVICE, dtype=torch.float32)
    target_bins = torch.randint(0, num_bins, (N, N), device=DEVICE, dtype=torch.long)

    # Correctness
    loss_ref = pytorch_distogram_tiled(U, V, w_bin, bias, target_bins)
    loss_full = pytorch_distogram_full(U, V, w_bin, bias, target_bins)
    loss_comp = compiled_distogram(U, V, w_bin, bias, target_bins)

    err_full = (loss_ref - loss_full).abs().item()
    err_comp = (loss_ref - loss_comp).abs().item()

    # Tiled timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        pytorch_distogram_tiled(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_distogram_tiled(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t_tiled = (time.perf_counter() - t0) / n_iter * 1000

    # Vectorized timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        pytorch_distogram_full(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_distogram_full(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / n_iter * 1000

    # Compiled timing
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        compiled_distogram(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compiled_distogram(U, V, w_bin, bias, target_bins)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / n_iter * 1000

    return {
        "N": N,
        "tiled_ms": t_tiled, "vectorized_ms": t_full, "compiled_ms": t_compiled,
        "speedup_vec": t_tiled / max(t_full, 1e-6),
        "speedup_comp": t_tiled / max(t_compiled, 1e-6),
        "err_full": err_full, "err_comp": err_comp,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*100}")

    # 1. Sinkhorn
    print("\n## 1. Flash-Sinkhorn: PyTorch (N×N materialized) vs Triton (tiled, no N×N)")
    print(f"{'N':>6} | {'PyTorch ms':>11} | {'Triton ms':>10} | {'Speedup':>8} | {'log_u err':>10} | {'O_avg err':>10} | {'Mem PT':>8} | {'Mem Tri':>8}")
    print("-" * 95)
    for N in [64, 128, 256, 512, 768, 1024]:
        try:
            r = bench_sinkhorn(N, K_iter=7)
            print(f"{r['N']:>6} | {r['pytorch_ms']:>11.2f} | {r['triton_ms']:>10.2f} | {r['speedup']:>7.2f}x | {r['log_u_err']:>10.2e} | {r['O_err']:>10.2e} | {r['mem_pytorch_MB']:>6.1f}MB | {r['mem_triton_MB']:>6.3f}MB")
        except Exception as e:
            print(f"{N:>6} | {'OOM/ERR':>11} | {'---':>10} | {'---':>8} | --- | --- | --- | ---  [{e.__class__.__name__}]")

    # 2. Co-evolution
    print(f"\n## 2. Co-evolution (S=128, R=16, D=512): Python tiling vs vectorized vs torch.compile")
    print(f"{'N':>6} | {'Tiled ms':>10} | {'Vector ms':>10} | {'Compile ms':>11} | {'Vec spdup':>10} | {'Comp spdup':>11} | {'err':>10}")
    print("-" * 90)
    for N in [64, 128, 256, 512]:
        r = bench_coevol(N)
        print(f"{r['N']:>6} | {r['tiled_ms']:>10.2f} | {r['vectorized_ms']:>10.2f} | {r['compiled_ms']:>11.2f} | {r['speedup_vec']:>9.2f}x | {r['speedup_comp']:>10.2f}x | {r['err_vec']:>10.2e}")

    # 3. Distogram
    print(f"\n## 3. Distogram Loss (d_low=64, 39 bins): Python tiling vs vectorized vs torch.compile")
    print(f"{'N':>6} | {'Tiled ms':>10} | {'Vector ms':>10} | {'Compile ms':>11} | {'Vec spdup':>10} | {'Comp spdup':>11} | {'err':>10}")
    print("-" * 90)
    for N in [64, 128, 256, 512]:
        r = bench_distogram(N)
        print(f"{r['N']:>6} | {r['tiled_ms']:>10.2f} | {r['vectorized_ms']:>10.2f} | {r['compiled_ms']:>11.2f} | {r['speedup_vec']:>9.2f}x | {r['speedup_comp']:>10.2f}x | {r['err_full']:>10.2e}")

    print(f"\n{'='*100}")
    print("Notes:")
    print("- Sinkhorn Triton: eliminates O(N²) cost matrix storage, computes distances on-the-fly per tile")
    print("- Co-evolution/Distogram: torch.compile fuses the Python tile loops automatically")
    print("- Memory savings are the primary goal; speed follows at larger N")


if __name__ == "__main__":
    main()
