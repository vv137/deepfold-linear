"""
Co-evolution tiling Triton kernel (SPEC §18 item 2).

Fuses the nested Python tile loops from MSA block §6.2 step 4:
  c_tile = einsum('sir,sjr->ijr', U_i, V_j) / S
  w_tile = sigmoid(Lin(16→1)(c_tile))
  h_agg[i] += w_tile @ h_coevol[j]
  c_bar_accum[i] += c_tile.sum(dim=1)

Vectorised: uses tl.dot for all matrix products, streams S in chunks,
and chunks over D to keep SRAM usage bounded.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _coevol_kernel(
    # U: (S, N, R), V: (S, N, R)
    U_ptr,
    V_ptr,
    # h_coevol: (N, D)
    H_COEVOL_ptr,
    # Scalar weight projection: weight (R,), bias (1,)
    W_WEIGHT_ptr,
    B_WEIGHT_ptr,
    # Outputs: h_agg (N, D), c_bar (N, R)
    H_AGG_ptr,
    C_BAR_ptr,
    # Dims
    S: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    # Strides for U, V: (S, N, R)
    stride_us,
    stride_un,
    stride_ur,
    stride_vs,
    stride_vn,
    stride_vr,
    # Strides for h_coevol: (N, D)
    stride_hn,
    stride_hd,
    # Strides for outputs
    stride_an,
    stride_ad,
    stride_cn,
    stride_cr,
    # Tile sizes
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    S_CHUNK: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    """Compute co-evolution aggregation in a single tiled pass.

    Grid: (ceil(N / BLOCK_I),).  Each program owns one row tile
    and iterates over all J tiles.

    For each (i-tile, j-tile) pair we compute:
      c_tile[:,:,r]  = (1/S) * U[:,i,r]^T @ V[:,j,r]   for r in 0..R-1
      w_score        = sum_r c_tile[:,:,r] * w_weight[r] + b_weight
      w_tile         = sigmoid(w_score)                  (BLOCK_I, BLOCK_J)
      h_agg[i]      += w_tile @ h_coevol[j]             (BLOCK_I, D)
      c_bar[i]      += c_tile.sum(dim=1)                (BLOCK_I, R)
    """
    pid_i = tl.program_id(0)
    i_start = pid_i * BLOCK_I
    i_idx = i_start + tl.arange(0, BLOCK_I)  # (BLOCK_I,)
    i_mask = i_idx < N

    inv_S = 1.0 / S

    # Load projection weights once — R is small (16)
    r_idx = tl.arange(0, R)  # (R,)
    w_weight = tl.load(W_WEIGHT_ptr + r_idx, mask=r_idx < R)  # (R,)
    b_weight = tl.load(B_WEIGHT_ptr)  # scalar

    # c_bar accumulator: (BLOCK_I, R) — stored across j-tile loop
    c_bar_acc = tl.zeros([BLOCK_I, R], dtype=tl.float32)

    # ---- iterate over j-tiles ----
    for j_start in range(0, N, BLOCK_J):
        j_idx = j_start + tl.arange(0, BLOCK_J)  # (BLOCK_J,)
        j_mask = j_idx < N

        # w_score accumulator: (BLOCK_I, BLOCK_J)
        w_score = tl.full([BLOCK_I, BLOCK_J], b_weight, dtype=tl.float32)

        # ---- iterate over r dimensions ----
        for r in range(R):
            # c_r will hold c_tile[:, :, r] = (1/S) * U[:,i,r]^T @ V[:,j,r]
            c_r = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)

            # Stream over S in chunks, using tl.dot for the matmul
            for s_start in range(0, S, S_CHUNK):
                s_idx = s_start + tl.arange(0, S_CHUNK)  # (S_CHUNK,)
                s_mask = s_idx < S

                # Load U_sr: (S_CHUNK, BLOCK_I) — U[s, i, r]
                # Pointer: U_ptr + s*stride_us + i*stride_un + r*stride_ur
                u_ptrs = (
                    U_ptr
                    + s_idx[:, None] * stride_us
                    + i_idx[None, :] * stride_un
                    + r * stride_ur
                )  # (S_CHUNK, BLOCK_I)
                u_chunk = tl.load(
                    u_ptrs, mask=s_mask[:, None] & i_mask[None, :], other=0.0
                )  # (S_CHUNK, BLOCK_I)

                # Load V_sr: (S_CHUNK, BLOCK_J) — V[s, j, r]
                v_ptrs = (
                    V_ptr
                    + s_idx[:, None] * stride_vs
                    + j_idx[None, :] * stride_vn
                    + r * stride_vr
                )  # (S_CHUNK, BLOCK_J)
                v_chunk = tl.load(
                    v_ptrs, mask=s_mask[:, None] & j_mask[None, :], other=0.0
                )  # (S_CHUNK, BLOCK_J)

                # c_r += U_chunk^T @ V_chunk  →  (BLOCK_I, BLOCK_J)
                c_r += tl.dot(tl.trans(u_chunk), v_chunk)

            c_r *= inv_S  # (BLOCK_I, BLOCK_J)

            # Accumulate weighted score: w_score += c_r * w_weight[r]
            w_r = tl.load(W_WEIGHT_ptr + r)  # scalar
            w_score += c_r * w_r

            # Accumulate c_bar: c_bar[i, r] += sum_j c_r[i, j]
            c_r_sum = tl.sum(c_r, axis=1)  # (BLOCK_I,)
            c_bar_acc += c_r_sum[:, None] * (r_idx[None, :] == r).to(tl.float32)

        # ---- sigmoid ----
        w_tile = tl.sigmoid(w_score)  # (BLOCK_I, BLOCK_J)
        w_tile = tl.where(j_mask[None, :], w_tile, 0.0)  # mask invalid j

        # ---- h_agg[i] += w_tile @ h_coevol[j] ----
        # (BLOCK_I, BLOCK_J) @ (BLOCK_J, D_CHUNK) → (BLOCK_I, D_CHUNK)
        # Chunk over D to keep register pressure bounded
        for d_start in range(0, D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)  # (D_CHUNK,)
            d_mask = d_idx < D

            # Load h_coevol[j, d]: (BLOCK_J, D_CHUNK)
            h_ptrs = (
                H_COEVOL_ptr + j_idx[:, None] * stride_hn + d_idx[None, :] * stride_hd
            )  # (BLOCK_J, D_CHUNK)
            h_chunk = tl.load(
                h_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0
            )  # (BLOCK_J, D_CHUNK)

            # agg_chunk = w_tile @ h_chunk  →  (BLOCK_I, D_CHUNK)
            agg_chunk = tl.dot(w_tile, h_chunk)

            # Atomic-add to output
            out_ptrs = (
                H_AGG_ptr + i_idx[:, None] * stride_an + d_idx[None, :] * stride_ad
            )  # (BLOCK_I, D_CHUNK)
            tl.atomic_add(out_ptrs, agg_chunk, mask=i_mask[:, None] & d_mask[None, :])

    # ---- store c_bar ----
    c_bar_ptrs = (
        C_BAR_ptr + i_idx[:, None] * stride_cn + r_idx[None, :] * stride_cr
    )  # (BLOCK_I, R)
    tl.store(c_bar_ptrs, c_bar_acc, mask=i_mask[:, None] & (r_idx[None, :] < R))


def triton_coevol(
    U: torch.Tensor,  # (S, N, R)
    V: torch.Tensor,  # (S, N, R)
    h_coevol: torch.Tensor,  # (N, D)
    w_weight: torch.Tensor,  # (R,)
    b_weight: torch.Tensor,  # (1,)
    BLOCK_I: int = 32,
    BLOCK_J: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton co-evolution aggregation.

    Returns:
        h_agg: (N, D) weighted aggregation
        c_bar: (N, R) co-evolution profile
    """
    S, N, R = U.shape
    D = h_coevol.shape[1]

    U = U.contiguous().float()
    V = V.contiguous().float()
    h_coevol = h_coevol.contiguous().float()
    w_weight = w_weight.contiguous().float()
    b_weight = b_weight.contiguous().float()

    h_agg = torch.zeros(N, D, device=U.device, dtype=torch.float32)
    c_bar = torch.zeros(N, R, device=U.device, dtype=torch.float32)

    # S_CHUNK must divide S or the kernel handles the tail via masking.
    # Pick 32 as a good default; for small S fall back to S itself.
    S_CHUNK = min(32, S)
    D_CHUNK = min(64, D)

    grid = ((N + BLOCK_I - 1) // BLOCK_I,)
    _coevol_kernel[grid](
        U,
        V,
        h_coevol,
        w_weight,
        b_weight,
        h_agg,
        c_bar,
        S,
        N,
        R,
        D,
        U.stride(0),
        U.stride(1),
        U.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        h_coevol.stride(0),
        h_coevol.stride(1),
        h_agg.stride(0),
        h_agg.stride(1),
        c_bar.stride(0),
        c_bar.stride(1),
        BLOCK_I=BLOCK_I,
        BLOCK_J=BLOCK_J,
        S_CHUNK=S_CHUNK,
        D_CHUNK=D_CHUNK,
    )

    return h_agg, c_bar
