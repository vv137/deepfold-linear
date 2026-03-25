"""
Co-evolution tiling Triton kernel with forward + backward (SPEC §18 item 2).

Fuses the nested Python tile loops from MSA block §6.2 step 4:
  c_tile = einsum('sir,sjr->ijr', U_i, V_j) / S
  w_tile = sigmoid(Lin(16->1)(c_tile)) * mask_j
  h_agg[i] += w_tile @ h_coevol[j]
  c_bar_accum[i] += c_tile.sum(dim=1) * mask_i

Forward: tiled with tl.dot, streams S in chunks, chunks D for SRAM.
         Caches w_tile (B, N, N) for backward reuse.
Backward: two kernels loading cached w_tile:
  - dU kernel (i-centric): dU + dw_weight + db_weight (c_r recomputed for dw_weight)
  - dV kernel (j-centric): dV + dh_coevol (fused D-chunk loop)

Supports batched inputs: U, V -> (B, S, N, R), h_coevol -> (B, N, D).
Optional padding mask (B, N) for variable-length proteins in a batch.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Forward kernel
# ============================================================================


@triton.jit
def _coevol_fwd_kernel(
    # Inputs
    U_ptr, V_ptr, H_COEVOL_ptr, W_WEIGHT_ptr, B_WEIGHT_ptr, MASK_ptr,
    # Outputs
    H_AGG_ptr, C_BAR_ptr, W_TILE_ptr,
    # Dims
    S: tl.constexpr, N: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    # Strides for U, V: (B, S, N, R)
    stride_ub, stride_us, stride_un, stride_ur,
    stride_vb, stride_vs, stride_vn, stride_vr,
    # Strides for h_coevol / h_agg: (B, N, D)
    stride_hb, stride_hn, stride_hd,
    stride_ab, stride_an, stride_ad,
    # Strides for c_bar: (B, N, R)
    stride_cb, stride_cn, stride_cr,
    # Strides for w_tile: (B, N, N)
    stride_wb, stride_wi, stride_wj,
    # Stride for mask: (B, N)
    stride_mb,
    # Tile sizes
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    S_CHUNK: tl.constexpr, D_CHUNK: tl.constexpr,
    # HAS_MASK/SAVE_W_TILE gate all accesses to MASK_ptr/W_TILE_ptr;
    # when False, those pointers are dummies and never dereferenced.
    HAS_MASK: tl.constexpr, SAVE_W_TILE: tl.constexpr,
):
    """Grid: (B, ceil(N / BLOCK_I)). Each program owns one i-tile, iterates j-tiles."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    i_start = pid_i * BLOCK_I
    i_idx = i_start + tl.arange(0, BLOCK_I)
    i_mask = i_idx < N

    U_batch = U_ptr + pid_b * stride_ub
    V_batch = V_ptr + pid_b * stride_vb
    H_batch = H_COEVOL_ptr + pid_b * stride_hb
    A_batch = H_AGG_ptr + pid_b * stride_ab
    C_batch = C_BAR_ptr + pid_b * stride_cb

    inv_S = 1.0 / S
    b_weight = tl.load(B_WEIGHT_ptr)
    r_idx = tl.arange(0, R)

    c_bar_acc = tl.zeros([BLOCK_I, R], dtype=tl.float32)

    for j_start in range(0, N, BLOCK_J):
        j_idx = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_idx < N

        w_score = tl.full([BLOCK_I, BLOCK_J], b_weight, dtype=tl.float32)

        for r in range(R):
            c_r = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            for s_start in range(0, S, S_CHUNK):
                s_idx = s_start + tl.arange(0, S_CHUNK)
                s_mask = s_idx < S
                u_ptrs = U_batch + s_idx[:, None] * stride_us + i_idx[None, :] * stride_un + r * stride_ur
                u_chunk = tl.load(u_ptrs, mask=s_mask[:, None] & i_mask[None, :], other=0.0)
                v_ptrs = V_batch + s_idx[:, None] * stride_vs + j_idx[None, :] * stride_vn + r * stride_vr
                v_chunk = tl.load(v_ptrs, mask=s_mask[:, None] & j_mask[None, :], other=0.0)
                c_r += tl.dot(tl.trans(u_chunk), v_chunk, input_precision="ieee")
            c_r *= inv_S

            w_r = tl.load(W_WEIGHT_ptr + r)
            w_score += c_r * w_r

            c_r_sum = tl.sum(c_r, axis=1)
            c_bar_acc += c_r_sum[:, None] * (r_idx[None, :] == r).to(tl.float32)

        w_tile = tl.sigmoid(w_score)
        w_tile = tl.where(j_mask[None, :], w_tile, 0.0)

        if HAS_MASK:
            j_pad = tl.load(MASK_ptr + pid_b * stride_mb + j_idx, mask=j_mask, other=0.0)
            w_tile = w_tile * j_pad[None, :]

        if SAVE_W_TILE:
            wt_ptrs = (W_TILE_ptr + pid_b * stride_wb
                       + i_idx[:, None] * stride_wi + j_idx[None, :] * stride_wj)
            tl.store(wt_ptrs, w_tile, mask=i_mask[:, None] & j_mask[None, :])

        for d_start in range(0, D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)
            d_mask = d_idx < D
            h_ptrs = H_batch + j_idx[:, None] * stride_hn + d_idx[None, :] * stride_hd
            h_chunk = tl.load(h_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0)
            agg_chunk = tl.dot(w_tile, h_chunk, input_precision="ieee")
            out_ptrs = A_batch + i_idx[:, None] * stride_an + d_idx[None, :] * stride_ad
            tl.atomic_add(out_ptrs, agg_chunk, mask=i_mask[:, None] & d_mask[None, :])

    if HAS_MASK:
        i_pad = tl.load(MASK_ptr + pid_b * stride_mb + i_idx, mask=i_mask, other=0.0)
        c_bar_acc = c_bar_acc * i_pad[:, None]

    c_bar_ptrs = C_batch + i_idx[:, None] * stride_cn + r_idx[None, :] * stride_cr
    tl.store(c_bar_ptrs, c_bar_acc, mask=i_mask[:, None] & (r_idx[None, :] < R))


# ============================================================================
# Backward kernel 1: dU + dw_weight + db_weight  (i-centric)
# ============================================================================


@triton.jit
def _coevol_bwd_dU_kernel(
    # Saved tensors
    U_ptr, V_ptr, H_COEVOL_ptr, W_WEIGHT_ptr, MASK_ptr, W_TILE_ptr,
    # Upstream gradients
    GRAD_HAGG_ptr, GRAD_CBAR_ptr,
    # Outputs
    DU_ptr, DW_WEIGHT_ptr, DB_WEIGHT_ptr,
    # Dims
    S: tl.constexpr, N: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    # U/V strides
    stride_ub, stride_us, stride_un, stride_ur,
    stride_vb, stride_vs, stride_vn, stride_vr,
    # h_coevol strides
    stride_hb, stride_hn, stride_hd,
    # grad_h_agg strides
    stride_gb, stride_gn, stride_gd,
    # grad_c_bar strides
    stride_gcb, stride_gcn, stride_gcr,
    # dU strides
    stride_dub, stride_dus, stride_dun, stride_dur,
    # w_tile strides
    stride_wb, stride_wi, stride_wj,
    # mask stride
    stride_mb,
    # Tile sizes
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    S_CHUNK: tl.constexpr, D_CHUNK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Grid: (B, ceil(N/BLOCK_I)). Loads cached w_tile, computes dU + dw_weight + db_weight."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    i_start = pid_i * BLOCK_I
    i_idx = i_start + tl.arange(0, BLOCK_I)
    i_mask = i_idx < N

    U_batch = U_ptr + pid_b * stride_ub
    V_batch = V_ptr + pid_b * stride_vb
    H_batch = H_COEVOL_ptr + pid_b * stride_hb
    G_batch = GRAD_HAGG_ptr + pid_b * stride_gb
    GC_batch = GRAD_CBAR_ptr + pid_b * stride_gcb
    DU_batch = DU_ptr + pid_b * stride_dub
    WT_batch = W_TILE_ptr + pid_b * stride_wb

    inv_S = 1.0 / S
    r_idx = tl.arange(0, R)

    # Preload grad_c_bar[i, :]
    gc_ptrs = GC_batch + i_idx[:, None] * stride_gcn + r_idx[None, :] * stride_gcr
    grad_cbar_i = tl.load(gc_ptrs, mask=i_mask[:, None] & (r_idx[None, :] < R), other=0.0)
    if HAS_MASK:
        i_pad = tl.load(MASK_ptr + pid_b * stride_mb + i_idx, mask=i_mask, other=0.0)
        grad_cbar_i = grad_cbar_i * i_pad[:, None]

    dw_acc = tl.zeros([R], dtype=tl.float32)
    db_acc = tl.zeros([1], dtype=tl.float32)

    for j_start in range(0, N, BLOCK_J):
        j_idx = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_idx < N

        # Load cached w_tile
        wt_ptrs = WT_batch + i_idx[:, None] * stride_wi + j_idx[None, :] * stride_wj
        w_tile = tl.load(wt_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

        # Compute dw_score: w_tile*(1-w_tile) is correct sigmoid derivative
        # because w_tile = sig*mask, and mask=0 => w_tile=0 => dw_score=0.
        dw_tile = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
        for d_start in range(0, D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)
            d_mask = d_idx < D
            g_ptrs = G_batch + i_idx[:, None] * stride_gn + d_idx[None, :] * stride_gd
            g_chunk = tl.load(g_ptrs, mask=i_mask[:, None] & d_mask[None, :], other=0.0)
            h_ptrs = H_batch + j_idx[:, None] * stride_hn + d_idx[None, :] * stride_hd
            h_chunk = tl.load(h_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0)
            dw_tile += tl.dot(g_chunk, tl.trans(h_chunk), input_precision="ieee")

        dw_score = dw_tile * w_tile * (1.0 - w_tile)
        db_acc += tl.sum(dw_score).to(tl.float32).expand_dims(0)

        # dU per r + dw_weight via c_r recomputation
        for r in range(R):
            w_r = tl.load(W_WEIGHT_ptr + r)
            gc_r = tl.sum(grad_cbar_i * (r_idx[None, :] == r).to(tl.float32), axis=1)
            dc_r = dw_score * w_r + gc_r[:, None]

            c_r = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            for s_start in range(0, S, S_CHUNK):
                s_idx = s_start + tl.arange(0, S_CHUNK)
                s_mask = s_idx < S

                u_ptrs = U_batch + s_idx[:, None] * stride_us + i_idx[None, :] * stride_un + r * stride_ur
                u_chunk = tl.load(u_ptrs, mask=s_mask[:, None] & i_mask[None, :], other=0.0)
                v_ptrs = V_batch + s_idx[:, None] * stride_vs + j_idx[None, :] * stride_vn + r * stride_vr
                v_chunk = tl.load(v_ptrs, mask=s_mask[:, None] & j_mask[None, :], other=0.0)

                # dU
                dU_delta = inv_S * tl.dot(v_chunk, tl.trans(dc_r), input_precision="ieee")
                du_ptrs = DU_batch + s_idx[:, None] * stride_dus + i_idx[None, :] * stride_dun + r * stride_dur
                old_dU = tl.load(du_ptrs, mask=s_mask[:, None] & i_mask[None, :], other=0.0)
                tl.store(du_ptrs, old_dU + dU_delta, mask=s_mask[:, None] & i_mask[None, :])

                # Reuse loaded U, V for c_r (dw_weight)
                c_r += tl.dot(tl.trans(u_chunk), v_chunk, input_precision="ieee")

            c_r *= inv_S
            dw_r_val = tl.sum(dw_score * c_r)
            dw_acc += dw_r_val * (r_idx == r).to(tl.float32)

    tl.atomic_add(DW_WEIGHT_ptr + r_idx, dw_acc, mask=r_idx < R)
    tl.atomic_add(DB_WEIGHT_ptr + tl.arange(0, 1), db_acc)


# ============================================================================
# Backward kernel 2: dV + dh_coevol  (j-centric, fused D-chunk loop)
# ============================================================================


@triton.jit
def _coevol_bwd_dV_kernel(
    # Saved tensors
    U_ptr, H_COEVOL_ptr, W_WEIGHT_ptr, MASK_ptr, W_TILE_ptr,
    # Upstream gradients
    GRAD_HAGG_ptr, GRAD_CBAR_ptr,
    # Outputs
    DV_ptr, DH_COEVOL_ptr,
    # Dims
    S: tl.constexpr, N: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    # U strides
    stride_ub, stride_us, stride_un, stride_ur,
    # h_coevol strides
    stride_hb, stride_hn, stride_hd,
    # grad_h_agg strides
    stride_gb, stride_gn, stride_gd,
    # grad_c_bar strides
    stride_gcb, stride_gcn, stride_gcr,
    # dV strides
    stride_dvb, stride_dvs, stride_dvn, stride_dvr,
    # dh_coevol strides
    stride_dhb, stride_dhn, stride_dhd,
    # w_tile strides
    stride_wb, stride_wi, stride_wj,
    # mask stride
    stride_mb,
    # Tile sizes
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    S_CHUNK: tl.constexpr, D_CHUNK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Grid: (B, ceil(N/BLOCK_J)). Loads cached w_tile, computes dV + dh_coevol."""
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    j_start = pid_j * BLOCK_J
    j_idx = j_start + tl.arange(0, BLOCK_J)
    j_mask = j_idx < N

    U_batch = U_ptr + pid_b * stride_ub
    H_batch = H_COEVOL_ptr + pid_b * stride_hb
    G_batch = GRAD_HAGG_ptr + pid_b * stride_gb
    GC_batch = GRAD_CBAR_ptr + pid_b * stride_gcb
    DV_batch = DV_ptr + pid_b * stride_dvb
    DH_batch = DH_COEVOL_ptr + pid_b * stride_dhb
    WT_batch = W_TILE_ptr + pid_b * stride_wb

    inv_S = 1.0 / S
    r_idx = tl.arange(0, R)

    for i_start in range(0, N, BLOCK_I):
        i_idx = i_start + tl.arange(0, BLOCK_I)
        i_mask = i_idx < N

        # Load cached w_tile
        wt_ptrs = WT_batch + i_idx[:, None] * stride_wi + j_idx[None, :] * stride_wj
        w_tile = tl.load(wt_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

        # Fused D-chunk loop: compute dw_tile (for dw_score) and dh_coevol simultaneously.
        # Both need grad_h_agg[i, d] chunks — loading once halves bandwidth.
        dw_tile = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
        for d_start in range(0, D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)
            d_mask = d_idx < D

            g_ptrs = G_batch + i_idx[:, None] * stride_gn + d_idx[None, :] * stride_gd
            g_chunk = tl.load(g_ptrs, mask=i_mask[:, None] & d_mask[None, :], other=0.0)
            h_ptrs = H_batch + j_idx[:, None] * stride_hn + d_idx[None, :] * stride_hd
            h_chunk = tl.load(h_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0)

            # dw_tile accumulation
            dw_tile += tl.dot(g_chunk, tl.trans(h_chunk), input_precision="ieee")

            # dh_coevol[j, d] += w_tile^T @ grad_h_agg[i, d]
            dh_chunk = tl.dot(tl.trans(w_tile), g_chunk, input_precision="ieee")
            dh_ptrs = DH_batch + j_idx[:, None] * stride_dhn + d_idx[None, :] * stride_dhd
            old_dh = tl.load(dh_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0)
            tl.store(dh_ptrs, old_dh + dh_chunk, mask=j_mask[:, None] & d_mask[None, :])

        dw_score = dw_tile * w_tile * (1.0 - w_tile)

        # dV per r
        gc_ptrs = GC_batch + i_idx[:, None] * stride_gcn + r_idx[None, :] * stride_gcr
        grad_cbar_i = tl.load(gc_ptrs, mask=i_mask[:, None] & (r_idx[None, :] < R), other=0.0)
        if HAS_MASK:
            i_pad = tl.load(MASK_ptr + pid_b * stride_mb + i_idx, mask=i_mask, other=0.0)
            grad_cbar_i = grad_cbar_i * i_pad[:, None]

        for r in range(R):
            w_r = tl.load(W_WEIGHT_ptr + r)
            gc_r = tl.sum(grad_cbar_i * (r_idx[None, :] == r).to(tl.float32), axis=1)
            dc_r = dw_score * w_r + gc_r[:, None]

            for s_start in range(0, S, S_CHUNK):
                s_idx = s_start + tl.arange(0, S_CHUNK)
                s_mask = s_idx < S

                u_ptrs = U_batch + s_idx[:, None] * stride_us + i_idx[None, :] * stride_un + r * stride_ur
                u_chunk = tl.load(u_ptrs, mask=s_mask[:, None] & i_mask[None, :], other=0.0)
                dV_delta = inv_S * tl.dot(u_chunk, dc_r, input_precision="ieee")

                dv_ptrs = DV_batch + s_idx[:, None] * stride_dvs + j_idx[None, :] * stride_dvn + r * stride_dvr
                old_dV = tl.load(dv_ptrs, mask=s_mask[:, None] & j_mask[None, :], other=0.0)
                tl.store(dv_ptrs, old_dV + dV_delta, mask=s_mask[:, None] & j_mask[None, :])


# ============================================================================
# Autograd function
# ============================================================================


def _compute_chunks(S, D):
    """Compute S_CHUNK and D_CHUNK respecting tl.dot K >= 16 constraint."""
    S_CHUNK = max(min(32, S), 16)
    D_CHUNK = max(min(64, D), 16)
    return S_CHUNK, D_CHUNK


def _launch_fwd(U, V, h_coevol, w_weight, b_weight, mask, BLOCK_I, BLOCK_J,
                save_w_tile=False):
    """Launch forward kernel, return (h_agg, c_bar, w_tile_or_None)."""
    B, S, N, R = U.shape
    D = h_coevol.shape[2]
    S_CHUNK, D_CHUNK = _compute_chunks(S, D)
    has_mask = mask is not None

    h_agg = torch.zeros(B, N, D, device=U.device, dtype=torch.float32)
    c_bar = torch.zeros(B, N, R, device=U.device, dtype=torch.float32)

    if save_w_tile:
        w_tile_buf = torch.empty(B, N, N, device=U.device, dtype=torch.float32)
    else:
        w_tile_buf = torch.empty(0, device=U.device, dtype=torch.float32)

    grid = (B, triton.cdiv(N, BLOCK_I))
    _coevol_fwd_kernel[grid](
        U, V, h_coevol, w_weight, b_weight,
        mask if has_mask else U,  # dummy, gated by HAS_MASK
        h_agg, c_bar,
        w_tile_buf if save_w_tile else h_agg,  # dummy, gated by SAVE_W_TILE
        S, N, R, D,
        U.stride(0), U.stride(1), U.stride(2), U.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        h_coevol.stride(0), h_coevol.stride(1), h_coevol.stride(2),
        h_agg.stride(0), h_agg.stride(1), h_agg.stride(2),
        c_bar.stride(0), c_bar.stride(1), c_bar.stride(2),
        w_tile_buf.stride(0) if save_w_tile else 0,
        w_tile_buf.stride(1) if save_w_tile else 0,
        w_tile_buf.stride(2) if save_w_tile else 0,
        mask.stride(0) if has_mask else 0,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
        S_CHUNK=S_CHUNK, D_CHUNK=D_CHUNK,
        HAS_MASK=has_mask, SAVE_W_TILE=save_w_tile,
    )
    return h_agg, c_bar, w_tile_buf if save_w_tile else None


class CoevolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, h_coevol, w_weight, b_weight, mask, BLOCK_I, BLOCK_J):
        h_agg, c_bar, w_tile_buf = _launch_fwd(
            U, V, h_coevol, w_weight, b_weight, mask, BLOCK_I, BLOCK_J,
            save_w_tile=True,
        )
        ctx.save_for_backward(
            U, V, h_coevol, w_weight, b_weight,
            mask if mask is not None else torch.empty(0, device=U.device),
            w_tile_buf,
        )
        ctx.BLOCK_I = BLOCK_I
        ctx.BLOCK_J = BLOCK_J
        ctx.has_mask = mask is not None
        return h_agg, c_bar

    @staticmethod
    def backward(ctx, grad_h_agg, grad_c_bar):
        U, V, h_coevol, w_weight, b_weight, mask_or_empty, w_tile_buf = ctx.saved_tensors
        has_mask = ctx.has_mask
        mask = mask_or_empty if has_mask else None
        BLOCK_I, BLOCK_J = ctx.BLOCK_I, ctx.BLOCK_J
        B, S, N, R = U.shape
        D = h_coevol.shape[2]
        S_CHUNK, D_CHUNK = _compute_chunks(S, D)

        grad_h_agg = grad_h_agg.contiguous().float()
        grad_c_bar = grad_c_bar.contiguous().float()

        dU = torch.zeros_like(U)
        dV = torch.zeros_like(V)
        dh_coevol = torch.zeros_like(h_coevol)
        dw_weight = torch.zeros_like(w_weight)
        db_weight = torch.zeros_like(b_weight)

        mask_ptr = mask if has_mask else U  # dummy, gated by HAS_MASK
        wt_strides = (w_tile_buf.stride(0), w_tile_buf.stride(1), w_tile_buf.stride(2))

        # Kernel 1: dU + dw_weight + db_weight (i-centric)
        grid_i = (B, triton.cdiv(N, BLOCK_I))
        _coevol_bwd_dU_kernel[grid_i](
            U, V, h_coevol, w_weight, mask_ptr, w_tile_buf,
            grad_h_agg, grad_c_bar,
            dU, dw_weight, db_weight,
            S=S, N=N, R=R, D=D,
            stride_ub=U.stride(0), stride_us=U.stride(1),
            stride_un=U.stride(2), stride_ur=U.stride(3),
            stride_vb=V.stride(0), stride_vs=V.stride(1),
            stride_vn=V.stride(2), stride_vr=V.stride(3),
            stride_hb=h_coevol.stride(0), stride_hn=h_coevol.stride(1),
            stride_hd=h_coevol.stride(2),
            stride_gb=grad_h_agg.stride(0), stride_gn=grad_h_agg.stride(1),
            stride_gd=grad_h_agg.stride(2),
            stride_gcb=grad_c_bar.stride(0), stride_gcn=grad_c_bar.stride(1),
            stride_gcr=grad_c_bar.stride(2),
            stride_dub=dU.stride(0), stride_dus=dU.stride(1),
            stride_dun=dU.stride(2), stride_dur=dU.stride(3),
            stride_wb=wt_strides[0], stride_wi=wt_strides[1], stride_wj=wt_strides[2],
            stride_mb=mask.stride(0) if has_mask else 0,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
            S_CHUNK=S_CHUNK, D_CHUNK=D_CHUNK,
            HAS_MASK=has_mask,
        )

        # Kernel 2: dV + dh_coevol (j-centric)
        grid_j = (B, triton.cdiv(N, BLOCK_J))
        _coevol_bwd_dV_kernel[grid_j](
            U, h_coevol, w_weight, mask_ptr, w_tile_buf,
            grad_h_agg, grad_c_bar,
            dV, dh_coevol,
            S=S, N=N, R=R, D=D,
            stride_ub=U.stride(0), stride_us=U.stride(1),
            stride_un=U.stride(2), stride_ur=U.stride(3),
            stride_hb=h_coevol.stride(0), stride_hn=h_coevol.stride(1),
            stride_hd=h_coevol.stride(2),
            stride_gb=grad_h_agg.stride(0), stride_gn=grad_h_agg.stride(1),
            stride_gd=grad_h_agg.stride(2),
            stride_gcb=grad_c_bar.stride(0), stride_gcn=grad_c_bar.stride(1),
            stride_gcr=grad_c_bar.stride(2),
            stride_dvb=dV.stride(0), stride_dvs=dV.stride(1),
            stride_dvn=dV.stride(2), stride_dvr=dV.stride(3),
            stride_dhb=dh_coevol.stride(0), stride_dhn=dh_coevol.stride(1),
            stride_dhd=dh_coevol.stride(2),
            stride_wb=wt_strides[0], stride_wi=wt_strides[1], stride_wj=wt_strides[2],
            stride_mb=mask.stride(0) if has_mask else 0,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
            S_CHUNK=S_CHUNK, D_CHUNK=D_CHUNK,
            HAS_MASK=has_mask,
        )

        return dU, dV, dh_coevol, dw_weight, db_weight, None, None, None


# ============================================================================
# Public API
# ============================================================================


def triton_coevol(
    U: torch.Tensor,
    V: torch.Tensor,
    h_coevol: torch.Tensor,
    w_weight: torch.Tensor,
    b_weight: torch.Tensor,
    mask: torch.Tensor | None = None,
    BLOCK_I: int = 32,
    BLOCK_J: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton co-evolution aggregation with batch support and autograd.

    Accepts both unbatched (S, N, R) and batched (B, S, N, R) inputs.
    When any input requires grad, caches w_tile for efficient backward.

    Returns:
        h_agg: (N, D) or (B, N, D) weighted aggregation
        c_bar: (N, R) or (B, N, R) co-evolution profile
    """
    unbatched = U.dim() == 3
    if unbatched:
        U = U.unsqueeze(0)
        V = V.unsqueeze(0)
        h_coevol = h_coevol.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    U = U.contiguous().float()
    V = V.contiguous().float()
    h_coevol = h_coevol.contiguous().float()
    w_weight = w_weight.contiguous().float()
    b_weight = b_weight.contiguous().float()
    if mask is not None:
        mask = mask.contiguous().float()

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in [U, V, h_coevol, w_weight, b_weight]
    )

    if needs_grad:
        h_agg, c_bar = CoevolFunction.apply(
            U, V, h_coevol, w_weight, b_weight, mask, BLOCK_I, BLOCK_J
        )
    else:
        h_agg, c_bar, _ = _launch_fwd(
            U, V, h_coevol, w_weight, b_weight, mask, BLOCK_I, BLOCK_J,
            save_w_tile=False,
        )

    if unbatched:
        return h_agg.squeeze(0), c_bar.squeeze(0)
    return h_agg, c_bar
