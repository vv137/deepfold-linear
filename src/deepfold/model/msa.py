"""MSA Module — 4 blocks (SPEC §6). ~2.8M params.

Operates on protein (and optionally RNA) tokens only.
Non-MSA tokens get no co-evolution signal and retain uniform UOT marginals.

Supports both unbatched and batched inputs via dual-mode pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU, LNLinear, zero_init_linear
from deepfold.model.position_encoding import PositionBias

try:
    from deepfold.model.kernels.coevol_kernel import triton_coevol

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class MSABlock(nn.Module):
    """Single MSA block (SPEC §6.2)."""

    def __init__(
        self,
        d_model: int = 512,
        d_msa: int = 64,
        h_msa: int = 8,
        h_res: int = 16,
        coevol_rank: int = 16,
        tile_size: int = 64,
    ):
        super().__init__()
        self.d_msa = d_msa
        self.h_msa = h_msa
        self.h_res = h_res
        self.d_head = d_msa // h_msa  # 8
        self.coevol_rank = coevol_rank
        self.tile_size = tile_size

        # 1. Single -> MSA injection
        self.single_to_msa = nn.Linear(d_model, d_msa)

        # 2. Row-wise attention — after ln_row, so bias=False (SPEC §0)
        self.ln_row = nn.LayerNorm(d_msa)
        self.w_q = nn.Linear(d_msa, d_msa, bias=False)
        self.w_k = nn.Linear(d_msa, d_msa, bias=False)
        self.w_v = nn.Linear(d_msa, d_msa, bias=False)
        self.w_g = nn.Linear(d_msa, d_msa, bias=False)
        self.w_o_row = nn.Linear(d_msa, d_msa, bias=False)

        # 3. Column weighted mean — after ln_col, so bias=False
        self.ln_col = nn.LayerNorm(d_msa)
        self.col_weight = nn.Linear(d_msa, 1, bias=False)
        self.col_to_single = nn.Linear(d_msa, d_model, bias=False)

        # 4. Low-rank co-evolution — after ln_coevol, so bias=False
        self.ln_coevol = nn.LayerNorm(d_msa)
        self.u_proj = nn.Linear(d_msa, coevol_rank, bias=False)
        self.v_proj = nn.Linear(d_msa, coevol_rank, bias=False)
        self.coevol_value = LNLinear(d_model, d_model)
        self.coevol_weight = nn.Linear(coevol_rank, 1)
        self.coevol_out = nn.Linear(d_model, d_model)

        # 5. Marginal update — zeros init (SPEC §14)
        self.mu_proj = zero_init_linear(d_model, h_res)
        self.nu_proj = zero_init_linear(d_model, h_res)
        self.coevol_to_marginal = nn.Linear(coevol_rank, h_res)

        # 6. SwiGLU transition
        self.ln_ff = nn.LayerNorm(d_msa)
        self.swiglu = SwiGLU(d_msa, d_msa * 4, d_msa)

    def forward(
        self,
        m: torch.Tensor,
        h_res: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        protein_mask: torch.Tensor,
        pos_bias: torch.Tensor,
        alpha_coevol: torch.Tensor,
        msa_pad_mask: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:            (S, N_prot, d_msa) or (B, S, N_prot, d_msa) MSA representation
            h_res:        (N, d_model) or (B, N, d_model) all tokens
            mu:           (H_res, N) or (B, H_res, N) row marginals
            nu:           (H_res, N) or (B, H_res, N) column marginals
            protein_mask: (N,) or (B, N) bool — True for tokens with MSA data
            pos_bias:     (H_msa, N_prot, N_prot) or (B, H_msa, N_prot, N_prot) position bias
            alpha_coevol: (H_res,) per-head coevol coefficient for this block
            msa_pad_mask: (B, N_prot) bool/float, 1=real 0=pad. Only for batched.
            training:     bool

        Returns:
            m, h_res, mu, nu
        """
        # Dual-mode: detect unbatched and add B dim
        unbatched = h_res.dim() == 2
        if unbatched:
            m = m.unsqueeze(0)
            h_res = h_res.unsqueeze(0)
            mu = mu.unsqueeze(0)
            nu = nu.unsqueeze(0)
            protein_mask = protein_mask.unsqueeze(0)
            pos_bias = pos_bias.unsqueeze(0)

        B, S, N_prot, _ = m.shape
        N = h_res.shape[1]
        H = self.h_msa
        d_h = self.d_head
        device = h_res.device

        if msa_pad_mask is None:
            msa_pad_mask = m.new_ones(B, N_prot)

        # Skip MSA processing if no protein/RNA tokens.
        if N_prot == 0:
            result_m, result_h = m, h_res + 0
            if unbatched:
                return (
                    result_m.squeeze(0),
                    result_h.squeeze(0),
                    mu.squeeze(0),
                    nu.squeeze(0),
                )
            return result_m, result_h, mu, nu

        # Precompute protein indices once for gather/scatter operations
        prot_indices = _build_protein_indices(protein_mask, N_prot)  # (B, N_prot)

        # ---- 1. Single -> MSA injection ----
        h_proj = self.single_to_msa(h_res)  # (B, N, d_msa)
        h_prot = _gather_at_indices(h_proj, prot_indices)  # (B, N_prot, d_msa)
        m = m + h_prot.unsqueeze(1)  # (B, 1, N_prot, d_msa) broadcast over S

        # ---- 2. Row-wise attention ----
        m_n = self.ln_row(m)  # (B, S, N_prot, d_msa)
        Q = self.w_q(m_n).view(B, S, N_prot, H, d_h)
        K = self.w_k(m_n).view(B, S, N_prot, H, d_h)
        V = self.w_v(m_n).view(B, S, N_prot, H, d_h)
        G = self.w_g(m_n).view(B, S, N_prot, H, d_h)

        # (B, S, H, N_prot, N_prot)
        scores = torch.einsum("bsihd,bsjhd->bshij", Q, K) / (d_h**0.5)
        scores = scores + pos_bias.unsqueeze(1)  # (B, 1, H, N_prot, N_prot)

        # Mask padded MSA positions
        msa_mask_f = msa_pad_mask.float()  # (B, N_prot)
        col_mask_bias = (1 - msa_mask_f)[:, None, None, None, :] * (-1e9)
        scores = scores + col_mask_bias

        attn = F.softmax(scores, dim=-1)
        att_out = torch.einsum("bshij,bsjhd->bsihd", attn, V)  # (B, S, N_prot, H, d_h)
        att_out = att_out.reshape(B, S, N_prot, -1)  # (B, S, N_prot, d_msa)
        G_flat = G.reshape(B, S, N_prot, -1)
        m = m + torch.sigmoid(G_flat) * self.w_o_row(att_out)

        # ---- 3. Column weighted mean ----
        m_n = self.ln_col(m)  # (B, S, N_prot, d_msa)
        alpha = F.softmax(
            self.col_weight(m_n), dim=1
        )  # (B, S, N_prot, 1) softmax over S
        col_agg = (alpha * m_n).sum(dim=1)  # (B, N_prot, d_msa)
        h_prot_update = self.col_to_single(col_agg)  # (B, N_prot, d_model)

        h_res = h_res.clone()
        _scatter_add_at_indices(h_res, h_prot_update, prot_indices)

        # ---- 4. Low-rank co-evolution (tiled) ----
        m_n = self.ln_coevol(m)  # (B, S, N_prot, d_msa)
        U = self.u_proj(m_n)  # (B, S, N_prot, r)
        V_ = self.v_proj(m_n)  # (B, S, N_prot, r)

        h_coevol = self.coevol_value(h_res)  # (B, N, d_model)

        # Triton path: CUDA only, supports both training (autograd) and inference
        use_triton = _HAS_TRITON and h_res.is_cuda

        h_agg = torch.zeros_like(h_res)  # (B, N, d_model)
        c_bar_accum = torch.zeros(
            B, N, self.coevol_rank, device=device, dtype=h_res.dtype
        )

        if use_triton:
            h_coevol_prot = _gather_at_indices(h_coevol, prot_indices)  # (B, N_prot, D)

            h_agg_prot, c_bar_prot = triton_coevol(
                U,  # (B, S, N_prot, R)
                V_,  # (B, S, N_prot, R)
                h_coevol_prot,  # (B, N_prot, D)
                self.coevol_weight.weight.squeeze(
                    0
                ),  # (R,) — Lin(R->1).weight is (1,R)
                self.coevol_weight.bias,  # (1,)
                mask=msa_pad_mask,  # (B, N_prot) or None
            )
            # Scatter protein results back to full token dimension
            _scatter_add_at_indices(h_agg, h_agg_prot.to(h_agg.dtype), prot_indices)
            _scatter_add_at_indices(
                c_bar_accum, c_bar_prot.to(c_bar_accum.dtype), prot_indices
            )
        else:
            TILE = self.tile_size
            for i0 in range(0, N_prot, TILE):
                ie = min(i0 + TILE, N_prot)
                U_i = U[:, :, i0:ie, :]  # (B, S, ti, r)

                for j0 in range(0, N_prot, TILE):
                    je = min(j0 + TILE, N_prot)
                    V_j = V_[:, :, j0:je, :]  # (B, S, tj, r)

                    c_tile = (
                        torch.einsum("bsir,bsjr->bijr", U_i, V_j) / S
                    )  # (B, ti, tj, r)

                    w_tile = torch.sigmoid(
                        self.coevol_weight(c_tile).squeeze(-1)
                    )  # (B, ti, tj)

                    # Mask padded positions in j dimension
                    w_tile = w_tile * msa_pad_mask[:, j0:je].unsqueeze(1)

                    h_j = _gather_at_indices(
                        h_coevol, prot_indices[:, j0:je]
                    )  # (B, tj, d_model)
                    tile_out = torch.bmm(w_tile, h_j)  # (B, ti, d_model)
                    _scatter_add_at_indices(h_agg, tile_out, prot_indices[:, i0:ie])

                    c_bar_tile = c_tile.sum(dim=2)  # (B, ti, r)
                    c_bar_tile = c_bar_tile * msa_pad_mask[:, i0:ie].unsqueeze(-1)
                    _scatter_add_at_indices(
                        c_bar_accum, c_bar_tile, prot_indices[:, i0:ie]
                    )

        h_res = h_res + self.coevol_out(h_agg / max(N_prot, 1))
        c_bar = c_bar_accum / max(N_prot, 1)  # (B, N, r)

        # ---- 5. Marginal update ----
        mu_logit = self.mu_proj(h_res)  # (B, N, H_res)
        nu_logit = self.nu_proj(h_res)  # (B, N, H_res)

        # Co-evolution bias masked to protein tokens
        coevol_bias = (
            self.coevol_to_marginal(c_bar) * protein_mask.unsqueeze(-1).float()
        )
        bias_t = (alpha_coevol[None, None, :] * coevol_bias).permute(
            0, 2, 1
        )  # (B, H_res, N)

        mu_new = F.softmax(mu_logit.permute(0, 2, 1) + bias_t, dim=-1)  # (B, H_res, N)
        nu_new = F.softmax(nu_logit.permute(0, 2, 1) + bias_t, dim=-1)  # (B, H_res, N)

        # ---- 6. SwiGLU transition ----
        m = m + self.swiglu(self.ln_ff(m))

        # ---- 7. Row dropout ----
        if training:
            drop_mask = torch.bernoulli(
                torch.full((B, S, 1, 1), 0.85, device=device, dtype=m.dtype)
            )
            m = m * drop_mask / 0.85

        if unbatched:
            return m.squeeze(0), h_res.squeeze(0), mu_new.squeeze(0), nu_new.squeeze(0)
        return m, h_res, mu_new, nu_new


def _build_protein_indices(protein_mask: torch.Tensor, N_prot: int) -> torch.Tensor:
    """Build (B, N_prot) index tensor of protein positions per batch element.

    Uses stable argsort to put True positions first, preserving relative order.
    Padded with index 0 if fewer than N_prot True positions exist.

    Args:
        protein_mask: (B, N) bool
        N_prot: int

    Returns:
        (B, N_prot) long tensor of indices
    """
    sorted_idx = torch.argsort(~protein_mask.bool(), dim=1, stable=True)  # (B, N)
    return sorted_idx[:, :N_prot].contiguous()


def _gather_at_indices(h: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather from h at given indices along dim=1.

    Args:
        h: (B, N, D)
        indices: (B, K) long

    Returns:
        (B, K, D)
    """
    return torch.gather(h, 1, indices.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


def _scatter_add_at_indices(
    target: torch.Tensor, source: torch.Tensor, indices: torch.Tensor
) -> None:
    """Scatter-add source into target at given indices along dim=1 (in-place).

    Args:
        target: (B, N, D)
        source: (B, K, D)
        indices: (B, K) long
    """
    target.scatter_add_(
        1, indices.unsqueeze(-1).expand_as(source), source.to(target.dtype)
    )


class MSAModule(nn.Module):
    """Stack of 4 MSA blocks (SPEC §6)."""

    def __init__(
        self,
        n_blocks: int = 4,
        d_model: int = 512,
        d_msa: int = 64,
        h_msa: int = 8,
        h_res: int = 16,
        coevol_rank: int = 16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MSABlock(d_model, d_msa, h_msa, h_res, coevol_rank)
                for _ in range(n_blocks)
            ]
        )

        # Per-block, per-head alpha_coevol — zeros init (SPEC §6.2 step 5)
        self.alpha_coevol = nn.Parameter(torch.zeros(n_blocks, h_res))

        # MSA position bias
        self.pos_bias = PositionBias(h_msa, 68)

    def forward(
        self,
        m: torch.Tensor,
        h_res: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        protein_mask: torch.Tensor,
        msa_bins: torch.Tensor,
        msa_pad_mask: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:            (S, N_prot, 64) or (B, S, N_prot, 64)
            h_res:        (N, 512) or (B, N, 512)
            mu:           (H_res, N) or (B, H_res, N)
            nu:           (H_res, N) or (B, H_res, N)
            protein_mask: (N,) or (B, N) bool
            msa_bins:     (N_prot, N_prot) or (B, N_prot, N_prot) int
            msa_pad_mask: (B, N_prot) bool/float or None
            training:     bool

        Returns:
            m, h_res, mu, nu
        """
        pos_bias = self.pos_bias(msa_bins)

        for i, block in enumerate(self.blocks):
            m, h_res, mu, nu = block(
                m,
                h_res,
                mu,
                nu,
                protein_mask=protein_mask,
                pos_bias=pos_bias,
                alpha_coevol=self.alpha_coevol[i],
                msa_pad_mask=msa_pad_mask,
                training=training,
            )

        return m, h_res, mu, nu
