"""MSA Module — 4 blocks (SPEC §6). ~2.8M params.

Operates on protein (and optionally RNA) tokens only.
Non-MSA tokens get no co-evolution signal and retain uniform UOT marginals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU, LNLinear, zero_init_linear
from deepfold.model.position_encoding import PositionBias


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
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:            (S, N_prot, d_msa) MSA representation
            h_res:        (N, d_model) all tokens
            mu:           (H_res, N) row marginals
            nu:           (H_res, N) column marginals
            protein_mask: (N,) bool — True for tokens with MSA data
            pos_bias:     (H_msa, N_prot, N_prot) position bias for MSA attention
            alpha_coevol: (H_res,) per-head coevol coefficient for this block
            training:     bool

        Returns:
            m, h_res, mu, nu
        """
        S, N_prot, _ = m.shape
        N = h_res.shape[0]
        H = self.h_msa
        d_h = self.d_head
        device = h_res.device

        # Skip MSA processing if no protein/RNA tokens.
        # Add zero to h_res to keep it in the autograd graph.
        if N_prot == 0:
            return m, h_res + 0, mu, nu

        # ---- 1. Single -> MSA injection ----
        h_prot = h_res[protein_mask]  # (N_prot, d_model)
        m = m + self.single_to_msa(h_prot).unsqueeze(0)  # broadcast (1, N_prot, d_msa)

        # ---- 2. Row-wise attention ----
        m_n = self.ln_row(m)  # (S, N_prot, d_msa)
        Q = self.w_q(m_n).view(S, N_prot, H, d_h)
        K = self.w_k(m_n).view(S, N_prot, H, d_h)
        V = self.w_v(m_n).view(S, N_prot, H, d_h)
        G = self.w_g(m_n).view(S, N_prot, H, d_h)

        # (S, H, N_prot, N_prot)
        scores = torch.einsum("sihd,sjhd->shij", Q, K) / (d_h**0.5)
        scores = scores + pos_bias.unsqueeze(0)  # (1, H, N_prot, N_prot)

        attn = F.softmax(scores, dim=-1)
        att_out = torch.einsum("shij,sjhd->sihd", attn, V)  # (S, N_prot, H, d_h)
        att_out = att_out.reshape(S, N_prot, -1)  # (S, N_prot, d_msa)
        G_flat = G.reshape(S, N_prot, -1)
        m = m + torch.sigmoid(G_flat) * self.w_o_row(att_out)

        # ---- 3. Column weighted mean ----
        m_n = self.ln_col(m)  # (S, N_prot, d_msa)
        alpha = F.softmax(self.col_weight(m_n), dim=0)  # (S, N_prot, 1) softmax over S
        col_agg = (alpha * m_n).sum(dim=0)  # (N_prot, d_msa)
        h_prot_update = self.col_to_single(col_agg)  # (N_prot, d_model)
        h_res = h_res.clone()
        h_res[protein_mask] = h_res[protein_mask] + h_prot_update

        # ---- 4. Low-rank co-evolution (tiled) ----
        m_n = self.ln_coevol(m)  # (S, N_prot, d_msa)
        U = self.u_proj(m_n)  # (S, N_prot, r)
        V_ = self.v_proj(m_n)  # (S, N_prot, r)

        h_coevol = self.coevol_value(h_res)  # (N, d_model)

        TILE = self.tile_size
        h_agg = torch.zeros_like(h_res)
        c_bar_accum = torch.zeros(N, self.coevol_rank, device=device, dtype=h_res.dtype)

        # Map protein indices to full token indices for tiled computation
        prot_indices = torch.where(protein_mask)[0]  # (N_prot,)

        for i0 in range(0, N_prot, TILE):
            ie = min(i0 + TILE, N_prot)
            U_i = U[:, i0:ie, :]  # (S, ti, r)

            for j0 in range(0, N_prot, TILE):
                je = min(j0 + TILE, N_prot)
                V_j = V_[:, j0:je, :]  # (S, tj, r)

                # Co-evolution vector: (ti, tj, r)
                c_tile = torch.einsum("sir,sjr->ijr", U_i, V_j) / S

                # Scalar weight for aggregation
                w_tile = torch.sigmoid(
                    self.coevol_weight(c_tile).squeeze(-1)
                )  # (ti, tj)

                # Map to full indices
                full_j_idx = prot_indices[j0:je]
                h_agg[prot_indices[i0:ie]] += w_tile @ h_coevol[full_j_idx]

                # r-dim profile accumulator
                c_bar_accum[prot_indices[i0:ie]] += c_tile.sum(dim=1)  # (ti, r)

        h_res = h_res + self.coevol_out(h_agg)
        c_bar = c_bar_accum / max(N_prot, 1)  # (N, r)

        # ---- 5. Marginal update ----
        mu_logit = self.mu_proj(h_res)  # (N, H_res)
        nu_logit = self.nu_proj(h_res)  # (N, H_res)

        # Co-evolution bias (protein tokens only)
        coevol_bias = torch.zeros(N, self.h_res, device=device, dtype=h_res.dtype)
        coevol_bias[protein_mask] = self.coevol_to_marginal(c_bar[protein_mask])

        # alpha_coevol gating: (H_res,) * (H_res, N)
        bias = alpha_coevol.unsqueeze(-1) * coevol_bias.T  # (H_res, N)

        mu_new = F.softmax(mu_logit.T + bias, dim=-1)  # (H_res, N)
        nu_new = F.softmax(nu_logit.T + bias, dim=-1)  # (H_res, N)

        # ---- 6. SwiGLU transition ----
        m_n = self.ln_ff(m)
        m = m + self.swiglu(m_n)

        # ---- 7. Row dropout ----
        if training:
            mask = torch.bernoulli(
                torch.full((S, 1, 1), 0.85, device=device, dtype=m.dtype)
            )
            m = m * mask / 0.85

        return m, h_res, mu_new, nu_new


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
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:            (S, N_prot, 64)
            h_res:        (N, 512)
            mu:           (H_res, N)
            nu:           (H_res, N)
            protein_mask: (N,) bool
            msa_bins:     (N_prot, N_prot) int, position bins for MSA tokens
            training:     bool

        Returns:
            m, h_res, mu, nu
        """
        pos_bias = self.pos_bias(msa_bins)  # (H_msa, N_prot, N_prot)

        for i, block in enumerate(self.blocks):
            m, h_res, mu, nu = block(
                m,
                h_res,
                mu,
                nu,
                protein_mask=protein_mask,
                pos_bias=pos_bias,
                alpha_coevol=self.alpha_coevol[i],
                training=training,
            )

        return m, h_res, mu, nu
