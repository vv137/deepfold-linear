"""Trunk / Recycling Loop (SPEC §5). Manages MSA + Token UOT+EGNN blocks.

Supports both unbatched and batched inputs via dual-mode pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.msa import MSAModule
from deepfold.model.primitives import zero_init_linear
from deepfold.model.trunk_block import TokenUOTBlock
from deepfold.model.position_encoding import compute_bins
from deepfold.model.input_embedding import (
    TokenSingleEmbedding,
    MSAEmbedding,
)
from deepfold.model.atom_encoder import AtomToTokenEncoder


class Trunk(nn.Module):
    """Full trunk: input embedding + recycling loop (SPEC §5.2)."""

    def __init__(
        self,
        d_model: int = 512,
        d_msa: int = 64,
        d_atom: int = 128,
        h_res: int = 16,
        h_msa: int = 8,
        n_msa_blocks: int = 4,
        n_uot_blocks: int = 48,
        sigma_data: float = 16.0,
        max_cycles: int = 5,
        inference_cycles: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.h_res = h_res
        self.sigma_data = sigma_data
        self.max_cycles = max_cycles
        self.inference_cycles = inference_cycles

        # Input embeddings
        self.token_embed = TokenSingleEmbedding(d_model)
        self.msa_embed = MSAEmbedding(d_msa)
        self.atom_encoder = AtomToTokenEncoder(d_atom, d_model)

        # MSA module
        self.msa_module = MSAModule(
            n_blocks=n_msa_blocks,
            d_model=d_model,
            d_msa=d_msa,
            h_msa=h_msa,
            h_res=h_res,
        )

        # Token UOT+EGNN blocks
        self.uot_blocks = nn.ModuleList(
            [
                TokenUOTBlock(d_model=d_model, n_heads=h_res, block_idx=i)
                for i in range(n_uot_blocks)
            ]
        )

        # Shared marginal projections — per-block mu/nu from current h_res
        # LN stabilizes input scale across 48 blocks of varying depth
        # bias=False after LN (SPEC §0); zero-init weight: uniform marginals at start
        self.ln_mu = nn.LayerNorm(d_model)
        self.mu_proj = nn.Linear(d_model, h_res, bias=False)
        self.nu_proj = nn.Linear(d_model, h_res, bias=False)

    def forward(
        self,
        token_type: torch.Tensor,
        profile: torch.Tensor,
        del_mean: torch.Tensor,
        has_msa: torch.Tensor,
        msa_feat: torch.Tensor,
        c_atom: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        token_idx: torch.Tensor,
        chain_id: torch.Tensor,
        global_idx: torch.Tensor,
        bond_matrix: torch.Tensor,
        msa_token_mask: torch.Tensor,
        token_pad_mask: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
        num_cycles: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_type:     (N,) or (B, N) int
            profile:        (N, 32) or (B, N, 32)
            del_mean:       (N, 1) or (B, N, 1)
            has_msa:        (N, 1) or (B, N, 1)
            msa_feat:       (C, S, N_prot, 34) or (B, C, S, N_prot, 34)
            c_atom:         (N_atom, 128) or (B, N_atom, 128)
            p_lm:           (n_pairs, 16) or (B, n_pairs, 16)
            p_lm_idx:       (n_pairs, 2) or (B, n_pairs, 2)
            token_idx:      (N_atom,) or (B, N_atom)
            chain_id:       (N,) or (B, N) int
            global_idx:     (N,) or (B, N) int
            bond_matrix:    (N, N) or (B, N, N) bool
            msa_token_mask:   (N,) or (B, N) bool
            token_pad_mask: (B, N) bool/float or None (1=real, 0=pad)
            msa_mask:       (B, C, S, N_prot) float or None (1=real, 0=pad)
            num_cycles:     int, number of recycling cycles to run.

        Returns:
            h_res:  (N, 512) or (B, N, 512)
            mu:     (H, N) or (B, H, N)
            nu:     (H, N) or (B, H, N)
            x_res:  (N, 3) or (B, N, 3)
        """
        # Dual-mode: detect unbatched
        unbatched = token_type.dim() == 1
        if unbatched:
            token_type = token_type.unsqueeze(0)
            profile = profile.unsqueeze(0)
            del_mean = del_mean.unsqueeze(0)
            has_msa = has_msa.unsqueeze(0)
            msa_feat = msa_feat.unsqueeze(0)
            c_atom = c_atom.unsqueeze(0)
            p_lm = p_lm.unsqueeze(0)
            p_lm_idx = p_lm_idx.unsqueeze(0)
            token_idx = token_idx.unsqueeze(0)
            chain_id = chain_id.unsqueeze(0)
            global_idx = global_idx.unsqueeze(0)
            bond_matrix = bond_matrix.unsqueeze(0)
            msa_token_mask = msa_token_mask.unsqueeze(0)

        device = token_type.device
        B, N = token_type.shape
        token_idx.shape[1]

        if token_pad_mask is None:
            token_pad_mask = token_type.new_ones(B, N).float()
        token_pad_mask_f = token_pad_mask.float()
        # Precompute frequently used mask shapes
        mask_3d = token_pad_mask_f.unsqueeze(-1)  # (B, N, 1)
        mask_sum = token_pad_mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

        # ---- Input embedding (SPEC §3) ----
        h_res = self.token_embed(token_type, profile, del_mean, has_msa)  # (B, N, 512)
        # msa_feat: (B, C, S, N_prot, 34) — per-cycle MSA, embedded inside cycle loop

        # Atom-to-token encoder (runs once, SPEC §3.5)
        # Per-sample loop: atom encoder uses scatter_mean with sample-specific indices
        atom_agg = torch.stack(
            [
                self.atom_encoder(c_atom[b], p_lm[b], p_lm_idx[b], token_idx[b], N)
                for b in range(B)
            ],
            dim=0,
        )  # (B, N, d_model)
        h_res = h_res + atom_agg

        # Initial coordinates: randn * sigma_data, centered per sample (SPEC §5.2)
        x_res = torch.randn(B, N, 3, device=device) * self.sigma_data * mask_3d
        x_res = x_res - (x_res * mask_3d).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(
            -1
        )
        x_res = x_res * mask_3d

        # Precompute position bins
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)  # (B, N, N)
        # msa_feat: (B, C, S, N_prot, 34) or (C, S, N_prot, 34)
        N_prot = msa_feat.shape[-2]
        if N_prot > 0:
            msa_bins = _compute_msa_bins_batched(
                chain_id, global_idx, bond_matrix, msa_token_mask, N_prot
            )
        else:
            msa_bins = torch.zeros(B, 0, 0, device=device, dtype=torch.long)

        # Initialize log dual variables
        H = self.h_res
        log_u_carry = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        log_v_carry = torch.zeros(B, H, N, device=device, dtype=torch.float32)

        for cycle in range(num_cycles):
            is_last = cycle == num_cycles - 1

            # ---- Per-cycle MSA embedding (fresh each cycle) ----
            n_msa_cycles = msa_feat.shape[-4]  # C dimension
            cycle_msa_feat = msa_feat[:, cycle % n_msa_cycles]  # (B, S, N_prot, 34)
            m = self.msa_embed(cycle_msa_feat)  # (B, S, N_prot, 64)

            # Per-cycle MSA mask
            cycle_msa_mask = None
            if msa_mask is not None:
                cycle_msa_mask = msa_mask[:, cycle % n_msa_cycles]  # (B, S, N_prot)

            # ---- MSA module: 4 blocks + coevol bias (SPEC §6) ----
            if is_last:
                m, h_res, coevol_bias = self.msa_module(
                    m,
                    h_res,
                    msa_token_mask,
                    msa_bins,
                    msa_mask=cycle_msa_mask,
                    training=self.training,
                )
            else:
                with torch.no_grad():
                    m, h_res, coevol_bias = self.msa_module(
                        m,
                        h_res,
                        msa_token_mask,
                        msa_bins,
                        msa_mask=cycle_msa_mask,
                        training=self.training,
                    )

            # ---- Token UOT+EGNN blocks x48 (SPEC §8) ----
            # Per-block marginals from current h_res + coevol_bias
            log_u_prev = log_u_carry
            log_v_prev = log_v_carry

            for block in self.uot_blocks:
                # Compute per-block marginals from current h_res
                h_n = self.ln_mu(h_res)
                mu_logit = self.mu_proj(h_n).permute(0, 2, 1)  # (B, H, N)
                nu_logit = self.nu_proj(h_n).permute(0, 2, 1)  # (B, H, N)
                mu = F.softmax(mu_logit + coevol_bias, dim=-1)
                nu = F.softmax(nu_logit + coevol_bias, dim=-1)
                mu = mu * token_pad_mask_f.unsqueeze(1)
                nu = nu * token_pad_mask_f.unsqueeze(1)

                if is_last:
                    h_res, x_res, log_u_prev, log_v_prev = block(
                        h_res,
                        x_res,
                        mu,
                        nu,
                        log_u_prev,
                        log_v_prev,
                        pos_bins,
                        mask=token_pad_mask,
                    )
                else:
                    with torch.no_grad():
                        h_res, x_res, log_u_prev, log_v_prev = block(
                            h_res,
                            x_res,
                            mu,
                            nu,
                            log_u_prev,
                            log_v_prev,
                            pos_bins,
                            mask=token_pad_mask,
                        )

            # Re-center x_res per sample (once per cycle, SPEC §5.2)
            x_sum = (x_res.float() * mask_3d).sum(dim=1, keepdim=True)
            x_res = x_res - (x_sum / mask_sum.unsqueeze(-1)).to(x_res.dtype)
            x_res = x_res * mask_3d

            if not is_last:
                h_res = h_res.detach()
                x_res = x_res.detach()
                log_u_carry = log_u_prev.detach() if log_u_prev is not None else None
                log_v_carry = log_v_prev.detach() if log_v_prev is not None else None
            else:
                log_u_carry = log_u_prev
                log_v_carry = log_v_prev

        if unbatched:
            return h_res.squeeze(0), mu.squeeze(0), nu.squeeze(0), x_res.squeeze(0)
        return h_res, mu, nu, x_res


def _compute_msa_bins_batched(
    chain_id: torch.Tensor,
    global_idx: torch.Tensor,
    bond_matrix: torch.Tensor,
    msa_token_mask: torch.Tensor,
    N_prot: int,
) -> torch.Tensor:
    """Compute MSA position bins for protein subset, batched.

    Args:
        chain_id: (B, N)
        global_idx: (B, N)
        bond_matrix: (B, N, N)
        msa_token_mask: (B, N) bool
        N_prot: int

    Returns:
        (B, N_prot, N_prot) int position bins
    """
    from deepfold.model.msa import _build_protein_indices

    B, N = chain_id.shape

    prot_indices = _build_protein_indices(msa_token_mask, N_prot)  # (B, N_prot)

    # Gather protein chain_id and global_idx
    prot_chain = torch.gather(chain_id, 1, prot_indices)  # (B, N_prot)
    prot_global = torch.gather(global_idx, 1, prot_indices)  # (B, N_prot)

    # Gather protein bond_matrix: (B, N_prot, N_prot)
    # First gather rows, then columns
    idx_row = prot_indices.unsqueeze(2).expand(B, N_prot, N)  # (B, N_prot, N)
    prot_bond_rows = torch.gather(bond_matrix, 1, idx_row)  # (B, N_prot, N)
    idx_col = prot_indices.unsqueeze(1).expand(B, N_prot, N_prot)  # (B, N_prot, N_prot)
    prot_bond = torch.gather(prot_bond_rows, 2, idx_col)  # (B, N_prot, N_prot)

    return compute_bins(prot_chain, prot_global, prot_bond)
