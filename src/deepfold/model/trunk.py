"""Trunk / Recycling Loop (SPEC §5). Manages MSA + Token UOT+EGNN blocks."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from deepfold.model.msa import MSAModule
from deepfold.model.trunk_block import TokenUOTBlock
from deepfold.model.position_encoding import PositionBias, compute_bins
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

        # Shared position bias for UOT blocks (SPEC §4.3)
        self.pos_bias = PositionBias(h_res, 68)

        # Per-head geometry weight — zeros init (SPEC §7.1)
        self.w_dist = nn.Parameter(torch.zeros(h_res))

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
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_type:   (N,) int
            profile:      (N, 32)
            del_mean:     (N, 1)
            has_msa:      (N, 1)
            msa_feat:     (S, N_prot, 34)
            c_atom:       (N_atom, 128) embedded atom features
            p_lm:         (n_pairs, 16) embedded atom pair features
            p_lm_idx:     (n_pairs, 2)
            token_idx:    (N_atom,) maps atoms to tokens
            chain_id:     (N,) int
            global_idx:   (N,) int
            bond_matrix:  (N, N) bool
            protein_mask: (N,) bool

        Returns:
            h_res:  (N, 512)
            mu:     (H, N)
            nu:     (H, N)
            x_res:  (N, 3)
        """
        device = token_type.device
        N = token_type.shape[0]
        N_atom = token_idx.shape[0]

        # ---- Input embedding (SPEC §3) ----
        h_res = self.token_embed(token_type, profile, del_mean, has_msa)
        m = self.msa_embed(msa_feat)

        # Atom-to-token encoder (runs once, SPEC §3.5)
        # c_atom and p_lm are already embedded by DeepFoldLinear.forward()
        atom_agg = self.atom_encoder(c_atom, p_lm, p_lm_idx, token_idx, N)
        h_res = h_res + atom_agg

        # Initial marginals
        mu = torch.ones(self.h_res, N, device=device) / N
        nu = torch.ones(self.h_res, N, device=device) / N

        # Initial coordinates: randn * sigma_data, centered (SPEC §5.2)
        x_res = torch.randn(N, 3, device=device) * self.sigma_data
        x_res = x_res - x_res.mean(dim=0, keepdim=True)

        # Precompute position bins
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)  # (N, N)

        # MSA position bins (protein subset)
        prot_indices = torch.where(protein_mask)[0]
        prot_chain_id = chain_id[prot_indices]
        prot_global_idx = global_idx[prot_indices]
        prot_bond = bond_matrix[prot_indices][:, prot_indices]
        msa_bins = compute_bins(prot_chain_id, prot_global_idx, prot_bond)

        # Precompute UOT position bias
        uot_pos_bias = self.pos_bias(pos_bins)  # (H, N, N)

        # ---- Sample cycle count (SPEC §5.3) ----
        if self.training:
            num_cycles = random.randint(1, self.max_cycles)
        else:
            num_cycles = self.inference_cycles

        # Initialize log dual variables as zeros (not None) so the code path
        # through Sinkhorn is deterministic — required for gradient checkpointing.
        H = self.h_res
        log_u_carry = torch.zeros(H, N, device=device, dtype=torch.float32)
        log_v_carry = torch.zeros(H, N, device=device, dtype=torch.float32)

        for cycle in range(num_cycles):
            is_last = cycle == num_cycles - 1

            # ---- MSA blocks x4 (SPEC §6) ----
            if is_last:
                m, h_res, mu_new, nu_new = self.msa_module(
                    m, h_res, mu, nu, protein_mask, msa_bins, training=self.training
                )
            else:
                with torch.no_grad():
                    m, h_res, mu_new, nu_new = self.msa_module(
                        m, h_res, mu, nu, protein_mask, msa_bins, training=self.training
                    )

            # Marginal carry: log-space geometric blend (SPEC §5.2)
            if cycle > 0:
                mu = F.softmax(
                    torch.log(mu_new + 1e-8) + 0.5 * torch.log(mu + 1e-8), dim=-1
                )
                nu = F.softmax(
                    torch.log(nu_new + 1e-8) + 0.5 * torch.log(nu + 1e-8), dim=-1
                )
            else:
                mu, nu = mu_new, nu_new

            # ---- Token UOT+EGNN blocks x48 (SPEC §8) ----
            log_u_prev = log_u_carry
            log_v_prev = log_v_carry

            for block in self.uot_blocks:
                if is_last:
                    h_res, x_res, log_u_prev, log_v_prev = checkpoint(
                        block,
                        h_res,
                        x_res,
                        mu,
                        nu,
                        log_u_prev,
                        log_v_prev,
                        uot_pos_bias,
                        self.w_dist,
                        pos_bins,
                        use_reentrant=True,
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
                            uot_pos_bias,
                            self.w_dist,
                            pos_bins,
                        )

            # Re-center x_res (once per cycle, SPEC §5.2)
            x_res = x_res - x_res.float().mean(dim=0, keepdim=True).to(x_res.dtype)

            if not is_last:
                h_res = h_res.detach()
                x_res = x_res.detach()
                mu = mu.detach()
                nu = nu.detach()
                log_u_carry = log_u_prev.detach() if log_u_prev is not None else None
                log_v_carry = log_v_prev.detach() if log_v_prev is not None else None
            else:
                log_u_carry = log_u_prev
                log_v_carry = log_v_prev

        return h_res, mu, nu, x_res
