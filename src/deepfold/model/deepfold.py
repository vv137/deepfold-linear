"""DeepFold-Linear: Full model combining trunk + diffusion (SPEC §2, v4.5).

v4.5: h_res NOT frozen before diffusion — end-to-end gradient from L_diff,
L_lddt through atom blocks → h_cond → h_res → trunk. No UOT blocks in
diffusion. ~220M parameters.
"""

import torch
import torch.nn as nn

from deepfold.model.trunk import Trunk
from deepfold.model.init import init_model
from deepfold.model.position_encoding import compute_bins
from deepfold.model.input_embedding import AtomSingleEmbedding, AtomPairEmbedding
from deepfold.model.diffusion import (
    DiffusionModule,
    sample_training_sigma,
    karras_schedule,
    SIGMA_DATA,
    SIGMA_MAX,
)
from deepfold.model.losses import (
    DistogramLoss,
    _atom_type_weights,
    edm_diffusion_loss,
    smooth_lddt,
    total_loss,
)
from deepfold.data.augment import batch_augment


class DeepFoldLinear(nn.Module):
    """Full DeepFold-Linear model (~220M params, SPEC §14 v4.5)."""

    def __init__(
        self,
        d_model: int = 512,
        d_msa: int = 64,
        d_atom: int = 128,
        h_res: int = 16,
        h_msa: int = 8,
        n_msa_blocks: int = 4,
        n_uot_blocks: int = 48,
        n_atom_blocks: int = 10,
        sigma_data: float = SIGMA_DATA,
        max_cycles: int = 5,
        inference_cycles: int = 3,
        diffusion_multiplicity: int = 16,
        s_trans: float = 1.0,
        loss_weights: dict[str, float] | None = None,
    ):
        super().__init__()

        # Atom embeddings (shared by trunk and diffusion, SPEC §3.3, §3.4)
        self.atom_single_embed = AtomSingleEmbedding(d_atom=d_atom)
        self.atom_pair_embed = AtomPairEmbedding(d_pair=16)

        # Trunk
        self.trunk = Trunk(
            d_model=d_model,
            d_msa=d_msa,
            d_atom=d_atom,
            h_res=h_res,
            h_msa=h_msa,
            n_msa_blocks=n_msa_blocks,
            n_uot_blocks=n_uot_blocks,
            sigma_data=sigma_data,
            max_cycles=max_cycles,
            inference_cycles=inference_cycles,
        )

        # Diffusion (v4.5: no UOT blocks, no marginals)
        self.diffusion = DiffusionModule(
            d_model=d_model,
            d_atom=d_atom,
            n_atom_blocks=n_atom_blocks,
            sigma_data=sigma_data,
        )

        # Distogram loss module (has learnable parameters)
        self.distogram_loss = DistogramLoss(d_model=d_model)

        self.sigma_data = sigma_data
        self.diffusion_multiplicity = diffusion_multiplicity
        self.s_trans = s_trans
        self.loss_weights = loss_weights or {}

        # Apply SPEC §14 initialization
        init_model(self)

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
        x_atom_true: torch.Tensor | None = None,
        x_res_true: torch.Tensor | None = None,
        atom_resolved_mask: torch.Tensor | None = None,
        token_resolved_mask: torch.Tensor | None = None,
        token_pad_mask: torch.Tensor | None = None,
        atom_pad_mask: torch.Tensor | None = None,
        pair_valid_mask: torch.Tensor | None = None,
        msa_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: trunk -> diffusion (training) or trunk -> sampling (inference).

        Supports both unbatched (N, ...) and batched (B, N, ...) inputs.
        Mask tensors (token_pad_mask, atom_pad_mask, pair_valid_mask, msa_pad_mask)
        are floats with 1.0=valid, 0.0=padding. They are only needed for batched
        inputs (B>1) where samples have been padded to the same length.

        Returns dict with 'x_atom_pred' and losses if training.
        """
        device = token_type.device
        is_batched = token_type.dim() >= 2

        # Embed atom features once — shared by trunk and diffusion (SPEC §3.3, §3.4)
        # nn.Linear handles arbitrary leading dims; [...] slicing works for both
        # unbatched (N, D) and batched (B, N, D) inputs.
        c_atom = self.atom_single_embed(c_atom)
        p_lm = self.atom_pair_embed(p_lm[..., :3], p_lm[..., 4:5])

        # ---- Trunk ----
        h_res, mu, nu, x_res = self.trunk(
            token_type,
            profile,
            del_mean,
            has_msa,
            msa_feat,
            c_atom,
            p_lm,
            p_lm_idx,
            token_idx,
            chain_id,
            global_idx,
            bond_matrix,
            protein_mask,
            token_pad_mask=token_pad_mask,
            # msa_pad_mask from collate is (B, S, N_prot); MSA module expects (B, N_prot)
            msa_pad_mask=msa_pad_mask.amax(dim=1) if (
                msa_pad_mask is not None and msa_pad_mask.dim() == 3
            ) else msa_pad_mask,
        )

        result = {"h_res": h_res, "mu": mu, "nu": nu, "x_res": x_res}

        if self.training and x_atom_true is not None and x_res_true is not None:
            # ---- Training: M augmented diffusion samples (Boltz-style multiplicity) ----
            # Trunk runs once; diffusion runs M times with independent augmentations + σ.
            M = self.diffusion_multiplicity

            # Per-atom loss weights by mol_type (Boltz-1: nucleotide 5×, ligand 10×)
            atom_weights = _atom_type_weights(token_idx, token_type)

            # Generate M augmented copies of atom coords
            # Unbatched: (N_atom, 3) → (M, N_atom, 3)
            # Batched:   (B, N_atom, 3) → (M, B, N_atom, 3)
            x_atom_aug = batch_augment(
                x_atom_true, M, s_trans=self.s_trans, training=True
            )

            # Sample M noise levels (one per augmented sample)
            sigmas = sample_training_sigma(M, device)  # (M,)

            # Run M diffusion forwards, accumulate losses.
            # Each diffusion call is checkpointed individually to bound peak
            # memory. No DDP issue: the loop runs in forward, checkpoint
            # replays inside backward but DDP hooks are already resolved.
            l_diff_parts = []
            l_lddt_parts = []

            for i in range(M):
                x_true_i = x_atom_aug[i]  # (N_atom, 3) or (B, N_atom, 3)
                sigma_i = sigmas[i]

                noise = torch.randn_like(x_true_i)
                x_noisy_i = x_true_i + sigma_i * noise

                x_pred_i = self.diffusion(
                    h_res,
                    c_atom,
                    p_lm,
                    p_lm_idx,
                    x_noisy_i,
                    sigma_i,
                    token_idx,
                    token_pad_mask,
                    atom_pad_mask,
                    pair_valid_mask,
                )

                l_diff_parts.append(edm_diffusion_loss(
                    x_pred_i, x_true_i, sigma_i,
                    resolved_mask=atom_resolved_mask,
                    atom_weights=atom_weights,
                ))
                l_lddt_parts.append(smooth_lddt(
                    x_pred_i, x_true_i,
                    resolved_mask=atom_resolved_mask,
                ))

            # Average over M samples
            l_diff = torch.stack(l_diff_parts).mean()
            l_lddt = torch.stack(l_lddt_parts).mean()

            # Trunk-only losses (not multiplied — computed once)
            if is_batched:
                # Batched: use token_pad_mask for distogram
                l_disto = self.distogram_loss(
                    h_res, x_res_true, token_pad_mask=token_pad_mask
                )
            else:
                # Unbatched: build (N, N) pair mask from resolved mask
                if token_resolved_mask is not None:
                    disto_mask = token_resolved_mask[:, None] * token_resolved_mask[None, :]
                else:
                    disto_mask = None
                l_disto = self.distogram_loss(h_res, x_res_true, valid_mask=disto_mask)

            l_trunk_coord = smooth_lddt(
                x_res, x_res_true, resolved_mask=token_resolved_mask
            )
            l_total = total_loss(
                l_diff, l_lddt, l_disto, l_trunk_coord, **self.loss_weights
            )

            result.update(
                {
                    "x_atom_pred": x_pred_i,  # last sample for logging
                    "loss": l_total,
                    "l_diff": l_diff,
                    "l_lddt": l_lddt,
                    "l_disto": l_disto,
                    "l_trunk_coord": l_trunk_coord,
                }
            )

        return result

    @torch.no_grad()
    def sample(
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
        n_steps: int = 200,
    ) -> torch.Tensor:
        """Inference: trunk + Heun 2nd-order diffusion sampling (SPEC §12)."""
        device = token_type.device
        N_atom = token_idx.shape[0]

        # Embed atom features once (SPEC §3.3, §3.4)
        c_atom = self.atom_single_embed(c_atom)
        p_lm = self.atom_pair_embed(p_lm[:, :3], p_lm[:, 4:5])

        # Precompute position bins
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)

        # Trunk
        h_res, mu, nu, x_res = self.trunk(
            token_type,
            profile,
            del_mean,
            has_msa,
            msa_feat,
            c_atom,
            p_lm,
            p_lm_idx,
            token_idx,
            chain_id,
            global_idx,
            bond_matrix,
            protein_mask,
        )

        # Initialize from noise
        sigmas = karras_schedule(n_steps, device)
        x_atom = torch.randn(N_atom, 3, device=device) * SIGMA_MAX

        # Heun 2nd-order sampling (probability flow ODE: dx/dσ = (x - D(x;σ)) / σ)
        for i in range(n_steps - 1):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Denoise: D(x; σ) ≈ x_clean
            denoised = self.diffusion(
                h_res,
                c_atom,
                p_lm,
                p_lm_idx,
                x_atom,
                sigma_cur,
                token_idx,
            )
            d_cur = (x_atom - denoised) / sigma_cur

            # Euler step
            x_next = x_atom + (sigma_next - sigma_cur) * d_cur

            # Heun correction (except last step)
            if sigma_next > 0 and i < n_steps - 2:
                denoised_next = self.diffusion(
                    h_res,
                    c_atom,
                    p_lm,
                    p_lm_idx,
                    x_next,
                    sigma_next,
                    token_idx,
                )
                d_next = (x_next - denoised_next) / sigma_next
                x_next = x_atom + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)

            x_atom = x_next

        return x_atom
