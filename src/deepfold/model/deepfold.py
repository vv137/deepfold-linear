"""DeepFold-Linear: Full model combining trunk + diffusion (SPEC §2, v5).

v5: Boltz-1-style diffusion with encoder-transformer-decoder.
24-layer DiffusionTransformer with 68-bin position bias (no pair repr).
Triton kernels for O(N) memory. Proper c_skip EDM preconditioning.
~375M parameters (~220M trunk + ~155M diffusion).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from deepfold.model.trunk import Trunk
from deepfold.model.init import init_model
from deepfold.model.position_encoding import compute_bins
from deepfold.model.input_embedding import AtomSingleEmbedding, AtomPairEmbedding
from deepfold.model.diffusion import (
    sample_training_sigma,
    karras_schedule,
    SIGMA_DATA,
    SIGMA_MAX,
)
from deepfold.model.diffusion_v2 import DiffusionModuleV2
from deepfold.model.losses import (
    DistogramLoss,
    _atom_type_weights,
    edm_diffusion_loss,
    smooth_lddt,
    total_loss,
)
from deepfold.data.augment import batch_augment
from deepfold.data import const


class DeepFoldLinear(nn.Module):
    """Full DeepFold-Linear model (~375M params, SPEC §14 v5)."""

    def __init__(
        self,
        d_model: int = 512,
        d_msa: int = 64,
        d_atom: int = 128,
        h_res: int = 16,
        h_msa: int = 8,
        n_msa_blocks: int = 4,
        n_uot_blocks: int = 48,
        # Diffusion v2 config
        n_diff_transformer_layers: int = 24,
        n_diff_encoder_blocks: int = 3,
        n_diff_decoder_blocks: int = 3,
        n_diff_heads: int = 16,
        d_fourier: int = 256,
        sigma_data: float = SIGMA_DATA,
        max_cycles: int = 5,
        inference_cycles: int = 3,
        diffusion_multiplicity: int = 16,
        s_trans: float = 1.0,
        loss_weights: dict[str, float] | None = None,
        # Init config
        gamma_std: float = 1e-4,
        adaln_gate_bias: float = -2.0,
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

        # Diffusion v2: encoder-transformer-decoder (Boltz-1 style)
        self.diffusion = DiffusionModuleV2(
            d_model=d_model,
            d_atom=d_atom,
            d_fourier=d_fourier,
            n_transformer_layers=n_diff_transformer_layers,
            n_encoder_blocks=n_diff_encoder_blocks,
            n_decoder_blocks=n_diff_decoder_blocks,
            n_diff_heads=n_diff_heads,
        )

        # Distogram loss module (has learnable parameters)
        self.distogram_loss = DistogramLoss(d_model=d_model)

        self.sigma_data = sigma_data
        self.diffusion_multiplicity = diffusion_multiplicity
        self.s_trans = s_trans
        self.loss_weights = loss_weights or {}

        # Apply SPEC §14 initialization
        init_model(self, gamma_std=gamma_std, adaln_gate_bias=adaln_gate_bias)

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
        token_atom_starts: torch.Tensor | None = None,
        token_atom_counts: torch.Tensor | None = None,
        x_atom_true: torch.Tensor | None = None,
        x_res_true: torch.Tensor | None = None,
        atom_resolved_mask: torch.Tensor | None = None,
        token_resolved_mask: torch.Tensor | None = None,
        token_pad_mask: torch.Tensor | None = None,
        atom_pad_mask: torch.Tensor | None = None,
        pair_valid_mask: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
        compute_losses: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: trunk -> diffusion (training) or trunk -> sampling (inference).

        Supports both unbatched (N, ...) and batched (B, N, ...) inputs.
        Mask tensors (token_pad_mask, atom_pad_mask, pair_valid_mask, msa_mask)
        are floats with 1.0=valid, 0.0=padding. They are only needed for batched
        inputs (B>1) where samples have been padded to the same length.

        Args:
            compute_losses: If True, compute and return losses even when not
                in training mode. Useful for validation with model.eval().

        Returns dict with 'x_atom_pred' and losses if training or compute_losses.
        """
        device = token_type.device
        is_batched = token_type.dim() >= 2

        # Embed atom features once — shared by trunk and diffusion (SPEC §3.3, §3.4)
        c_atom = self.atom_single_embed(c_atom)
        p_lm = self.atom_pair_embed(p_lm[..., :3], p_lm[..., 4:5])

        # Compute s_inputs for diffusion conditioning (token embedding before trunk)
        s_inputs = self.trunk.token_embed(token_type, profile, del_mean, has_msa)

        # Compute pos_bins for diffusion transformer position bias
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)  # (B, N, N) int32

        # ---- Sample cycle count (SPEC §5.3) ----
        if self.training:
            _nc = torch.randint(1, self.trunk.max_cycles + 1, (1,), device=device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(_nc, src=0)
            num_cycles = int(_nc.item())
        else:
            num_cycles = self.trunk.inference_cycles

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
            msa_token_mask,
            token_pad_mask=token_pad_mask,
            msa_mask=msa_mask,
            num_cycles=num_cycles,
        )

        # Trunk squeezes back to unbatched when input was unbatched.
        # Ensure everything is batched (B, ...) for diffusion v2.
        if not is_batched:
            def _unsqueeze(t):
                return t.unsqueeze(0) if t is not None else None
            h_res, mu, nu, x_res = _unsqueeze(h_res), _unsqueeze(mu), _unsqueeze(nu), _unsqueeze(x_res)
            c_atom, token_idx, token_type, s_inputs, pos_bins = (
                _unsqueeze(c_atom), _unsqueeze(token_idx), _unsqueeze(token_type),
                _unsqueeze(s_inputs), _unsqueeze(pos_bins),
            )
            token_atom_starts = _unsqueeze(token_atom_starts)
            token_atom_counts = _unsqueeze(token_atom_counts)
            x_atom_true, x_res_true = _unsqueeze(x_atom_true), _unsqueeze(x_res_true)
            atom_resolved_mask = _unsqueeze(atom_resolved_mask)
            token_resolved_mask = _unsqueeze(token_resolved_mask)
            token_pad_mask, atom_pad_mask = _unsqueeze(token_pad_mask), _unsqueeze(atom_pad_mask)
            is_batched = True

        # Compute token_atom_starts/counts from token_idx if not provided
        if token_atom_starts is None or token_atom_counts is None:
            B_cur = token_idx.shape[0]
            N_cur = h_res.shape[1]
            counts_list, starts_list = [], []
            for b in range(B_cur):
                c = torch.bincount(token_idx[b], minlength=N_cur).to(torch.int32)
                counts_list.append(c)
                starts_list.append(torch.cumsum(c, dim=0) - c)
            token_atom_starts = torch.stack(starts_list)
            token_atom_counts = torch.stack(counts_list)

        result = {"h_res": h_res, "mu": mu, "nu": nu, "x_res": x_res, "num_cycles": num_cycles}

        if (self.training or compute_losses) and x_atom_true is not None and x_res_true is not None:
            # ---- Training: M augmented diffusion samples (Boltz-style multiplicity) ----
            # Trunk runs once; diffusion runs M times with independent augmentations + σ.
            M = self.diffusion_multiplicity

            # Per-atom loss weights by mol_type (Boltz-1: nucleotide 5×, ligand 10×)
            atom_weights = _atom_type_weights(token_idx, token_type)

            # Per-atom / per-token nucleotide flag for smooth LDDT cutoff
            token_is_nuc = (
                (token_type == const.MOL_DNA) | (token_type == const.MOL_RNA)
            ).float()
            if token_idx.dim() == 1:
                atom_is_nuc = token_is_nuc[token_idx]
            else:
                atom_is_nuc = torch.gather(token_is_nuc, 1, token_idx)

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

                # Checkpoint each diffusion call: only inputs saved, activations
                # recomputed during backward. Peak memory = 1 sample at a time
                # instead of M=16 simultaneously. use_reentrant=False avoids
                # DDP "marked ready twice" (no hook replay).
                x_pred_i = checkpoint(
                    self.diffusion,
                    h_res,
                    s_inputs,
                    c_atom,
                    x_noisy_i,
                    sigma_i,
                    token_idx,
                    pos_bins,
                    token_atom_starts,
                    token_atom_counts,
                    token_pad_mask,
                    atom_pad_mask,
                    use_reentrant=False,
                )

                l_diff_parts.append(
                    edm_diffusion_loss(
                        x_pred_i,
                        x_true_i,
                        sigma_i,
                        resolved_mask=atom_resolved_mask,
                        atom_weights=atom_weights,
                    )
                )
                l_lddt_parts.append(
                    smooth_lddt(
                        x_pred_i,
                        x_true_i,
                        resolved_mask=atom_resolved_mask,
                        is_nucleotide=atom_is_nuc,
                    )
                )

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
                    disto_mask = (
                        token_resolved_mask[:, None] * token_resolved_mask[None, :]
                    )
                else:
                    disto_mask = None
                l_disto = self.distogram_loss(h_res, x_res_true, valid_mask=disto_mask)

            l_trunk_coord = smooth_lddt(
                x_res, x_res_true, resolved_mask=token_resolved_mask,
                is_nucleotide=token_is_nuc,
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
        msa_token_mask: torch.Tensor,
        token_atom_starts: torch.Tensor | None = None,
        token_atom_counts: torch.Tensor | None = None,
        n_steps: int = 200,
    ) -> torch.Tensor:
        """Inference: trunk + Heun 2nd-order diffusion sampling (SPEC §12)."""
        device = token_type.device
        is_unbatched = token_type.dim() == 1

        # Embed atom features once (SPEC §3.3, §3.4)
        c_atom = self.atom_single_embed(c_atom)
        p_lm = self.atom_pair_embed(p_lm[..., :3], p_lm[..., 4:5])

        # Compute s_inputs and pos_bins
        s_inputs = self.trunk.token_embed(token_type, profile, del_mean, has_msa)
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)

        # Trunk
        h_res, mu, nu, x_res = self.trunk(
            token_type, profile, del_mean, has_msa, msa_feat,
            c_atom, p_lm, p_lm_idx, token_idx,
            chain_id, global_idx, bond_matrix, msa_token_mask,
            num_cycles=self.trunk.inference_cycles,
        )

        # Ensure batched
        if is_unbatched:
            def _unsq(t):
                return t.unsqueeze(0) if t is not None else None
            h_res, s_inputs, c_atom, token_idx = (
                _unsq(h_res), _unsq(s_inputs), _unsq(c_atom), _unsq(token_idx))
            pos_bins = _unsq(pos_bins)
            token_atom_starts, token_atom_counts = _unsq(token_atom_starts), _unsq(token_atom_counts)

        B = h_res.shape[0]
        M = token_idx.shape[1]
        N = h_res.shape[1]

        # Compute starts/counts if not provided
        if token_atom_starts is None or token_atom_counts is None:
            cl, sl = [], []
            for b in range(B):
                c = torch.bincount(token_idx[b], minlength=N).to(torch.int32)
                cl.append(c)
                sl.append(torch.cumsum(c, dim=0) - c)
            token_atom_starts = torch.stack(sl)
            token_atom_counts = torch.stack(cl)

        # Initialize from noise
        sigmas = karras_schedule(n_steps, device)
        x_atom = torch.randn(B, M, 3, device=device) * SIGMA_MAX

        def _denoise(x, sigma):
            return self.diffusion(
                h_res, s_inputs, c_atom, x, sigma,
                token_idx, pos_bins,
                token_atom_starts, token_atom_counts,
            )

        # Heun 2nd-order sampling (probability flow ODE)
        for i in range(n_steps - 1):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            denoised = _denoise(x_atom, sigma_cur)
            d_cur = (x_atom - denoised) / sigma_cur

            x_next = x_atom + (sigma_next - sigma_cur) * d_cur

            if sigma_next > 0 and i < n_steps - 2:
                denoised_next = _denoise(x_next, sigma_next)
                d_next = (x_next - denoised_next) / sigma_next
                x_next = x_atom + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)

            x_atom = x_next

        if is_unbatched:
            x_atom = x_atom.squeeze(0)
        return x_atom
