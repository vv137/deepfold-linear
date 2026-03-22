"""Weight initialization (SPEC §14).

Two-tier philosophy:
  1. Representation path (h_res): near-identity across all blocks.
     W_O scaled by 1/sqrt(L) per module depth. Each block starts weak.
  2. Coordinate path (x_res): completely dormant at init.
     gamma=0, w_dist=0, w_rel=0, alpha_coevol=0.
     Content-only attention first; geometry earns its influence.
"""

import math

import torch.nn as nn


def _is_zero_init(name: str) -> bool:
    """Parameters that must stay at zero (dormant at init)."""
    return any(tag in name for tag in ("gamma", "w_dist", "alpha_coevol"))


def _is_position_bias(name: str) -> bool:
    """PositionBias weights — zeros init, learned."""
    return "pos_bias.weight" in name


def init_model(model: nn.Module) -> None:
    """Apply SPEC §14 initialization to a DeepFoldLinear model.

    - Xavier normal for all 2D weight matrices (except zero-init ones).
    - Attention output (w_o) scaled by 1/sqrt(L) for residual depth control.
    - SwiGLU output NOT depth-scaled (product structure self-suppresses).
    - Zeros for gamma, w_dist, w_rel, alpha_coevol, zero_init_linear outputs.
    - LayerNorm: weight=1, bias=0 (PyTorch default, but explicit).
    - Biases: zeros (PyTorch default).
    """
    # Collect depth info: module name -> number of sibling blocks
    # trunk.uot_blocks.X  -> 48 blocks
    # diffusion.diff_uot_blocks.X -> 2 blocks
    # msa_module.blocks.X -> 4 blocks (MSA)
    # diffusion.atom_blocks.X -> 3 blocks
    # atom_encoder -> 1 block (no depth scaling needed)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Scalars and 1D params
            if _is_zero_init(name) or _is_position_bias(name):
                nn.init.zeros_(param)
            elif (
                "layernorm" in name.lower()
                or ".ln_" in name
                or name.endswith(".ln.weight")
            ):
                # LayerNorm gamma=1 already default, but be explicit
                if "weight" in name:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)
            # All other 1D (biases) stay at PyTorch default (zeros)
            continue

        # 2D weight matrices
        if _is_zero_init(name) or _is_position_bias(name):
            nn.init.zeros_(param)
            continue

        if "coord_out" in name:
            # Diffusion coordinate output — zero init (EDM identity at init)
            nn.init.zeros_(param)
            continue

        # Xavier normal for everything else
        nn.init.xavier_normal_(param)

        # Depth-scaled W_O for residual stream control (GPT-2/3 style)
        depth = _get_residual_depth(name)
        if depth is not None and depth > 1 and ".w_o" in name:
            param.data /= math.sqrt(depth)


def _get_residual_depth(name: str) -> int | None:
    """Return the number of sibling blocks for residual depth scaling.

    Only applies to repeated block stacks where residual contributions
    accumulate. Returns None for non-block parameters.
    """
    if ".uot_blocks." in name and "diff_uot_blocks" not in name:
        return 48  # trunk UOT+EGNN blocks
    if ".diff_uot_blocks." in name:
        return 2  # diffusion UOT blocks
    if "msa_module.blocks." in name or "msa_module.blocks" in name:
        return 4  # MSA blocks
    if ".atom_blocks." in name:
        return 10  # diffusion atom blocks (SPEC v4.4)
    return None
