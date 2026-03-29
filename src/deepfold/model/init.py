"""Weight initialization (SPEC §14).

Two-tier philosophy:
  1. Representation path (h_res): near-identity across all blocks.
     W_O scaled by 1/sqrt(L) per module depth. Each block starts weak.
  2. Coordinate path (x_res): dormant at init.
     alpha_h=0 (cost weight midpoint 0.5), w_gate bias=0 (tanh gate starts at 0),
     lambda_h=1 (unit displacement scale), pos_bias=0.
"""

import math

import torch.nn as nn


def _is_zero_init(name: str) -> bool:
    """Parameters that must stay at zero (dormant at init)."""
    return any(k in name for k in ("alpha_h",))


def _is_adaln_zero_gate(name: str) -> bool:
    """AdaLN-Zero gates: weight=0, bias=-2.0 (set in AtomBlock.__init__)."""
    return "_attn_gate" in name or "_transition_gate" in name


def _is_position_bias(name: str) -> bool:
    """PositionBias weights — zeros init, learned."""
    return "pos_bias.weight" in name


def init_model(
    model: nn.Module,
    adaln_gate_bias: float = -2.0,
    **kwargs,
) -> None:
    """Apply SPEC §14 initialization to a DeepFoldLinear model.

    - Xavier normal for all 2D weight matrices (except zero-init ones).
    - Attention output (w_o) scaled by 1/sqrt(L) for residual depth control.
    - SwiGLU output NOT depth-scaled (product structure self-suppresses).
    - alpha_h=0 (cost weight sigmoid midpoint = 0.5).
    - lambda_h=1 (unit displacement scale, PyTorch default).
    - Zeros for pos_bias weights.
    - LayerNorm: weight=1, bias=0 (PyTorch default, but explicit).
    - Biases: zeros (PyTorch default).
    """
    # Collect depth info: module name -> number of sibling blocks
    # trunk.trunk_blocks.X  -> 48 blocks
    # msa_module.blocks.X -> 4 blocks (MSA)
    # diffusion.atom_blocks.X -> 3 blocks
    # atom_encoder -> 1 block (no depth scaling needed)

    for name, param in model.named_parameters():
        # AdaLN-Zero gates: weight=0, bias from config
        if _is_adaln_zero_gate(name):
            if "bias" in name:
                nn.init.constant_(param, adaln_gate_bias)
            else:
                nn.init.zeros_(param)
            continue

        # Skip frozen Fourier embedding weights
        if "fourier_embed" in name:
            continue

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
            # All other 1D (biases, lambda_h, r_h) stay at PyTorch default
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
    if ".trunk_blocks." in name:
        return 48  # trunk OT blocks
    if "msa_module.blocks." in name or "msa_module.blocks" in name:
        return 4  # MSA blocks
    if ".atom_blocks." in name:
        return 10  # diffusion atom blocks (SPEC v4.4)
    return None
