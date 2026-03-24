"""Core primitives used throughout DeepFold-Linear (SPEC §0, §1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward: gate * SiLU(value) -> out. SPEC uses this everywhere."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        # SwiGLU always follows LN -> bias=False (SPEC §0 bias convention)
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.value_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class LNLinear(nn.Module):
    """LayerNorm then Linear composite (LN_Lin in SPEC §0)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


def zero_init_linear(d_in: int, d_out: int) -> nn.Linear:
    """Linear layer with zero-initialized weights and bias."""
    lin = nn.Linear(d_in, d_out)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def adaln_zero_gate(d_in: int, d_out: int) -> nn.Linear:
    """AdaLN-Zero gate: w=0, b=-2 → sigmoid(-2) ≈ 0.12 at init."""
    lin = nn.Linear(d_in, d_out)
    nn.init.zeros_(lin.weight)
    nn.init.constant_(lin.bias, -2.0)
    return lin
