"""Atom-to-Token Encoder (SPEC §3.5).

Aggregates atom-level chemical information into the token representation.
Runs once before recycling. ~400K params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU
from deepfold.utils.scatter import scatter_mean


class AtomToTokenEncoder(nn.Module):
    """
    Single block of local gated self-attention, then mean-pool to token level.

    c_atom:    (N_atom, 128)  frozen atom reference features
    p_lm:      (local, 16)    frozen intra-token atom pair features
    token_idx: (N_atom,)      maps each atom to its token
    N:         int             number of tokens
    Returns:   (N, 512)       per-token atom summary
    """

    def __init__(self, d_atom: int = 128, d_model: int = 512, n_heads: int = 4):
        super().__init__()
        self.d_atom = d_atom
        self.n_heads = n_heads
        self.d_head = d_atom // n_heads  # 32

        # Self-attention projections — after LN, so bias=False (SPEC §0)
        self.ln_attn = nn.LayerNorm(d_atom)
        self.w_q = nn.Linear(d_atom, d_atom, bias=False)
        self.w_k = nn.Linear(d_atom, d_atom, bias=False)
        self.w_v = nn.Linear(d_atom, d_atom, bias=False)
        self.w_g = nn.Linear(d_atom, d_atom, bias=False)
        self.w_o = nn.Linear(d_atom, d_atom, bias=False)

        # Pair bias: (16 -> n_heads) for intra-token pair bias
        self.pair_bias_proj = nn.Linear(16, n_heads)

        # SwiGLU transition
        self.ln_ff = nn.LayerNorm(d_atom)
        self.swiglu = SwiGLU(d_atom, d_atom * 4, d_atom)

        # Project to token dim
        self.to_token = nn.Linear(d_atom, d_model)

    def forward(
        self,
        c_atom: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        token_idx: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        """
        Args:
            c_atom:    (N_atom, 128) frozen atom reference features
            p_lm:      (n_pairs, 16) frozen atom pair features
            p_lm_idx:  (n_pairs, 2) indices [i, j] into atom dimension
            token_idx: (N_atom,) maps each atom to its token
            n_tokens:  int, number of tokens N

        Returns:
            (N, 512) per-token atom summary
        """
        N_atom = c_atom.shape[0]
        H = self.n_heads
        d_h = self.d_head
        q = c_atom  # (N_atom, 128)

        # --- Gated local self-attention ---
        q_n = self.ln_attn(q)  # (N_atom, 128)
        Q = self.w_q(q_n).view(N_atom, H, d_h)  # (N_atom, 4, 32)
        K = self.w_k(q_n).view(N_atom, H, d_h)
        V = self.w_v(q_n).view(N_atom, H, d_h)
        G = self.w_g(q_n).view(N_atom, H, d_h)

        # Compute attention scores: (H, N_atom, N_atom)
        scores = torch.einsum("ihd,jhd->hij", Q, K) / (d_h**0.5)  # (H, N_atom, N_atom)

        # Add pair bias for intra-token pairs
        if p_lm.shape[0] > 0:
            bias = self.pair_bias_proj(p_lm)  # (n_pairs, H)
            # Scatter pair bias into attention scores
            src_idx = p_lm_idx[:, 0]  # (n_pairs,)
            dst_idx = p_lm_idx[:, 1]  # (n_pairs,)
            pair_bias = torch.zeros(
                H, N_atom, N_atom, device=q.device, dtype=bias.dtype
            )
            pair_bias[:, src_idx, dst_idx] = bias.T  # (H, n_pairs)
            scores = scores + pair_bias

        # Mask: only attend within same token (local attention)
        token_mask = token_idx.unsqueeze(0) == token_idx.unsqueeze(
            1
        )  # (N_atom, N_atom)
        scores = scores.masked_fill(~token_mask[None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (H, N_atom, N_atom)
        att_out = torch.einsum("hij,jhd->ihd", attn, V)  # (H, N_atom, d_h)
        att_out = att_out.permute(1, 0, 2).reshape(N_atom, -1)  # (N_atom, 128)

        G_flat = G.reshape(N_atom, -1)  # (N_atom, 128)
        q = q + torch.sigmoid(G_flat) * self.w_o(att_out)

        # --- SwiGLU transition ---
        q = q + self.swiglu(self.ln_ff(q))

        # --- Project and scatter_mean to token level ---
        atom_feat = self.to_token(q)  # (N_atom, 512)
        return scatter_mean(atom_feat, token_idx, n_tokens)  # (N, 512)
