"""Atom-to-Token Encoder (SPEC §3.5).

Aggregates atom-level chemical information into the token representation.
Runs once before recycling. ~400K params.

Supports both unbatched and batched inputs via dual-mode pattern:
detect unbatched (c_atom.dim()==2) -> unsqueeze(0), process batched, squeeze back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU
from deepfold.utils.scatter import scatter_mean


class AtomToTokenEncoder(nn.Module):
    """
    Single block of local gated self-attention, then mean-pool to token level.

    Unbatched:
        c_atom:    (N_atom, 128)  frozen atom reference features
        p_lm:      (n_pairs, 16)  frozen intra-token atom pair features
        token_idx: (N_atom,)      maps each atom to its token
        N:         int             number of tokens
        Returns:   (N, 512)       per-token atom summary

    Batched:
        c_atom:         (B, N_atom, 128)
        p_lm:           (B, n_pairs, 16)
        p_lm_idx:       (B, n_pairs, 2)
        token_idx:      (B, N_atom)
        n_tokens:       int
        atom_pad_mask:  (B, N_atom) bool, True for real atoms
        pair_pad_mask:  (B, n_pairs) bool, True for real pairs
        token_pad_mask: (B, N) bool, True for real tokens
        Returns:        (B, N, 512)
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
        atom_pad_mask: torch.Tensor | None = None,
        pair_pad_mask: torch.Tensor | None = None,
        token_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            c_atom:         (N_atom, 128) or (B, N_atom, 128)
            p_lm:           (n_pairs, 16) or (B, n_pairs, 16)
            p_lm_idx:       (n_pairs, 2) or (B, n_pairs, 2) indices [i, j] into atom dim
            token_idx:      (N_atom,) or (B, N_atom) maps each atom to its token
            n_tokens:       int, number of tokens N
            atom_pad_mask:  (B, N_atom) bool, True for real atoms (batched only)
            pair_pad_mask:  (B, n_pairs) bool, True for real pairs (batched only)
            token_pad_mask: (B, N) bool, True for real tokens (batched only)

        Returns:
            (N, 512) or (B, N, 512)
        """
        unbatched = c_atom.dim() == 2
        if unbatched:
            c_atom = c_atom.unsqueeze(0)
            p_lm = p_lm.unsqueeze(0)
            p_lm_idx = p_lm_idx.unsqueeze(0)
            token_idx = token_idx.unsqueeze(0)

        B, N_atom, _ = c_atom.shape
        H = self.n_heads
        d_h = self.d_head
        n_pairs = p_lm.shape[1]
        q = c_atom  # (B, N_atom, 128)

        # --- Gated local self-attention ---
        q_n = self.ln_attn(q)  # (B, N_atom, 128)
        Q = self.w_q(q_n).view(B, N_atom, H, d_h)  # (B, N_atom, H, d_h)
        K = self.w_k(q_n).view(B, N_atom, H, d_h)
        V = self.w_v(q_n).view(B, N_atom, H, d_h)
        G = self.w_g(q_n).view(B, N_atom, H, d_h)

        # Attention scores: (B, H, N_atom, N_atom)
        scores = torch.einsum("bihd,bjhd->bhij", Q, K) / (d_h**0.5)

        # Add pair bias for intra-token pairs
        if n_pairs > 0:
            bias = self.pair_bias_proj(p_lm)  # (B, n_pairs, H)
            pair_bias = torch.zeros(
                B, H, N_atom, N_atom, device=q.device, dtype=bias.dtype
            )
            b_idx = (
                torch.arange(B, device=q.device).unsqueeze(1).expand(-1, n_pairs)
            )  # (B, n_pairs)
            src_idx = p_lm_idx[..., 0]  # (B, n_pairs)
            dst_idx = p_lm_idx[..., 1]  # (B, n_pairs)

            if pair_pad_mask is not None:
                b_idx = b_idx[pair_pad_mask]
                src_idx = src_idx[pair_pad_mask]
                dst_idx = dst_idx[pair_pad_mask]
                bias_flat = bias[pair_pad_mask]  # (valid_pairs, H)
            else:
                b_idx = b_idx.reshape(-1)
                src_idx = src_idx.reshape(-1)
                dst_idx = dst_idx.reshape(-1)
                bias_flat = bias.reshape(-1, H)  # (B*n_pairs, H)

            # Vectorized scatter: h_idx broadcasts over all heads
            h_idx = torch.arange(H, device=q.device)
            pair_bias[
                b_idx[:, None], h_idx[None, :], src_idx[:, None], dst_idx[:, None]
            ] = bias_flat

            scores = scores + pair_bias

        # Mask: only attend within same token (local attention)
        token_mask = (
            token_idx[:, :, None] == token_idx[:, None, :]
        )  # (B, N_atom, N_atom)

        # Apply atom padding mask if batched
        if atom_pad_mask is not None:
            # Both query and key must be real atoms
            pair_valid = (
                atom_pad_mask[:, :, None] & atom_pad_mask[:, None, :]
            )  # (B, N_atom, N_atom)
            token_mask = token_mask & pair_valid

        scores = scores.masked_fill(~token_mask[:, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, N_atom, N_atom)
        # Replace NaN from all-inf rows (padded atoms) with 0
        attn = attn.nan_to_num(0.0)

        att_out = torch.einsum("bhij,bjhd->bihd", attn, V)  # (B, N_atom, H, d_h)
        att_out = att_out.reshape(B, N_atom, -1)  # (B, N_atom, 128)

        G_flat = G.reshape(B, N_atom, -1)  # (B, N_atom, 128)
        q = q + torch.sigmoid(G_flat) * self.w_o(att_out)

        # --- SwiGLU transition ---
        q = q + self.swiglu(self.ln_ff(q))

        # --- Project and scatter_mean to token level ---
        atom_feat = self.to_token(q)  # (B, N_atom, 512)
        out = scatter_mean(atom_feat, token_idx, n_tokens)  # (B, N, 512)

        # Zero-pad token positions where token_pad_mask == False
        if token_pad_mask is not None:
            out = out * token_pad_mask.unsqueeze(-1).float()

        if unbatched:
            out = out.squeeze(0)
        return out
