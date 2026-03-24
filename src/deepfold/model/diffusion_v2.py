"""Diffusion Module v2 — Boltz-1-style encoder-transformer-decoder (SPEC §9 v5).

Replaces the flat 10-AtomBlock design with:
  1. SingleConditioning: σ-aware token conditioning
  2. AtomEncoder: 3× (windowed atom self-attn + atom→token cross-attn)
  3. DiffusionTransformer: 24× token self-attn with 68-bin position bias
  4. AtomDecoder: 3× (token→atom cross-attn + windowed atom self-attn)

All attention uses custom Triton kernels for O(N) memory.
Proper EDM preconditioning with c_skip.
~115M params (vs ~4.2M in v1).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU, zero_init_linear, adaln_zero_gate
from deepfold.model.position_encoding import PositionBias
from deepfold.model.diffusion import (
    FourierEmbedding,
    AdaLN,
    edm_preconditioning,
    sample_training_sigma,
    karras_schedule,
    SIGMA_DATA,
    SIGMA_MAX,
    SIGMA_MIN,
)

try:
    from deepfold.model.kernels.flash_diffusion_attn import flash_diffusion_attn
    from deepfold.model.kernels.flash_atom_attn import flash_atom_attn
    from deepfold.model.kernels.cross_attn_kernel import (
        atom_to_token_attn,
        token_to_atom_attn,
    )
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ============================================================================
# SingleConditioning
# ============================================================================


class _FP32FourierEmbedding(FourierEmbedding):
    """FourierEmbedding that stays float32 even when parent module is cast."""

    def _apply(self, fn):
        # Override to prevent dtype casting of frozen weights
        super()._apply(fn)
        self.proj.weight.data = self.proj.weight.data.float()
        self.proj.bias.data = self.proj.bias.data.float()
        return self


class SingleConditioning(nn.Module):
    """σ-conditioned token features (Boltz-1 Algorithm 21).

    s = h_res + s_inputs + broadcast(Linear(LN(FourierEmbed(σ))))
    s = s + SwiGLU_transition(LN(s))  ×2
    """

    def __init__(self, d_model: int = 512, d_fourier: int = 256):
        super().__init__()
        self.fourier_embed = _FP32FourierEmbedding(d_fourier)
        self.ln_fourier = nn.LayerNorm(d_fourier)
        self.t_proj = nn.Linear(d_fourier, d_model, bias=False)

        # 2× SwiGLU transition blocks with residual
        self.ln1 = nn.LayerNorm(d_model)
        self.swiglu1 = SwiGLU(d_model, d_model * 4, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.swiglu2 = SwiGLU(d_model, d_model * 4, d_model)

    def forward(
        self,
        h_res: torch.Tensor,
        s_inputs: torch.Tensor,
        c_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_res: (B, N, 512) trunk output (NOT detached)
            s_inputs: (B, N, 512) cached token embedding (before trunk)
            c_noise: (B,) log-scaled noise level

        Returns:
            s: (B, N, 512) σ-conditioned token features
        """
        t_emb = self.fourier_embed(c_noise.float()).to(h_res.dtype)  # (B, d_fourier)
        t_proj = self.t_proj(self.ln_fourier(t_emb))           # (B, d_model)
        s = h_res + s_inputs + t_proj.unsqueeze(1)             # (B, N, d_model)
        s = s + self.swiglu1(self.ln1(s))
        s = s + self.swiglu2(self.ln2(s))
        return s


# ============================================================================
# DiffusionTransformerLayer
# ============================================================================


class DiffusionTransformerLayer(nn.Module):
    """Single token-level transformer layer for diffusion.

    Pre-norm self-attention with 68-bin position bias + AdaLN-Zero gating.
    Uses flash_diffusion_attn Triton kernel for O(N) memory.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Pre-norm with AdaLN conditioning
        self.adaln_attn = AdaLN(d_model, d_model)

        # Self-attention projections (after AdaLN → bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # AdaLN-Zero gate for attention output
        self._attn_gate = adaln_zero_gate(d_model, d_model)

        # SwiGLU transition
        self.adaln_ff = AdaLN(d_model, d_model)
        self.swiglu = SwiGLU(d_model, d_model * 4, d_model)

        self._ff_gate = adaln_zero_gate(d_model, d_model)

    def forward(
        self,
        s: torch.Tensor,
        s_cond: torch.Tensor,
        pos_weight: torch.Tensor,
        pos_bins: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s: (B, N, d_model) token features
            s_cond: (B, N, d_model) conditioning signal
            pos_weight: (H, 68) position bias weights (shared across layers)
            pos_bins: (B, N, N) int32 bin indices
            mask: (B, N) float, 1=valid 0=pad

        Returns:
            s: (B, N, d_model) updated token features
        """
        B, N, _ = s.shape
        H = self.n_heads
        d_h = self.d_head

        # --- Self-attention ---
        s_n = self.adaln_attn(s, s_cond)
        Q = self.w_q(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)  # (B, H, N, d_h)
        K = self.w_k(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)

        if _HAS_TRITON and s.is_cuda:
            att_out = flash_diffusion_attn(Q, K, V, pos_weight, pos_bins, mask)
        else:
            from deepfold.model.kernels.flash_diffusion_attn import flash_diff_attn_ref
            att_out = flash_diff_attn_ref(Q.float(), K.float(), V.float(),
                                          pos_weight, pos_bins, mask).to(Q.dtype)

        att_out = att_out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, d_model)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, N, -1)
        attn_update = torch.sigmoid(G_flat) * self.w_o(att_out)

        s = s + torch.sigmoid(self._attn_gate(s_cond)) * attn_update
        s = s * mask.unsqueeze(-1)

        # --- SwiGLU transition ---
        ff_update = self.swiglu(self.adaln_ff(s, s_cond))
        s = s + torch.sigmoid(self._ff_gate(s_cond)) * ff_update
        s = s * mask.unsqueeze(-1)

        return s


# ============================================================================
# Atom Self-Attention Block (windowed, reuses AdaLN pattern from v1 AtomBlock)
# ============================================================================


class WindowedAtomBlock(nn.Module):
    """Windowed atom self-attention (W=32 queries, H=128 keys).

    Same AdaLN + AdaLN-Zero gating as v1 AtomBlock, but uses Triton
    windowed kernel instead of ±16 dense attention.
    """

    def __init__(self, d_atom: int = 128, d_cond: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_atom = d_atom
        self.n_heads = n_heads
        self.d_head = d_atom // n_heads

        self.adaln1 = AdaLN(d_atom, d_cond)
        self.adaln2 = AdaLN(d_atom, d_cond)

        self.w_q = nn.Linear(d_atom, d_atom, bias=True)  # Q has bias (AF3)
        self.w_k = nn.Linear(d_atom, d_atom, bias=False)
        self.w_v = nn.Linear(d_atom, d_atom, bias=False)
        self.w_g = nn.Linear(d_atom, d_atom, bias=False)
        self.w_o = nn.Linear(d_atom, d_atom, bias=False)

        self._attn_gate = adaln_zero_gate(d_cond, d_atom)

        self.swiglu = SwiGLU(d_atom, d_atom * 4, d_atom)

        self._ff_gate = adaln_zero_gate(d_cond, d_atom)

    def forward(
        self,
        a: torch.Tensor,
        cond: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a: (B, M, d_atom) atom features
            cond: (B, M, d_cond) per-atom conditioning
            atom_mask: (B, M) float, 1=valid 0=pad

        Returns:
            a: (B, M, d_atom) updated atom features
        """
        B, M, _ = a.shape
        H = self.n_heads
        d_h = self.d_head

        # --- Windowed self-attention ---
        a_n = self.adaln1(a, cond)
        Q = self.w_q(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)
        K = self.w_k(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)

        if _HAS_TRITON and a.is_cuda:
            att_out = flash_atom_attn(Q, K, V, atom_mask)
        else:
            from deepfold.model.kernels.flash_atom_attn import flash_atom_attn_ref
            att_out = flash_atom_attn_ref(Q.float(), K.float(), V.float(),
                                          atom_mask).to(Q.dtype)

        att_out = att_out.permute(0, 2, 1, 3).reshape(B, M, -1)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, M, -1)
        attn_update = torch.sigmoid(G_flat) * self.w_o(att_out)

        a = a + torch.sigmoid(self._attn_gate(cond)) * attn_update
        a = a * atom_mask.unsqueeze(-1)

        # --- SwiGLU transition ---
        ff_update = self.swiglu(self.adaln2(a, cond))
        a = a + torch.sigmoid(self._ff_gate(cond)) * ff_update
        a = a * atom_mask.unsqueeze(-1)

        return a


# ============================================================================
# Cross-Attention Modules
# ============================================================================


class AtomToTokenCrossAttn(nn.Module):
    """Sparse cross-attention: tokens query their own atoms (encoder)."""

    def __init__(self, d_token: int = 512, d_atom: int = 128, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_token, d_atom, bias=False)
        self.w_k = nn.Linear(d_atom, d_atom, bias=False)
        self.w_v = nn.Linear(d_atom, d_atom, bias=False)
        self.w_g = nn.Linear(d_token, d_atom, bias=False)
        self.w_o = nn.Linear(d_atom, d_token, bias=False)

        self.ln_q = nn.LayerNorm(d_token)
        self.ln_kv = nn.LayerNorm(d_atom)

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        token_atom_starts: torch.Tensor,
        token_atom_counts: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s: (B, N, d_token) token features (queries)
            a: (B, M, d_atom) atom features (keys/values)
            token_atom_starts: (B, N) int32
            token_atom_counts: (B, N) int32
            token_mask: (B, N) float

        Returns:
            update: (B, N, d_token) to add to s
        """
        B, N, _ = s.shape
        M = a.shape[1]
        H = self.n_heads
        d_atom = a.shape[-1]
        d_h = d_atom // H

        s_n = self.ln_q(s)
        a_n = self.ln_kv(a)

        Q = self.w_q(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)  # (B, H, N, d_h)
        K = self.w_k(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)  # (B, H, M, d_h)
        V = self.w_v(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)

        if _HAS_TRITON and s.is_cuda:
            att_out = atom_to_token_attn(Q, K, V, token_atom_starts,
                                         token_atom_counts, token_mask)
        else:
            from deepfold.model.kernels.cross_attn_kernel import atom_to_token_ref
            att_out = atom_to_token_ref(Q.float(), K.float(), V.float(),
                                        token_atom_starts, token_atom_counts,
                                        token_mask).to(Q.dtype)

        att_out = att_out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, d_atom)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.w_o(torch.sigmoid(G_flat) * att_out)


class TokenToAtomCrossAttn(nn.Module):
    """Dense cross-attention: atoms query all tokens (decoder)."""

    def __init__(self, d_token: int = 512, d_atom: int = 128, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_atom, d_atom, bias=False)
        self.w_k = nn.Linear(d_token, d_atom, bias=False)
        self.w_v = nn.Linear(d_token, d_atom, bias=False)
        self.w_g = nn.Linear(d_atom, d_atom, bias=False)
        self.w_o = nn.Linear(d_atom, d_atom, bias=False)

        self.ln_q = nn.LayerNorm(d_atom)
        self.ln_kv = nn.LayerNorm(d_token)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        atom_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a: (B, M, d_atom) atom features (queries)
            s: (B, N, d_token) token features (keys/values)
            atom_mask: (B, M) float
            token_mask: (B, N) float

        Returns:
            update: (B, M, d_atom) to add to a
        """
        B, M, _ = a.shape
        N = s.shape[1]
        H = self.n_heads
        d_atom = a.shape[-1]
        d_h = d_atom // H

        a_n = self.ln_q(a)
        s_n = self.ln_kv(s)

        Q = self.w_q(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)
        K = self.w_k(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(s_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(a_n).view(B, M, H, d_h).permute(0, 2, 1, 3)

        if _HAS_TRITON and a.is_cuda:
            att_out = token_to_atom_attn(Q, K, V, atom_mask, token_mask)
        else:
            from deepfold.model.kernels.cross_attn_kernel import token_to_atom_ref
            att_out = token_to_atom_ref(Q.float(), K.float(), V.float(),
                                        atom_mask, token_mask).to(Q.dtype)

        att_out = att_out.permute(0, 2, 1, 3).reshape(B, M, -1)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, M, -1)
        return self.w_o(torch.sigmoid(G_flat) * att_out)


# ============================================================================
# DiffusionModule v2
# ============================================================================


class DiffusionModuleV2(nn.Module):
    """Diffusion module v2: encoder-transformer-decoder (Boltz-1 style).

    Architecture:
      1. SingleConditioning: h_res + s_inputs + FourierEmbed(σ) + 2× transition
      2. AtomEncoder: 3× (WindowedAtomBlock + AtomToTokenCrossAttn)
      3. DiffusionTransformer: 24× (self-attn with 68-bin pos bias + SwiGLU)
      4. AtomDecoder: 3× (TokenToAtomCrossAttn + WindowedAtomBlock)
      5. EDM output: c_skip · x_noisy + c_out · Δx
    """

    def __init__(
        self,
        d_model: int = 512,
        d_atom: int = 128,
        d_fourier: int = 256,
        n_transformer_layers: int = 24,
        n_encoder_blocks: int = 3,
        n_decoder_blocks: int = 3,
        n_diff_heads: int = 16,
        n_atom_heads: int = 4,
        n_cross_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_atom = d_atom

        # 1. SingleConditioning
        self.conditioning = SingleConditioning(d_model, d_fourier)

        # Atom coordinate embedding
        self.atom_coord_proj = nn.Linear(3, d_atom)

        # Atom conditioning projection: s_atoms from s via token_idx
        self.atom_cond_proj = nn.Linear(d_model, d_atom)

        # 2. AtomEncoder
        self.encoder_atom_blocks = nn.ModuleList(
            [WindowedAtomBlock(d_atom, d_atom, n_atom_heads)
             for _ in range(n_encoder_blocks)]
        )
        self.encoder_cross_attns = nn.ModuleList(
            [AtomToTokenCrossAttn(d_model, d_atom, n_cross_heads)
             for _ in range(n_encoder_blocks)]
        )

        # 3. DiffusionTransformer
        self.transformer_layers = nn.ModuleList(
            [DiffusionTransformerLayer(d_model, n_diff_heads)
             for _ in range(n_transformer_layers)]
        )
        # Shared position bias weights across all transformer layers
        self.pos_bias = PositionBias(n_diff_heads, num_bins=68)

        # Post-transformer norm
        self.ln_transformer = nn.LayerNorm(d_model)

        # 4. AtomDecoder
        self.decoder_cross_attns = nn.ModuleList(
            [TokenToAtomCrossAttn(d_model, d_atom, n_cross_heads)
             for _ in range(n_decoder_blocks)]
        )
        self.decoder_atom_blocks = nn.ModuleList(
            [WindowedAtomBlock(d_atom, d_atom, n_atom_heads)
             for _ in range(n_decoder_blocks)]
        )

        # Token-to-atom broadcast projection (for skip connection init)
        self.token_to_atom_proj = nn.Linear(d_model, d_atom, bias=False)
        nn.init.zeros_(self.token_to_atom_proj.weight)

        # 5. Coordinate output — zero init
        self.ln_out = nn.LayerNorm(d_atom)
        self.coord_out = zero_init_linear(d_atom, 3)

    def forward(
        self,
        h_res: torch.Tensor,
        s_inputs: torch.Tensor,
        c_atom: torch.Tensor,
        x_atom_noisy: torch.Tensor,
        sigma: torch.Tensor,
        token_idx: torch.Tensor,
        pos_bins: torch.Tensor,
        token_atom_starts: torch.Tensor,
        token_atom_counts: torch.Tensor,
        token_pad_mask: torch.Tensor | None = None,
        atom_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_res:              (B, N, 512) trunk output (NOT detached)
            s_inputs:           (B, N, 512) cached token embedding
            c_atom:             (B, M, 128) frozen atom reference
            x_atom_noisy:       (B, M, 3) noisy atom coordinates
            sigma:              (B,) or scalar noise level
            token_idx:          (B, M) atom-to-token mapping
            pos_bins:           (B, N, N) int32, 68-bin position encoding
            token_atom_starts:  (B, N) int32
            token_atom_counts:  (B, N) int32
            token_pad_mask:     (B, N) float, 1=valid 0=pad
            atom_pad_mask:      (B, M) float, 1=valid 0=pad

        Returns:
            x_pred: (B, M, 3) denoised coordinates
        """
        B = h_res.shape[0]
        N = h_res.shape[1]
        M = x_atom_noisy.shape[1]

        if token_pad_mask is None:
            token_pad_mask = h_res.new_ones(B, N)
        if atom_pad_mask is None:
            atom_pad_mask = x_atom_noisy.new_ones(B, M)

        # Ensure sigma is (B,)
        if sigma.dim() == 0:
            sigma = sigma.expand(B)

        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma)

        # ---- 1. SingleConditioning ----
        s = self.conditioning(h_res, s_inputs, c_noise)  # (B, N, d_model)

        # ---- 2. AtomEncoder ----
        # Embed noisy coordinates
        a = c_atom + self.atom_coord_proj(
            x_atom_noisy * c_in[:, None, None]
        )  # (B, M, d_atom)

        # Per-atom conditioning from s via token_idx
        d_model = s.shape[-1]
        s_atoms = torch.gather(
            s, 1, token_idx.unsqueeze(-1).expand(-1, -1, d_model)
        )
        atom_cond = self.atom_cond_proj(s_atoms)  # (B, M, d_atom)

        for atom_block, cross_attn in zip(
            self.encoder_atom_blocks, self.encoder_cross_attns
        ):
            a = atom_block(a, atom_cond, atom_pad_mask)
            s = s + cross_attn(s, a, token_atom_starts, token_atom_counts,
                               token_pad_mask)

        # Cache encoder atom repr for decoder skip
        a_enc = a

        # ---- 3. DiffusionTransformer ----
        pos_weight = self.pos_bias.weight  # (H, 68)
        for layer in self.transformer_layers:
            s = layer(s, s, pos_weight, pos_bins, token_pad_mask)
        s = self.ln_transformer(s)

        # ---- 4. AtomDecoder ----
        # Gather refined s to atom level (shared for skip init + conditioning)
        s_at_atoms = torch.gather(
            s, 1, token_idx.unsqueeze(-1).expand(-1, -1, d_model)
        )
        a = a_enc + self.token_to_atom_proj(s_at_atoms)
        atom_cond_dec = self.atom_cond_proj(s_at_atoms)

        for cross_attn, atom_block in zip(
            self.decoder_cross_attns, self.decoder_atom_blocks
        ):
            a = a + cross_attn(a, s, atom_pad_mask, token_pad_mask)
            a = atom_block(a, atom_cond_dec, atom_pad_mask)

        # ---- 5. Coordinate output ----
        delta_x = self.coord_out(self.ln_out(a))  # (B, M, 3)

        # Proper EDM: x_pred = c_skip · x_noisy + c_out · F_θ
        x_pred = (
            c_skip[:, None, None] * x_atom_noisy
            + c_out[:, None, None] * delta_x
        )

        # Re-center per sample
        mask_f = atom_pad_mask.float()
        mask_sum = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_f32 = x_pred.float()
        com = (x_f32 * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        x_pred = (x_f32 - com).to(x_pred.dtype)
        x_pred = x_pred * atom_pad_mask.unsqueeze(-1)

        return x_pred
