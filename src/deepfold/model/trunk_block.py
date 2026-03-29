"""Token OT Block — MHA + Balanced Sinkhorn Transport (SPEC §8, v6.0).

48 blocks in trunk. Each block:
  [Step 1] MHA with 68-bin position bias → h update + SwiGLU FFN
  [Step 2] Balanced Sinkhorn transport → coordinate update

Supports both unbatched (N, d) and batched (B, N, d) inputs via dual-mode pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.kernels.flash_sinkhorn_transport import balanced_sinkhorn_transport
from deepfold.model.position_encoding import PositionBias
from deepfold.model.primitives import SwiGLU

try:
    from deepfold.model.kernels.flash_diffusion_attn import flash_diffusion_attn
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

K_ITER = 20


class TokenOTBlock(nn.Module):
    """Single Token OT block: MHA + balanced Sinkhorn transport (SPEC §8)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        r_0: float = 10.0,
        dropout: float = 0.0,
        block_idx: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 32
        self.block_idx = block_idx

        # ---- [Step 1] MHA Track ----
        self.ln_mha = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)  # MHA output gate
        self.pos_bias = PositionBias(n_heads, 68)

        # ---- FFN (SwiGLU) ----
        self.ln_ff = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_model * 4, d_model)
        self.ff_dropout = nn.Dropout(dropout)

        # ---- [Step 3] Sinkhorn Transport Track ----
        self.ln_sink = nn.LayerNorm(d_model)
        self.w_q_sink = nn.Linear(d_model, d_model, bias=False)
        self.w_k_sink = nn.Linear(d_model, d_model, bias=False)

        # Mixing gate α_h: sigmoid(alpha_h) blends feat vs geo
        # Init -1.0 → sigmoid(-1) ≈ 0.27 → start sequence/feature-heavy
        self.alpha_h = nn.Parameter(torch.full((n_heads,), -1.0))

        # Per-head characteristic distance (learnable, init r_0)
        self.r_h = nn.Parameter(torch.full((n_heads,), r_0))

        # Per-head entropic regularization (fixed buffer)
        nr = n_heads // 4
        self.register_buffer(
            "eps",
            torch.tensor(
                [0.5] * nr + [1.0] * nr + [2.0] * nr + [4.0] * nr,
                dtype=torch.float32,
            ),
        )

        # ---- [Step 4] Coordinate Update ----
        # Mobility gate G_i: sigmoid(Linear(h)) → per-residue ∈ (0, 1)
        self.ln_gate = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, 1, bias=True)
        # Intensity gate λ_h: tanh(param)/H → per-head bounded scale
        self.lambda_h_raw = nn.Parameter(torch.zeros(n_heads))  # tanh(0)=0 → dormant at init

    def forward(
        self,
        h: torch.Tensor,
        x_res: torch.Tensor,
        pos_bins: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:        (N, 512) or (B, N, 512) token representation
            x_res:    (N, 3) or (B, N, 3) token coordinates
            pos_bins: (N, N) or (B, N, N) int, 68-bin position encoding
            mask:     (B, N) bool/float, 1=real 0=pad. Only for batched.

        Returns:
            h, x_res
        """
        # Dual-mode: detect unbatched and add B dim
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)
            x_res = x_res.unsqueeze(0)
            pos_bins = pos_bins.unsqueeze(0)

        B, N, _ = h.shape
        H = self.n_heads
        d_h = self.d_head

        if mask is None:
            mask = h.new_ones(B, N)

        # ================================================================
        # [Step 1] MHA: Contextualization with 68-bin position bias
        # ================================================================
        h_n = self.ln_mha(h)
        Q = self.w_q(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        K = self.w_k(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)

        # Flash attention with 68-bin position bias
        G = self.w_g(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)  # (B, H, N, d_h)

        if _HAS_TRITON and h.is_cuda:
            att_out = flash_diffusion_attn(Q, K, V, self.pos_bias.weight,
                                           pos_bins, mask)
        else:
            from deepfold.model.kernels.flash_diffusion_attn import flash_diff_attn_ref
            att_out = flash_diff_attn_ref(Q.float(), K.float(), V.float(),
                                          self.pos_bias.weight, pos_bins,
                                          mask).to(Q.dtype)

        att_out = att_out.permute(0, 2, 1, 3).reshape(B, N, H * d_h)  # (B, N, d)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, N, H * d_h)
        h_update = torch.sigmoid(G_flat) * self.w_o(att_out)
        h_update = h_update * mask.unsqueeze(-1)
        h = h + h_update

        # ---- FFN (SwiGLU) ----
        ff_update = self.ff_dropout(self.swiglu(self.ln_ff(h)))
        ff_update = ff_update * mask.unsqueeze(-1)
        h = h + ff_update

        # ================================================================
        # [Step 2–3] Balanced Sinkhorn Transport
        # ================================================================
        h_s = self.ln_sink(h)
        Q_s = self.w_q_sink(h_s).view(B, N, H, d_h).permute(0, 2, 1, 3)
        K_s = self.w_k_sink(h_s).view(B, N, H, d_h).permute(0, 2, 1, 3)

        # L2-normalize for cosine cost
        Q_s = F.normalize(Q_s, dim=-1)
        K_s = F.normalize(K_s, dim=-1)

        # Balanced Sinkhorn → centroid (O(N) saved memory via checkpoint)
        x_centroid = balanced_sinkhorn_transport(
            Q_s, K_s, x_res, self.eps, self.alpha_h, self.r_h,
            K_iter=K_ITER, mask=mask,
        )
        # x_centroid: (B, H, N, 3)

        # ================================================================
        # [Step 4] Gated Coordinate Update
        # ================================================================
        # vec_h = centroid - x → displacement toward transport target
        vec_h = x_centroid - x_res.unsqueeze(1).to(x_centroid.dtype)  # (B, H, N, 3)

        # Mobility gate G_i: sigmoid → per-residue ∈ (0, 1)
        gate = torch.sigmoid(self.w_gate(self.ln_gate(h)))  # (B, N, 1)

        # Intensity gate λ_h: tanh(raw) / H → per-head bounded, averaged
        lam = torch.tanh(self.lambda_h_raw).view(1, H, 1, 1) / H
        weighted_disp = (lam * vec_h).sum(dim=1)  # (B, N, 3)

        x_update = gate * weighted_disp  # (B, N, 3)
        x_update = x_update * mask.unsqueeze(-1)
        x_res = x_res + x_update.to(x_res.dtype)

        if unbatched:
            return h.squeeze(0), x_res.squeeze(0)
        return h, x_res
