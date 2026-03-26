"""Token UOT+EGNN Block (SPEC §8). 48 blocks in trunk.

Supports both unbatched (N, d) and batched (B, N, d) inputs via dual-mode pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.kernels.flash_sinkhorn_attn import flash_sinkhorn_attn
from deepfold.model.position_encoding import PositionBias
from deepfold.model.primitives import SwiGLU

K_ITER = 20


class TokenUOTBlock(nn.Module):
    """Single Token UOT+EGNN block (SPEC §8)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        r_0: float = 10.0,
        block_idx: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % 4 == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 32
        self.r_0 = r_0
        self.block_idx = block_idx

        # Attention projections — after LN, so bias=False (SPEC §0)
        self.ln_attn = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # EGNN: per-head signed geometry step size — noise init (SPEC §8.2)
        self.gamma = nn.Parameter(torch.randn(n_heads) * 1e-4)

        # Per-layer position bias (SPEC §4.3)
        self.pos_bias = PositionBias(n_heads, 68)

        # Per-head geometry weight for cost matrix — sigmoid bounded (0,1)
        # Init -2.0 → sigmoid ≈ 0.12 (weak geometry bias initially)
        self.w_dist_logit = nn.Parameter(torch.full((n_heads,), -2.0))

        # Per-head entropic regularization (fixed buffer, SPEC §7.2)
        nr = n_heads // 4
        self.register_buffer(
            "eps",
            torch.tensor(
                [0.5] * nr + [1.0] * nr + [2.0] * nr + [4.0] * nr,
                dtype=torch.float32,
            ),
        )

        # SwiGLU transition (512 -> 2048 -> 512)
        self.ln_ff = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_model * 4, d_model)

    def forward(
        self,
        h: torch.Tensor,
        x_res: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        log_u_prev: torch.Tensor | None,
        log_v_prev: torch.Tensor | None,
        pos_bins: torch.Tensor,
        geo_gate: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h:          (N, 512) or (B, N, 512) token representation
            x_res:      (N, 3) or (B, N, 3) token coordinates
            mu:         (H, N) or (B, H, N) row marginals
            nu:         (H, N) or (B, H, N) column marginals
            log_u_prev: (H, N) or (B, H, N) or None — warm-start
            log_v_prev: (H, N) or (B, H, N) or None
            pos_bins:   (N, N) or (B, N, N) int, 68-bin position encoding
            geo_gate:   (H,) or None — sigma-gated for diffusion blocks
            mask:       (B, N) bool/float, 1=real 0=pad. Only for batched.

        Returns:
            h, x_res, log_u, log_v
        """
        # Dual-mode: detect unbatched and add B dim
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)
            x_res = x_res.unsqueeze(0)
            mu = mu.unsqueeze(0)
            nu = nu.unsqueeze(0)
            if log_u_prev is not None:
                log_u_prev = log_u_prev.unsqueeze(0)
            if log_v_prev is not None:
                log_v_prev = log_v_prev.unsqueeze(0)
            pos_bins = pos_bins.unsqueeze(0)

        B, N, _ = h.shape
        H = self.n_heads
        d_h = self.d_head

        if mask is None:
            mask = h.new_ones(B, N)

        # ---- Pre-norm ----
        h_n = self.ln_attn(h)  # (B, N, d_model)
        Q = self.w_q(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)  # (B, H, N, d_h)
        K = self.w_k(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(h_n).view(B, N, H, d_h).permute(0, 2, 1, 3)

        # ---- Flash Sinkhorn Attention: O(N) memory ----
        Q_ln = F.layer_norm(Q, [d_h])
        K_ln = F.layer_norm(K, [d_h])

        pos_weight = self.pos_bias.weight  # (H, 68)

        w_dist = self.w_dist_logit.sigmoid()  # (H,) bounded (0, 1)
        effective_w_dist = w_dist
        if geo_gate is not None:
            effective_w_dist = geo_gate * w_dist

        # FP32 for log marginals — BF16 underflows the softmax Jacobian
        # in the backward, killing gradient to mu_proj/nu_proj.
        log_mu = torch.log(mu.float().clamp(min=1e-8))  # (B, H, N)
        log_nu = torch.log(nu.float().clamp(min=1e-8))

        # Flash Sinkhorn: O(N) memory for both training and inference.
        # Triton forward computes cost tiles on the fly (content + pos_bias + geo_bias).
        # Triton backward uses exact unrolled gradients through K Sinkhorn iterations
        # (not IFT/CG approximate), giving correct gradients for all parameters
        # including mu_proj, nu_proj, coevol_to_marginal, w_dist_logit, pos_bias.
        o, x_centroid, log_u, log_v = flash_sinkhorn_attn(
            Q_ln,
            K_ln,
            V,
            G,
            x_res,
            pos_weight,
            pos_bins,
            self.eps,
            effective_w_dist,
            log_mu,
            log_nu,
            K_iter=K_ITER,
            lam=1.0,
            r_0=self.r_0,
            log_u_init=log_u_prev,
            log_v_init=log_v_prev,
            mask=mask,
        )
        # o: (B, N, H*d_h), x_centroid: (B, H, N, 3)
        # log_u: (B, H, N), log_v: (B, H, N)

        # h update (invariant)
        h_update = self.w_o(o.to(h.dtype))  # (B, N, d_model)
        # Zero out padded positions
        h_update = h_update * mask.unsqueeze(-1)
        h = h + h_update

        # ---- EGNN x_res update (equivariant, SPEC §8) ----
        delta = x_res.unsqueeze(1).float() - x_centroid  # (B, H, N, 3)
        x_update = torch.einsum("h,bhnc->bnc", self.gamma.tanh() / self.n_heads, delta)  # (B, N, 3)
        x_update = x_update * mask.unsqueeze(-1)
        x_res = x_res + x_update.to(x_res.dtype)

        # ---- SwiGLU transition ----
        ff_update = self.swiglu(self.ln_ff(h))
        ff_update = ff_update * mask.unsqueeze(-1)
        h = h + ff_update

        log_u_out = log_u.to(h.dtype)
        log_v_out = log_v.to(h.dtype)

        if unbatched:
            return (
                h.squeeze(0),
                x_res.squeeze(0),
                log_u_out.squeeze(0),
                log_v_out.squeeze(0),
            )
        return h, x_res, log_u_out, log_v_out
