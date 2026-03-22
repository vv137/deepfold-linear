"""Token UOT+EGNN Block (SPEC §8). 48 blocks in trunk, 2 in diffusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.sinkhorn import sinkhorn_solve, compute_transport_output
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

        # EGNN: per-head signed geometry step size — zeros init (SPEC §8.2)
        self.gamma = nn.Parameter(torch.zeros(n_heads))

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
        w_rel_res: torch.Tensor,
        w_dist: torch.Tensor,
        pos_bins: torch.Tensor,
        geo_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h:          (N, 512) token representation
            x_res:      (N, 3) token coordinates
            mu:         (H, N) row marginals
            nu:         (H, N) column marginals
            log_u_prev: (H, N) or None — warm-start
            log_v_prev: (H, N) or None
            w_rel_res:  PositionBias module or (H, N, N) precomputed
            w_dist:     (H,) per-head geometry weight
            pos_bins:   (N, N) int, 68-bin position encoding
            geo_gate:   (H,) or None — sigma-gated for diffusion blocks

        Returns:
            h, x_res, log_u, log_v
        """
        N = h.shape[0]
        H = self.n_heads
        d_h = self.d_head

        # ---- Pre-norm ----
        h_n = self.ln_attn(h)
        Q = self.w_q(h_n).view(N, H, d_h)
        K = self.w_k(h_n).view(N, H, d_h)
        V = self.w_v(h_n).view(N, H, d_h)
        G = self.w_g(h_n).view(N, H, d_h)

        # Transpose to (H, N, d_h)
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        G = G.permute(1, 0, 2)

        # ---- Cost matrix (SPEC §7.1) ----
        # LN on Q, K for Sinkhorn stability
        Q_ln = F.layer_norm(Q, [d_h])
        K_ln = F.layer_norm(K, [d_h])
        content = -torch.einsum("hid,hjd->hij", Q_ln, K_ln) / (d_h**0.5)  # (H, N, N)

        # Position bias
        if isinstance(w_rel_res, torch.Tensor) and w_rel_res.dim() == 3:
            pos_bias = w_rel_res
        else:
            pos_bias = w_rel_res(pos_bins)  # (H, N, N)

        # Geometry bias
        dist = torch.cdist(x_res, x_res)  # (N, N)
        f_dist = dist / (self.r_0 + dist)  # (N, N), bounded [0, 1)

        # Per-head geometry weight
        effective_w_dist = w_dist  # (H,)
        if geo_gate is not None:
            effective_w_dist = geo_gate * w_dist  # sigma-gated for diffusion

        geo_bias = effective_w_dist[:, None, None] * f_dist.unsqueeze(0)  # (H, N, N)

        C = content + pos_bias + geo_bias  # (H, N, N)

        # ---- Sinkhorn ----
        init_u = log_u_prev
        init_v = log_v_prev

        log_mu = torch.log(mu.clamp(min=1e-8))
        log_nu = torch.log(nu.clamp(min=1e-8))

        # FP32 for Sinkhorn log-domain stability (SPEC §18).
        # .float() casts are differentiable — gradients flow through dtype conversion.
        C_fp32 = C.float()
        log_mu_fp32 = log_mu.float()
        log_nu_fp32 = log_nu.float()
        eps_fp32 = self.eps.float()
        init_u_fp32 = init_u.float()
        init_v_fp32 = init_v.float()

        log_u, log_v = sinkhorn_solve(
            C_fp32,
            log_mu_fp32,
            log_nu_fp32,
            eps=eps_fp32,
            lam=1.0,
            K=K_ITER,
            log_u_init=init_u_fp32,
            log_v_init=init_v_fp32,
        )

        # ---- Transport output + EGNN ----
        o, T_norm, x_centroid = compute_transport_output(
            V.float(), G.float(), log_u, log_v, C_fp32, eps_fp32, x_res.float()
        )

        # h update (invariant)
        h = h + self.w_o(o.to(h.dtype))

        # ---- EGNN x_res update (equivariant, SPEC §8) ----
        # Three gradient sources flow through x_res per the backward analysis:
        # 1. EGNN: ∂L/∂x through centroid displacement (via gamma)
        # 2. Geometry cost: ∂L/∂x through distance term in C_ij (via IFT → grad_C)
        # 3. Pass-through from next block
        delta = x_res.unsqueeze(0).float() - x_centroid  # (H, N, 3)
        x_update = torch.einsum("h,hnc->nc", self.gamma, delta)  # (N, 3)
        x_res = x_res + x_update.to(x_res.dtype)

        # ---- SwiGLU transition ----
        h = h + self.swiglu(self.ln_ff(h))

        return h, x_res, log_u.to(h.dtype), log_v.to(h.dtype)
