"""Diffusion Module (SPEC §9, §12 — v4.5). ~4.2M params.

v4.5: No UOT blocks in diffusion. Trunk h_res provides all inter-token
context. End-to-end gradient from diffusion loss through h_res into trunk.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.model.primitives import SwiGLU, zero_init_linear


# ============================================================================
# EDM Schedule (SPEC §12)
# ============================================================================

SIGMA_DATA = 16.0
SIGMA_MAX = 160.0
SIGMA_MIN = 0.002
P_MEAN = -1.2
P_STD = 1.2
RHO = 7.0


def edm_preconditioning(sigma: torch.Tensor):
    """EDM preconditioning functions (SPEC §11.1)."""
    sigma_data = SIGMA_DATA
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1.0 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = sigma.log() / 4.0
    return c_skip, c_out, c_in, c_noise


def sample_training_sigma(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample training noise levels: ln(sigma) ~ N(P_mean, P_std^2) (SPEC §12)."""
    ln_sigma = torch.randn(batch_size, device=device) * P_STD + P_MEAN
    return ln_sigma.exp()


def karras_schedule(n_steps: int, device: torch.device) -> torch.Tensor:
    """Karras noise schedule for inference (SPEC §12)."""
    rho_inv = 1.0 / RHO
    sigmas = (
        SIGMA_MAX**rho_inv
        + torch.arange(n_steps, device=device)
        / (n_steps - 1)
        * (SIGMA_MIN**rho_inv - SIGMA_MAX**rho_inv)
    ) ** RHO
    return sigmas


def timestep_fourier_embedding(c_noise: torch.Tensor, d: int = 128) -> torch.Tensor:
    """Fourier embedding for timestep conditioning."""
    half_d = d // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_d, device=c_noise.device, dtype=c_noise.dtype)
        / half_d
    )
    args = c_noise.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (*, d)


# ============================================================================
# Atom Block (SPEC §9.3)
# ============================================================================


class AtomBlock(nn.Module):
    """Single atom-level block with AdaLN conditioning (SPEC §9.3)."""

    def __init__(self, d_atom: int = 128, d_model: int = 512, n_heads: int = 4):
        super().__init__()
        self.d_atom = d_atom
        self.n_heads = n_heads
        self.d_head = d_atom // n_heads  # 32

        # AdaLN
        self.cond_proj = nn.Linear(d_model, d_atom)
        self.adaln1 = nn.Linear(d_atom, d_atom * 2)
        self.adaln2 = nn.Linear(d_atom, d_atom * 2)

        # Self-attention — after AdaLN (includes LN), so bias=False (SPEC §0)
        self.ln_attn = nn.LayerNorm(d_atom)
        self.w_q = nn.Linear(d_atom, d_atom, bias=False)
        self.w_k = nn.Linear(d_atom, d_atom, bias=False)
        self.w_v = nn.Linear(d_atom, d_atom, bias=False)
        self.w_g = nn.Linear(d_atom, d_atom, bias=False)
        self.w_o = nn.Linear(d_atom, d_atom, bias=False)

        # Pair bias
        self.pair_bias = nn.Linear(16, n_heads)

        # Conditioned gating
        self.cond_gate1 = nn.Linear(d_atom, d_atom)
        self.cond_gate2 = nn.Linear(d_atom, d_atom)

        # SwiGLU transition
        self.swiglu = SwiGLU(d_atom, d_atom * 4, d_atom)

    def forward(
        self,
        q: torch.Tensor,
        c_atom: torch.Tensor,
        h_cond: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        t_emb: torch.Tensor,
        token_idx: torch.Tensor,
        atom_pad_mask: torch.Tensor | None = None,
        pair_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q:              (B, N_atom, 128) or (N_atom, 128) atom representation
            c_atom:         (B, N_atom, 128) or (N_atom, 128) frozen reference
            h_cond:         (B, N, 512) or (N, 512) timestep-conditioned token repr
            p_lm:           (B, n_pairs, 16) or (n_pairs, 16) frozen atom pair features
            p_lm_idx:       (B, n_pairs, 2) or (n_pairs, 2) pair indices
            t_emb:          (B, 128) or (128,) timestep embedding
            token_idx:      (B, N_atom) or (N_atom,) atom-to-token mapping
            atom_pad_mask:  (B, N_atom) float, 1=valid 0=pad. None means all valid.
            pair_valid_mask: (B, n_pairs) float, 1=valid 0=pad. None means all valid.
        """
        # Dual-mode: handle unbatched input
        if q.dim() == 2:
            return self._forward_unbatched(
                q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx
            )

        B, N_atom, _ = q.shape
        H = self.n_heads
        d_h = self.d_head

        if atom_pad_mask is None:
            atom_pad_mask = q.new_ones(B, N_atom)

        # Gather h_cond at token_idx: (B, N_atom, d_model)
        d_model = h_cond.shape[-1]
        h_cond_atoms = torch.gather(
            h_cond, 1, token_idx.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (B, N_atom, d_model)

        # AdaLN conditioning: timestep + token info
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0).expand(B, -1)  # (B, 128)
        cond = t_emb.unsqueeze(1) + self.cond_proj(h_cond_atoms)  # (B, N_atom, 128)
        gamma1, beta1 = self.adaln1(cond).chunk(2, dim=-1)

        # AdaLN-normalized input
        q_n = (1 + gamma1) * self.ln_attn(q) + beta1  # (B, N_atom, 128)

        # Self-attention
        Q = self.w_q(q_n).view(B, N_atom, H, d_h).permute(0, 2, 1, 3)  # (B,H,N,d_h)
        K = self.w_k(q_n).view(B, N_atom, H, d_h).permute(0, 2, 1, 3)
        V = self.w_v(q_n).view(B, N_atom, H, d_h).permute(0, 2, 1, 3)
        G = self.w_g(q_n).view(B, N_atom, H, d_h).permute(0, 2, 1, 3)

        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / (d_h**0.5)  # (B,H,N,N)

        # Pair bias for local atom pairs
        if p_lm.shape[-2] > 0:
            bias = self.pair_bias(p_lm)  # (B, n_pairs, H)
            pair_bias = torch.zeros(
                B, H, N_atom, N_atom, device=q.device, dtype=bias.dtype
            )
            # Scatter pair biases per batch element
            for b in range(B):
                if pair_valid_mask is not None:
                    valid = pair_valid_mask[b] > 0  # (n_pairs,)
                    idx = p_lm_idx[b][valid]  # (n_valid, 2)
                    b_bias = bias[b][valid]  # (n_valid, H)
                else:
                    idx = p_lm_idx[b]
                    b_bias = bias[b]
                if idx.shape[0] > 0:
                    pair_bias[b, :, idx[:, 0], idx[:, 1]] = b_bias.T
            scores = scores + pair_bias

        # Window mask: attend within ±16 atoms
        atom_pos = torch.arange(N_atom, device=q.device)
        window_mask = (atom_pos.unsqueeze(0) - atom_pos.unsqueeze(1)).abs() <= 16
        scores = scores.masked_fill(~window_mask[None, None, :, :], float("-inf"))

        # Atom padding mask: mask out padded key positions
        mask_bias = (1 - atom_pad_mask.float())[:, None, None, :] * (-1e9)
        scores = scores + mask_bias

        attn = F.softmax(scores, dim=-1)  # (B, H, N_atom, N_atom)
        att_out = torch.einsum("bhij,bhjd->bhid", attn, V)  # (B,H,N,d_h)
        att_out = att_out.permute(0, 2, 1, 3).reshape(B, N_atom, -1)  # (B,N,H*d_h)
        G_flat = G.permute(0, 2, 1, 3).reshape(B, N_atom, -1)
        q = q + torch.sigmoid(G_flat) * self.w_o(att_out)

        # Conditioned gating (AF3 style)
        q = q + torch.sigmoid(self.cond_gate1(c_atom)) * q

        # SwiGLU + AdaLN
        gamma2, beta2 = self.adaln2(cond).chunk(2, dim=-1)
        q_n2 = (1 + gamma2) * F.layer_norm(q, [self.d_atom]) + beta2
        q = q + torch.sigmoid(self.cond_gate2(c_atom)) * self.swiglu(q_n2)

        # Zero out padded positions
        q = q * atom_pad_mask.unsqueeze(-1)

        return q

    def _forward_unbatched(
        self,
        q: torch.Tensor,
        c_atom: torch.Tensor,
        h_cond: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        t_emb: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Original unbatched forward path for backward compatibility."""
        N_atom = q.shape[0]
        H = self.n_heads
        d_h = self.d_head

        # AdaLN conditioning: timestep + token info
        cond = t_emb.unsqueeze(0) + self.cond_proj(h_cond[token_idx])  # (N_atom, 128)
        gamma1, beta1 = self.adaln1(cond).chunk(2, dim=-1)

        # AdaLN-normalized input
        q_n = (1 + gamma1) * self.ln_attn(q) + beta1  # (N_atom, 128)

        # Self-attention (local window via token masking)
        Q = self.w_q(q_n).view(N_atom, H, d_h)
        K = self.w_k(q_n).view(N_atom, H, d_h)
        V = self.w_v(q_n).view(N_atom, H, d_h)
        G = self.w_g(q_n).view(N_atom, H, d_h)

        scores = torch.einsum("ihd,jhd->hij", Q, K) / (d_h**0.5)  # (H, N_atom, N_atom)

        # Pair bias for local atom pairs
        if p_lm.shape[0] > 0:
            bias = self.pair_bias(p_lm)  # (n_pairs, H)
            pair_bias = torch.zeros(
                H, N_atom, N_atom, device=q.device, dtype=bias.dtype
            )
            pair_bias[:, p_lm_idx[:, 0], p_lm_idx[:, 1]] = bias.T
            scores = scores + pair_bias

        # Window mask: attend within ±16 atoms (window=32)
        atom_pos = torch.arange(N_atom, device=q.device)
        window_mask = (atom_pos.unsqueeze(0) - atom_pos.unsqueeze(1)).abs() <= 16
        scores = scores.masked_fill(~window_mask[None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (H, N_atom, N_atom)
        att_out = torch.einsum("hij,jhd->ihd", attn, V)  # (H, N_atom, d_h)
        att_out = att_out.permute(1, 0, 2).reshape(N_atom, -1)  # (N_atom, H*d_h)
        G_flat = G.reshape(N_atom, -1)  # G is (N_atom, H, d_h)
        q = q + torch.sigmoid(G_flat) * self.w_o(att_out)

        # Conditioned gating (AF3 style)
        q = q + torch.sigmoid(self.cond_gate1(c_atom)) * q

        # SwiGLU + AdaLN
        gamma2, beta2 = self.adaln2(cond).chunk(2, dim=-1)
        q_n2 = (1 + gamma2) * F.layer_norm(q, [self.d_atom]) + beta2
        q = q + torch.sigmoid(self.cond_gate2(c_atom)) * self.swiglu(q_n2)

        return q


# ============================================================================
# Diffusion Module (SPEC §9.2 — v4.5)
# ============================================================================


class DiffusionModule(nn.Module):
    """Diffusion module: 10 atom blocks, no UOT blocks (SPEC §9, v4.5).

    v4.5: Trunk h_res already contains all inter-token structural context
    from 48 UOT+EGNN blocks. No need to re-run UOT per diffusion step.
    h_res is NOT detached — end-to-end gradient flows back into trunk.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_atom: int = 128,
        n_atom_blocks: int = 10,
        sigma_data: float = SIGMA_DATA,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_atom = d_atom
        self.sigma_data = sigma_data

        # Timestep -> token-level conditioning (SPEC §9.2)
        self.t_proj = nn.Linear(d_atom, d_model)

        # Atom coordinate embedding
        self.atom_coord_proj = nn.Linear(3, d_atom)

        # Atom blocks
        self.atom_blocks = nn.ModuleList(
            [AtomBlock(d_atom, d_model) for _ in range(n_atom_blocks)]
        )

        # Final coordinate output — zero init (SPEC §9.2)
        self.ln_out = nn.LayerNorm(d_atom)
        self.coord_out = zero_init_linear(d_atom, 3)

    def forward(
        self,
        h_res: torch.Tensor,
        c_atom: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        x_atom_noisy: torch.Tensor,
        sigma: torch.Tensor,
        token_idx: torch.Tensor,
        token_pad_mask: torch.Tensor | None = None,
        atom_pad_mask: torch.Tensor | None = None,
        pair_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single diffusion denoising step (SPEC §9.2, v4.5).

        Args:
            h_res:          (B, N, 512) or (N, 512) from trunk (NOT detached)
            c_atom:         (B, N_atom, 128) or (N_atom, 128) frozen reference
            p_lm:           (B, n_pairs, 16) or (n_pairs, 16) frozen atom pairs
            p_lm_idx:       (B, n_pairs, 2) or (n_pairs, 2)
            x_atom_noisy:   (B, N_atom, 3) or (N_atom, 3) noisy coordinates
            sigma:          (B,) or scalar, current noise level
            token_idx:      (B, N_atom) or (N_atom,) atom-to-token
            token_pad_mask: (B, N) float, 1=valid 0=pad. None means all valid.
            atom_pad_mask:  (B, N_atom) float, 1=valid 0=pad. None means all valid.
            pair_valid_mask: (B, n_pairs) float, 1=valid 0=pad. None means all valid.

        Returns:
            x_atom_new: (B, N_atom, 3) or (N_atom, 3) denoised coordinates
        """
        # Dual-mode: handle unbatched input
        if h_res.dim() == 2:
            return self._forward_unbatched(
                h_res, c_atom, p_lm, p_lm_idx, x_atom_noisy, sigma, token_idx
            )

        B = h_res.shape[0]

        if atom_pad_mask is None:
            atom_pad_mask = x_atom_noisy.new_ones(x_atom_noisy.shape[:2])

        # 1. Timestep embedding
        # Ensure sigma is (B,)
        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        _, c_out, _, c_noise = edm_preconditioning(sigma)  # each (B,)
        t_emb = timestep_fourier_embedding(c_noise, d=self.d_atom)  # (B, d_atom)

        # 2. Token-level timestep conditioning (SPEC §9.2 v4.5)
        # t_proj(t_emb): (B, d_model) -> broadcast to (B, 1, d_model)
        h_cond = h_res + self.t_proj(t_emb).unsqueeze(1)  # (B, N, 512)

        # 3. Atom embedding
        q = c_atom + self.atom_coord_proj(
            x_atom_noisy / self.sigma_data
        )  # (B, N_atom, 128)

        # 4. Atom blocks
        for block in self.atom_blocks:
            q = block(
                q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx,
                atom_pad_mask=atom_pad_mask,
                pair_valid_mask=pair_valid_mask,
            )

        # 5. Coordinate update
        delta_x = self.coord_out(self.ln_out(q))  # (B, N_atom, 3)
        # c_out is (B,) -> (B, 1, 1) for broadcast
        x_atom_new = x_atom_noisy + c_out[:, None, None] * delta_x

        # Re-center per sample (float32 hygiene, prevents drift)
        mask_f = atom_pad_mask.float()  # (B, N_atom)
        mask_sum = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        x_f32 = x_atom_new.float()
        com = (x_f32 * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        x_atom_new = (x_f32 - com).to(x_atom_new.dtype)

        # Zero out padded atom positions
        x_atom_new = x_atom_new * atom_pad_mask.unsqueeze(-1)

        return x_atom_new

    def _forward_unbatched(
        self,
        h_res: torch.Tensor,
        c_atom: torch.Tensor,
        p_lm: torch.Tensor,
        p_lm_idx: torch.Tensor,
        x_atom_noisy: torch.Tensor,
        sigma: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Original unbatched forward for backward compatibility."""
        # 1. Timestep embedding
        _, c_out, _, c_noise = edm_preconditioning(sigma)
        t_emb = timestep_fourier_embedding(c_noise, d=self.d_atom)  # (d_atom,)

        # 2. Token-level timestep conditioning (SPEC §9.2 v4.5)
        h_cond = h_res + self.t_proj(t_emb)  # (N, 512) broadcast

        # 3. Atom embedding
        q = c_atom + self.atom_coord_proj(
            x_atom_noisy / self.sigma_data
        )  # (N_atom, 128)

        # 4. Atom blocks
        for block in self.atom_blocks:
            q = block(q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx)

        # 5. Coordinate update
        delta_x = self.coord_out(self.ln_out(q))  # (N_atom, 3)
        x_atom_new = x_atom_noisy + c_out * delta_x

        # Re-center (float32 hygiene, prevents drift during multi-step sampling)
        x_atom_new = x_atom_new - x_atom_new.float().mean(dim=0, keepdim=True).to(
            x_atom_new.dtype
        )

        return x_atom_new
