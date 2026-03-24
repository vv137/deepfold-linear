"""UOT-Sinkhorn solver with unrolled differentiable backward (SPEC §7.2-7.3, v4.4).

v4.4 change: IFT backward replaced with plain autograd through K iterations.
Standard torch.autograd differentiates through the logsumexp operations directly,
giving exact gradients for the actual K-step computation performed.

Convergence check follows flash-sinkhorn: L∞ on potential change, checked every
`check_every` iterations to amortise the GPU→CPU sync cost.

Supports both unbatched (H,N,N) and batched (B,H,N,N) inputs with optional
padding mask for variable-length sequences in a batch.
"""

import torch


def sinkhorn_solve(
    C: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    eps: torch.Tensor,
    lam: float = 1.0,
    K: int = 7,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
    threshold: float | None = None,
    check_every: int = 2,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log-domain Sinkhorn iterations — fully differentiable (SPEC §7.3).

    Standard autograd propagates through all executed iterations. With gradient
    checkpointing per trunk block, only one block's intermediates exist at a time.

    Early stopping (optional): when ``threshold`` is not None, checks
    max(|Δlog_u|∞, |Δlog_v|∞) < threshold every ``check_every`` iterations.
    Each check requires a GPU→CPU sync (~2ms), so keep check_every ≥ 2.

    Args:
        C:            (H, N, N) or (B, H, N, N) cost matrix
        log_mu:       (H, N) or (B, H, N) log row marginals
        log_nu:       (H, N) or (B, H, N) log column marginals
        eps:          (H,) fixed per-head entropic regularization
        lam:          scalar marginal penalty (1.0)
        K:            max number of iterations
        log_u_init, log_v_init: warm-start (H, N) / (B, H, N) or None
        threshold:    early-stop L∞ tolerance (None = fixed K iterations)
        check_every:  check convergence every N iterations
        mask:         (B, N) bool/float, 1=real 0=pad. Only used with batched input.

    Returns:
        log_u, log_v: (H, N) or (B, H, N) dual variables
    """
    unbatched = C.dim() == 3
    if unbatched:
        C = C.unsqueeze(0)
        log_mu = log_mu.unsqueeze(0)
        log_nu = log_nu.unsqueeze(0)
        if log_u_init is not None:
            log_u_init = log_u_init.unsqueeze(0)
        if log_v_init is not None:
            log_v_init = log_v_init.unsqueeze(0)

    # kappa: (H,) — broadcasts over batch
    kappa = lam / (lam + eps)  # (H,)

    log_u = log_u_init if log_u_init is not None else torch.zeros_like(log_mu)
    log_v = log_v_init if log_v_init is not None else torch.zeros_like(log_nu)

    log_K = -C / eps[None, :, None, None]  # (1, H, N, N) broadcasts to (B, H, N, N)

    # Precompute mask bias for masked logsumexp
    if mask is not None:
        mask_f = mask.float()
        col_mask_bias = (1 - mask_f)[:, None, None, :] * (-1e9)  # (B, 1, 1, N)
        row_mask_bias = (1 - mask_f)[:, None, :, None] * (-1e9)  # (B, 1, N, 1)
    else:
        col_mask_bias = 0
        row_mask_bias = 0

    for i in range(K):
        log_u_prev = log_u
        log_v_prev = log_v

        # Row update — mask columns before logsumexp over dim=-1
        log_u = kappa[None, :, None] * (
            log_mu
            - torch.logsumexp(log_K + log_v[:, :, None, :] + col_mask_bias, dim=-1)
        )
        # Column update — mask rows before logsumexp over dim=-2
        log_v = kappa[None, :, None] * (
            log_nu
            - torch.logsumexp(log_K + log_u[:, :, :, None] + row_mask_bias, dim=-2)
        )

        # Early stopping: L∞ on potential change (flash-sinkhorn convention)
        if threshold is not None and (i + 1) % check_every == 0:
            u_change = (log_u - log_u_prev).abs().max().item()
            v_change = (log_v - log_v_prev).abs().max().item()
            if max(u_change, v_change) < threshold:
                break

    # Zero out padded positions in output duals
    if mask is not None:
        mask_bhn = mask[:, None, :]  # (B, 1, N)
        log_u = log_u * mask_bhn
        log_v = log_v * mask_bhn

    if unbatched:
        return log_u.squeeze(0), log_v.squeeze(0)
    return log_u, log_v


def compute_transport_output(
    V: torch.Tensor,
    G: torch.Tensor,
    log_u: torch.Tensor,
    log_v: torch.Tensor,
    C: torch.Tensor,
    eps: torch.Tensor,
    x_res: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Transport-weighted average with running-max numerical stability (SPEC §7.4).

    Args:
        V:     (H, N, d_h) or (B, H, N, d_h) value vectors
        G:     (H, N, d_h) or (B, H, N, d_h) gate vectors
        log_u: (H, N) or (B, H, N) row dual
        log_v: (H, N) or (B, H, N) column dual
        C:     (H, N, N) or (B, H, N, N) cost matrix
        eps:   (H,) per-head regularization
        x_res: (N, 3) or (B, N, 3) optional coordinates for EGNN centroid
        mask:  (B, N) bool/float, 1=real 0=pad. Only used with batched input.

    Returns:
        o:       (N, H*d_h) or (B, N, H*d_h) gated output
        T_norm:  (H, N, N) or (B, H, N, N) normalized transport (for EGNN)
        x_centroid: (H, N, 3) or (B, H, N, 3) or None if x_res is None
    """
    unbatched = C.dim() == 3
    if unbatched:
        C = C.unsqueeze(0)
        V = V.unsqueeze(0)
        G = G.unsqueeze(0)
        log_u = log_u.unsqueeze(0)
        log_v = log_v.unsqueeze(0)
        if x_res is not None:
            x_res = x_res.unsqueeze(0)

    log_K = -C / eps[None, :, None, None]  # (B, H, N, N)

    # Running-max trick
    log_score = log_u[:, :, :, None] + log_K + log_v[:, :, None, :]  # (B, H, N, N)

    # Mask padded columns with -inf before exp
    if mask is not None:
        mask_f = mask.float()
        col_mask_bias = (1 - mask_f)[:, None, None, :] * (-1e9)  # (B, 1, 1, N)
        log_score = log_score + col_mask_bias

    row_max = log_score.max(dim=-1, keepdim=True).values  # (B, H, N, 1)

    T = torch.exp(log_score - row_max)  # (B, H, N, N) safe: <= 1
    T_sum = T.sum(dim=-1, keepdim=True)  # (B, H, N, 1)
    T_norm = T / (T_sum + 1e-6)  # (B, H, N, N) row-stochastic

    # Note: ∂T_norm/∂log_u = 0 (log_u cancels in T/row_sum). The gradient
    # to log_mu flows through g_v → Sinkhorn iterations → log_mu, NOT
    # through T_norm directly. This is correct with unrolled autograd.

    # Zero out padded rows
    if mask is not None:
        T_norm = T_norm * mask_f[:, None, :, None]

    # Transport-weighted average of values
    O_avg = torch.einsum("bhnm,bhmd->bhnd", T_norm, V)  # (B, H, N, d_h)

    # Gating
    o = torch.sigmoid(G) * O_avg  # (B, H, N, d_h)
    B, H, N, d_h = o.shape
    o = o.permute(0, 2, 1, 3).reshape(B, N, H * d_h)  # (B, N, H*d_h)

    # Zero out padded positions in output
    if mask is not None:
        o = o * mask_f[:, :, None]

    # EGNN centroid
    x_centroid = None
    if x_res is not None:
        x_centroid = torch.einsum("bhnm,bmc->bhnc", T_norm, x_res)  # (B, H, N, 3)

    if unbatched:
        xc_out = x_centroid.squeeze(0) if x_centroid is not None else None
        return o.squeeze(0), T_norm.squeeze(0), xc_out
    return o, T_norm, x_centroid
