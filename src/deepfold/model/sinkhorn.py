"""UOT-Sinkhorn solver with unrolled differentiable backward (SPEC §7.2-7.3, v4.4).

v4.4 change: IFT backward replaced with plain autograd through K iterations.
Standard torch.autograd differentiates through the logsumexp operations directly,
giving exact gradients for the actual K-step computation performed.

Convergence check follows flash-sinkhorn: L∞ on potential change, checked every
`check_every` iterations to amortise the GPU→CPU sync cost.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log-domain Sinkhorn iterations — fully differentiable (SPEC §7.3).

    Standard autograd propagates through all executed iterations. With gradient
    checkpointing per trunk block, only one block's intermediates exist at a time.

    Early stopping (optional): when ``threshold`` is not None, checks
    max(|Δlog_u|∞, |Δlog_v|∞) < threshold every ``check_every`` iterations.
    Each check requires a GPU→CPU sync (~2ms), so keep check_every ≥ 2.

    Args:
        C:            (H, N, N) cost matrix
        log_mu:       (H, N) log row marginals
        log_nu:       (H, N) log column marginals
        eps:          (H,) fixed per-head entropic regularization
        lam:          scalar marginal penalty (1.0)
        K:            max number of iterations
        log_u_init, log_v_init: warm-start (H, N) or None
        threshold:    early-stop L∞ tolerance (None = fixed K iterations)
        check_every:  check convergence every N iterations

    Returns:
        log_u, log_v: (H, N) dual variables
    """
    kappa = lam / (lam + eps)  # (H,)

    log_u = log_u_init if log_u_init is not None else torch.zeros_like(log_mu)
    log_v = log_v_init if log_v_init is not None else torch.zeros_like(log_nu)

    log_K = -C / eps[:, None, None]  # (H, N, N)

    for i in range(K):
        log_u_prev = log_u
        log_v_prev = log_v

        # Row update
        log_u = kappa[:, None] * (
            log_mu - torch.logsumexp(log_K + log_v[:, None, :], dim=-1)
        )
        # Column update
        log_v = kappa[:, None] * (
            log_nu - torch.logsumexp(log_K + log_u[:, :, None], dim=-2)
        )

        # Early stopping: L∞ on potential change (flash-sinkhorn convention)
        if threshold is not None and (i + 1) % check_every == 0:
            u_change = (log_u - log_u_prev).abs().max().item()
            v_change = (log_v - log_v_prev).abs().max().item()
            if max(u_change, v_change) < threshold:
                break

    return log_u, log_v


def compute_transport_output(
    V: torch.Tensor,
    G: torch.Tensor,
    log_u: torch.Tensor,
    log_v: torch.Tensor,
    C: torch.Tensor,
    eps: torch.Tensor,
    x_res: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Transport-weighted average with running-max numerical stability (SPEC §7.4).

    Args:
        V:     (H, N, d_h) value vectors
        G:     (H, N, d_h) gate vectors
        log_u: (H, N) row dual
        log_v: (H, N) column dual
        C:     (H, N, N) cost matrix
        eps:   (H,) per-head regularization
        x_res: (N, 3) optional coordinates for EGNN centroid

    Returns:
        o:       (N, H*d_h) gated output
        T_norm:  (H, N, N) row-normalized transport (for EGNN)
        x_centroid: (H, N, 3) or None if x_res is None
    """
    log_K = -C / eps[:, None, None]  # (H, N, N)

    # Running-max trick
    log_score = log_u[:, :, None] + log_K + log_v[:, None, :]  # (H, N, N)
    row_max = log_score.max(dim=-1, keepdim=True).values  # (H, N, 1)

    T = torch.exp(log_score - row_max)  # (H, N, N) safe: <= 1
    T_sum = T.sum(dim=-1, keepdim=True)  # (H, N, 1)
    T_norm = T / (T_sum + 1e-6)  # (H, N, N) row-stochastic

    # Transport-weighted average of values
    O_avg = torch.einsum("hnm,hmd->hnd", T_norm, V)  # (H, N, d_h)

    # Gating
    o = torch.sigmoid(G) * O_avg  # (H, N, d_h)
    H, N, d_h = o.shape
    o = o.permute(1, 0, 2).reshape(N, H * d_h)  # (N, H*d_h)

    # EGNN centroid
    x_centroid = None
    if x_res is not None:
        x_centroid = torch.einsum("hnm,mc->hnc", T_norm, x_res)  # (H, N, 3)

    return o, T_norm, x_centroid
