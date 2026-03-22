"""Conjugate gradient solvers.

Adapted from flash-sinkhorn (references/flash-sinkhorn/).
Includes:
- Standard CG for symmetric positive definite systems (Ax = b)
- Steihaug-CG for trust-region subproblems (handles indefinite Hessians)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class CGInfo:
    """Convergence info from CG solve."""

    converged: bool
    iters: int
    residual: float
    initial_residual: float


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    max_iter: int = 300,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    stabilise_every: int = 0,
) -> tuple[torch.Tensor, CGInfo]:
    """CG for symmetric positive definite systems.

    Solves A @ x = b given matvec = A @ v.

    Args:
        matvec: Matrix-vector product function A @ v.
        b:      Right-hand side vector.
        x0:     Initial guess (warm-start). None → zeros.
        max_iter: Maximum iterations.
        rtol, atol: Convergence tolerances on residual norm.
        preconditioner: M^{-1} @ v function. None → identity.
        stabilise_every: Recompute true residual every N iters (0 = disabled).

    Returns:
        x: Solution vector.
        info: CGInfo with convergence details.
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - matvec(x)

    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = torch.dot(r, z)

    init_res = float(torch.linalg.norm(r).detach().cpu())
    tol = max(atol, rtol * init_res)

    for it in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if denom.abs() == 0:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if stabilise_every and (it + 1) % stabilise_every == 0:
            r = b - matvec(x)

        res = float(torch.linalg.norm(r).detach().cpu())
        if res <= tol:
            return x, CGInfo(
                converged=True, iters=it + 1, residual=res, initial_residual=init_res
            )

        z = preconditioner(r) if preconditioner is not None else r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    res = float(torch.linalg.norm(r).detach().cpu())
    return x, CGInfo(
        converged=False, iters=int(max_iter), residual=res, initial_residual=init_res
    )


@dataclass(frozen=True)
class SteihaugCGInfo:
    """Convergence info from Steihaug-CG solve."""

    converged: bool
    iters: int
    residual: float
    termination_reason: str  # "converged", "negative_curvature", "boundary", "max_iter"
    hit_boundary: bool
    negative_curvature_detected: bool
    predicted_reduction: float


def _solve_trust_region_boundary(
    x: torch.Tensor, p: torch.Tensor, delta: float
) -> float:
    """Find τ > 0 such that ||x + τp|| = delta."""
    xx = torch.dot(x, x).item()
    xp = torch.dot(x, p).item()
    pp = torch.dot(p, p).item()
    disc = xp * xp - pp * (xx - delta * delta)
    if disc < 0:
        return delta / math.sqrt(pp + 1e-10)
    sqrt_disc = math.sqrt(max(disc, 0))
    tau1 = (-xp + sqrt_disc) / (pp + 1e-10)
    tau2 = (-xp - sqrt_disc) / (pp + 1e-10)
    if tau1 > 0 and tau2 > 0:
        return min(tau1, tau2)
    return max(tau1, tau2) if max(tau1, tau2) > 0 else abs(tau1)


def steihaug_cg(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    grad: torch.Tensor,
    delta: float,
    *,
    max_iter: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    neg_curv_tol: float = 1e-10,
) -> tuple[torch.Tensor, SteihaugCGInfo]:
    """Steihaug-CG for trust-region subproblems.

    Solves: min_p  g^T p + 0.5 p^T H p  s.t. ||p|| ≤ δ

    Handles indefinite Hessians (saddle escape) and trust-region truncation.

    Args:
        hvp_fn: Hessian-vector product H @ v.
        grad:   Gradient at current point.
        delta:  Trust region radius.
        max_iter: Maximum CG iterations.
        rtol, atol: Convergence tolerances.
        neg_curv_tol: Threshold for negative curvature detection.

    Returns:
        p: Step direction.
        info: SteihaugCGInfo with details.
    """
    p = torch.zeros_like(grad)
    r = -grad.clone()
    d = r.clone()

    r_norm = torch.linalg.norm(r).item()
    init_res = r_norm
    tol = max(atol, rtol * init_res)

    if r_norm < tol:
        return p, SteihaugCGInfo(
            converged=True,
            iters=0,
            residual=r_norm,
            termination_reason="converged",
            hit_boundary=False,
            negative_curvature_detected=False,
            predicted_reduction=0.0,
        )

    neg_curv = False

    for it in range(max_iter):
        Hd = hvp_fn(d)
        dHd = torch.dot(d, Hd).item()

        if dHd <= neg_curv_tol:
            neg_curv = True
            tau = _solve_trust_region_boundary(p, d, delta)
            p_final = p + tau * d
            Hp = hvp_fn(p_final)
            pred = (
                -torch.dot(grad, p_final).item() - 0.5 * torch.dot(p_final, Hp).item()
            )
            return p_final, SteihaugCGInfo(
                converged=False,
                iters=it + 1,
                residual=r_norm,
                termination_reason="negative_curvature",
                hit_boundary=True,
                negative_curvature_detected=True,
                predicted_reduction=pred,
            )

        rr = torch.dot(r, r).item()
        alpha = rr / dHd
        p_new = p + alpha * d

        if torch.linalg.norm(p_new).item() >= delta:
            tau = _solve_trust_region_boundary(p, d, delta)
            p_final = p + tau * d
            Hp = hvp_fn(p_final)
            pred = (
                -torch.dot(grad, p_final).item() - 0.5 * torch.dot(p_final, Hp).item()
            )
            return p_final, SteihaugCGInfo(
                converged=False,
                iters=it + 1,
                residual=r_norm,
                termination_reason="boundary",
                hit_boundary=True,
                negative_curvature_detected=neg_curv,
                predicted_reduction=pred,
            )

        p = p_new
        r = r - alpha * Hd
        r_norm = torch.linalg.norm(r).item()

        if r_norm < tol:
            Hp = hvp_fn(p)
            pred = -torch.dot(grad, p).item() - 0.5 * torch.dot(p, Hp).item()
            return p, SteihaugCGInfo(
                converged=True,
                iters=it + 1,
                residual=r_norm,
                termination_reason="converged",
                hit_boundary=False,
                negative_curvature_detected=neg_curv,
                predicted_reduction=pred,
            )

        rr_new = torch.dot(r, r).item()
        beta = rr_new / rr
        d = r + beta * d

    Hp = hvp_fn(p)
    pred = -torch.dot(grad, p).item() - 0.5 * torch.dot(p, Hp).item()
    p_norm = torch.linalg.norm(p).item()
    return p, SteihaugCGInfo(
        converged=False,
        iters=max_iter,
        residual=r_norm,
        termination_reason="max_iter",
        hit_boundary=(p_norm >= delta * 0.99),
        negative_curvature_detected=neg_curv,
        predicted_reduction=pred,
    )
