"""Training loop (SPEC §13). Pure PyTorch, no Lightning."""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from deepfold.data.crop import get_crop_size  # noqa: F401 — re-export

logger = logging.getLogger(__name__)


def _log_grad_spike(model: nn.Module, step: int, grad_norm: float) -> None:
    """Log per-module grad norms when a spike is detected."""
    raw = model.module if hasattr(model, "module") else model
    module_norms = {}
    for name, param in raw.named_parameters():
        if param.grad is None:
            continue
        # Group by top-level module: trunk.trunk_blocks.3.w_q → trunk
        parts = name.split(".")
        key = parts[0] if len(parts) > 1 else name
        g = param.grad.norm().item()
        if key not in module_norms or g > module_norms[key][0]:
            module_norms[key] = (g, name)
    top = sorted(module_norms.items(), key=lambda x: -x[1][0])[:5]
    lines = [f"  {mod}: max_grad={gn:.1f} ({pname})" for mod, (gn, pname) in top]
    logger.warning(
        "Grad spike at step %d (norm=%.1f). Top modules:\n%s",
        step, grad_norm, "\n".join(lines),
    )


# ============================================================================
# EMA
# ============================================================================


class EMA:
    """Exponential moving average of model parameters (SPEC §13.2, v4.5).

    During warmup, shadow tracks raw params directly (no blending).
    After warmup, standard EMA: θ_ema ← decay · θ_ema + (1-decay) · θ.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 1000
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    def update(self, model: nn.Module):
        self.step += 1
        if self.step <= self.warmup_steps:
            # Copy current params directly during warmup
            for name, p in model.named_parameters():
                self.shadow[name].copy_(p.data)
        else:
            for name, p in model.named_parameters():
                self.shadow[name].lerp_(p.data, 1 - self.decay)

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (for inference)."""
        self.backup = {name: p.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])

    def state_dict(self) -> dict:
        return {"shadow": {k: v.clone() for k, v in self.shadow.items()}, "step": self.step}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.step = state["step"]


# ============================================================================
# Optimizer setup
# ============================================================================


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """Build AdamW optimizer with 2 param groups (SPEC §13.1).

    Decay: all weight matrices (MHA Q/K/V/O, Sinkhorn Q/K, SwiGLU, w_gate).
    No decay: LN γ/β, biases, small scalars (alpha_h, r_h, lambda_h_raw),
              position bias (Swin convention).
    """
    # Small per-head scalars — no decay
    _NO_DECAY_SCALARS = {"alpha_h", "r_h", "lambda_h_raw", "eps"}

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "layernorm" in name.lower() or ".ln_" in name or name.endswith(".ln.weight") or name.endswith(".ln.bias"):
            no_decay_params.append(param)
        elif "bias" in name:
            no_decay_params.append(param)
        elif "pos_bias" in name:
            no_decay_params.append(param)
        elif any(name.endswith(f".{k}") for k in _NO_DECAY_SCALARS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=betas, eps=1e-8)


# ============================================================================
# Metric aggregation
# ============================================================================

_METRIC_KEYS = ("loss", "l_diff", "l_lddt", "l_disto", "l_trunk_slddt", "l_trunk_logmse")


def _reduce_metrics(
    metrics: dict[str, float], device: torch.device
) -> dict[str, float]:
    """All-reduce metrics across DDP ranks (mean)."""
    if not dist.is_initialized():
        return metrics
    vals = torch.tensor(
        [metrics[k] for k in _METRIC_KEYS], device=device, dtype=torch.float64
    )
    dist.all_reduce(vals, op=dist.ReduceOp.AVG)
    for i, k in enumerate(_METRIC_KEYS):
        metrics[k] = vals[i].item()
    return metrics


# ============================================================================
# Training step (with gradient accumulation)
# ============================================================================


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: AdamW,
    step: int,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    scheduler: object | None = None,
    is_accumulating: bool = False,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    """Single training step with mixed precision (SPEC §13.2, §18).

    When is_accumulating=True, skips optimizer step (for gradient accumulation).
    The caller should call with is_accumulating=False on the last micro-batch.

    Returns None if the batch was skipped (OOM or other error).
    """
    model.train()

    lr = optimizer.param_groups[0]["lr"]

    device = next(model.parameters()).device

    try:
        # Mixed precision forward
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            outputs = model(**batch)
            loss = outputs["loss"] / grad_accum_steps

        # Backward
        grad_norm = float("nan")
        if scaler is not None:
            scaler.scale(loss).backward()
            if not is_accumulating:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                ).item()
                if grad_norm > max_grad_norm * 10:
                    _log_grad_spike(model, step, grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]
        else:
            loss.backward()
            if not is_accumulating:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                ).item()
                if grad_norm > max_grad_norm * 10:
                    _log_grad_spike(model, step, grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]

    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM at step %d — skipping batch, clearing cache", step)
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("OOM at step %d — skipping batch, clearing cache", step)
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            return None
        raise

    metrics = {
        "loss": outputs["loss"].item(),
        "l_diff": outputs.get("l_diff", torch.tensor(0.0)).item(),
        "l_lddt": outputs.get("l_lddt", torch.tensor(0.0)).item(),
        "l_disto": outputs.get("l_disto", torch.tensor(0.0)).item(),
        "l_trunk_slddt": outputs.get("l_trunk_slddt", torch.tensor(0.0)).item(),
        "l_trunk_logmse": outputs.get("l_trunk_logmse", torch.tensor(0.0)).item(),
        "num_cycles": outputs.get("num_cycles", 0),
        "lr": lr,
        "grad_norm": grad_norm,
    }

    # Aggregate across DDP ranks
    return _reduce_metrics(metrics, device)


# ============================================================================
# Validation step
# ============================================================================


@torch.no_grad()
def val_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Single validation step with mixed precision.

    Uses model.eval() to disable dropout and stochastic behavior.
    compute_losses=True ensures losses are computed despite eval mode.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        outputs = model(**batch, compute_losses=True)

    metrics = {
        "loss": outputs["loss"].item(),
        "l_diff": outputs.get("l_diff", torch.tensor(0.0)).item(),
        "l_lddt": outputs.get("l_lddt", torch.tensor(0.0)).item(),
        "l_disto": outputs.get("l_disto", torch.tensor(0.0)).item(),
        "l_trunk_slddt": outputs.get("l_trunk_slddt", torch.tensor(0.0)).item(),
        "l_trunk_logmse": outputs.get("l_trunk_logmse", torch.tensor(0.0)).item(),
    }
    return _reduce_metrics(metrics, device)
