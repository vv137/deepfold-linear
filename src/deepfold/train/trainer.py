"""Training loop (SPEC §13). Pure PyTorch, no Lightning."""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from deepfold.data.crop import get_crop_size  # noqa: F401 — re-export

logger = logging.getLogger(__name__)


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
# LR Schedule
# ============================================================================


def get_lr(
    step: int,
    warmup_steps: int = 5000,
    total_steps: int = 500_000,
    base_lr: float = 1e-4,
) -> float:
    """Linear warmup + cosine decay (SPEC §13.1)."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================================
# Optimizer setup
# ============================================================================


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """Build AdamW optimizer with 3 param groups (SPEC §13.1).

    1. Standalone weight matrices (not post-LN): weight_decay
    2. No decay: LN γ/β, biases, post-LN projections (w_q/w_k/w_v/w_g/w_o,
       SwiGLU — scale-invariant under LN), bounded params (w_dist_logit,
       alpha_coevol, pos_bias)
    3. EGNN γ: weight_decay (pull toward zero = no coordinate update)
    """
    # Post-LN projection names — scale-invariant, no decay
    _POST_LN_NAMES = {"w_q", "w_k", "w_v", "w_g", "w_o", "swiglu"}
    # Bounded or zeros-init gating params — no decay
    _BOUNDED_NAMES = {"w_dist_logit", "alpha_coevol", "pos_bias"}

    decay_params = []
    no_decay_params = []
    gamma_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gamma" in name and "layernorm" not in name.lower():
            # EGNN γ — separate group with decay
            gamma_params.append(param)
        elif "layernorm" in name.lower() or "ln" in name.lower() or "bias" in name:
            no_decay_params.append(param)
        elif any(k in name for k in _POST_LN_NAMES):
            no_decay_params.append(param)
        elif any(k in name for k in _BOUNDED_NAMES):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": gamma_params, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, lr=lr, betas=betas, eps=1e-8)


# ============================================================================
# Metric aggregation
# ============================================================================

_METRIC_KEYS = ("loss", "l_diff", "l_lddt", "l_disto", "l_trunk_coord")


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
    warmup_steps: int = 5000,
    total_steps: int = 500_000,
    base_lr: float = 1e-4,
    is_accumulating: bool = False,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    """Single training step with mixed precision (SPEC §13.2, §18).

    When is_accumulating=True, skips optimizer step (for gradient accumulation).
    The caller should call with is_accumulating=False on the last micro-batch.

    Returns None if the batch was skipped (OOM or other error).
    """
    model.train()

    if not is_accumulating:
        lr = get_lr(step, warmup_steps, total_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    else:
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
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if not is_accumulating:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                ).item()
                optimizer.step()
                optimizer.zero_grad()

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
        "l_trunk_coord": outputs.get("l_trunk_coord", torch.tensor(0.0)).item(),
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
    """Single validation step with mixed precision."""
    model.eval()
    device = next(model.parameters()).device

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        outputs = model(**batch)

    metrics = {
        "loss": outputs["loss"].item(),
        "l_diff": outputs.get("l_diff", torch.tensor(0.0)).item(),
        "l_lddt": outputs.get("l_lddt", torch.tensor(0.0)).item(),
        "l_disto": outputs.get("l_disto", torch.tensor(0.0)).item(),
        "l_trunk_coord": outputs.get("l_trunk_coord", torch.tensor(0.0)).item(),
    }
    return _reduce_metrics(metrics, device)
