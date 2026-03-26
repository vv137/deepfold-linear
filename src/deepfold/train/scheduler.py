"""AF3 learning rate scheduler: warmup + plateau + exponential decay.

Adapted from Boltz-1 (references/boltz/src/boltz/model/optim/scheduler.py).
"""

import torch


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup → plateau → step exponential decay (AF3).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    base_lr : float
        Starting LR for warmup (typically 0.0).
    max_lr : float
        Peak LR reached after warmup.
    warmup_steps : int
        Steps for linear warmup.
    start_decay_after : int
        Step at which decay begins.
    decay_every : int
        Steps between each decay application.
    decay_factor : float
        Multiplicative factor per decay interval.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 0.0,
        max_lr: float = 1.8e-3,
        warmup_steps: int = 1000,
        start_decay_after: int = 50_000,
        decay_every: int = 50_000,
        decay_factor: float = 0.95,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps > start_decay_after:
            raise ValueError("warmup_steps must not exceed start_decay_after")

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.start_decay_after = start_decay_after
        self.decay_every = decay_every
        self.decay_factor = decay_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step <= self.warmup_steps:
            lr = self.base_lr + (step / max(self.warmup_steps, 1)) * self.max_lr
        elif step > self.start_decay_after:
            steps_since = step - self.start_decay_after
            exp = (steps_since // self.decay_every) + 1
            lr = self.max_lr * (self.decay_factor ** exp)
        else:
            lr = self.max_lr
        return [lr for _ in self.optimizer.param_groups]
