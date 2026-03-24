"""Configuration loader for DeepFold-Linear training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    d_model: int = 512
    d_msa: int = 64
    d_atom: int = 128
    h_res: int = 16
    h_msa: int = 8
    n_msa_blocks: int = 4
    n_uot_blocks: int = 48
    n_atom_blocks: int = 10
    sigma_data: float = 16.0
    max_cycles: int = 5
    inference_cycles: int = 3


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    warmup_steps: int = 5000
    total_steps: int = 500_000
    ema_decay: float = 0.999
    ema_warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    batch_size: int = 1
    grad_accum_steps: int = 1
    crop_schedule: list = field(
        default_factory=lambda: [
            [0, 256],
            [100_000, 384],
            [300_000, 512],
            [500_000, 768],
        ]
    )


@dataclass
class LossWeights:
    w_diff: float = 1.0
    w_lddt: float = 1.0
    w_disto: float = 0.2
    w_trunk_coord: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "w_diff": self.w_diff,
            "w_lddt": self.w_lddt,
            "w_disto": self.w_disto,
            "w_trunk_coord": self.w_trunk_coord,
        }


@dataclass
class MSAConfig:
    max_depth: int = 128  # max MSA rows after uniform subsampling


@dataclass
class DiffusionConfig:
    multiplicity: int = 16
    sigma_max: float = 160.0
    sigma_min: float = 0.002
    sigma_data: float = 16.0
    p_mean: float = -1.2
    p_std: float = 1.2
    rho: float = 7.0
    inference_steps: int = 200


@dataclass
class SamplerConfig:
    type: str = "cluster"  # "cluster" or "random"
    alpha_prot: float = 3.0
    alpha_nucl: float = 3.0
    alpha_ligand: float = 1.0
    beta: float = 1.0
    samples_per_epoch: int = 100_000


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "deepfold-linear"
    entity: str | None = None
    name: str | None = None  # run name; None = auto-generated
    tags: list[str] = field(default_factory=list)
    log_every: int = 1  # log every N optimizer steps


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _apply_dict(obj, d: dict) -> None:
    """Set attributes on obj from dict, skipping unknown keys."""
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    cfg = Config()
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "model" in raw:
        _apply_dict(cfg.model, raw["model"])
    if "training" in raw:
        _apply_dict(cfg.training, raw["training"])
    if "loss_weights" in raw:
        _apply_dict(cfg.loss_weights, raw["loss_weights"])
    if "sampler" in raw:
        _apply_dict(cfg.sampler, raw["sampler"])
    if "diffusion" in raw:
        _apply_dict(cfg.diffusion, raw["diffusion"])
    if "msa" in raw:
        _apply_dict(cfg.msa, raw["msa"])
    if "wandb" in raw:
        _apply_dict(cfg.wandb, raw["wandb"])

    return cfg
