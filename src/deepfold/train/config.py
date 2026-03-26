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
    # Diffusion v2
    n_diff_transformer_layers: int = 24
    n_diff_encoder_blocks: int = 3
    n_diff_decoder_blocks: int = 3
    n_diff_heads: int = 16
    d_fourier: int = 256
    sigma_data: float = 16.0
    max_cycles: int = 3
    inference_cycles: int = 3


@dataclass
class DataConfig:
    num_workers: int = 4
    release_cutoff: str | None = None  # YYYY-MM-DD or None
    seed: int = 42


@dataclass
class LoggingConfig:
    save_every: int = 10_000
    save_every_minutes: int = 60  # time-based checkpoint interval (0 = disabled)
    log_every: int = 100
    extra_log_every: int = 1_000


@dataclass
class InitConfig:
    gamma_std: float = 1e-4       # EGNN gamma N(0, std) noise init
    adaln_gate_bias: float = -2.0 # AdaLN-Zero gate bias; sigmoid(-2.0) ≈ 0.12


@dataclass
class TrainingConfig:
    lr: float = 1e-3           # max_lr (peak after warmup)
    base_lr: float = 0.0       # starting LR for warmup
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    warmup_steps: int = 1000
    start_decay_after: int = 50_000
    decay_every: int = 50_000
    decay_factor: float = 0.95
    total_steps: int = 500_000
    ema_decay: float = 0.999
    ema_warmup_steps: int = 1000
    max_grad_norm: float = 10.0
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
class ValidationConfig:
    val_every: int = 1_000      # validate every N steps
    val_batches: int = 50       # max batches per validation run (0 = all)
    validate_first: bool = False # run full validation before training starts


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
    min_depth: int = 1    # minimum random depth during training


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
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    init: InitConfig = field(default_factory=InitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
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
    if "data" in raw:
        _apply_dict(cfg.data, raw["data"])
    if "logging" in raw:
        _apply_dict(cfg.logging, raw["logging"])
    if "init" in raw:
        _apply_dict(cfg.init, raw["init"])
    if "training" in raw:
        _apply_dict(cfg.training, raw["training"])
    if "validation" in raw:
        _apply_dict(cfg.validation, raw["validation"])
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
