"""Training script for DeepFold-Linear."""

import argparse
import gc
import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from deepfold.data.crop import get_crop_size, set_crop_schedule
from deepfold.data.dataset import DeepFoldDataset, collate_fn
from deepfold.data.sampler import ClusterWeightedSampler, load_manifest
from deepfold.model.deepfold import DeepFoldLinear
from deepfold.train.config import load_config
from deepfold.train.scheduler import AlphaFoldLRScheduler
from deepfold.train.trainer import (
    EMA,
    build_optimizer,
    train_step,
    val_step,
)
load_dotenv()  # load WANDB_API_KEY from .env

# Python 3.14 defaults to forkserver which requires picklable worker args.
# Use fork for compatibility with closures and CUDA context sharing.
multiprocessing.set_start_method("fork", force=True)

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)


def _run_validation(raw_model, ema, val_loader, device, max_batches):
    """Run validation epoch with EMA weights, return averaged metrics or None."""
    gc.collect()
    torch.cuda.empty_cache()
    ema.apply(raw_model)
    metrics_sum = {
        k: 0.0 for k in ("loss", "l_diff", "l_lddt", "l_disto", "l_trunk_slddt", "l_trunk_logmse")
    }
    n_val = 0
    max_val = max_batches if max_batches > 0 else len(val_loader)
    for val_batch in val_loader:
        if n_val >= max_val:
            break
        val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
        vm = val_step(raw_model, val_batch)
        for k in metrics_sum:
            metrics_sum[k] += vm[k]
        n_val += 1
        del val_batch, vm
    ema.restore(raw_model)
    raw_model.train()
    gc.collect()
    torch.cuda.empty_cache()
    if n_val > 0:
        return {k: v / n_val for k, v in metrics_sum.items()}
    return None


def _log_val_metrics(val_avg, step, label, wandb):
    """Log validation metrics to console and optionally wandb."""
    logger.info(
        "[%s] step=%d loss=%.4f diff=%.4f lddt=%.4f disto=%.4f slddt=%.4f logmse=%.4f",
        label, step,
        val_avg["loss"], val_avg["l_diff"], val_avg["l_lddt"],
        val_avg["l_disto"], val_avg["l_trunk_slddt"], val_avg["l_trunk_logmse"],
    )
    if wandb is not None:
        wandb.log({f"val/{k}": v for k, v in val_avg.items()}, step=step)


def _heatmap(data, title, xlabel, ylabel, cmap="RdBu_r", figsize=(8, 12)):
    """Create a heatmap figure with adaptive symmetric range and stats annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vmax = max(float(np.abs(data).max()), 1e-6)
    stats = (f"min={data.min():.3e}  max={data.max():.3e}  "
             f"mean={data.mean():.3e}  std={data.std():.3e}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{stats}", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _log_extra(model, step, wandb):
    """Log per-head/per-layer scalar heatmaps and summary scalars."""
    import numpy as np

    raw = model.module if hasattr(model, "module") else model
    blocks = raw.trunk.uot_blocks

    # ---- Collect per-layer per-head arrays ----
    # Base gamma: tanh(w_gamma.bias) — residue-independent component (n_layers, H)
    gamma_bias_map = torch.stack(
        [b.w_gamma.bias for b in blocks]
    ).detach().cpu().tanh().numpy()

    # Runtime gamma gate stats from last forward pass (if available)
    gamma_gate_stats = {}
    if hasattr(blocks[0], "_last_gamma_gate"):
        gates = [b._last_gamma_gate for b in blocks if hasattr(b, "_last_gamma_gate")]
        if gates:
            # Each gate: (B, N, H) — compute per-head stats across B,N
            gate_mean = torch.stack([g.mean(dim=(0, 1)) for g in gates]).cpu().numpy()  # (L, H)
            gate_std = torch.stack([g.std(dim=(0, 1)) for g in gates]).cpu().numpy()
            gate_sign = torch.stack([g.sign().mean(dim=(0, 1)) for g in gates]).cpu().numpy()
            gamma_gate_stats = {
                "mean": gate_mean, "std": gate_std, "sign_mean": gate_sign,
            }

    from deepfold.model.primitives import algebraic_sigmoid
    wdist_map = algebraic_sigmoid(
        torch.stack([b.w_dist_raw for b in blocks])
    ).detach().cpu().numpy()
    # pos_bias: (n_layers, H, 68)
    pos_map = torch.stack([b.pos_bias.weight for b in blocks]).detach().cpu().numpy()

    # ---- Summary scalars (cheap trend lines) ----
    scalars = {
        "params/gamma_bias_abs_mean": float(np.abs(gamma_bias_map).mean()),
        "params/gamma_bias_abs_max": float(np.abs(gamma_bias_map).max()),
        "params/w_dist_mean": float(wdist_map.mean()),
        "params/w_dist_max": float(wdist_map.max()),
        "params/pos_bias_abs_max": float(np.abs(pos_map).max()),
    }
    if gamma_gate_stats:
        scalars["params/gamma_gate_mean_abs"] = float(np.abs(gamma_gate_stats["mean"]).mean())
        scalars["params/gamma_gate_std_mean"] = float(gamma_gate_stats["std"].mean())
        scalars["params/gamma_gate_sign_mean"] = float(gamma_gate_stats["sign_mean"].mean())
    wandb.log(scalars, step=step)

    import matplotlib.pyplot as plt

    # ---- Heatmaps ----
    # Base gamma: tanh(w_gamma.bias) — (n_layers, n_heads)
    fig = _heatmap(gamma_bias_map, f"tanh(γ_bias) — step {step}", "Head", "Layer")
    wandb.log({"heatmap/gamma_bias": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Runtime gamma gate stats (if available)
    if gamma_gate_stats:
        for stat_name, data in gamma_gate_stats.items():
            fig = _heatmap(data, f"γ_gate {stat_name} — step {step}", "Head", "Layer",
                           cmap="RdBu_r" if stat_name == "sign_mean" else "viridis")
            wandb.log({f"heatmap/gamma_gate_{stat_name}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    # w_dist (algebraic sigmoid): (n_layers, n_heads)
    fig = _heatmap(wdist_map, f"w_dist — step {step}", "Head", "Layer",
                   cmap="viridis")
    wandb.log({"heatmap/w_dist": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # trunk pos_bias: per-head heatmap (n_layers, 68) × n_heads
    n_heads = pos_map.shape[1]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    vmax = max(float(np.abs(pos_map).max()), 1e-6)
    stats = (f"min={pos_map.min():.3e}  max={pos_map.max():.3e}  "
             f"std={pos_map.std():.3e}")
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 0.15 * len(blocks) + 2 * rows),
                              sharex=True, sharey=True, layout="constrained")
    if n_heads == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for h in range(n_heads):
        ax = axes_flat[h]
        ax.imshow(pos_map[:, h, :], aspect="auto", cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        # Vertical lines separating same-chain sep | inter-chain | bond bins
        ax.axvline(64.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(65.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"Head {h}", fontsize=9)
        if h % cols == 0:
            ax.set_ylabel("Layer")
    for i in range(n_heads, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"trunk pos_bias — step {step}\n{stats}", fontsize=10)
    wandb.log({"heatmap/trunk_pos_bias": wandb.Image(fig)}, step=step)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON manifest for cluster-weighted sampling",
    )
    parser.add_argument(
        "--msa-dir",
        type=str,
        default=None,
        help="Directory with MSA NPZ files ({pdb}_{chain}.npz)",
    )
    parser.add_argument("--val-data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Base directory for run outputs (auto-generates timestamped subdir)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: <run_dir>/checkpoints)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--extra-log-every", type=int, default=None,
                        help="Interval for expensive wandb logs (heatmaps, etc.)")
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None,
                        help="Max batches per validation run (0=all)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=0)
    # CLI overrides for config values
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--release-cutoff", type=str, default=None,
        help="Only train on structures before this date (YYYY-MM-DD)",
    )
    parser.add_argument("--validate-first", action="store_true", default=None,
                        help="Run full validation before training starts")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config from YAML
    cfg = load_config(args.config)
    # CLI overrides for training
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.total_steps is not None:
        cfg.training.total_steps = args.total_steps
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    # CLI overrides for data
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.release_cutoff is not None:
        cfg.data.release_cutoff = args.release_cutoff
    if args.seed is not None:
        cfg.data.seed = args.seed

    # CLI overrides for logging
    if args.save_every is not None:
        cfg.logging.save_every = args.save_every
    if args.log_every is not None:
        cfg.logging.log_every = args.log_every
    if args.extra_log_every is not None:
        cfg.logging.extra_log_every = args.extra_log_every

    # CLI overrides for validation
    if args.val_every is not None:
        cfg.validation.val_every = args.val_every
    if args.val_batches is not None:
        cfg.validation.val_batches = args.val_batches
    if args.validate_first:
        cfg.validation.validate_first = True

    # Reproducibility
    seed = cfg.data.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set crop schedule from config
    set_crop_schedule(cfg.training.crop_schedule)

    # DDP setup
    use_ddp = "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    rank0 = local_rank == 0

    # ---- Output directory setup (rank 0 generates name, broadcast to all) ----
    if rank0:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_name = ""
    if use_ddp:
        name_list = [run_name]
        dist.broadcast_object_list(name_list, src=0)
        run_name = name_list[0]
    run_dir = Path(args.output_dir) / run_name
    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    )

    if rank0:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # File logging
        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FMT))
        logging.getLogger().addHandler(file_handler)

        # Save resolved config for reproducibility
        import yaml as _yaml

        def _section(obj):
            return {k: getattr(obj, k) for k in vars(obj)}

        config_snapshot = {
            "model": _section(cfg.model),
            "data": _section(cfg.data),
            "logging": _section(cfg.logging),
            "training": _section(cfg.training),
            "validation": _section(cfg.validation),
            "loss_weights": cfg.loss_weights.to_dict(),
            "sampler": _section(cfg.sampler),
            "diffusion": _section(cfg.diffusion),
            "msa": _section(cfg.msa),
            "wandb": _section(cfg.wandb),
        }
        with open(run_dir / "config.yaml", "w") as f:
            _yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

        # Save CLI args
        with open(run_dir / "args.txt", "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")

        logger.info("Run directory: %s", run_dir)

    # ---- WandB init (rank 0 only) ----
    use_wandb = cfg.wandb.enabled and rank0
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name or run_name,
            tags=cfg.wandb.tags,
            config=config_snapshot,
            dir=str(run_dir),
        )
        logger.info("WandB initialized: %s/%s", cfg.wandb.project, wandb.run.name)
    else:
        wandb = None  # type: ignore

    # Build model from config (with loss weights)
    model = DeepFoldLinear(
        d_model=cfg.model.d_model,
        d_msa=cfg.model.d_msa,
        d_atom=cfg.model.d_atom,
        h_res=cfg.model.h_res,
        h_msa=cfg.model.h_msa,
        n_msa_blocks=cfg.model.n_msa_blocks,
        n_uot_blocks=cfg.model.n_uot_blocks,
        n_diff_transformer_layers=cfg.model.n_diff_transformer_layers,
        n_diff_encoder_blocks=cfg.model.n_diff_encoder_blocks,
        n_diff_decoder_blocks=cfg.model.n_diff_decoder_blocks,
        n_diff_heads=cfg.model.n_diff_heads,
        d_fourier=cfg.model.d_fourier,
        sigma_data=cfg.model.sigma_data,
        max_cycles=cfg.model.max_cycles,
        inference_cycles=cfg.model.inference_cycles,
        diffusion_multiplicity=cfg.diffusion.multiplicity,
        loss_weights=cfg.loss_weights.to_dict(),
    ).to(device)

    if rank0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %.1fM", n_params / 1e6)
        logger.info("Loss weights: %s", cfg.loss_weights.to_dict())

    optimizer = build_optimizer(
        model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )
    ema = EMA(
        model, decay=cfg.training.ema_decay, warmup_steps=cfg.training.ema_warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = AlphaFoldLRScheduler(
        optimizer,
        base_lr=cfg.training.base_lr,
        max_lr=cfg.training.lr,
        warmup_steps=cfg.training.warmup_steps,
        start_decay_after=cfg.training.start_decay_after,
        decay_every=cfg.training.decay_every,
        decay_factor=cfg.training.decay_factor,
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Fast-forward scheduler to match resumed step
            scheduler.last_epoch = start_step
        # Restore RNG state on rank 0; other ranks get fresh seeds offset by
        # rank (same as initial setup) to maintain data diversity across GPUs.
        if "rng_state" in ckpt and not use_ddp:
            torch.set_rng_state(ckpt["rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
            np.random.set_state(ckpt["np_rng_state"])
        elif use_ddp:
            # DDP: seed deterministically from step + rank for reproducibility
            resume_seed = seed + start_step * 1000 + local_rank
            torch.manual_seed(resume_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(resume_seed)
            np.random.seed(resume_seed % (2**31))
        if rank0:
            restored = ["model", "optimizer", "ema"]
            if "scaler" in ckpt:
                restored.append("scaler")
            if "rng_state" in ckpt:
                restored.append("rng" if not use_ddp else "rng(reseeded)")
            logger.info(
                "Resumed from %s at step %d (restored: %s)",
                args.resume,
                start_step,
                ", ".join(restored),
            )

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Data loaders ----
    data_dir = Path(args.data_dir)
    train_paths = sorted(data_dir.glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    # Filter by release date cutoff if manifest + cutoff provided
    # Structures <= cutoff → train, structures > cutoff → auto val (if no --val-data-dir)
    val_paths_from_cutoff = []
    if cfg.data.release_cutoff and args.manifest:
        import json

        with open(args.manifest) as f:
            manifest = json.load(f)
        release_by_id = {
            r["id"]: r.get("structure", {}).get("released", "9999") for r in manifest
        }
        before = len(train_paths)
        val_paths_from_cutoff = [
            p
            for p in train_paths
            if p.stem in release_by_id and release_by_id[p.stem] > cfg.data.release_cutoff
        ]
        train_paths = [
            p
            for p in train_paths
            if p.stem in release_by_id and release_by_id[p.stem] <= cfg.data.release_cutoff
        ]
        if rank0:
            logger.info(
                "Release cutoff %s: %d → %d train, %d val (%.1f%% train)",
                cfg.data.release_cutoff,
                before,
                len(train_paths),
                len(val_paths_from_cutoff),
                100 * len(train_paths) / before,
            )
        if not train_paths:
            raise FileNotFoundError(
                f"No structures before cutoff {cfg.data.release_cutoff}"
            )

    train_dataset = DeepFoldDataset(
        data_paths=train_paths,
        max_tokens=get_crop_size(start_step),
        max_msa_seqs=cfg.msa.max_depth,
        min_msa_seqs=cfg.msa.min_depth,
        msa_dir=args.msa_dir,
        max_msa_cycles=cfg.model.max_cycles,
        training=True,
        seed=seed,
    )

    # Build sampler: cluster-weighted (if manifest provided) or random
    train_sampler = None
    # Build path lookup for alignment between sampler records and dataset
    path_by_stem = {p.stem: p for p in train_paths}
    if args.manifest and cfg.sampler.type == "cluster":
        records = load_manifest(args.manifest)
        # Filter records to match date-filtered dataset and re-order train_paths
        # to match records so sampler index i → data_paths[i]
        records = [r for r in records if r.id in path_by_stem]
        train_paths = [path_by_stem[r.id] for r in records]
        # Rebuild dataset with aligned paths
        train_dataset = DeepFoldDataset(
            data_paths=train_paths,
            max_tokens=get_crop_size(start_step),
            max_msa_seqs=cfg.msa.max_depth,
            min_msa_seqs=cfg.msa.min_depth,
            msa_dir=args.msa_dir,
            training=True,
            seed=seed,
        )
        # Offset seed by rank so DDP ranks sample different structures
        sampler_seed = seed + (local_rank if use_ddp else 0)
        train_sampler = ClusterWeightedSampler(
            records=records,
            alpha_prot=cfg.sampler.alpha_prot,
            alpha_nucl=cfg.sampler.alpha_nucl,
            alpha_ligand=cfg.sampler.alpha_ligand,
            beta=cfg.sampler.beta,
            seed=sampler_seed,
            samples_per_epoch=cfg.sampler.samples_per_epoch,
        )
        if rank0:
            logger.info("Using cluster-weighted sampler (%d records)", len(records))
    elif use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        if rank0:
            logger.info("Using DistributedSampler")

    def _worker_init_fn(worker_id):
        """Seed each DataLoader worker uniquely to avoid duplicate augmentations."""
        np.random.seed(seed + worker_id + local_rank * 1000)

    batch_size = cfg.training.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=_worker_init_fn if cfg.data.num_workers > 0 else None,
    )

    val_loader = None
    if args.val_data_dir:
        val_paths = sorted(Path(args.val_data_dir).glob("*.npz"))
    elif val_paths_from_cutoff:
        val_paths = val_paths_from_cutoff
    else:
        val_paths = []
    if val_paths:
        val_dataset = DeepFoldDataset(
            data_paths=val_paths, max_tokens=get_crop_size(start_step), training=True
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=True) if use_ddp else None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    if rank0:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch = batch_size * cfg.training.grad_accum_steps * world_size
        logger.info(
            "Training on %d structures (batch_size=%d, accum=%d, gpus=%d, effective=%d)",
            len(train_paths),
            batch_size,
            cfg.training.grad_accum_steps,
            world_size,
            effective_batch,
        )
        if val_loader:
            logger.info("Validation on %d structures", len(val_paths))

    # ---- Validate first (optional) ----
    if cfg.validation.validate_first and val_loader:
        if rank0:
            logger.info("Running pre-training validation...")
        raw_model = model.module if use_ddp else model
        val_avg = _run_validation(raw_model, ema, val_loader, device,
                                  cfg.validation.val_batches)
        if rank0 and val_avg:
            _log_val_metrics(val_avg, start_step, "val-init",
                             wandb if use_wandb else None)
    elif cfg.validation.validate_first and not val_loader:
        if rank0:
            logger.warning("--validate-first requested but no validation data available")

    # ---- Training loop ----
    grad_accum = cfg.training.grad_accum_steps
    train_iter = iter(train_loader)
    prev_crop = get_crop_size(start_step)
    oom_count = 0
    last_ckpt_time = time.monotonic()

    for step in range(start_step + 1, cfg.training.total_steps + 1):
        # Update crop size when schedule changes
        crop_size = get_crop_size(step)
        if crop_size != prev_crop:
            train_dataset.set_crop_size(crop_size)
            prev_crop = crop_size
            if rank0:
                logger.info("Crop size changed to %d at step %d", crop_size, step)

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(step)

        # Gradient accumulation loop
        metrics = None
        step_oom = False
        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            is_last_micro = micro == grad_accum - 1
            result = train_step(
                model,
                batch,
                optimizer,
                step,
                scaler,
                max_grad_norm=cfg.training.max_grad_norm,
                scheduler=scheduler,
                is_accumulating=not is_last_micro,
                grad_accum_steps=grad_accum,
            )

            del batch
            if result is None:
                step_oom = True
                break
            metrics = result

        # DDP: broadcast OOM across all ranks so everyone skips together
        # (prevents deadlock when one rank OOMs but others wait for all-reduce)
        if use_ddp:
            oom_flag = torch.tensor([1 if step_oom else 0], device=device)
            dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            step_oom = oom_flag.item() > 0

        if step_oom:
            oom_count += 1
            optimizer.zero_grad(set_to_none=True)
            if rank0:
                logger.warning("OOM skip #%d at step %d", oom_count, step)
            continue

        ema.update(model.module if use_ddp else model)

        if rank0 and step % cfg.logging.log_every == 0:
            logger.info(
                "step=%d loss=%.4f diff=%.4f lddt=%.4f disto=%.4f "
                "slddt=%.4f logmse=%.4f grad_norm=%.4f lr=%.6f crop=%d",
                step,
                metrics["loss"],
                metrics["l_diff"],
                metrics["l_lddt"],
                metrics["l_disto"],
                metrics["l_trunk_slddt"],
                metrics["l_trunk_logmse"],
                metrics["grad_norm"],
                metrics["lr"],
                crop_size,
            )
            if use_wandb and step % cfg.wandb.log_every == 0:
                wandb.log(
                    {
                        "train/loss": metrics["loss"],
                        "train/l_diff": metrics["l_diff"],
                        "train/l_lddt": metrics["l_lddt"],
                        "train/l_disto": metrics["l_disto"],
                        "train/l_trunk_slddt": metrics["l_trunk_slddt"],
                        "train/l_trunk_logmse": metrics["l_trunk_logmse"],
                        "train/grad_norm": metrics["grad_norm"],
                        "train/lr": metrics["lr"],
                        "train/crop_size": crop_size,
                        "train/num_cycles": metrics["num_cycles"],
                    },
                    step=step,
                )
            if use_wandb and step % cfg.logging.extra_log_every == 0:
                _log_extra(model, step, wandb)

        # Checkpointing (before validation — save consistent state)
        # Rank 0 decides; broadcast to all ranks to avoid DDP deadlock
        # (time.monotonic() can differ across processes)
        now = time.monotonic()
        step_trigger = step % cfg.logging.save_every == 0
        time_trigger = (
            cfg.logging.save_every_minutes > 0
            and (now - last_ckpt_time) >= cfg.logging.save_every_minutes * 60
        )
        should_save = step_trigger or time_trigger
        if use_ddp:
            _save_flag = torch.tensor([int(should_save)], device=device)
            dist.broadcast(_save_flag, src=0)
            should_save = _save_flag.item() > 0
        if should_save:
            last_ckpt_time = now
            if use_ddp:
                dist.barrier()
        if rank0 and should_save:
            raw_model = model.module if use_ddp else model
            path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            ckpt_data = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
            }
            ckpt_data["scheduler"] = scheduler.state_dict()
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            if torch.cuda.is_available():
                ckpt_data["cuda_rng_state"] = torch.cuda.get_rng_state()
            torch.save(ckpt_data, path)
            latest = os.path.join(checkpoint_dir, "latest.pt")
            tmp_link = latest + ".tmp"
            os.symlink(os.path.basename(path), tmp_link)
            os.replace(tmp_link, latest)
            logger.info("Saved checkpoint: %s", path)

        # Validation
        if val_loader and step % cfg.validation.val_every == 0:
            raw_model = model.module if use_ddp else model
            val_avg = _run_validation(raw_model, ema, val_loader, device,
                                      cfg.validation.val_batches)
            if rank0 and val_avg:
                _log_val_metrics(val_avg, step, "val",
                                 wandb if use_wandb else None)

    if rank0 and oom_count > 0:
        logger.info("Total OOM skips: %d", oom_count)

    if use_wandb:
        wandb.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
