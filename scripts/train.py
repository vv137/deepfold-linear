"""Training script for DeepFold-Linear."""

import argparse
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from deepfold.data.crop import get_crop_size, set_crop_schedule
from deepfold.data.dataset import DeepFoldDataset, collate_fn
from deepfold.data.sampler import ClusterWeightedSampler, load_manifest
from deepfold.model.deepfold import DeepFoldLinear
from deepfold.train.config import load_config
from deepfold.train.trainer import (
    EMA,
    build_optimizer,
    train_step,
    val_step,
)

# Python 3.14 defaults to forkserver which requires picklable worker args.
# Use fork for compatibility with closures and CUDA context sharing.
multiprocessing.set_start_method("fork", force=True)

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None,
                        help="JSON manifest for cluster-weighted sampling")
    parser.add_argument("--msa-dir", type=str, default=None,
                        help="Directory with MSA NPZ files ({pdb}_{chain}.npz)")
    parser.add_argument("--val-data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base directory for run outputs (auto-generates timestamped subdir)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: YYYYMMDD_HHMMSS)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: <run_dir>/checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=0)
    # CLI overrides for config values
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--release-cutoff", type=str, default=None,
                        help="Only train on structures released before this date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config from YAML
    cfg = load_config(args.config)
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.total_steps is not None:
        cfg.training.total_steps = args.total_steps
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    # ---- Output directory setup (rank 0 only) ----
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"

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
        config_snapshot = {
            "model": {k: getattr(cfg.model, k) for k in vars(cfg.model)},
            "training": {k: getattr(cfg.training, k) for k in vars(cfg.training)},
            "loss_weights": cfg.loss_weights.to_dict(),
            "sampler": {k: getattr(cfg.sampler, k) for k in vars(cfg.sampler)},
            "diffusion": {k: getattr(cfg.diffusion, k) for k in vars(cfg.diffusion)},
            "msa": {k: getattr(cfg.msa, k) for k in vars(cfg.msa)},
        }
        with open(run_dir / "config.yaml", "w") as f:
            _yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

        # Save CLI args
        with open(run_dir / "args.txt", "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")

        logger.info("Run directory: %s", run_dir)

    # Build model from config (with loss weights)
    model = DeepFoldLinear(
        d_model=cfg.model.d_model,
        d_msa=cfg.model.d_msa,
        d_atom=cfg.model.d_atom,
        h_res=cfg.model.h_res,
        h_msa=cfg.model.h_msa,
        n_msa_blocks=cfg.model.n_msa_blocks,
        n_uot_blocks=cfg.model.n_uot_blocks,
        n_atom_blocks=cfg.model.n_atom_blocks,
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
        model, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )
    ema = EMA(
        model, decay=cfg.training.ema_decay, warmup_steps=cfg.training.ema_warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        if rank0:
            logger.info("Resumed from %s at step %d", args.resume, start_step)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Checkpoint dir already created by rank 0 above; non-rank-0 needs the path
    if not rank0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data loaders ----
    data_dir = Path(args.data_dir)
    train_paths = sorted(data_dir.glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    # Filter by release date cutoff if manifest + cutoff provided
    if args.release_cutoff and args.manifest:
        import json
        with open(args.manifest) as f:
            manifest = json.load(f)
        valid_ids = {
            r["id"] for r in manifest
            if r.get("structure", {}).get("released", "9999") <= args.release_cutoff
        }
        before = len(train_paths)
        train_paths = [p for p in train_paths if p.stem in valid_ids]
        if rank0:
            logger.info(
                "Release cutoff %s: %d → %d structures (%.1f%%)",
                args.release_cutoff, before, len(train_paths),
                100 * len(train_paths) / before,
            )
        if not train_paths:
            raise FileNotFoundError(f"No structures before cutoff {args.release_cutoff}")

    train_dataset = DeepFoldDataset(
        data_paths=train_paths,
        max_tokens=get_crop_size(start_step),
        max_msa_seqs=cfg.msa.max_depth,
        msa_dir=args.msa_dir,
        training=True,
        seed=args.seed,
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
            msa_dir=args.msa_dir,
            training=True,
            seed=args.seed,
        )
        # Offset seed by rank so DDP ranks sample different structures
        sampler_seed = args.seed + (local_rank if use_ddp else 0)
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
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        if rank0:
            logger.info("Using DistributedSampler")

    def _worker_init_fn(worker_id):
        """Seed each DataLoader worker uniquely to avoid duplicate augmentations."""
        np.random.seed(args.seed + worker_id + local_rank * 1000)

    batch_size = cfg.training.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    val_loader = None
    if args.val_data_dir:
        val_dir = Path(args.val_data_dir)
        val_paths = sorted(val_dir.glob("*.npz"))
        if val_paths:
            val_dataset = DeepFoldDataset(
                data_paths=val_paths, max_tokens=get_crop_size(start_step), training=True
            )
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

    if rank0:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch = batch_size * cfg.training.grad_accum_steps * world_size
        logger.info(
            "Training on %d structures (batch_size=%d, accum=%d, gpus=%d, effective=%d)",
            len(train_paths), batch_size, cfg.training.grad_accum_steps,
            world_size, effective_batch,
        )
        if val_loader:
            logger.info("Validation on %d structures", len(val_paths))

    # ---- Training loop ----
    grad_accum = cfg.training.grad_accum_steps
    train_iter = iter(train_loader)
    prev_crop = get_crop_size(start_step)
    oom_count = 0

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
                warmup_steps=cfg.training.warmup_steps,
                total_steps=cfg.training.total_steps,
                base_lr=cfg.training.lr,
                is_accumulating=not is_last_micro,
            )

            if result is None:
                # OOM — discard entire step (no optimizer step was taken)
                metrics = None
                oom_count += 1
                if rank0:
                    logger.warning("OOM skip #%d at step %d", oom_count, step)
                break
            metrics = result

        if metrics is None:
            continue  # entire step skipped due to OOM

        ema.update(model.module if use_ddp else model)

        if rank0 and step % args.log_every == 0:
            logger.info(
                "step=%d loss=%.4f diff=%.4f lddt=%.4f disto=%.4f trunk=%.4f "
                "grad_norm=%.4f lr=%.6f crop=%d",
                step, metrics["loss"], metrics["l_diff"], metrics["l_lddt"],
                metrics["l_disto"], metrics["l_trunk_coord"],
                metrics["grad_norm"], metrics["lr"], crop_size,
            )

        # Validation
        if val_loader and step % args.val_every == 0:
            raw_model = model.module if use_ddp else model
            ema.apply(raw_model)
            val_metrics_sum = {k: 0.0 for k in ("loss", "l_diff", "l_lddt", "l_disto", "l_trunk_coord")}
            n_val = 0
            for val_batch in val_loader:
                val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
                vm = val_step(raw_model, val_batch)
                for k in val_metrics_sum:
                    val_metrics_sum[k] += vm[k]
                n_val += 1
            ema.restore(raw_model)
            if rank0 and n_val > 0:
                logger.info(
                    "[val] step=%d loss=%.4f diff=%.4f lddt=%.4f disto=%.4f trunk=%.4f",
                    step,
                    val_metrics_sum["loss"] / n_val,
                    val_metrics_sum["l_diff"] / n_val,
                    val_metrics_sum["l_lddt"] / n_val,
                    val_metrics_sum["l_disto"] / n_val,
                    val_metrics_sum["l_trunk_coord"] / n_val,
                )

        # Checkpointing
        if rank0 and step % args.save_every == 0:
            raw_model = model.module if use_ddp else model
            path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
            }, path)
            logger.info("Saved checkpoint: %s", path)

    if rank0 and oom_count > 0:
        logger.info("Total OOM skips: %d", oom_count)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
