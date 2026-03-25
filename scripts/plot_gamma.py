#!/usr/bin/env python
"""Plot tanh(gamma) heatmap for all checkpoints in a run directory.

Usage:
    uv run python scripts/plot_gamma.py runs/my_run
    uv run python scripts/plot_gamma.py runs/my_run -o /tmp/gamma_plots
    uv run python scripts/plot_gamma.py runs/my_run --global-vlim
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_gamma(ckpt_path):
    """Load only gamma tensors + step from checkpoint (minimal memory)."""
    import pickle
    import zipfile

    step = 0
    gammas = {}

    # PyTorch checkpoints are zip files — scan the pickle header for step
    # and selectively load only gamma tensors via mmap.
    with open(ckpt_path, "rb") as f:
        # Try zip-based checkpoint (torch.save default)
        try:
            zf = zipfile.ZipFile(f)
        except zipfile.BadZipFile:
            zf = None

    if zf is not None:
        # Use torch.load with a custom unpickler isn't worth it;
        # just lazy-load via mmap to avoid resident memory.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False,
                          mmap=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", 0)
    state = ckpt["model"]

    for k, v in state.items():
        if k.startswith("trunk.uot_blocks.") and k.endswith(".gamma"):
            idx = int(k.split(".")[2])
            gammas[idx] = v.clone()  # clone off mmap before closing

    # Free the large checkpoint immediately
    del ckpt, state

    if not gammas:
        raise RuntimeError(f"No trunk.uot_blocks.*.gamma found in {ckpt_path}")

    n_layers = max(gammas) + 1
    gamma_map = torch.stack([gammas[i] for i in range(n_layers)]).tanh().numpy()
    return step, gamma_map


def print_gamma_stats(gamma_map, step):
    """Print mean/std statistics: total, per head, per layer."""
    n_layers, n_heads = gamma_map.shape
    total_mean, total_std = gamma_map.mean(), gamma_map.std()
    head_mean = gamma_map.mean(axis=0)   # (n_heads,)
    head_std = gamma_map.std(axis=0)
    layer_mean = gamma_map.mean(axis=1)  # (n_layers,)
    layer_std = gamma_map.std(axis=1)

    print(f"  Step {step}  total: mean={total_mean:+.4f} std={total_std:.4f}")
    head_parts = [f"h{h}={head_mean[h]:+.4f}±{head_std[h]:.4f}" for h in range(n_heads)]
    print(f"    per head:  {', '.join(head_parts)}")
    # Layer stats: show first 6, last 6 with ... in between if many
    def _fmt_layer(i):
        return f"L{i}={layer_mean[i]:+.4f}±{layer_std[i]:.4f}"
    if n_layers <= 12:
        layer_parts = [_fmt_layer(i) for i in range(n_layers)]
        print(f"    per layer: {', '.join(layer_parts)}")
    else:
        first = [_fmt_layer(i) for i in range(6)]
        last = [_fmt_layer(i) for i in range(n_layers - 6, n_layers)]
        print(f"    per layer: {', '.join(first)}, ..., {', '.join(last)}")


def plot_gamma(gamma_map, step, vlim, out_path):
    """Save a single gamma heatmap with marginal mean/std annotations."""
    n_layers, n_heads = gamma_map.shape
    head_mean = gamma_map.mean(axis=0)
    head_std = gamma_map.std(axis=0)
    layer_mean = gamma_map.mean(axis=1)
    layer_std = gamma_map.std(axis=1)
    total_mean, total_std = gamma_map.mean(), gamma_map.std()

    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig = plt.figure(figsize=(10, 13), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[6, 1.5], height_ratios=[1.5, 6],
                              hspace=0.05, wspace=0.05)

        # Main heatmap
        ax_main = fig.add_subplot(gs[1, 0])
        im = ax_main.imshow(
            gamma_map, aspect="auto", cmap="RdBu_r",
            vmin=-vlim, vmax=vlim, interpolation="nearest",
        )
        ax_main.set_xlabel(r"Head $h$")
        ax_main.set_ylabel(r"Layer $\ell$")

        # Top panel: per-head mean ± std
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        x = np.arange(n_heads)
        ax_top.bar(x, head_mean, yerr=head_std, color="steelblue", alpha=0.7,
                   capsize=2, edgecolor="none")
        ax_top.axhline(0, color="k", lw=0.5)
        ax_top.set_ylabel(r"mean $\pm$ std")
        ax_top.set_title(
            rf"$\tanh(\gamma_{{h,\ell}})$ — step {step}"
            f"   (total: {total_mean:+.4f} ± {total_std:.4f})",
        )
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Right panel: per-layer mean ± std
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        y = np.arange(n_layers)
        ax_right.barh(y, layer_mean, xerr=layer_std, color="steelblue", alpha=0.7,
                      capsize=2, edgecolor="none")
        ax_right.axvline(0, color="k", lw=0.5)
        ax_right.set_xlabel(r"mean $\pm$ std")
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # Colorbar in the empty top-right cell
        ax_cb = fig.add_subplot(gs[0, 1])
        ax_cb.axis("off")
        fig.colorbar(im, ax=ax_cb, fraction=0.9, label=r"$\tanh(\gamma)$")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot tanh(gamma) heatmap per checkpoint")
    parser.add_argument("run_dir", type=Path, help="Run output directory (e.g. runs/my_run)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <run_dir>/gamma_plots)")
    parser.add_argument("--global-vlim", action="store_true",
                        help="Use a single shared color limit across all steps")
    args = parser.parse_args()

    ckpt_dir = args.run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Collect step_*.pt files, sorted by step number
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_files:
        raise FileNotFoundError(f"No step_*.pt files in {ckpt_dir}")

    out_dir = args.output_dir or args.run_dir / "gamma_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global vlim: quick first pass (gamma tensors are tiny, ~3KB each)
    global_vlim = None
    if args.global_vlim:
        global_max = 0.0
        for path in ckpt_files:
            _, gm = extract_gamma(path)
            global_max = max(global_max, float(np.abs(gm).max()))
        global_vlim = global_max if global_max > 0 else 1.0
        print(f"Global vlim=±{global_vlim:.4f}")

    # Plot one checkpoint at a time
    count = 0
    for path in ckpt_files:
        step, gamma_map = extract_gamma(path)
        vlim = global_vlim if global_vlim is not None else float(np.abs(gamma_map).max())
        if vlim == 0:
            vlim = 1.0
        out_path = out_dir / f"gamma_step_{step}.png"
        print_gamma_stats(gamma_map, step)
        plot_gamma(gamma_map, step, vlim, out_path)
        print(f"    → {out_path.name}  (vlim=±{vlim:.4f})")
        count += 1

    print(f"\nDone — {count} plots in {out_dir}")


if __name__ == "__main__":
    main()
