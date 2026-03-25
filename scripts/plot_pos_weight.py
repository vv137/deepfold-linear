#!/usr/bin/env python
"""Plot per-head position bias weights (H, 68) for all checkpoints.

68 bins: 0-64 = same-chain offsets (-32..+32), 65 = diff-chain, 66 = bonded, 67 = bonded-cross.
Each head gets its own subplot with shared y-axis across all heads.

Usage:
    uv run python scripts/plot_pos_weight.py runs/my_run
    uv run python scripts/plot_pos_weight.py runs/my_run --latest --table
"""
import argparse
from pathlib import Path

import torch

from _plot_utils import load_checkpoint, iter_checkpoints, print_table


BIN_LABELS = {
    0: "-32", 16: "-16", 32: "0", 48: "+16", 64: "+32",
    65: "X", 66: "B", 67: "BX",
}
# X=cross-chain, B=bonded, BX=bonded cross-chain


def extract_pos_weight(ckpt_path):
    """Extract trunk pos_bias.weight (H, 68) from checkpoint."""
    step, state = load_checkpoint(ckpt_path)
    key = "trunk.pos_bias.weight"
    if key not in state:
        raise RuntimeError(f"{key} not found in {ckpt_path}")
    w = state[key].float().numpy()
    del state
    return step, w


def plot_pos_weight(data, step, out_path, global_ylim=None):
    """Plot (H, 68) as H subplots sharing y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_heads, n_bins = data.shape
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                              sharex=True, sharey=True, layout="constrained")
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    x = np.arange(n_bins)
    ylim = global_ylim or max(abs(data.max()), abs(data.min()), 0.01)

    for h in range(n_heads):
        ax = axes[h]
        # Color: same-chain offsets blue, special bins distinct
        colors = ["steelblue"] * 65 + ["#e74c3c", "#2ecc71", "#9b59b6"]
        ax.bar(x, data[h], color=colors, width=0.8, edgecolor="none")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"Head {h}", fontsize=10)

        if h >= n_heads - cols:
            # Bottom row: add x-tick labels
            ticks = sorted(BIN_LABELS.keys())
            ax.set_xticks(ticks)
            ax.set_xticklabels([BIN_LABELS[t] for t in ticks], fontsize=7, rotation=45, ha="right")
        if h % cols == 0:
            ax.set_ylabel("weight")

    # Hide unused axes
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Position bias weights — step {step}", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot per-head position bias weights")
    parser.add_argument("run_dir", type=Path, help="Run output directory")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--global-vlim", action="store_true",
                        help="Use shared y-axis limit across all steps")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--table", action="store_true",
                        help="Print values as a table instead of generating plots")
    parser.add_argument("--latest", action="store_true",
                        help="Only process the latest checkpoint")
    args = parser.parse_args()

    ckpt_files = iter_checkpoints(args.run_dir)
    if args.latest:
        ckpt_files = ckpt_files[-1:]

    if args.table:
        for path in ckpt_files:
            step, w = extract_pos_weight(path)
            # Transpose so rows=bins, cols=heads for readability
            print(f"  Step {step} [pos_weight]  {w.shape[0]} heads x {w.shape[1]} bins")
            print(f"  {'bin':>5s}  {'label':>6s}  " + "  ".join(f"{'h'+str(h):>8s}" for h in range(w.shape[0])))
            for b in range(w.shape[1]):
                label = BIN_LABELS.get(b, str(b - 32))
                vals = "  ".join(f"{w[h, b]:+8.4f}" for h in range(w.shape[0]))
                print(f"  {b:5d}  {label:>6s}  {vals}")
            print()
        return

    out_dir = args.output_dir or args.run_dir / "pos_weight_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global ylim
    import numpy as np
    cache = {}
    global_ylim = None
    if args.global_vlim:
        global_max = 0.0
        for path in ckpt_files:
            step, w = extract_pos_weight(path)
            cache[path] = (step, w)
            global_max = max(global_max, float(np.abs(w).max()))
        global_ylim = global_max if global_max > 0 else 1.0
        print(f"Global ylim=±{global_ylim:.4f}")

    count, skipped = 0, 0
    for path in ckpt_files:
        step_from_name = int(path.stem.split("_")[1])
        out_path = out_dir / f"pos_weight_step_{step_from_name}.png"
        if args.skip and out_path.exists():
            skipped += 1
            continue
        step, w = cache.pop(path, None) or extract_pos_weight(path)
        print(f"  Step {step}: max={abs(w).max():.4f}, mean={w.mean():.4f}")
        plot_pos_weight(w, step, out_path, global_ylim)
        print(f"    → {out_path.name}")
        count += 1

    msg = f"\nDone — {count} plots in {out_dir}"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg)


if __name__ == "__main__":
    main()
