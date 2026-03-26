#!/usr/bin/env python
"""Plot per-head position bias weights for all checkpoints.

68 bins: 0-64 = same-chain offsets (-32..+32), 65 = diff-chain, 66 = bonded, 67 = bonded-cross.

Trunk pos_bias is per-layer (48 layers × 16 heads × 68 bins).
MSA and diffusion pos_bias are shared (H × 68).

Usage:
    uv run python scripts/plot_pos_weight.py runs/my_run
    uv run python scripts/plot_pos_weight.py runs/my_run --module msa --latest --table
    uv run python scripts/plot_pos_weight.py runs/my_run --module all --latest
    uv run python scripts/plot_pos_weight.py runs/my_run --layer 0,23,47 --latest
"""
import argparse
import re
from pathlib import Path


from _plot_utils import load_checkpoint, iter_checkpoints


BIN_LABELS = {
    0: "-32", 16: "-16", 32: "0", 48: "+16", 64: "+32",
    65: "X", 66: "B", 67: "BX",
}
# X=cross-chain, B=bonded, BX=bonded cross-chain

# MSA and diffusion have a single shared pos_bias
SHARED_KEYS = {
    "msa": "trunk.msa_module.pos_bias.weight",
    "diffusion": "diffusion.pos_bias.weight",
}

# Trunk has per-layer pos_bias: trunk.uot_blocks.{i}.pos_bias.weight
TRUNK_PATTERN = re.compile(r"trunk\.uot_blocks\.(\d+)\.pos_bias\.weight")


def extract_all_pos_weights(ckpt_path, layers=None):
    """Extract all pos_bias weights from checkpoint."""
    step, state = load_checkpoint(ckpt_path)
    results = {}
    # Trunk: per-layer
    trunk_layers = {}
    for key in state:
        m = TRUNK_PATTERN.match(key)
        if m:
            idx = int(m.group(1))
            if layers is None or idx in layers:
                trunk_layers[idx] = state[key].float().numpy()
    if trunk_layers:
        results["trunk"] = trunk_layers
    # Shared modules
    for module, key in SHARED_KEYS.items():
        if key in state:
            results[module] = state[key].float().numpy()
    del state
    return step, results


def plot_shared(data, step, out_path, module, global_ylim=None):
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
        colors = ["steelblue"] * 65 + ["#e74c3c", "#2ecc71", "#9b59b6"]
        ax.bar(x, data[h], color=colors, width=0.8, edgecolor="none")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"Head {h}", fontsize=10)
        if h >= n_heads - cols:
            ticks = sorted(BIN_LABELS.keys())
            ax.set_xticks(ticks)
            ax.set_xticklabels([BIN_LABELS[t] for t in ticks], fontsize=7, rotation=45, ha="right")
        if h % cols == 0:
            ax.set_ylabel("weight")

    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{module} position bias — step {step}", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trunk_heatmap(layer_weights, step, out_path, global_ylim=None):
    """Plot trunk per-layer pos_bias as (n_layers, 68) heatmap per head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_layers = sorted(layer_weights.keys())
    n_layers = len(sorted_layers)
    # Stack: (n_layers, H, 68)
    stacked = np.stack([layer_weights[i] for i in sorted_layers])
    n_heads = stacked.shape[1]

    vlim = global_ylim or max(abs(stacked.max()), abs(stacked.min()), 0.01)

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 0.15 * n_layers * rows + 2),
                              sharex=True, sharey=True, layout="constrained")
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(stacked[:, h, :], aspect="auto", cmap="RdBu_r",
                        vmin=-vlim, vmax=vlim, interpolation="nearest")
        ax.set_title(f"Head {h}", fontsize=10)
        if h >= n_heads - cols:
            ticks = sorted(BIN_LABELS.keys())
            ax.set_xticks(ticks)
            ax.set_xticklabels([BIN_LABELS[t] for t in ticks], fontsize=7, rotation=45, ha="right")
        if h % cols == 0:
            ax.set_ylabel("Layer")
            # Show layer ticks at intervals
            layer_ticks = list(range(0, n_layers, max(1, n_layers // 8)))
            ax.set_yticks(layer_ticks)
            ax.set_yticklabels([str(sorted_layers[i]) for i in layer_ticks], fontsize=7)

    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    fig.colorbar(im, ax=axes[:n_heads].tolist(), shrink=0.6, label="weight")
    fig.suptitle(f"trunk per-layer position bias — step {step}", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_one_table(step, w, module, layer=None):
    prefix = f"{module}" + (f" L{layer}" if layer is not None else "")
    print(f"  Step {step} [{prefix}]  {w.shape[0]} heads x {w.shape[1]} bins")
    print(f"  {'bin':>5s}  {'label':>6s}  " + "  ".join(f"{'h'+str(h):>8s}" for h in range(w.shape[0])))
    for b in range(w.shape[1]):
        label = BIN_LABELS.get(b, str(b - 32))
        vals = "  ".join(f"{w[h, b]:+8.4f}" for h in range(w.shape[0]))
        print(f"  {b:5d}  {label:>6s}  {vals}")
    print()


def print_table(ckpt_files, modules, layers=None):
    """Print pos_bias values as a table."""
    for path in ckpt_files:
        step, all_w = extract_all_pos_weights(path, layers=layers)
        for mod in modules:
            if mod == "trunk" and "trunk" in all_w:
                for idx in sorted(all_w["trunk"]):
                    _print_one_table(step, all_w["trunk"][idx], "trunk", layer=idx)
            elif mod in all_w:
                _print_one_table(step, all_w[mod], mod)
            else:
                print(f"  Step {step} [{mod}]  not found in checkpoint\n")


def generate_plots(ckpt_files, modules, out_dir, args, layers=None):
    """Generate plot PNGs for each checkpoint and module."""
    out_dir.mkdir(parents=True, exist_ok=True)

    count, skipped = 0, 0
    for path in ckpt_files:
        step_from_name = int(path.stem.split("_")[1])
        step, all_w = extract_all_pos_weights(path, layers=layers)

        for mod in modules:
            suffix = f"_{mod}" if len(modules) > 1 else ""
            out_path = out_dir / f"pos_weight{suffix}_step_{step_from_name}.png"
            if args.skip and out_path.exists():
                skipped += 1
                continue

            if mod == "trunk" and "trunk" in all_w:
                layer_weights = all_w["trunk"]
                print(f"  Step {step} [trunk]: {len(layer_weights)} layers, "
                      f"max={max(abs(w).max() for w in layer_weights.values()):.4f}")
                plot_trunk_heatmap(layer_weights, step, out_path)
                print(f"    → {out_path.name}")
                count += 1
            elif mod in all_w:
                w = all_w[mod]
                print(f"  Step {step} [{mod}]: max={abs(w).max():.4f}, mean={w.mean():.4f}")
                plot_shared(w, step, out_path, module=mod)
                print(f"    → {out_path.name}")
                count += 1

    msg = f"\nDone — {count} plots in {out_dir}"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg)


def parse_layers(s):
    """Parse comma-separated layer indices."""
    if s is None:
        return None
    return set(int(x) for x in s.split(","))


def main():
    parser = argparse.ArgumentParser(description="Plot per-head position bias weights")
    parser.add_argument("run_dir", type=Path, help="Run output directory")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-m", "--module", default="trunk",
                        choices=["trunk", "msa", "diffusion", "all"],
                        help="Which pos_bias to plot (default: trunk)")
    parser.add_argument("--layer", type=str, default=None,
                        help="Comma-separated trunk layer indices (default: all)")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--table", action="store_true",
                        help="Print values as a table instead of generating plots")
    parser.add_argument("--latest", action="store_true",
                        help="Only process the latest checkpoint")
    args = parser.parse_args()

    all_modules = list(SHARED_KEYS.keys()) + ["trunk"]
    modules = all_modules if args.module == "all" else [args.module]
    layers = parse_layers(args.layer)

    ckpt_files = iter_checkpoints(args.run_dir)
    if args.latest:
        ckpt_files = ckpt_files[-1:]

    if args.table:
        print_table(ckpt_files, modules, layers=layers)
        return

    out_dir = args.output_dir or args.run_dir / "pos_weight_plots"
    generate_plots(ckpt_files, modules, out_dir, args, layers=layers)


if __name__ == "__main__":
    main()
