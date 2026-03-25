"""Shared utilities for per-checkpoint parameter heatmap scripts."""
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_checkpoint(ckpt_path):
    """Load checkpoint via mmap, return (step, state_dict)."""
    with open(ckpt_path, "rb") as f:
        try:
            zipfile.ZipFile(f)
            is_zip = True
        except zipfile.BadZipFile:
            is_zip = False

    ckpt = torch.load(
        ckpt_path, map_location="cpu", weights_only=False,
        **({"mmap": True} if is_zip else {}),
    )
    step = ckpt.get("step", 0)
    state = ckpt["model"]
    del ckpt
    return step, state


def extract_param(ckpt_path, suffix, activation):
    """Extract per-block trunk parameter from checkpoint.

    Args:
        suffix: Parameter name suffix, e.g. ".gamma" or ".w_dist_logit".
        activation: Callable applied to stacked tensor, e.g. torch.tanh.

    Returns:
        (step, param_map) where param_map is (n_layers, n_heads) numpy array.
    """
    step, state = load_checkpoint(ckpt_path)

    params = {}
    for k, v in state.items():
        if k.startswith("trunk.uot_blocks.") and k.endswith(suffix):
            idx = int(k.split(".")[2])
            params[idx] = v.clone()

    del state

    if not params:
        raise RuntimeError(f"No trunk.uot_blocks.*{suffix} found in {ckpt_path}")

    n_layers = max(params) + 1
    param_map = activation(torch.stack([params[i] for i in range(n_layers)])).numpy()
    return step, param_map


def iter_checkpoints(run_dir):
    """Return sorted list of step_*.pt checkpoint paths."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_files:
        raise FileNotFoundError(f"No step_*.pt files in {ckpt_dir}")
    return ckpt_files


def compute_stats(data):
    """Compute marginal statistics for (n_layers, n_heads) array.

    Returns dict with keys: total_mean, total_std, head_mean, head_std,
    layer_mean, layer_std.
    """
    return {
        "total_mean": data.mean(),
        "total_std": data.std(),
        "head_mean": data.mean(axis=0),
        "head_std": data.std(axis=0),
        "layer_mean": data.mean(axis=1),
        "layer_std": data.std(axis=1),
    }


def print_stats(stats, step, name="param"):
    """Print mean/std statistics from precomputed stats dict."""
    n_heads = len(stats["head_mean"])
    n_layers = len(stats["layer_mean"])

    print(f"  Step {step} [{name}]  total: mean={stats['total_mean']:+.4f} std={stats['total_std']:.4f}")
    head_parts = [f"h{h}={stats['head_mean'][h]:+.4f}±{stats['head_std'][h]:.4f}" for h in range(n_heads)]
    for i in range(0, n_heads, 8):
        prefix = "    per head:  " if i == 0 else "               "
        print(f"{prefix}{', '.join(head_parts[i:i+8])}")

    def _fmt_layer(i):
        return f"L{i}={stats['layer_mean'][i]:+.4f}±{stats['layer_std'][i]:.4f}"
    layer_parts = [_fmt_layer(i) for i in range(n_layers)]
    # Wrap at 6 per line for readability
    for i in range(0, n_layers, 6):
        prefix = "    per layer: " if i == 0 else "               "
        print(f"{prefix}{', '.join(layer_parts[i:i+6])}")


def plot_heatmap(data, step, out_path, stats, *, vmin, vmax, cmap="RdBu_r",
                 title_expr, cb_label):
    """Save a heatmap with marginal mean/std bar charts.

    Args:
        data: (n_layers, n_heads) numpy array.
        stats: Precomputed stats dict from compute_stats().
        title_expr: LaTeX expression for the title (without $).
        cb_label: Colorbar label string.
    """
    n_layers, n_heads = data.shape

    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig = plt.figure(figsize=(10, 13), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[6, 1.5], height_ratios=[1.5, 6],
                              hspace=0.05, wspace=0.05)

        # Main heatmap
        ax_main = fig.add_subplot(gs[1, 0])
        im = ax_main.imshow(
            data, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax_main.set_xlabel(r"Head $h$")
        ax_main.set_ylabel(r"Layer $\ell$")

        # Top panel: per-head mean ± std
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        x = np.arange(n_heads)
        ax_top.bar(x, stats["head_mean"], yerr=stats["head_std"],
                   color="steelblue", alpha=0.7, capsize=2, edgecolor="none")
        ax_top.axhline(0, color="k", lw=0.5)
        ax_top.set_ylabel(r"mean $\pm$ std")
        ax_top.set_title(
            rf"${title_expr}$ — step {step}"
            f"   (total: {stats['total_mean']:+.4f} ± {stats['total_std']:.4f})",
        )
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Right panel: per-layer mean ± std
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        y = np.arange(n_layers)
        ax_right.barh(y, stats["layer_mean"], xerr=stats["layer_std"],
                      color="steelblue", alpha=0.7, capsize=2, edgecolor="none")
        ax_right.axvline(0, color="k", lw=0.5)
        ax_right.set_xlabel(r"mean $\pm$ std")
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # Colorbar in the empty top-right cell
        ax_cb = fig.add_subplot(gs[0, 1])
        ax_cb.axis("off")
        fig.colorbar(im, ax=ax_cb, fraction=0.9, label=cb_label)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
