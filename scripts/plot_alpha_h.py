#!/usr/bin/env python
"""Plot sigmoid(alpha_h) heatmap for all checkpoints in a run directory.

Usage:
    uv run python scripts/plot_alpha_h.py runs/my_run
    uv run python scripts/plot_alpha_h.py runs/my_run -o /tmp/alpha_h_plots
    uv run python scripts/plot_alpha_h.py runs/my_run --global-vlim
"""
import argparse
from pathlib import Path

import torch

from _plot_utils import extract_param, compute_stats, print_stats, print_table, plot_heatmap, iter_checkpoints


def extract_alpha_h(ckpt_path):
    return extract_param(ckpt_path, ".alpha_h", torch.sigmoid)


def main():
    parser = argparse.ArgumentParser(description="Plot sigmoid(alpha_h) heatmap per checkpoint")
    parser.add_argument("run_dir", type=Path, help="Run output directory (e.g. runs/my_run)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <run_dir>/alpha_h_plots)")
    parser.add_argument("--global-vlim", action="store_true",
                        help="Use a single shared color limit across all steps")
    parser.add_argument("--skip", action="store_true",
                        help="Skip checkpoints whose output PNG already exists")
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
            step, alpha_map = extract_alpha_h(path)
            print_table(alpha_map, step, name="sigmoid(alpha_h)")
            print()
        return

    out_dir = args.output_dir or args.run_dir / "alpha_h_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    global_vlim = None
    if args.global_vlim:
        global_max = 0.0
        for path in ckpt_files:
            step, am = extract_alpha_h(path)
            cache[path] = (step, am)
            global_max = max(global_max, float(am.max()))
        global_vlim = global_max if global_max > 0 else 1.0
        print(f"Global vlim={global_vlim:.4f}")

    count, skipped = 0, 0
    for path in ckpt_files:
        step_from_name = int(path.stem.split("_")[1])
        out_path = out_dir / f"alpha_h_step_{step_from_name}.png"
        if args.skip and out_path.exists():
            skipped += 1
            continue
        step, alpha_map = cache.pop(path, None) or extract_alpha_h(path)
        stats = compute_stats(alpha_map)
        vlim = global_vlim if global_vlim is not None else float(alpha_map.max())
        if vlim == 0:
            vlim = 1.0
        print_stats(stats, step, name="sigmoid(alpha_h)")
        plot_heatmap(
            alpha_map, step, out_path, stats,
            vmin=0, vmax=vlim, cmap="YlOrRd",
            title_expr=r"\sigma(\alpha_{h,\ell})",
            cb_label=r"$\sigma(\alpha_h)$: feature $\leftarrow$ | $\rightarrow$ geometry",
        )
        print(f"    -> {out_path.name}  (vlim={vlim:.4f})")
        count += 1

    msg = f"\nDone — {count} plots in {out_dir}"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg)


if __name__ == "__main__":
    main()
