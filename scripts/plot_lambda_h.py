#!/usr/bin/env python
"""Plot tanh(lambda_h_raw)/H (intensity gate) heatmap for all checkpoints.

Usage:
    uv run python scripts/plot_lambda_h.py runs/my_run
    uv run python scripts/plot_lambda_h.py runs/my_run -o /tmp/lambda_h_plots
    uv run python scripts/plot_lambda_h.py runs/my_run --global-vlim
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from _plot_utils import extract_param, compute_stats, print_stats, print_table, plot_heatmap, iter_checkpoints


def extract_lambda_h(ckpt_path):
    # tanh(raw) / H — matches the actual scale used in forward()
    step, state = __import__("_plot_utils").load_checkpoint(ckpt_path)

    params = {}
    for k, v in state.items():
        if k.startswith("trunk.trunk_blocks.") and k.endswith(".lambda_h_raw"):
            idx = int(k.split(".")[2])
            params[idx] = v.clone()

    del state
    if not params:
        raise RuntimeError(f"No trunk.trunk_blocks.*.lambda_h_raw found in {ckpt_path}")

    n_layers = max(params) + 1
    H = params[0].numel()
    stacked = torch.stack([params[i] for i in range(n_layers)])
    lam = torch.tanh(stacked) / H
    return step, lam.numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot tanh(lambda_h)/H heatmap per checkpoint")
    parser.add_argument("run_dir", type=Path, help="Run output directory (e.g. runs/my_run)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <run_dir>/lambda_h_plots)")
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
            step, lam_map = extract_lambda_h(path)
            print_table(lam_map, step, name="tanh(lambda_h)/H")
            print()
        return

    out_dir = args.output_dir or args.run_dir / "lambda_h_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    global_vlim = None
    if args.global_vlim:
        global_max = 0.0
        for path in ckpt_files:
            step, lm = extract_lambda_h(path)
            cache[path] = (step, lm)
            global_max = max(global_max, float(np.abs(lm).max()))
        global_vlim = global_max if global_max > 0 else 1.0
        print(f"Global vlim=+-{global_vlim:.6f}")

    count, skipped = 0, 0
    for path in ckpt_files:
        step_from_name = int(path.stem.split("_")[1])
        out_path = out_dir / f"lambda_h_step_{step_from_name}.png"
        if args.skip and out_path.exists():
            skipped += 1
            continue
        step, lam_map = cache.pop(path, None) or extract_lambda_h(path)
        stats = compute_stats(lam_map)
        vlim = global_vlim if global_vlim is not None else float(np.abs(lam_map).max())
        if vlim == 0:
            vlim = 1.0
        print_stats(stats, step, name="tanh(lambda_h)/H")
        plot_heatmap(
            lam_map, step, out_path, stats,
            vmin=-vlim, vmax=vlim, cmap="RdBu_r",
            title_expr=r"\tanh(\lambda_{h,\ell}^{\mathrm{raw}})/H",
            cb_label=r"$\tanh(\lambda_h)/H$: intensity gate (dormant at 0)",
        )
        print(f"    -> {out_path.name}  (vlim=+-{vlim:.6f})")
        count += 1

    msg = f"\nDone — {count} plots in {out_dir}"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg)


if __name__ == "__main__":
    main()
