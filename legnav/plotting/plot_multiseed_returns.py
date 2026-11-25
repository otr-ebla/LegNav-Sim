"""
plot_multiseed_returns.py — Episode return vs environment steps over the multi-seed runs

One line per method (mean over seeds of the per-seed smoothed training return) with a lighter band for
the 95% confidence interval of that mean (Student t over the seeds). Reads the
checkpoints/multiseed/<method>_<ped>_s<seed>_curve.csv files written by train_multiseed.

Usage:
    python -m legnav.plotting.plot_multiseed_returns [--ped nsg]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from legnav import paths
from legnav.plotting.plot_reward_comparison import _apply_rcparams, smooth_window

MULTISEED = os.path.join(paths.REPO_ROOT, "checkpoints", "multiseed")
METHODS = [  # (curve prefix, label, color)
    ("calf_ppo", "PPO", "#1F77B4"),         # blue
    ("calf_sac", "SAC", "#FF7F0E"),         # orange
    ("calf_tqc", "TQC", "#2CA02C"),         # green
    ("vanilla", "VanillaE2E", "#9467BD"),   # purple
]
SMOOTH = 5     # per-seed rolling window (curve points are every ~0.5M steps)
GRID_N = 400


def load_method(prefix, ped):
    files = sorted(glob.glob(os.path.join(MULTISEED, f"{prefix}_{ped}_s[0-9]_curve.csv")))
    runs = []
    for f in files:
        d = pd.read_csv(f)
        d = d[(d.n_episodes > 0) & d.mean_return.notna()].sort_values("step_actual")  # SAC/TQC log an empty final row
        runs.append((d.step_actual.to_numpy() / 1e6, smooth_window(d.mean_return.to_numpy(), SMOOTH)))
    if not runs:
        return None
    lo = max(r[0][0] for r in runs); hi = min(r[0][-1] for r in runs)
    x = np.linspace(lo, hi, GRID_N)
    Y = np.stack([np.interp(x, s, y) for s, y in runs])
    n = len(runs)
    mean = Y.mean(0)
    half = stats.t.ppf(0.975, n - 1) * Y.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return x, mean, half, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ped", default="nsg", help="pedestrian model: nsg (legs) or disc")
    args = ap.parse_args()
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix, label, color in METHODS:
        r = load_method(prefix, args.ped)
        if r is None:
            print(f"[skip] no curves for {prefix}_{args.ped}"); continue
        x, mean, half, n = r
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.22, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.2, label=label)
        print(f"{label}: {n} seeds, {x[-1]:.1f}M steps, final return {mean[-1]:.2f} ± {half[-1]:.2f}")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Episode return")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.7)
    fig.tight_layout()
    os.makedirs(paths.FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = paths.figure(f"multiseed_return_{args.ped}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
