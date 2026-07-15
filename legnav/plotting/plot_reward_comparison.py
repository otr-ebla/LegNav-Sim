"""
plot_reward_comparison.py — Episode Reward during Training (PPO · SAC · TQC)

Produces a single figure with three vertically-stacked panels, one per
algorithm, sharing the x-axis (millions of env steps).  Each panel shows the
raw mean-episode-reward curve plus an EMA-smoothed overlay.

If a training log does not exist yet, the corresponding panel shows a
"no data" message so the figure can still be generated with partial results.

Usage:
    python -m legnav.plotting.plot_reward_comparison
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from legnav import paths

# ── Log file locations (must match the training scripts) ──────────────────────
# "max_steps_m": maximum number of env steps (in millions) to plot for that
# algorithm. Set to None to plot the full log. Edit these to control how far
# each curve extends along the x-axis.
ALGOS = [
    {
        "name":  "PPO",
        "path":  paths.checkpoint("ppo", "ppo_training_log.csv"),
        "color": "#1F77B4",   # blue
        "max_steps_m": None,
    },
    {
        "name":  "SAC",
        "path":  paths.checkpoint("sac", "sac_training_log.csv"),
        "color": "#FF7F0E",   # orange
        "max_steps_m": 20,
    },
    {
        "name":  "TQC",
        "path":  paths.checkpoint("tqc", "tqc_training_log.csv"),
        "color": "#2CA02C",   # green
        "max_steps_m": 20,
    },
]

EMA_WEIGHT = 0.90       # smoothing factor (higher → smoother)
ALPHA_RAW  = 0.20       # opacity for the raw curve behind the EMA


def _read_csv_columns(filepath, col_step="step", col_reward="mean_ep_reward"):
    """Read two columns from a CSV using only the stdlib csv module."""
    steps, rewards = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row[col_step]))
            rewards.append(float(row[col_reward]))
    return np.array(steps), np.array(rewards)


def smooth_ema(values, weight=EMA_WEIGHT):
    """Exponential moving average, same style as TensorBoard."""
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smoothed[i - 1] * weight + (1 - weight) * values[i]
    return smoothed


def main():
    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
        "font.size":       15,
        "axes.titlesize":  17,
        "axes.titleweight": "bold",
        "axes.labelsize":  18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    # Independent x-axes (each panel autoscales to its own step range, so a
    # 20M-step run fills the full panel width), but a shared y-axis so the
    # reward scale is identical across all three algorithms.
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(10, 9),
        sharex=False, sharey=True, gridspec_kw={"hspace": 0.35},
    )

    any_data = False

    for ax, algo in zip(axes, ALGOS):
        name  = algo["name"]
        color = algo["color"]
        log_path = algo["path"]

        if not os.path.exists(log_path):
            ax.text(0.5, 0.5, f"{name}  —  no training log found",
                    ha="center", va="center", fontsize=12,
                    color="#9CA3AF", style="italic",
                    transform=ax.transAxes)
            ax.set_ylabel("Episode Reward")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            continue

        steps, rewards = _read_csv_columns(log_path)
        steps_m = steps / 1e6                       # → millions

        # optional cutoff along the x-axis
        max_steps_m = algo.get("max_steps_m")
        if max_steps_m is not None:
            mask = steps_m <= max_steps_m
            steps_m, rewards = steps_m[mask], rewards[mask]

        if steps_m.size == 0:
            ax.text(0.5, 0.5, f"{name}  —  no data in range",
                    ha="center", va="center", fontsize=12,
                    color="#9CA3AF", style="italic",
                    transform=ax.transAxes)
            ax.set_ylabel("Episode Reward")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            continue

        # raw curve (faint)
        ax.plot(steps_m, rewards, color=color, alpha=ALPHA_RAW, linewidth=0.8)
        # EMA overlay
        ax.plot(steps_m, smooth_ema(rewards), color=color, linewidth=2.2)

        ax.set_ylabel("Episode Reward")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        any_data = True

    # Independent x-axes: every panel keeps its own tick labels, but the
    # "Training Steps (M)" axis label is shown only once, on the bottom panel.
    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.tick_params(labelbottom=True)
        ax.set_ylim(-50, 20)
    axes[-1].set_xlabel("Training Steps (M)")

    if not any_data:
        print("⚠  No training logs found for any algorithm. Nothing to plot.")
        sys.exit(1)

    fig.tight_layout()

    os.makedirs(paths.FIGURES_DIR, exist_ok=True)
    out_path = paths.figure("reward_comparison_ppo_sac_tqc.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
