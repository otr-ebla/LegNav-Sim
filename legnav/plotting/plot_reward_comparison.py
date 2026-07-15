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
ALGOS = [
    {
        "name":  "PPO (Legs)",
        "path":  paths.checkpoint("ppo", "ppo_training_log.csv"),
        "color": "#3B82F6",   # blue
    },
    {
        "name":  "SAC",
        "path":  paths.checkpoint("sac", "sac_training_log.csv"),
        "color": "#F59E0B",   # amber
    },
    {
        "name":  "TQC",
        "path":  paths.checkpoint("tqc", "tqc_training_log.csv"),
        "color": "#10B981",   # emerald
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

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(10, 9),
        sharex=True, gridspec_kw={"hspace": 0.25},
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

        # raw curve (faint)
        ax.plot(steps_m, rewards, color=color, alpha=ALPHA_RAW, linewidth=0.8)
        # EMA overlay
        ax.plot(steps_m, smooth_ema(rewards), color=color, linewidth=2.2,
                label=f"{name} (EMA {EMA_WEIGHT})")

        ax.set_ylabel("Episode Reward")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        any_data = True

    axes[-1].set_xlabel("Training Steps (M)")
    axes[-1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

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
