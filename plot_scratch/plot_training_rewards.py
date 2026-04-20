import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))

LOG_FILES = {
    "PPO": os.path.join(project_root, "src/jax_env/checkpoints/ppo_training_log.csv"),
    "SAC": os.path.join(project_root, "src/jax_env/checkpoints_sac/sac_training_log.csv"),
    "TQC": os.path.join(project_root, "src/jax_env/checkpoints_tqc/tqc_training_log.csv"),
}

COLORS = {
    "PPO": "#2196F3",
    "SAC": "#4CAF50",
    "TQC": "#FF5722",
}

SMOOTH_WINDOW = 10  # rolling mean window (set to 1 to disable)


def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=False)
fig.suptitle("Raw Episode Reward During Training", fontsize=15, fontweight="bold", y=0.98)

for ax, (algo, path) in zip(axes, LOG_FILES.items()):
    df = pd.read_csv(path)
    steps = df["step"].values
    rewards = df["mean_ep_reward"]

    raw = rewards.values
    smoothed = smooth(rewards, SMOOTH_WINDOW).values

    steps_m = steps / 1e6  # convert to millions

    ax.fill_between(steps_m, raw, alpha=0.25, color=COLORS[algo], label="_nolegend_")
    ax.plot(steps_m, raw, color=COLORS[algo], alpha=0.45, linewidth=0.8, label="raw")
    ax.plot(steps_m, smoothed, color=COLORS[algo], linewidth=2.0, label=f"smoothed (w={SMOOTH_WINDOW})")

    ax.set_title(algo, fontsize=13, fontweight="bold", loc="left", pad=6)
    ax.set_ylabel("Mean Episode Reward", fontsize=10)
    ax.set_xlabel("Environment Steps (M)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(current_dir, "training_rewards_ppo_sac_tqc.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
