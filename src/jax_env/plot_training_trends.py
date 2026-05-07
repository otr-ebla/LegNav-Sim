"""
plot_training_trends.py
Generates comparison plots for Episode Reward and Log Sigma trends between two PPO variants.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for the two training logs
LOG_INSIDE = "checkpoints/ppo_tanh_inside_log.csv"
LOG_OUTSIDE = "checkpoints/ppo_tanh_outside_log.csv"

def load_data():
    if not os.path.exists(LOG_INSIDE) or not os.path.exists(LOG_OUTSIDE):
        raise FileNotFoundError("Missing one or both CSV logs. Make sure to run both trainings first.")
    
    df_in = pd.read_csv(LOG_INSIDE)
    df_out = pd.read_csv(LOG_OUTSIDE)
    
    # Calculate Log Sigma directly from the saved Sigma values
    for df in [df_in, df_out]:
        df["log_sigma_v"] = np.log(df["sigma_v"])
        df["log_sigma_w"] = np.log(df["sigma_w"])
        
    return df_in, df_out

def smooth_curve(scalars, weight=0.85):
    """Exponential moving average smoothing (similar to TensorBoard)."""
    last = scalars.iloc[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    print("📊 Loading training logs...")
    df_in, df_out = load_data()
    
    # Common plot settings
    plt.style.use('seaborn-v0_8-whitegrid')
    color_in = "dodgerblue"
    color_out = "crimson"
    alpha_raw = 0.25
    
    # =========================================================================
    # FIGURE 1: Episode Reward Trend
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Tanh Inside
    ax1.plot(df_in["step"], df_in["mean_ep_reward"], color=color_in, alpha=alpha_raw)
    ax1.plot(df_in["step"], smooth_curve(df_in["mean_ep_reward"]), color=color_in, linewidth=2.5, label="Tanh Inside Network")
    
    # Tanh Outside
    ax1.plot(df_out["step"], df_out["mean_ep_reward"], color=color_out, alpha=alpha_raw)
    ax1.plot(df_out["step"], smooth_curve(df_out["mean_ep_reward"]), color=color_out, linewidth=2.5, label="Tanh Outside Network")
    
    ax1.set_title("Episode Reward during Training", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Environment Steps", fontsize=12)
    ax1.set_ylabel("Mean Episode Reward", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    
    fig1.tight_layout()
    fig1.savefig("trend_reward_comparison.png", dpi=300)
    print("✅ Saved 'trend_reward_comparison.png'")
    
    # =========================================================================
    # FIGURE 2: Log Sigma Trends (Subplots for V and W)
    # =========================================================================
    fig2, (ax_v, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Exploratory Noise (Log Sigma) Decay during Training", fontsize=16, fontweight="bold")
    
    # Subplot A: Linear Velocity (V)
    ax_v.plot(df_in["step"], df_in["log_sigma_v"], color=color_in, linewidth=2, label="Tanh Inside")
    ax_v.plot(df_out["step"], df_out["log_sigma_v"], color=color_out, linewidth=2, label="Tanh Outside")
    ax_v.set_title("Log Sigma - Linear Velocity (v)", fontsize=13)
    ax_v.set_xlabel("Environment Steps", fontsize=11)
    ax_v.set_ylabel("Log(Sigma)", fontsize=11)
    ax_v.legend(fontsize=11)
    
    # Subplot B: Angular Velocity (W)
    ax_w.plot(df_in["step"], df_in["log_sigma_w"], color=color_in, linewidth=2, label="Tanh Inside")
    ax_w.plot(df_out["step"], df_out["log_sigma_w"], color=color_out, linewidth=2, label="Tanh Outside")
    ax_w.set_title("Log Sigma - Angular Velocity (w)", fontsize=13)
    ax_w.set_xlabel("Environment Steps", fontsize=11)
    ax_w.legend(fontsize=11)
    
    fig2.tight_layout()
    fig2.savefig("trend_logsigma_comparison.png", dpi=300)
    print("✅ Saved 'trend_logsigma_comparison.png'")

if __name__ == "__main__":
    main()