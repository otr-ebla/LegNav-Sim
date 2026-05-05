"""
plot_shoe_dist.py — Optimized evaluation of shoe-distance distributions.
"""

import os
# Force GPU allocation BEFORE importing JAX
os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import argparse
import warnings
from functools import partial

import jax
# Lock JAX to the CUDA device
jax.config.update("jax_default_device", jax.devices("cuda")[0])

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. Environment Setup (Must precede imports)
import jax_env as _jax_env
_jax_env.USE_LEGS = True
_jax_env.SENSOR_NOISE = True

from jax_env import MAX_STEPS
from paper_comparison_eval import (
    _reset_stacked, _step_stacked, _build_ppo_act_vmap
)

# Number of testing episodes = v_max values * scenarios * envs per batch = 3 * 6 * 2048 = 36864 episodes, which is feasible on GPU with JIT and vmap optimizations.

# 2. Hyper-Optimized Rollout
@partial(jax.jit, static_argnums=(0, 1))
def rollout_fixed_vmax(act_vmap, n_envs, rng_key, scenario_idx, v_max_val):
    rng_key, rng_v = jax.random.split(rng_key)
    # Passed dynamically so we don't re-JIT per speed
    v_max_batch = jnp.full((n_envs,), v_max_val, dtype=jnp.float32)
    
    reset_keys = jax.random.split(rng_key, n_envs)
    obs, state = jax.vmap(_reset_stacked, in_axes=(0, 0, None))(
        reset_keys, v_max_batch, scenario_idx
    )

    carry0 = (state, obs, jnp.full(n_envs, 100.0), jnp.ones(n_envs, dtype=jnp.bool_))

    def _step(carry, step_idx):
        state, obs, mhd, active = carry
        k = jax.random.fold_in(rng_key, step_idx)
        
        actions = act_vmap(obs)
        step_keys = jax.random.split(k, n_envs)
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(
            step_keys, state, actions
        )

        ch = info["closest_shoe_surface"]
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g = info["goal_reached"] & active
        c = info["collision"] & active

        next_active = active & ~done
        return (next_state, next_obs, mhd, next_active), (g, c)

    final_carry, step_data = jax.lax.scan(
        _step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )

    _, _, final_mhd, _ = final_carry
    goals, cols = step_data

    ep_goal = goals.any(axis=0)
    ep_col = cols.any(axis=0)
    success = ep_goal & ~ep_col

    return final_mhd, success, ep_col

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default=2048, type=int, help="Envs per batch for smooth distributions")
    args = parser.parse_args()

    ckpt = "checkpoints/ppo_attn_final.msgpack"
    print(f"Loading PPO checkpoint: {ckpt}")
    act_vmap = _build_ppo_act_vmap(ckpt)

    v_max_list = [0.5, 1.0, 2.0]
    scenarios = [7, 8, 9, 10, 11, 12]
    rng = jax.random.PRNGKey(42)
    results = []

    print("Evaluating PPO over testing scenarios...")
    for v in v_max_list:
        v_jax = jnp.float32(v)
        for scen in scenarios:
            print(f" -> v_max={v:^4} m/s | Scenario={scen:^2} | Envs={args.envs}")
            rng, k = jax.random.split(rng)
            mhd, success, collision = jax.device_get(
                rollout_fixed_vmax(act_vmap, args.envs, k, jnp.int32(scen), v_jax)
            )

            for i in range(args.envs):
                results.append({
                    "v_max": v,
                    "scenario": scen,
                    "min_dist": float(mhd[i]),
                    "success": bool(success[i]),
                    "collision": bool(collision[i])
                })

    df = pd.DataFrame(results)

    # 3. Plotting the reality
    # 3. Plotting the distribution
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    df_success = df[df["success"] == True]
    if len(df_success) > 0:
        sns.kdeplot(
            data=df_success, x="min_dist", hue="v_max",
            palette="viridis", fill=False, common_norm=True, ax=ax,
            linewidth=2, clip=(0.0, None)
        )
    ax.set_title("Minimum Shoe Distance Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Surface-to-Surface Distance (m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # --- Find peaks and draw vertical lines with staggered labels ---
    peaks = []
    # Seaborn adds the KDE curves to ax.lines. We extract the raw x,y data from them.
    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) > 0:
            max_idx = np.argmax(y_data)
            peaks.append((x_data[max_idx], y_data[max_idx], line.get_color()))

    # Sort peaks left-to-right to reliably stagger the text
    peaks.sort(key=lambda p: p[0])

    y_min, y_max_axis = ax.get_ylim()
    
    for i, (x_max, y_max, color) in enumerate(peaks):
        # Draw the vertical dashed line from the peak to the x-axis
        ax.vlines(x=x_max, ymin=0, ymax=y_max, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Stagger text heights to prevent overlap (pushes labels down in a staircase)
        offset_multiplier = i + 1 
        y_text = 0 - (offset_multiplier * 0.05 * y_max_axis)

        # Annotate with a small connector line pointing to the exact x-axis location
        ax.annotate(
            f"{x_max:.2f}m",
            xy=(x_max, 0),
            xytext=(x_max, y_text),
            color=color,
            ha="center", va="top",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, shrinkA=0, shrinkB=0)
        )
    # ----------------------------------------------------------------

    plt.tight_layout()
    # Use bbox_inches='tight' so the saved image doesn't cut off the annotations hanging below the axis
    plt.savefig("shoe_distance_distribution.png", dpi=300, bbox_inches='tight')
    print("\nSaved 'shoe_distance_distribution.png'")

if __name__ == "__main__":
    main()