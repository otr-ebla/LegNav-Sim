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
def rollout_random_vmax(act_vmap, n_envs, rng_key, scenario_idx):
    rng_key, rng_v = jax.random.split(rng_key)
    # It samples a random linear v_max for each environment in the batch, uniformly from 0.2 to 2.0 m/s.
    v_max_batch = jax.random.uniform(rng_v, shape=(n_envs,), minval=0.2, maxval=2.0)
    
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

    return final_mhd, success, ep_col, v_max_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default=2048, type=int, help="Envs per batch for smooth distributions")
    args = parser.parse_args()

    ckpt = "checkpoints/ppo_attn_final.msgpack"
    print(f"Loading PPO checkpoint: {ckpt}")
    act_vmap = _build_ppo_act_vmap(ckpt)

    scenarios = [7, 8, 9, 10, 11, 12]
    rng = jax.random.PRNGKey(42)
    results = []

    print("Evaluating PPO over testing scenarios with Random v_max [0.2 - 2.0]...")
    for scen in scenarios:
        print(f" -> Scenario={scen:^2} | Envs={args.envs}")
        rng, k = jax.random.split(rng)
        
        # Ora riceviamo 4 valori, incluso l'array delle velocità campionate
        mhd, success, collision, sampled_vmax = jax.device_get(
            rollout_random_vmax(act_vmap, args.envs, k, jnp.int32(scen))
        )

        for i in range(args.envs):
            results.append({
                "v_max": float(sampled_vmax[i]), # Salviamo il valore esatto generato
                "scenario": scen,
                "min_dist": float(mhd[i]),
                "success": bool(success[i]),
                "collision": bool(collision[i])
            })

    df = pd.DataFrame(results)

    # 3. Plotting the distribution
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    df_success = df[df["success"] == True]
    if len(df_success) > 0:
        sns.kdeplot(
            data=df_success, x="min_dist",
            color="dodgerblue", fill=True, alpha=0.3, ax=ax,
            linewidth=2, clip=(0.0, None)
        )
    ax.set_xlim(left=0.0)
    ax.set_title("Minimum Shoe Distance Distribution (Random v_max 0.2 - 2.0 m/s)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Surface-to-Surface Distance (m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # --- Calculate MEAN and draw vertical line ---
    if len(df_success) > 0:
        mean_x = df_success["min_dist"].mean()
        color = "dodgerblue" # Colore fisso per coerenza con l'area
        
        y_min, y_max_axis = ax.get_ylim()
        
        # axvline disegna la linea da cima a fondo ignorando la geometria della curva
        ax.axvline(x=mean_x, color=color, linestyle='--', linewidth=2.5, alpha=0.9)
        
        # Annota il valore della media appena sotto l'asse X
        y_text = 0 - (0.05 * y_max_axis)
        ax.annotate(
            f"Mean: {mean_x:.2f}m",
            xy=(mean_x, 0),
            xytext=(mean_x, y_text),
            color=color,
            ha="center", va="top",
            fontweight="bold",
            fontsize=11,
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, shrinkA=0, shrinkB=0)
        )
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    plt.tight_layout()
    # Use bbox_inches='tight' so the saved image doesn't cut off the annotations hanging below the axis
    plt.savefig("shoe_distance_distribution.png", dpi=300, bbox_inches='tight')
    print("\nSaved 'shoe_distance_distribution.png'")

if __name__ == "__main__":
    main()