"""
benchmark_eval.py — Massive Headless Evaluation Dashboard
==========================================================
Tests PPO, SAC, and TQC models across all 7 scenarios and multiple max_v limits.
Runs entirely on the GPU using JAX vectorization, generating a massive 
dataset in seconds, and outputs a high-res comparison dashboard image.
"""

import os
# Force GPU for massive parallel evaluation
os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import Environment Logic
from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT, MAX_STEPS, STATE_VEC_SIZE
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState

OBS_SIZE   = 342
ACTION_DIM = 2
N_ENVS     = 1000  # Number of parallel episodes per (Policy + Scenario + Max_V) combination
MAX_V_TESTS = [1.0, 1.5, 2.0]
SCENARIOS  = list(range(7))

# ── 1. NETWORK DEFINITIONS ───────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108
    @nn.compact
    def __call__(self, x):
        pose_size = 3 * self.stack_dim
        state_size = STATE_VEC_SIZE
        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]
        
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))
        
        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))
        
        fused = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        return nn.relu(nn.Dense(128)(shared))   

class PolicyNetwork(nn.Module):
    """Universal Actor wrapper matching PPO, SAC, and TQC."""
    action_dim: int = ACTION_DIM
    @nn.compact
    def __call__(self, obs):
        feat = ObsEncoder()(obs)
        mean = nn.Dense(self.action_dim)(feat)
        return mean

# Action Squashing
def scale_action_ppo(mean, max_v):
    v = jax.nn.sigmoid(mean[..., 0]) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)

def scale_action_sac_tqc(mean, max_v):
    tanh_mean = jnp.tanh(mean)
    v = (tanh_mean[..., 0] + 1.0) * 0.5 * max_v
    w = tanh_mean[..., 1]
    return jnp.stack([v, w], axis=-1)


# ── 2. DYNAMIC ENVIRONMENT WRAPPERS ──────────────────────────────────────────
@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    """Resets env and injects the target max_v immediately."""
    base_obs, base_state = reset_env(key, min_dist, scen_idx)
    
    # Force the specific target_max_v
    base_state = base_state.replace(max_v=target_max_v)
    
    # Recompute state vector with new max_v
    dx = base_state.goal_x - base_state.x
    dy = base_state.goal_y - base_state.y
    max_goal_dist = jnp.sqrt(ROOM_W**2 + ROOM_H**2)
    goal_dist  = jnp.sqrt(dx**2 + dy**2)
    goal_angle = jnp.arctan2(dy, dx)
    goal_align = (goal_angle - base_state.theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    
    new_state_vec = jnp.array([
        0.0, 0.0, (target_max_v - 0.2) / 1.8,
        goal_dist / max_goal_dist, goal_align / jnp.pi,
        base_obs[14], base_obs[15], base_obs[16], base_obs[17] # rear prox
    ])
    
    pose      = base_obs[0:9]
    lidar     = base_obs[18:]
    
    lidar_stack = jnp.tile(lidar[None, :], (3, 1))
    pose_stack  = jnp.tile(pose[:3][None,  :], (3, 1))
    
    stacked_state = StackedEnvState(env_state=base_state, lidar_stack=lidar_stack, pose_stack=pose_stack)
    flat_obs = jnp.concatenate([pose_stack.flatten(), new_state_vec, lidar_stack.flatten()])
    return flat_obs, stacked_state

@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)
    new_pose      = base_obs[0:9][:3]
    new_state_vec = base_obs[9:18]
    new_lidar     = base_obs[18:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)

    new_stacked_state = StackedEnvState(env_state=new_base_state, lidar_stack=new_lidar_stack, pose_stack=new_pose_stack)
    flat_obs = jnp.concatenate([new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()])
    return flat_obs, new_stacked_state, reward, done, info


# ── 3. MASSIVE EVALUATION LOOP ───────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(1,))
def evaluate_chunk(params, net_type: str, scen_idx, target_max_v, rng_key):
    """Runs N_ENVS episodes to completion for a specific scenario and policy."""
    reset_keys = jax.random.split(rng_key, N_ENVS)
    
    # Vectorized Initialisation
    obs, state = jax.vmap(dynamic_reset_stacked, in_axes=(0, None, None, None))(
        reset_keys, 3.0, scen_idx, target_max_v
    )

    # Initial Euclidean Distance for SPL calculation
    init_dx = state.env_state.goal_x - state.env_state.x
    init_dy = state.env_state.goal_y - state.env_state.y
    init_dist = jnp.sqrt(init_dx**2 + init_dy**2)

    # Carry shape: (state, obs, rng, v, a_v, w, a_w, path_len, min_h_dist, active)
    carry = (
        state, obs, jax.random.split(rng_key)[0],
        jnp.zeros(N_ENVS), jnp.zeros(N_ENVS), jnp.zeros(N_ENVS), jnp.zeros(N_ENVS),
        jnp.zeros(N_ENVS), jnp.full(N_ENVS, 100.0), jnp.ones(N_ENVS, dtype=jnp.bool_)
    )

    def _step(carry, _):
        state, obs, key, v_p, av_p, w_p, aw_p, pl, mhd, active = carry
        key, k_step = jax.random.split(key, 2)

        # Infer action
        mean = PolicyNetwork().apply({"params": params}, obs)
        if net_type == "PPO":
            action = jax.vmap(scale_action_ppo)(mean, state.env_state.max_v)
        else:
            action = jax.vmap(scale_action_sac_tqc)(mean, state.env_state.max_v)

        # Step Environment
        step_keys = jax.random.split(k_step, N_ENVS)
        next_obs, next_state, _, done, info = jax.vmap(step_stacked_headless)(step_keys, state, action)

        # Calculate Kinematics only if episode is still running
        v = next_state.env_state.v
        w = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT
        jerk_v = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0)
        jerk_w = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)

        # Update accumulators
        pl  = pl + jnp.where(active, v * DT, 0.0)
        ch  = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        # Track Events
        g  = info["goal_reached"]
        c  = info["collision"]
        pc = info["passive_col"]

        step_data = (active, done, g, c, pc, jerk_v, jerk_w)
        
        # Once done is True, active becomes False forever
        next_active = active & ~done
        return (next_state, next_obs, key, v, av, w, aw, pl, mhd, next_active), step_data

    # Scan over time
    final_carry, step_data = jax.lax.scan(_step, carry, None, length=MAX_STEPS)
    _, _, _, _, _, _, _, final_pl, final_mhd, _ = final_carry
    active_mask, dones, goals, cols, pcols, jerks_v, jerks_w = step_data

    # Aggregate Statistics
    ep_lens = active_mask.sum(axis=0)
    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)

    # Mutually Exclusive Categories
    act_col  = ep_col & ~ep_pcol & ~ep_goal
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    # Averages
    avg_jerk = (jerks_v.sum(axis=0) + jerks_w.sum(axis=0)) / jnp.maximum(ep_lens, 1)
    spl      = ep_goal * (init_dist / jnp.maximum(final_pl, init_dist))
    time_g   = jnp.where(ep_goal, ep_lens * DT, jnp.nan)

    return {
        "success": ep_goal.astype(jnp.float32),
        "act_col": act_col.astype(jnp.float32),
        "pass_col": pass_col.astype(jnp.float32),
        "timeout": tmo.astype(jnp.float32),
        "spl": spl,
        "jerk": avg_jerk,
        "min_dist": final_mhd,
        "time": time_g
    }


# ── 4. EXECUTION & PLOTTING ──────────────────────────────────────────────────
def load_checkpoint_safe(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: raw = f.read()
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle.get("actor_params", bundle.get("params")) # Handles all 3 structures

def main():
    print(f"🚀 Initializing Headless Evaluation Dashboard...")
    
    policies = {
        "PPO": ("PPO", load_checkpoint_safe("checkpoints/ppo_model_best.msgpack")),
        "SAC": ("SAC", load_checkpoint_safe("checkpoints_sac/sac_best.msgpack")),
        "TQC": ("TQC", load_checkpoint_safe("checkpoints_tqc/tqc_best.msgpack"))
    }

    results = []
    rng = jax.random.PRNGKey(42)
    
    # Pre-compile the graph with dummy call to avoid timing pollution
    print("Compiling JAX evaluation graphs...")
    for net_type in ["PPO", "SAC"]: # SAC and TQC share the same net_type string internally
        dummy_params = PolicyNetwork().init(rng, jnp.zeros((1, OBS_SIZE)))["params"]
        _ = evaluate_chunk(dummy_params, net_type, 0, 1.5, rng)

    start_time = time.time()

    for p_name, (net_type, params) in policies.items():
        if params is None:
            print(f"⚠️  Missing checkpoint for {p_name}, skipping...")
            continue
            
        print(f"Evaluating {p_name} ", end="", flush=True)
        for scen in SCENARIOS:
            for v_max in MAX_V_TESTS:
                rng, sub_rng = jax.random.split(rng)
                metrics = evaluate_chunk(params, net_type, scen, v_max, sub_rng)
                metrics = jax.device_get(metrics) # Pull from GPU
                
                # Expand to Pandas Rows
                for i in range(N_ENVS):
                    results.append({
                        "Policy": p_name,
                        "Scenario": scen,
                        "Max_V": v_max,
                        "Success": metrics["success"][i],
                        "Active Col": metrics["act_col"][i],
                        "Passive Col": metrics["pass_col"][i],
                        "Timeout": metrics["timeout"][i],
                        "SPL": metrics["spl"][i],
                        "Jerk": metrics["jerk"][i],
                        "Min Dist": metrics["min_dist"][i],
                        "Time to Goal": metrics["time"][i]
                    })
            print(".", end="", flush=True)
        print(" Done!")

    df = pd.DataFrame(results)
    
    # Save Raw Data
    df.to_csv("evaluation_raw_data.csv", index=False)
    print(f"\n✅ Evaluated {len(df):,} episodes in {time.time()-start_time:.1f} seconds.")

    # ── GENERATE DASHBOARD IMAGE ──
    print("Generating Dashboard Plots...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard", fontsize=24, weight="bold")

    # Grouped data for rates
    rate_df = df.groupby(["Policy"])[["Success", "Active Col", "Passive Col", "Timeout"]].mean().reset_index()
    rate_df_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_df_melt["Rate"] *= 100 # To percentage

    # Plot 1: Overall Outcome Rates (Stacked/Grouped Bar)
    ax1 = plt.subplot(2, 3, 1)
    sns.barplot(data=rate_df_melt, x="Outcome", y="Rate", hue="Policy", ax=ax1)
    ax1.set_title("Overall Episode Outcomes (%)", fontsize=16)
    ax1.set_ylim(0, 100)

    # Plot 2: Success Rate per Scenario
    scen_df = df.groupby(["Scenario", "Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_names = {0:"Random", 1:"Parallel", 2:"Perpend", 3:"Circular", 4:"Bottleneck", 5:"Intersect", 6:"Groups"}
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(scen_names)
    
    ax2 = plt.subplot(2, 3, 2)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax2)
    ax2.set_title("Success Rate by Layout Topology", fontsize=16)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    ax2.set_ylim(0, 100)

    # Plot 3: Success Rate by Max Velocity
    v_df = df.groupby(["Max_V", "Policy"])["Success"].mean().reset_index()
    v_df["Success"] *= 100
    ax3 = plt.subplot(2, 3, 3)
    sns.lineplot(data=v_df, x="Max_V", y="Success", hue="Policy", marker="o", linewidth=3, markersize=10, ax=ax3)
    ax3.set_title("Success Rate vs. Robot Max Speed", fontsize=16)
    ax3.set_xticks(MAX_V_TESTS)
    ax3.set_ylim(0, 100)

    # Plot 4: Efficiency (SPL) Boxplot
    ax4 = plt.subplot(2, 3, 4)
    # Only plot SPL for successful episodes to judge strictly path efficiency
    sns.boxplot(data=df[df["Success"] == 1.0], x="Policy", y="SPL", hue="Policy", ax=ax4, showfliers=False)
    ax4.set_title("Success weighted by Path Length (SPL)", fontsize=16)

    # Plot 5: Time to Goal Boxplot
    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=df[df["Success"] == 1.0], x="Policy", y="Time to Goal", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Time to Reach Goal (Seconds)", fontsize=16)

    # Plot 6: Average Jerk (Smoothness)
    ax6 = plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy", ax=ax6, showfliers=False)
    ax6.set_title("Average Kinematic Jerk (Smoothness)", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("✅ Saved dashboard to 'Evaluation_Dashboard.png'")


if __name__ == "__main__":
    main()