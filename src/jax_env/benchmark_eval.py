"""
benchmark_eval.py — High-Speed Evaluation Dashboard
===================================================
Evaluates RL models across all 7 scenarios and generates a visual dashboard.

OOM FIX: Removed the nested vmap over (N_SCENARIOS × N_SPEEDS) that tried to
allocate ~5.4 GiB for a single compiled graph. Evaluation is now a sequential
Python loop over (scenario, speed) pairs; each iteration dispatches a single
vmap over N_ENVS environments, which is the actual parallelism budget the GPU
can handle. Compile time drops to seconds and VRAM stays under 2 GiB.

TRAINING CURVES: A 9th panel plots episode reward over training steps, loaded
from CSV logs written by jax_ppo.py / SACjax.py / TQCjac.py. If a log file is
missing the panel shows a "no data" notice instead of crashing.
"""

import os
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import jax_env
jax_env.USE_LEGS = True

from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT, MAX_STEPS
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState
from jax_legs import LEG_RADIUS
from jax_network import EndToEndActorCritic
from SACnetwork import SACActorNetwork
from TQCeval import TQCActorNetwork

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_SIZE   = 342
ACTION_DIM = 2
N_ENVS     = 4096   # parallelism per (scenario, speed) cell — fits in VRAM

MAX_V_TESTS = [0.2, 0.5, 0.75, 1.0, 1.33, 1.66, 2.0]
N_SCENARIOS = 7
N_SPEEDS    = len(MAX_V_TESTS)

# Training-curve CSV paths written by each trainer
TRAINING_LOG_PATHS = {
    "PPO": "checkpoints/ppo_training_log.csv",
    "SAC": "checkpoints_sac/sac_training_log.csv",
    "TQC": "checkpoints_tqc/tqc_training_log.csv",
}

_ppo_net = EndToEndActorCritic(action_dim=ACTION_DIM)
_sac_net = SACActorNetwork(action_dim=ACTION_DIM)
_tqc_net = TQCActorNetwork(action_dim=ACTION_DIM)

# ── Action Squashing ──────────────────────────────────────────────────────────
def _squash_ppo(mean, max_v):
    v = jax.nn.sigmoid(mean[..., 0]) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)

def _squash_sac_tqc(mean, max_v):
    t = jnp.tanh(mean)
    v = (t[..., 0] + 1.0) * 0.5 * max_v
    w = t[..., 1]
    return jnp.stack([v, w], axis=-1)

# ── Environment Wrappers ──────────────────────────────────────────────────────
@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    base_obs, base_state = reset_env(key, min_dist, scen_idx)
    pose      = base_obs[0:3]
    state_vec = base_obs[3:12]
    lidar     = base_obs[12:]

    base_state = base_state.replace(max_v=target_max_v)
    new_state_vec = jnp.array([
        0.0, 0.0, (target_max_v - 0.2) / 1.8,
        state_vec[3], state_vec[4],
        state_vec[5], state_vec[6], state_vec[7], state_vec[8],
    ])

    lidar_stack = jnp.tile(lidar[None, :], (3, 1))
    pose_stack  = jnp.tile(pose[None, :],  (3, 1))
    stacked_state = StackedEnvState(
        env_state=base_state, lidar_stack=lidar_stack, pose_stack=pose_stack
    )
    flat_obs = jnp.concatenate([pose_stack.flatten(), new_state_vec, lidar_stack.flatten()])
    return flat_obs, stacked_state


@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)
    new_pose      = base_obs[0:3]
    new_state_vec = base_obs[3:12]
    new_lidar     = base_obs[12:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)
    new_stacked_state = StackedEnvState(
        env_state=new_base_state, lidar_stack=new_lidar_stack, pose_stack=new_pose_stack
    )
    flat_obs = jnp.concatenate([new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()])
    return flat_obs, new_stacked_state, reward, done, info


# ── Core Evaluation Kernel ────────────────────────────────────────────────────
# Each network type gets its OWN JIT-compiled function so that `params` always
# belongs to exactly one architecture.  Applying all three networks to the same
# params (as the previous version did with jnp.where) causes Flax to look for
# kernel shapes that don't exist in the checkpoint → ScopeParamNotFoundError.
#
# All three kernels share the same rollout body; only the forward-pass line
# differs.  Each is compiled once on first call and reused for all 49 cells.

def _rollout_body(net_apply_fn, squash_fn, params, scen_idx, target_max_v, rng_key):
    """Inner rollout used by all three per-network eval functions."""
    reset_keys = jax.random.split(rng_key, N_ENVS)
    obs, state = jax.vmap(dynamic_reset_stacked, in_axes=(0, None, None, None))(
        reset_keys, 3.0, scen_idx, target_max_v
    )

    init_dist = jnp.sqrt(
        (state.env_state.goal_x - state.env_state.x) ** 2 +
        (state.env_state.goal_y - state.env_state.y) ** 2
    )

    carry = (
        state, obs,
        jnp.zeros(N_ENVS), jnp.zeros(N_ENVS),   # v_p, av_p
        jnp.zeros(N_ENVS), jnp.zeros(N_ENVS),   # w_p, aw_p
        jnp.zeros(N_ENVS),                        # path_len
        jnp.full(N_ENVS, 100.0),                  # min_human_dist
        jnp.ones(N_ENVS, dtype=jnp.bool_),        # active
    )

    human_r = LEG_RADIUS if jax_env.USE_LEGS else PEOPLE_RADIUS

    def _step(carry, step_idx):
        state, obs, v_p, av_p, w_p, aw_p, pl, mhd, active = carry
        k_step = jax.random.fold_in(rng_key, step_idx)

        raw_out = net_apply_fn({"params": params}, obs)
        # PPO returns (mean, logstd, value); SAC/TQC return (mean, log_std)
        mean = raw_out[0]
        action = jax.vmap(squash_fn)(mean, state.env_state.max_v)

        step_keys = jax.random.split(k_step, N_ENVS)
        next_obs, next_state, _, done, info = jax.vmap(step_stacked_headless)(
            step_keys, state, action
        )

        v  = next_state.env_state.v
        w  = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT

        jerk_v = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0)
        jerk_w = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)
        pl     = pl + jnp.where(active, v * DT, 0.0)

        ch  = info["closest_human"] - ROBOT_RADIUS - human_r
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"] & active
        c  = info["collision"]    & active
        pc = info["passive_col"]  & active

        step_data   = (active, done, g, c, pc, jerk_v, jerk_w)
        next_active = active & ~done
        return (next_state, next_obs, v, av, w, aw, pl, mhd, next_active), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    _, _, _, _, _, _, final_pl, final_mhd, _ = final_carry
    active_mask, _, goals, cols, pcols, jerks_v, jerks_w = step_data

    ep_lens = active_mask.sum(axis=0)
    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)

    act_col  = ep_col  & ~ep_pcol & ~ep_goal
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    avg_jerk = (jerks_v.sum(axis=0) + jerks_w.sum(axis=0)) / jnp.maximum(ep_lens, 1)
    spl      = ep_goal * (init_dist / jnp.maximum(final_pl, init_dist))
    time_g   = jnp.where(ep_goal, ep_lens * DT, jnp.nan)

    return {
        "success":  ep_goal.astype(jnp.float32),
        "act_col":  act_col.astype(jnp.float32),
        "pass_col": pass_col.astype(jnp.float32),
        "timeout":  tmo.astype(jnp.float32),
        "spl": spl, "jerk": avg_jerk, "min_dist": final_mhd, "time": time_g,
    }


# One JIT kernel per network — params stay within their own architecture scope.
@jax.jit
def evaluate_cell_ppo(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_ppo_net.apply, _squash_ppo, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_sac(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_sac_net.apply, _squash_sac_tqc, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_tqc(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_tqc_net.apply, _squash_sac_tqc, params,
                         scen_idx, target_max_v, rng_key)

_EVAL_FN = {"PPO": evaluate_cell_ppo, "SAC": evaluate_cell_sac, "TQC": evaluate_cell_tqc}


# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_checkpoint_safe(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle.get("actor_params", bundle.get("params"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]

    raw_policies = {
        "PPO": load_checkpoint_safe("checkpoints/ppo_model_best.msgpack"),
        "SAC": load_checkpoint_safe("checkpoints_sac/sac_best.msgpack"),
        "TQC": load_checkpoint_safe("checkpoints_tqc/tqc_best.msgpack"),
    }
    policies = {
        name: jax.device_put(p, gpu)
        for name, p in raw_policies.items()
        if p is not None
    }

    if not policies:
        print("No valid checkpoints found. Please train a model first.")
        return

    rng = jax.random.PRNGKey(42)

    # ── Warm-up: compile only the kernel(s) we actually have checkpoints for ──
    print("Compiling evaluation kernels (one per available policy, ~30 s each)...")
    for p_name, params in policies.items():
        rng, k_warmup = jax.random.split(rng)
        jax.block_until_ready(_EVAL_FN[p_name](params, 0, 1.0, k_warmup))
        print(f"  {p_name} kernel compiled.")
    print()

    # ── Sequential evaluation loop ────────────────────────────────────────────
    results    = []
    scen_names = {
        0: "Random", 1: "Parallel", 2: "Perpend", 3: "Circular",
        4: "Bottleneck", 5: "Intersect", 6: "Groups",
    }

    print("Executing evaluation grid (sequential cells, parallel envs)...")
    for p_name, params in policies.items():
        eval_fn  = _EVAL_FN[p_name]
        t_policy = time.time()
        for si in range(N_SCENARIOS):
            for vi, v_max in enumerate(MAX_V_TESTS):
                rng, cell_rng = jax.random.split(rng)
                cell = jax.device_get(jax.block_until_ready(
                    eval_fn(params, si, float(v_max), cell_rng)
                ))
                for i in range(N_ENVS):
                    results.append({
                        "Policy":       p_name,
                        "Scenario":     si,
                        "Max_V":        v_max,
                        "Success":      cell["success"][i],
                        "Active Col":   cell["act_col"][i],
                        "Passive Col":  cell["pass_col"][i],
                        "Timeout":      cell["timeout"][i],
                        "SPL":          cell["spl"][i],
                        "Jerk":         cell["jerk"][i],
                        "Min Dist":     cell["min_dist"][i],
                        "Time to Goal": cell["time"][i],
                    })
        print(f"  {p_name} done in {time.time() - t_policy:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv("evaluation_raw_data.csv", index=False)
    print("Saved evaluation_raw_data.csv\n")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    print("Generating dashboard...")
    sns.set_theme(style="whitegrid", palette="muted")

    # 3 rows × 3 cols = 9 panels; last panel is training curves
    fig = plt.figure(figsize=(27, 21))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard",
                 fontsize=26, weight="bold")

    # 1. Overall Outcomes
    rate_df   = df.groupby("Policy")[["Success","Active Col","Passive Col","Timeout"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax1 = plt.subplot(3, 3, 1)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax1)
    ax1.set_title("Overall Episode Outcomes (%)", fontsize=13)
    ax1.set_ylim(0, 100)

    # 2. Success by Scenario
    scen_df = df.groupby(["Scenario","Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(scen_names)
    ax2 = plt.subplot(3, 3, 2)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax2)
    ax2.set_title("Success Rate by Layout Topology", fontsize=13)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    ax2.set_ylim(0, 100)

    # 3. Success vs Speed
    v_df = df.groupby(["Max_V","Policy"])["Success"].mean().reset_index()
    v_df["Success"] *= 100
    ax3 = plt.subplot(3, 3, 3)
    sns.lineplot(data=v_df, x="Max_V", y="Success", hue="Policy",
                 marker="o", linewidth=3, markersize=8, ax=ax3)
    ax3.set_title("Success Rate vs. Robot Max Speed", fontsize=13)
    ax3.set_xticks(MAX_V_TESTS)
    ax3.set_ylim(0, 100)

    # 4. Active Collisions vs Speed
    v_col_df = df.groupby(["Max_V","Policy"])["Active Col"].mean().reset_index()
    v_col_df["Active Col"] *= 100
    ax4 = plt.subplot(3, 3, 4)
    sns.lineplot(data=v_col_df, x="Max_V", y="Active Col", hue="Policy",
                 marker="X", linewidth=3, markersize=8, ax=ax4)
    ax4.set_title("Active Collisions vs. Robot Max Speed", fontsize=13)
    ax4.set_xticks(MAX_V_TESTS)
    ax4.set_ylim(0, 100)

    suc = df[df["Success"] == 1.0]

    # 5. SPL
    ax5 = plt.subplot(3, 3, 5)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Success-weighted Path Length (SPL)", fontsize=13)

    # 6. Time to Goal
    ax6 = plt.subplot(3, 3, 6)
    sns.boxplot(data=suc, x="Policy", y="Time to Goal", hue="Policy",
                ax=ax6, showfliers=False)
    ax6.set_title("Time to Reach Goal (seconds)", fontsize=13)

    # 7. Safety Margin vs Speed
    ax7 = plt.subplot(3, 3, 7)
    sns.lineplot(data=df, x="Max_V", y="Min Dist", hue="Policy",
                 marker="^", linewidth=3, markersize=8, ax=ax7)
    ax7.set_title("Safety Margin (Min Human Dist) vs. Speed", fontsize=13)
    ax7.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Collision Threshold")
    ax7.set_xticks(MAX_V_TESTS)
    ax7.legend()

    # 8. Jerk
    ax8 = plt.subplot(3, 3, 8)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy",
                ax=ax8, showfliers=False)
    ax8.set_title("Average Kinematic Jerk (Smoothness)", fontsize=13)

    # 9. Training Curves — episode reward over environment steps
    # All three trainers now log 'step' as total_env_steps so the x-axis is
    # comparable: PPO = update × NUM_ENVS × ROLLOUT_STEPS, SAC/TQC = total_steps.
    ax9 = plt.subplot(3, 3, 9)
    any_log = False
    POLICY_COLORS = {"PPO": "#4C72B0", "SAC": "#DD8452", "TQC": "#55A868"}
    for p_name, log_path in TRAINING_LOG_PATHS.items():
        if os.path.exists(log_path):
            try:
                log_df = pd.read_csv(log_path)
                # Convert raw step count → millions of env steps for readability
                x_millions = log_df["step"] / 1e6
                # Rolling-window smooth (window proportional to log density)
                w = max(5, len(log_df) // 30)
                smoothed = log_df["mean_ep_reward"].rolling(window=w, min_periods=1).mean()
                # Faint raw trace
                ax9.plot(
                    x_millions, log_df["mean_ep_reward"],
                    alpha=0.18, linewidth=1,
                    color=POLICY_COLORS.get(p_name),
                )
                # Bold smoothed trace
                ax9.plot(
                    x_millions, smoothed,
                    label=p_name, linewidth=2.5,
                    color=POLICY_COLORS.get(p_name),
                )
                any_log = True
            except Exception as e:
                print(f"  Warning: could not read {log_path}: {e}")

    if any_log:
        ax9.set_xlabel("Environment Steps (millions)", fontsize=11)
        ax9.set_ylabel("Mean Episode Reward", fontsize=11)
        ax9.set_title("Episode Reward During Training", fontsize=13)
        ax9.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax9.legend(fontsize=10)
    else:
        ax9.text(0.5, 0.5,
                 "No training logs found.\nRun trainers to generate\n"
                 "checkpoints/*/training_log.csv",
                 ha="center", va="center", transform=ax9.transAxes,
                 fontsize=11, color="gray")
        ax9.set_title("Episode Reward During Training", fontsize=13)
        ax9.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("Saved 'Evaluation_Dashboard.png'")


if __name__ == "__main__":
    main()