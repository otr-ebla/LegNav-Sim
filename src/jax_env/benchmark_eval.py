"""
benchmark_eval.py — High-Speed Evaluation Dashboard
===================================================
Evaluates RL models across all 7 scenarios and generates a visual dashboard.

OOM FIX: Removed the nested vmap over (N_SCENARIOS x N_SPEEDS) that tried to
allocate ~5.4 GiB for a single compiled graph. Evaluation is now a sequential
Python loop over (scenario, speed) pairs; each iteration dispatches a single
vmap over N_ENVS environments, which is the actual parallelism budget the GPU
can handle. Compile time drops to seconds and VRAM stays under 2 GiB.

TRAINING CURVES: A 12th panel plots episode reward over training steps, loaded
from CSV logs written by jax_ppo.py / SACjax.py / TQCjac.py.
"""

import os
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import flax.linen as nn
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
from jax_network import SharedEncoder, EndToEndActorCritic, scale_action_to_env

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_SIZE   = 662
ACTION_DIM = 2
N_ENVS     = 4096

MAX_V_TESTS = [0.2, 0.5, 0.75, 1.0, 1.33, 1.66, 2.0]
N_SCENARIOS = 7
N_SPEEDS    = len(MAX_V_TESTS)

TRAINING_LOG_PATHS = {
    "PPO": "checkpoints/ppo_training_log.csv",
    "SAC": "checkpoints_sac/sac_training_log.csv",
    "TQC": "checkpoints_tqc/tqc_training_log.csv",
}

SCEN_NAMES = {
    0: "Random", 1: "Parallel", 2: "Perpend", 3: "Circular",
    4: "Bottleneck", 5: "Intersect", 6: "Groups",
}

POLICY_COLORS = {"PPO": "#4C72B0", "SAC": "#DD8452", "TQC": "#55A868"}

# ── Network definitions (must match training code exactly) ────────────────────

# PPO: EndToEndActorCritic (monolithic encoder+actor+critic)
_ppo_net = EndToEndActorCritic(action_dim=ACTION_DIM)

# SAC: SharedEncoder + SACActorHead (LOG_STD_MAX=2.0, named Dense layers)
class _SACActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0

    @nn.compact
    def __call__(self, feat):
        mean    = nn.Dense(self.action_dim, name='mean')(feat)
        log_std = nn.Dense(self.action_dim, name='log_std')(feat)
        return mean, jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

# TQC: SharedEncoder + TQCActorHead (LOG_STD_MAX=0.5, unnamed Dense layers)
class _TQCActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat):
        mean    = nn.Dense(self.action_dim)(feat)
        log_std = nn.Dense(self.action_dim)(feat)
        return mean.astype(jnp.float32), jnp.clip(log_std.astype(jnp.float32),
                                                   self.LOG_STD_MIN, self.LOG_STD_MAX)

_shared_enc  = SharedEncoder()
_sac_head    = _SACActorHead()
_tqc_head    = _TQCActorHead()


# ── Unified apply functions ──────────────────────────────────────────────────
# All apply functions take ({"params": params}, obs) and return a tuple whose
# first element is the action mean, matching _rollout_body expectations.

def _sac_apply(variables, obs):
    p = variables["params"]
    feat = _shared_enc.apply({"params": p["enc"]}, obs)
    return _sac_head.apply({"params": p["head"]}, feat)

def _tqc_apply(variables, obs):
    p = variables["params"]
    feat = _shared_enc.apply({"params": p["enc"]}, obs)
    return _tqc_head.apply({"params": p["head"]}, feat)


# ── Action squashing (same formula for all three) ────────────────────────────
def _squash_action(mean, max_v):
    v = (jnp.tanh(mean[..., 0]) * 0.5 + 0.5) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)


# ── Environment Wrappers ────────────────────────────────────────────────────
from jax_env import STATE_VEC_SIZE as _SVS

POSE_SIZE  = 3
STACK_DIM  = 3

@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    base_obs, base_state = reset_env(key, min_dist, scen_idx)
    pose      = base_obs[0:POSE_SIZE]
    state_vec = base_obs[POSE_SIZE : POSE_SIZE + _SVS]
    lidar     = base_obs[POSE_SIZE + _SVS:]

    base_state = base_state.replace(max_v=target_max_v)
    # state_vec layout: [v, w, max_v_norm, goal_dist, goal_align]
    new_state_vec = jnp.array([
        0.0, 0.0, (target_max_v - 0.2) / 1.8,
        state_vec[3], state_vec[4],
    ])

    lidar_stack = jnp.tile(lidar[None, :], (STACK_DIM, 1))
    pose_stack  = jnp.tile(pose[None, :],  (STACK_DIM, 1))
    stacked_state = StackedEnvState(
        env_state=base_state, lidar_stack=lidar_stack, pose_stack=pose_stack
    )
    flat_obs = jnp.concatenate([pose_stack.flatten(), new_state_vec, lidar_stack.flatten()])
    return flat_obs, stacked_state


@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)
    new_pose      = base_obs[0:POSE_SIZE]
    new_state_vec = base_obs[POSE_SIZE : POSE_SIZE + _SVS]
    new_lidar     = base_obs[POSE_SIZE + _SVS:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)
    new_stacked_state = StackedEnvState(
        env_state=new_base_state, lidar_stack=new_lidar_stack, pose_stack=new_pose_stack
    )
    flat_obs = jnp.concatenate([new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()])
    return flat_obs, new_stacked_state, reward, done, info


# ── Core Evaluation Kernel ───────────────────────────────────────────────────

YIELD_DIST = 1.5
YIELD_FOV  = 1.57

def _rollout_body(net_apply_fn, squash_fn, params, scen_idx, target_max_v, rng_key):
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
        jnp.zeros(N_ENVS),                        # yield_zone_steps
        jnp.zeros(N_ENVS),                        # yield_comply_steps
    )

    human_r = LEG_RADIUS if jax_env.USE_LEGS else PEOPLE_RADIUS

    def _step(carry, step_idx):
        state, obs, v_p, av_p, w_p, aw_p, pl, mhd, active, yz_steps, yc_steps = carry
        k_step = jax.random.fold_in(rng_key, step_idx)

        raw_out = net_apply_fn({"params": params}, obs)
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

        # Yielding score
        ppl    = next_state.env_state.people
        dp_x   = ppl[:, :, 0] - next_state.env_state.x[:, None]
        dp_y   = ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang = jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None]
        rel_ang = (rel_ang + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        active_p  = ppl[:, :, 10] >= 0.0
        in_yz     = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & active_p
        any_in_yz = jnp.any(in_yz, axis=1)
        robot_stopped = v <= 0.1

        new_yz_steps = yz_steps + jnp.where(active & any_in_yz, 1.0, 0.0)
        new_yc_steps = yc_steps + jnp.where(active & any_in_yz & robot_stopped, 1.0, 0.0)

        step_data   = (active, done, g, c, pc, jerk_v, jerk_w, ch, v)
        next_active = active & ~done
        return (next_state, next_obs, v, av, w, aw, pl, mhd, next_active,
                new_yz_steps, new_yc_steps), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    _, _, _, _, _, _, final_pl, final_mhd, _, final_yz, final_yc = final_carry
    active_mask, _, goals, cols, pcols, jerks_v, jerks_w, step_dists, step_vs = step_data

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

    yield_score = jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan)

    step_dists = jnp.where(active_mask, step_dists, jnp.nan)
    step_vs    = jnp.where(active_mask, step_vs,    jnp.nan)

    return {
        "success":     ep_goal.astype(jnp.float32),
        "act_col":     act_col.astype(jnp.float32),
        "pass_col":    pass_col.astype(jnp.float32),
        "timeout":     tmo.astype(jnp.float32),
        "spl":         spl,
        "jerk":        avg_jerk,
        "min_dist":    final_mhd,
        "time":        time_g,
        "yield_score": yield_score,
        "step_dists":  step_dists,
        "step_vs":     step_vs,
    }


# One JIT kernel per network architecture.
@jax.jit
def evaluate_cell_ppo(params, scen_idx, target_max_v, rng_key):
    return _rollout_body(_ppo_net.apply, _squash_action, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_sac(params, scen_idx, target_max_v, rng_key):
    return _rollout_body(_sac_apply, _squash_action, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_tqc(params, scen_idx, target_max_v, rng_key):
    return _rollout_body(_tqc_apply, _squash_action, params,
                         scen_idx, target_max_v, rng_key)

_EVAL_FN = {"PPO": evaluate_cell_ppo, "SAC": evaluate_cell_sac, "TQC": evaluate_cell_tqc}


# ── Checkpoint loading ───────────────────────────────────────────────────────

def _load_raw(path):
    with open(path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def load_ppo(path):
    """PPO checkpoint: {"params": ..., "opt_state": ...}"""
    bundle = _load_raw(path)
    return bundle.get("params", bundle)


def load_sac(path):
    """SAC checkpoint: {"enc_params": ..., "actor_head_params": ...}"""
    bundle = _load_raw(path)
    return {"enc": bundle["enc_params"], "head": bundle["actor_head_params"]}


def load_tqc(path):
    """TQC checkpoint: {"enc_params": ..., "actor_params": ...}"""
    bundle = _load_raw(path)
    return {"enc": bundle["enc_params"], "head": bundle["actor_params"]}


_CKPT_PATHS = {
    "PPO": "checkpoints/ppo_attn_best.msgpack",
    "SAC": "checkpoints_sac/sac_best.msgpack",
    "TQC": "checkpoints_tqc/tqc_best.msgpack",
}

_LOADERS = {"PPO": load_ppo, "SAC": load_sac, "TQC": load_tqc}


# ── Dashboard plotting ──────────────────────────────────────────────────────

def _plot_dashboard(df, scatter_data):
    sns.set_theme(style="whitegrid", palette="muted")

    fig = plt.figure(figsize=(36, 21))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard",
                 fontsize=28, weight="bold")

    R, C = 3, 4

    # ── Row 1: outcome rates vs Max_V ────────────────────────────────────────
    _outcome_vs_speed(fig, R, C, 1, df, "Success",     "Success Rate vs. Max Speed",      "o")
    _outcome_vs_speed(fig, R, C, 2, df, "Active Col",  "Active Collisions vs. Max Speed",  "X")
    _outcome_vs_speed(fig, R, C, 3, df, "Passive Col", "Passive Collisions vs. Max Speed", "o")
    _outcome_vs_speed(fig, R, C, 4, df, "Timeout",     "Timeout Rate vs. Max Speed",       "s")

    # ── Row 2: quality metrics ───────────────────────────────────────────────
    suc = df[df["Success"] == 1.0]

    ax5 = plt.subplot(R, C, 5)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Success-weighted Path Length (SPL)", fontsize=13)
    ax5.set_ylabel("SPL")

    ax6 = plt.subplot(R, C, 6)
    sns.boxplot(data=suc, x="Policy", y="Time to Goal", hue="Policy",
                ax=ax6, showfliers=False)
    ax6.set_title("Time to Reach Goal (seconds)", fontsize=13)
    ax6.set_ylabel("seconds")

    ax7 = plt.subplot(R, C, 7)
    sns.lineplot(data=df, x="Max_V", y="Min Dist", hue="Policy",
                 marker="^", linewidth=3, markersize=8, ax=ax7)
    ax7.set_title("Safety Margin vs. Max Speed", fontsize=13)
    ax7.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Collision Threshold")
    ax7.set_xticks(MAX_V_TESTS)
    ax7.set_xlabel("Max Linear Speed (m/s)")
    ax7.set_ylabel("Min Human Distance (m)")
    ax7.legend(fontsize=9)

    ax8 = plt.subplot(R, C, 8)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy", ax=ax8, showfliers=False)
    ax8.set_title("Average Kinematic Jerk (Smoothness)", fontsize=13)
    ax8.set_ylabel("Jerk (m/s^3 + rad/s^3)")

    # ── Row 3: yielding, scenario breakdown, overall, training ───────────────
    v_ys_df = df.groupby(["Max_V","Policy"])["Yield Score"].mean().reset_index()
    ax9 = plt.subplot(R, C, 9)
    sns.lineplot(data=v_ys_df, x="Max_V", y="Yield Score", hue="Policy",
                 marker="D", linewidth=3, markersize=8, ax=ax9)
    ax9.set_title("Yielding Score vs. Max Speed", fontsize=13)
    ax9.set_xticks(MAX_V_TESTS)
    ax9.set_ylim(0, 1)
    ax9.set_xlabel("Max Linear Speed (m/s)")
    ax9.set_ylabel("Yield Compliance (0-1)")
    ax9.axhline(1.0, color="green", linestyle=":", alpha=0.4)
    ax9.axhline(0.5, color="gray",  linestyle=":", alpha=0.4)

    scen_df = df.groupby(["Scenario","Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(SCEN_NAMES)
    ax10 = plt.subplot(R, C, 10)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax10)
    ax10.set_title("Success Rate by Layout Topology", fontsize=13)
    ax10.set_xticklabels(ax10.get_xticklabels(), rotation=30, ha="right")
    ax10.set_ylim(0, 100)
    ax10.set_ylabel("Success Rate (%)")

    rate_df   = df.groupby("Policy")[["Success","Active Col","Passive Col","Timeout"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax11 = plt.subplot(R, C, 11)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax11)
    ax11.set_title("Overall Episode Outcomes (%)", fontsize=13)
    ax11.set_ylim(0, 100)
    ax11.set_ylabel("Rate (%)")

    _plot_training_curves(plt.subplot(R, C, 12))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("Saved 'Evaluation_Dashboard.png'")

    _plot_proximity_speed(scatter_data)


def _outcome_vs_speed(fig, R, C, pos, df, col, title, marker):
    grp = df.groupby(["Max_V","Policy"])[col].mean().reset_index()
    grp[col] *= 100
    ax = plt.subplot(R, C, pos)
    sns.lineplot(data=grp, x="Max_V", y=col, hue="Policy",
                 marker=marker, linewidth=3, markersize=8, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max Linear Speed (m/s)")
    ax.set_ylabel("Rate (%)")


def _plot_training_curves(ax):
    any_log = False
    for p_name, log_path in TRAINING_LOG_PATHS.items():
        if not os.path.exists(log_path):
            continue
        try:
            log_df = pd.read_csv(log_path)
            x_millions = log_df["step"] / 1e6
            w = max(5, len(log_df) // 30)
            smoothed = log_df["mean_ep_reward"].rolling(window=w, min_periods=1).mean()
            color = POLICY_COLORS.get(p_name)
            ax.plot(x_millions, log_df["mean_ep_reward"],
                    alpha=0.18, linewidth=1, color=color)
            ax.plot(x_millions, smoothed,
                    label=p_name, linewidth=2.5, color=color)
            any_log = True
        except Exception as e:
            print(f"  Warning: could not read {log_path}: {e}")

    if any_log:
        ax.set_xlabel("Environment Steps (millions)", fontsize=11)
        ax.set_ylabel("Mean Episode Reward", fontsize=11)
        ax.set_title("Episode Reward During Training", fontsize=13)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5,
                "No training logs found.\nRun trainers to generate\n"
                "checkpoints/*/training_log.csv",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("Episode Reward During Training", fontsize=13)
        ax.set_axis_off()


def _plot_proximity_speed(scatter_data):
    policies = list(scatter_data.keys())
    if not policies:
        return

    MAX_PTS = 120_000

    fig, axes = plt.subplots(
        1, len(policies),
        figsize=(8 * len(policies), 7),
        sharey=True, squeeze=False
    )
    fig.suptitle(
        "Linear Speed vs. Distance to Nearest Human\n"
        "(all active timesteps, all scenarios, all speed caps)",
        fontsize=15, weight="bold"
    )

    for col_idx, p_name in enumerate(policies):
        ax    = axes[0, col_idx]
        color = POLICY_COLORS.get(p_name, "#888888")
        dists, vs = scatter_data[p_name]

        ok    = (dists >= -0.3) & (dists <= 4.0) & (vs >= 0.0) & (vs <= 2.05)
        dists, vs = dists[ok], vs[ok]
        n = len(dists)

        if n > MAX_PTS:
            idx = np.random.default_rng(0).choice(n, MAX_PTS, replace=False)
            dists, vs = dists[idx], vs[idx]

        ax.scatter(dists, vs, s=1.5, alpha=0.08, color=color, rasterized=True)
        ax.axvline(0.0, color="red",    lw=1.5, ls="--", alpha=0.8, label="Collision (0 m)")
        ax.axvline(0.5, color="orange", lw=1.2, ls=":",  alpha=0.7, label="Comfort (0.5 m)")
        ax.set_xlim(-0.2, 3.8)
        ax.set_ylim(-0.05, 2.1)
        ax.set_xlabel("Surface distance to nearest human (m)", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Linear speed (m/s)", fontsize=12)
        ax.set_title(f"{p_name}  -  {n:,} pts", fontsize=13, weight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("proximity_speed_scatter.png", dpi=180)
    print("Saved 'proximity_speed_scatter.png'")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]

    # Load available checkpoints
    policies = {}
    for name, path in _CKPT_PATHS.items():
        if not os.path.exists(path):
            print(f"  {name}: checkpoint not found at {path}, skipping.")
            continue
        try:
            params = _LOADERS[name](path)
            policies[name] = jax.device_put(params, gpu)
            print(f"  {name}: loaded from {path}")
        except Exception as e:
            print(f"  {name}: failed to load ({e}), skipping.")

    if not policies:
        print("No valid checkpoints found. Please train a model first.")
        return

    rng = jax.random.PRNGKey(42)

    # Warm-up: compile kernels (pass jnp arrays so one compilation covers all cells)
    print("\nCompiling evaluation kernels (~30s each)...")
    for p_name, params in policies.items():
        rng, k_warmup = jax.random.split(rng)
        jax.block_until_ready(_EVAL_FN[p_name](
            params, jnp.int32(0), jnp.float32(1.0), k_warmup))
        print(f"  {p_name} kernel compiled.")
    print()

    # Sequential evaluation loop
    all_frames   = []
    scatter_data = {}
    total_cells  = N_SCENARIOS * N_SPEEDS

    print(f"Executing evaluation grid ({N_SCENARIOS} scenarios x {N_SPEEDS} speeds = {total_cells} cells, {N_ENVS} envs each)...")
    for p_name, params in policies.items():
        eval_fn  = _EVAL_FN[p_name]
        t_policy = time.time()
        sd_list, sv_list = [], []
        cell_idx = 0

        for si in range(N_SCENARIOS):
            t_scen = time.time()
            for vi, v_max in enumerate(MAX_V_TESTS):
                rng, cell_rng = jax.random.split(rng)
                cell = jax.device_get(jax.block_until_ready(
                    eval_fn(params, jnp.int32(si), jnp.float32(v_max), cell_rng)
                ))
                cell_idx += 1

                sd = cell["step_dists"].ravel()
                sv = cell["step_vs"].ravel()
                ok = np.isfinite(sd) & np.isfinite(sv)
                sd_list.append(sd[ok])
                sv_list.append(sv[ok])

                # Vectorized DataFrame construction (no per-env Python loop)
                n = len(cell["success"])
                cell_df = pd.DataFrame({
                    "Policy":        p_name,
                    "Scenario":      si,
                    "Max_V":         v_max,
                    "Success":       cell["success"],
                    "Active Col":    cell["act_col"],
                    "Passive Col":   cell["pass_col"],
                    "Timeout":       cell["timeout"],
                    "SPL":           cell["spl"],
                    "Jerk":          cell["jerk"],
                    "Min Dist":      cell["min_dist"],
                    "Time to Goal":  cell["time"],
                    "Yield Score":   cell["yield_score"],
                })
                all_frames.append(cell_df)

            suc_pct = np.mean([f["Success"].mean() for f in all_frames[-N_SPEEDS:]]) * 100
            print(f"    {p_name} | {SCEN_NAMES[si]:<11s} "
                  f"({cell_idx:>2d}/{total_cells}) "
                  f"suc={suc_pct:5.1f}%  "
                  f"{time.time() - t_scen:.1f}s")

        scatter_data[p_name] = (np.concatenate(sd_list), np.concatenate(sv_list))
        print(f"  {p_name} total: {time.time() - t_policy:.1f}s\n")

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv("evaluation_raw_data.csv", index=False)
    print("Saved evaluation_raw_data.csv\n")

    print("Generating dashboard...")
    _plot_dashboard(df, scatter_data)


if __name__ == "__main__":
    main()
