"""
benchmark_eval.py — High-Speed Evaluation Dashboard
===================================================
Evaluates RL models across all 7 scenarios and generates a visual dashboard.

OOM FIX: Removed the nested vmap over (N_SCENARIOS x N_SPEEDS) that tried to
allocate ~5.4 GiB for a single compiled graph. Evaluation is now a sequential
Python loop over (scenario, speed) pairs; each iteration dispatches a single
vmap over N_ENVS environments, which is the actual parallelism budget the GPU
can handle. Compile time drops to seconds and VRAM stays under 2 GiB.

"""

import os
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

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
from jax_network import SharedEncoder, EndToEndActorCritic, scale_action_to_env

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_SIZE   = 662
ACTION_DIM = 2
N_ENVS     = 4096

MAX_V_TESTS = [0.2, 0.5, 0.75, 1.0, 1.33, 1.66, 2.0]
N_SCENARIOS = 7
N_SPEEDS    = len(MAX_V_TESTS)

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
    base_obs, base_state = reset_env(key, min_dist, scen_idx, 0.0)  # ghost_prob=0.0: pedestrians always avoid the robot (matches jax_eval_multi)
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
        reset_keys, 9.0, scen_idx, target_max_v  # max_goal_dist=9.0 matches jax_eval_multi
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

        if jax_env.USE_LEGS:
            ch = info["closest_shoe_surface"]
        else:
            ch = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
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

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard",
                 fontsize=14, weight="bold")

    R, C = 3, 4
    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7
    FS_LEG   = 6
    LW = 2
    MS = 5

    # ── Row 1: outcome rates vs Max_V ────────────────────────────────────────
    _outcome_vs_speed(fig, R, C, 1, df, "Success",     "Success Rate vs. Max Speed",      "o")
    _outcome_vs_speed(fig, R, C, 2, df, "Active Col",  "Active Collisions vs. Max Speed",  "X")
    _outcome_vs_speed(fig, R, C, 3, df, "Passive Col", "Passive Collisions vs. Max Speed", "o")
    _outcome_vs_speed(fig, R, C, 4, df, "Timeout",     "Timeout Rate vs. Max Speed",       "s")

    # ── Row 2: quality metrics ───────────────────────────────────────────────
    suc = df[df["Success"] == 1.0]

    ax5 = plt.subplot(R, C, 5)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Success-weighted Path Length (SPL)", fontsize=FS_TITLE)
    ax5.set_ylabel("SPL", fontsize=FS_LABEL)
    ax5.tick_params(labelsize=FS_TICK)

    ax6 = plt.subplot(R, C, 6)
    sns.boxplot(data=suc, x="Policy", y="Time to Goal", hue="Policy",
                ax=ax6, showfliers=False)
    ax6.set_title("Time to Reach Goal (seconds)", fontsize=FS_TITLE)
    ax6.set_ylabel("seconds", fontsize=FS_LABEL)
    ax6.tick_params(labelsize=FS_TICK)

    ax7 = plt.subplot(R, C, 7)
    sns.lineplot(data=df, x="Max_V", y="Min Dist", hue="Policy",
                 marker="^", linewidth=LW, markersize=MS, ax=ax7)
    ax7.set_title("Safety Margin vs. Max Speed", fontsize=FS_TITLE)
    ax7.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Collision Threshold")
    ax7.set_xticks(MAX_V_TESTS)
    ax7.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax7.set_ylabel("Min Human Distance (m)", fontsize=FS_LABEL)
    ax7.tick_params(labelsize=FS_TICK)
    ax7.legend(fontsize=FS_LEG)

    ax8 = plt.subplot(R, C, 8)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy", ax=ax8, showfliers=False)
    ax8.set_title("Average Kinematic Jerk (Smoothness)", fontsize=FS_TITLE)
    ax8.set_ylabel("Jerk (m/s^3 + rad/s^3)", fontsize=FS_LABEL)
    ax8.tick_params(labelsize=FS_TICK)

    # ── Row 3: yielding, scenario breakdown, overall ─────────────────────────
    v_ys_df = df.groupby(["Max_V","Policy"])["Yield Score"].mean().reset_index()
    ax9 = plt.subplot(R, C, 9)
    sns.lineplot(data=v_ys_df, x="Max_V", y="Yield Score", hue="Policy",
                 marker="D", linewidth=LW, markersize=MS, ax=ax9)
    ax9.set_title("Yielding Score vs. Max Speed", fontsize=FS_TITLE)
    ax9.set_xticks(MAX_V_TESTS)
    ax9.set_ylim(0, 1)
    ax9.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax9.set_ylabel("Yield Compliance (0-1)", fontsize=FS_LABEL)
    ax9.tick_params(labelsize=FS_TICK)
    ax9.axhline(1.0, color="green", linestyle=":", alpha=0.4)
    ax9.axhline(0.5, color="gray",  linestyle=":", alpha=0.4)
    ax9.legend(fontsize=FS_LEG)

    scen_df = df.groupby(["Scenario","Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(SCEN_NAMES)
    ax10 = plt.subplot(R, C, 10)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax10)
    ax10.set_title("Success Rate by Layout Topology", fontsize=FS_TITLE)
    ax10.set_xticklabels(ax10.get_xticklabels(), rotation=30, ha="right", fontsize=FS_TICK)
    ax10.set_ylim(0, 100)
    ax10.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax10.tick_params(labelsize=FS_TICK)

    rate_df   = df.groupby("Policy")[["Success","Active Col","Passive Col","Timeout"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax11 = plt.subplot(R, C, 11)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax11)
    ax11.set_title("Overall Episode Outcomes (%)", fontsize=FS_TITLE)
    ax11.set_ylim(0, 100)
    ax11.set_ylabel("Rate (%)", fontsize=FS_LABEL)
    ax11.tick_params(labelsize=FS_TICK)

    # Position 12 intentionally left empty (training curves removed)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("Saved 'Evaluation_Dashboard.png'")

    _plot_proximity_speed(scatter_data)


def _outcome_vs_speed(fig, R, C, pos, df, col, title, marker):
    grp = df.groupby(["Max_V","Policy"])[col].mean().reset_index()
    grp[col] *= 100
    ax = plt.subplot(R, C, pos)
    sns.lineplot(data=grp, x="Max_V", y=col, hue="Policy",
                 marker=marker, linewidth=2, markersize=5, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=8)
    ax.set_ylabel("Rate (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)


def _plot_proximity_speed(scatter_data):
    policies = list(scatter_data.keys())
    if not policies:
        return

    MAX_PTS = 120_000

    fig, axes = plt.subplots(
        1, len(policies),
        figsize=(4 * len(policies), 3.5),
        sharey=True, squeeze=False
    )
    fig.suptitle(
        "Linear Speed vs. Distance to Nearest Human",
        fontsize=10, weight="bold"
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

        ax.scatter(dists, vs, s=1.0, alpha=0.08, color=color, rasterized=True)
        ax.axvline(0.0, color="red",    lw=1.0, ls="--", alpha=0.8, label="Collision (0 m)")
        ax.axvline(0.5, color="orange", lw=0.8, ls=":",  alpha=0.7, label="Comfort (0.5 m)")
        ax.set_xlim(-0.2, 3.8)
        ax.set_ylim(-0.05, 2.1)
        ax.set_xlabel("Surface distance to nearest human (m)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Linear speed (m/s)", fontsize=8)
        ax.set_title(f"{p_name}  -  {n:,} pts", fontsize=9, weight="bold")
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("proximity_speed_scatter.png", dpi=180)
    print("Saved 'proximity_speed_scatter.png'")
    plt.close(fig)


# ── Test Scenario Configuration ──────────────────────────────────────────────
from jax_scenarios import TEST_ROBOT_WAYPOINTS, TEST_SCENARIO_NAMES

N_TEST_ENVS   = 512
TEST_SCEN_IDS = sorted(TEST_SCENARIO_NAMES.keys())   # [7, 8, 9, 10, 11, 12]
N_TEST_SCENS  = len(TEST_SCEN_IDS)


# ── Per-waypoint Segment Rollout ──────────────────────────────────────────────
#
# Runs MAX_STEPS steps from a provided (obs, state) without calling reset.
# Saves the stacked state at the moment the goal is first reached (pre-done,
# i.e. before step_env internally resets the env) so the outer Python loop can
# set a new goal and continue from the arrival position.

def _segment_core(net_apply_fn, squash_fn, params,
                  init_obs, init_state, rng_key, step_offset):
    init_dist = jnp.sqrt(
        (init_state.env_state.goal_x - init_state.env_state.x) ** 2 +
        (init_state.env_state.goal_y - init_state.env_state.y) ** 2
    )

    carry = (
        init_state, init_obs,
        jnp.zeros(N_TEST_ENVS),           # path_len
        jnp.full(N_TEST_ENVS, 100.0),     # min_human_dist
        jnp.zeros(N_TEST_ENVS),           # v_p
        jnp.zeros(N_TEST_ENVS),           # w_p
        jnp.zeros(N_TEST_ENVS),           # av_p
        jnp.zeros(N_TEST_ENVS),           # aw_p
        jnp.ones(N_TEST_ENVS,  dtype=jnp.bool_),   # active
        jnp.zeros(N_TEST_ENVS, dtype=jnp.bool_),   # gr_flag (goal reached ever)
        init_state,                        # gr_state (pre-done state at first goal)
        init_obs,                          # gr_obs
    )

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p,
         active, gr_flag, gr_state, gr_obs) = carry

        k_step = jax.random.fold_in(rng_key, step_offset + step_idx)

        raw_out = net_apply_fn({"params": params}, obs)
        mean    = raw_out[0]
        action  = jax.vmap(squash_fn)(mean, state.env_state.max_v)

        step_keys = jax.random.split(k_step, N_TEST_ENVS)
        next_obs, next_state, _, done, info = jax.vmap(step_stacked_headless)(
            step_keys, state, action
        )

        v  = next_state.env_state.v
        w  = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT

        pl  = pl  + jnp.where(active, v * DT, 0.0)
        if jax_env.USE_LEGS:
            ch = info["closest_shoe_surface"]
        else:
            ch = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"] & active
        c  = info["collision"]    & active
        pc = info["passive_col"]  & active

        # On the first step where goal is reached, save the PRE-step state/obs.
        # (next_state is already reset by step_env; we need the arrival state.)
        first_goal = g & ~gr_flag

        def _sel(new_a, old_a):
            if new_a.ndim == 1:
                return jnp.where(first_goal, new_a, old_a)
            return jnp.where(
                first_goal.reshape([-1] + [1] * (new_a.ndim - 1)), new_a, old_a
            )

        new_gr_state = jax.tree_util.tree_map(_sel, state, gr_state)
        new_gr_obs   = _sel(obs, gr_obs)
        new_gr_flag  = gr_flag | g

        next_active = active & ~done
        step_data   = (active, g, c, pc, v, ch)
        return (
            next_state, next_obs, pl, mhd, v, av, w, aw,
            next_active, new_gr_flag, new_gr_state, new_gr_obs
        ), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    (_, _, final_pl, final_mhd, _, _, _, _,
     _, final_gr_flag, final_gr_state, final_gr_obs) = final_carry
    active_mask, goals, cols, pcols, step_vs, step_dists = step_data

    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)

    act_col  = ep_col  & ~ep_pcol & ~ep_goal
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    spl = ep_goal * (init_dist / jnp.maximum(final_pl, init_dist))

    metrics = {
        "goal_reached": ep_goal,
        "act_col":      act_col.astype(jnp.float32),
        "pass_col":     pass_col.astype(jnp.float32),
        "timeout":      tmo.astype(jnp.float32),
        "spl":          spl,
        "min_dist":     final_mhd,
    }
    return metrics, final_gr_state, final_gr_obs, final_gr_flag


@jax.jit
def segment_ppo(params, init_obs, init_state, rng_key, step_offset):
    return _segment_core(_ppo_net.apply, _squash_action, params,
                         init_obs, init_state, rng_key, step_offset)

@jax.jit
def segment_sac(params, init_obs, init_state, rng_key, step_offset):
    return _segment_core(_sac_apply, _squash_action, params,
                         init_obs, init_state, rng_key, step_offset)

@jax.jit
def segment_tqc(params, init_obs, init_state, rng_key, step_offset):
    return _segment_core(_tqc_apply, _squash_action, params,
                         init_obs, init_state, rng_key, step_offset)

_SEG_FN = {"PPO": segment_ppo, "SAC": segment_sac, "TQC": segment_tqc}


def _run_single_wp_cell(seg_fn, params, scen_id, v_max, cell_rng):
    """Single-waypoint scenario: one dispatch, no chaining."""
    reset_keys = jax.random.split(cell_rng, N_TEST_ENVS)
    obs, state = jax.vmap(
        dynamic_reset_stacked, in_axes=(0, None, None, None)
    )(reset_keys, 9.0, jnp.int32(scen_id), jnp.float32(v_max))
    # Returns future (not blocked)
    return seg_fn(params, obs, state, cell_rng, jnp.int32(0))


def _run_multi_wp_cell(seg_fn, params, scen_id, v_max, cell_rng, waypoints):
    """Multi-waypoint scenario: must chain waypoints sequentially."""
    n_wp = len(waypoints)
    reset_keys = jax.random.split(cell_rng, N_TEST_ENVS)
    obs, state = jax.vmap(
        dynamic_reset_stacked, in_axes=(0, None, None, None)
    )(reset_keys, 9.0, jnp.int32(scen_id), jnp.float32(v_max))

    still_alive  = np.ones(N_TEST_ENVS,  dtype=bool)
    overall_col  = np.zeros(N_TEST_ENVS, dtype=bool)
    overall_pcol = np.zeros(N_TEST_ENVS, dtype=bool)

    for wp_idx in range(n_wp):
        step_off = jnp.int32(wp_idx * MAX_STEPS)
        metrics, gr_state, gr_obs, gr_flag = jax.device_get(
            seg_fn(params, obs, state, cell_rng, step_off)
        )

        m_col  = metrics["act_col"].astype(bool)
        m_pcol = metrics["pass_col"].astype(bool)
        m_goal = metrics["goal_reached"]

        overall_col  |= m_col  & still_alive
        overall_pcol |= m_pcol & still_alive

        if wp_idx < n_wp - 1:
            next_gx, next_gy = waypoints[wp_idx + 1]
            new_env_state = gr_state.env_state.replace(
                goal_x=jnp.where(gr_flag, next_gx,
                                 gr_state.env_state.goal_x),
                goal_y=jnp.where(gr_flag, next_gy,
                                 gr_state.env_state.goal_y),
                max_v=jnp.full(N_TEST_ENVS, v_max),
            )
            state = gr_state.replace(env_state=new_env_state)
            obs   = gr_obs
            still_alive = still_alive & m_goal

    final_success = still_alive & metrics["goal_reached"]
    tmo = ~final_success & ~overall_col & ~overall_pcol
    return {
        "goal_reached": final_success,
        "act_col": overall_col,
        "pass_col": overall_pcol,
        "timeout": tmo,
        "spl": metrics["spl"],
        "min_dist": metrics["min_dist"],
    }


def run_test_scenarios(policies, rng):
    """
    Evaluates all policies on test scenarios 7-12.
    Single-waypoint scenarios are dispatched without blocking for pipelining.
    Multi-waypoint scenarios chain waypoints sequentially per cell.
    """
    all_frames  = []

    for p_name, params in policies.items():
        seg_fn   = _SEG_FN[p_name]
        t_policy = time.time()

        # Pre-split all RNG keys for this policy
        n_total = N_TEST_SCENS * N_SPEEDS
        rng, batch_rng = jax.random.split(rng)
        cell_keys = jax.random.split(batch_rng, n_total)
        key_idx = 0

        for scen_id in TEST_SCEN_IDS:
            waypoints = TEST_ROBOT_WAYPOINTS[scen_id]
            n_wp      = len(waypoints)
            t_scen    = time.time()

            if n_wp == 1:
                # Single-waypoint: dispatch all speeds, then collect
                futures = []
                for vi, v_max in enumerate(MAX_V_TESTS):
                    fut = _run_single_wp_cell(
                        seg_fn, params, scen_id, v_max, cell_keys[key_idx])
                    futures.append((v_max, fut))
                    key_idx += 1

                for v_max, fut in futures:
                    res = jax.device_get(fut)
                    m = res[0]  # metrics dict
                    all_frames.append(pd.DataFrame({
                        "Policy":        p_name,
                        "Scenario":      scen_id,
                        "Scenario_Name": TEST_SCENARIO_NAMES[scen_id],
                        "Max_V":         v_max,
                        "N_Waypoints":   n_wp,
                        "Success":       np.array(m["goal_reached"]).astype(float),
                        "Active Col":    np.array(m["act_col"]),
                        "Passive Col":   np.array(m["pass_col"]),
                        "Timeout":       np.array(m["timeout"]),
                        "SPL":           np.array(m["spl"]),
                        "Min Dist":      np.array(m["min_dist"]),
                    }))
            else:
                # Multi-waypoint: sequential per cell, but dispatch resets
                for vi, v_max in enumerate(MAX_V_TESTS):
                    m = _run_multi_wp_cell(
                        seg_fn, params, scen_id, v_max,
                        cell_keys[key_idx], waypoints)
                    key_idx += 1
                    all_frames.append(pd.DataFrame({
                        "Policy":        p_name,
                        "Scenario":      scen_id,
                        "Scenario_Name": TEST_SCENARIO_NAMES[scen_id],
                        "Max_V":         v_max,
                        "N_Waypoints":   n_wp,
                        "Success":       np.array(m["goal_reached"]).astype(float),
                        "Active Col":    np.array(m["act_col"]).astype(float),
                        "Passive Col":   np.array(m["pass_col"]).astype(float),
                        "Timeout":       np.array(m["timeout"]).astype(float),
                        "SPL":           np.array(m["spl"]),
                        "Min Dist":      np.array(m["min_dist"]),
                    }))

            suc_pct = np.mean(
                [f["Success"].mean() for f in all_frames[-N_SPEEDS:]]
            ) * 100
            print(f"    {p_name} | {TEST_SCENARIO_NAMES[scen_id]:<22s} "
                  f"suc={suc_pct:5.1f}%  "
                  f"{time.time() - t_scen:.1f}s")

        print(f"  {p_name} test total: {time.time() - t_policy:.1f}s\n")

    return pd.concat(all_frames, ignore_index=True)


def _plot_test_dashboard(test_df):
    """Compact dashboard for the 6 test scenarios."""
    sns.set_theme(style="whitegrid", palette="muted")

    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7
    FS_LEG   = 6
    LW = 2
    MS = 5

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Test Scenario Evaluation Dashboard (Scenarios 7-12)",
                 fontsize=12, weight="bold")

    # ── (0,0) Success rate per scenario ─────────────────────────────────────
    scen_suc = (test_df.groupby(["Scenario_Name", "Policy"])["Success"]
                .mean().reset_index())
    scen_suc["Success"] *= 100
    ax = axes[0, 0]
    sns.barplot(data=scen_suc, x="Scenario_Name", y="Success", hue="Policy", ax=ax)
    ax.set_title("Success Rate by Test Scenario", fontsize=FS_TITLE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS_TICK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # ── (0,1) Overall outcome breakdown ─────────────────────────────────────
    rate_df = (test_df.groupby("Policy")[
        ["Success", "Active Col", "Passive Col", "Timeout"]
    ].mean().reset_index())
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax = axes[0, 1]
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax)
    ax.set_title("Overall Outcomes", fontsize=FS_TITLE)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # ── (0,2) Success vs max speed ───────────────────────────────────────────
    spd_suc = (test_df.groupby(["Max_V", "Policy"])["Success"]
               .mean().reset_index())
    spd_suc["Success"] *= 100
    ax = axes[0, 2]
    sns.lineplot(data=spd_suc, x="Max_V", y="Success", hue="Policy",
                 marker="o", linewidth=LW, markersize=MS, ax=ax)
    ax.set_title("Success Rate vs. Max Speed", fontsize=FS_TITLE)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # ── (1,0) SPL (successful episodes only) ────────────────────────────────
    suc_only = test_df[test_df["Success"] == 1.0]
    ax = axes[1, 0]
    if not suc_only.empty:
        sns.boxplot(data=suc_only, x="Policy", y="SPL", hue="Policy",
                    ax=ax, showfliers=False)
    ax.set_title("SPL (successful eps)", fontsize=FS_TITLE)
    ax.set_ylabel("SPL", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)

    # ── (1,1) Safety margin vs speed ────────────────────────────────────────
    ax = axes[1, 1]
    sns.lineplot(data=test_df, x="Max_V", y="Min Dist", hue="Policy",
                 marker="^", linewidth=LW, markersize=MS, ax=ax)
    ax.set_title("Safety Margin vs. Max Speed", fontsize=FS_TITLE)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Collision")
    ax.set_xticks(MAX_V_TESTS)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax.set_ylabel("Min Human Distance (m)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # ── (1,2) Multi-waypoint full-path success ───────────────────────────────
    multi_wp = test_df[test_df["N_Waypoints"] > 1]
    ax = axes[1, 2]
    if not multi_wp.empty:
        mw_suc = (multi_wp.groupby(["Scenario_Name", "Policy"])["Success"]
                  .mean().reset_index())
        mw_suc["Success"] *= 100
        sns.barplot(data=mw_suc, x="Scenario_Name", y="Success", hue="Policy", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=FS_TICK)
    ax.set_title("Multi-Waypoint: Full-Path Success", fontsize=FS_TITLE)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Full-Path Success (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Test_Scenario_Dashboard.png", dpi=300)
    print("Saved 'Test_Scenario_Dashboard.png'")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]
    print(f"Running on: {gpu}\n")

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

    # ── Warm-up: compile all kernels ────────────────────────────────────────
    # Dispatch all warmup calls without blocking, then wait once at the end.
    print("\nCompiling evaluation kernels (all policies)...")
    t_compile = time.time()
    warmup_futures = {}
    for p_name, params in policies.items():
        rng, k_warmup = jax.random.split(rng)
        warmup_futures[p_name] = _EVAL_FN[p_name](
            params, jnp.int32(0), jnp.float32(1.0), k_warmup)

    rng, k_wu_reset = jax.random.split(rng)
    wu_reset_keys = jax.random.split(k_wu_reset, N_TEST_ENVS)
    wu_obs, wu_state = jax.vmap(
        dynamic_reset_stacked, in_axes=(0, None, None, None)
    )(wu_reset_keys, 9.0, jnp.int32(7), jnp.float32(1.0))

    seg_futures = {}
    for p_name, params in policies.items():
        rng, k_seg = jax.random.split(rng)
        seg_futures[p_name] = _SEG_FN[p_name](
            params, wu_obs, wu_state, k_seg, jnp.int32(0))

    # Single barrier for all compilations
    for p_name in policies:
        jax.block_until_ready(warmup_futures[p_name])
        jax.block_until_ready(seg_futures[p_name])
    t_compile = time.time() - t_compile
    print(f"  All kernels compiled in {t_compile:.1f}s\n")

    # ── Training-scenario evaluation (pipelined dispatch) ───────────────────
    # Pre-generate all RNG keys so dispatch is not blocked by Python key-splits.
    total_cells = N_SCENARIOS * N_SPEEDS
    all_frames   = []
    scatter_data = {}

    print(f"Executing evaluation grid ({N_SCENARIOS} scenarios x {N_SPEEDS} speeds "
          f"= {total_cells} cells, {N_ENVS} envs each)...")
    t_train_eval = time.time()

    for p_name, params in policies.items():
        eval_fn  = _EVAL_FN[p_name]
        t_policy = time.time()

        # Pre-split all RNG keys for this policy
        n_cells = N_SCENARIOS * N_SPEEDS
        rng, batch_rng = jax.random.split(rng)
        cell_keys = jax.random.split(batch_rng, n_cells)

        # Dispatch all cells without blocking — GPU stays saturated
        futures = []
        for si in range(N_SCENARIOS):
            for vi, v_max in enumerate(MAX_V_TESTS):
                idx = si * N_SPEEDS + vi
                fut = eval_fn(params, jnp.int32(si),
                              jnp.float32(v_max), cell_keys[idx])
                futures.append((si, v_max, fut))

        # Collect results (device_get implicitly waits)
        sd_list, sv_list = [], []
        for si, v_max, fut in futures:
            cell = jax.device_get(fut)

            sd = cell["step_dists"].ravel()
            sv = cell["step_vs"].ravel()
            ok = np.isfinite(sd) & np.isfinite(sv)
            sd_list.append(sd[ok])
            sv_list.append(sv[ok])

            all_frames.append(pd.DataFrame({
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
            }))

        scatter_data[p_name] = (np.concatenate(sd_list), np.concatenate(sv_list))
        suc_pct = np.mean([f["Success"].mean() for f in all_frames[-n_cells:]]) * 100
        print(f"  {p_name}: suc={suc_pct:5.1f}%  {time.time() - t_policy:.1f}s")

    t_train_eval = time.time() - t_train_eval

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv("evaluation_raw_data.csv", index=False)
    print("Saved evaluation_raw_data.csv\n")

    print("Generating training-scenario dashboard...")
    _plot_dashboard(df, scatter_data)

    # ── Test scenario evaluation ───────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Executing test scenario grid "
          f"({N_TEST_SCENS} scenarios x {N_SPEEDS} speeds = "
          f"{N_TEST_SCENS * N_SPEEDS} cells, {N_TEST_ENVS} envs each)...")
    print("NOTE: multi-waypoint scenarios (7, 9, 12) chain segments —")
    print("      success requires reaching the *last* waypoint.\n")

    t_test_eval = time.time()
    test_df = run_test_scenarios(policies, rng)
    t_test_eval = time.time() - t_test_eval

    test_df.to_csv("test_evaluation_raw_data.csv", index=False)
    print("Saved test_evaluation_raw_data.csv\n")

    print("Generating test-scenario dashboard...")
    _plot_test_dashboard(test_df)

    # ── Timing summary ─────────────────────────────────────────────────────
    t_total = time.time() - t_total
    print("\n" + "="*60)
    print("GPU BENCHMARK TIMING SUMMARY")
    print("="*60)
    print(f"  Compilation:           {t_compile:8.1f}s")
    print(f"  Training-scenario eval:{t_train_eval:8.1f}s")
    print(f"  Test-scenario eval:    {t_test_eval:8.1f}s")
    print(f"  Total (incl. plots):   {t_total:8.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()
