"""
benchmark_eval.py — High-Speed Evaluation Dashboard
===================================================
Evaluates RL models across the testing scenarios and generates a png 
dashboard with comprehensive metrics. Uses JAX for fast parallel rollouts on GPU.

"""

import os
import time
import warnings
from legnav import paths

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

import legnav.core.jax_env as jax_env
jax_env.USE_LEGS = True

from legnav.core.jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT, MAX_STEPS, get_obs
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import StackedEnvState, _ego_deltas, assemble_stacked_obs
from legnav.config import RobotConfig
from legnav.core.jax_network import SharedEncoder, EndToEndActorCritic, scale_action_to_env, USE_TANH_INSIDE

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_SIZE   = 668   # kin(5) + goal_stack(2*3) + ego_deltas(3*3) + lidar_stack(216*3)
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

# SAC: SharedEncoder + SACActorHead (named 'mean' Dense, STATE-DEPENDENT
# 'log_std' Dense). Must match SACjax.SACActorHead / jax_eval_multi._SACActorHead
# exactly — SAC's log_std is a Dense layer, not a global param vector, so the
# checkpoint stores log_std/{kernel,bias}.
class _SACActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5
    tanh_inside: bool  = False  # set at build time from jax_network.USE_TANH_INSIDE

    @nn.compact
    def __call__(self, feat):
        raw_mean = nn.Dense(self.action_dim, name="mean")(feat)
        if self.tanh_inside:
            v_mean = jnp.tanh(raw_mean[..., 0]) * 0.5 + 0.5
            w_mean = jnp.tanh(raw_mean[..., 1])
            actor_mean = jnp.stack([v_mean, w_mean], axis=-1)
        else:
            actor_mean = raw_mean
        logstd_pre = nn.Dense(self.action_dim, name="log_std")(feat)
        actor_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (jnp.tanh(logstd_pre) + 1.0)
        return actor_mean, actor_logstd

# TQC: SharedEncoder + TQCActorHead (unnamed Dense mean, 'log_std' param vector).
# Must match TQCjac.TQCActorHead / jax_eval_multi._TQCActorHead exactly.
class _TQCActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5
    tanh_inside: bool  = False  # set at build time from jax_network.USE_TANH_INSIDE

    @nn.compact
    def __call__(self, feat):
        raw_mean = nn.Dense(self.action_dim)(feat)
        if self.tanh_inside:
            v_mean = jnp.tanh(raw_mean[..., 0]) * 0.5 + 0.5
            w_mean = jnp.tanh(raw_mean[..., 1])
            actor_mean = jnp.stack([v_mean, w_mean], axis=-1)
        else:
            actor_mean = raw_mean
        logstd_param = self.param('log_std', nn.initializers.constant(1.0), (self.action_dim,))
        actor_logstd_raw = jnp.broadcast_to(logstd_param, actor_mean.shape)
        actor_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (jnp.tanh(actor_logstd_raw) + 1.0)
        return actor_mean.astype(jnp.float32), actor_logstd.astype(jnp.float32)

_shared_enc  = SharedEncoder()
_sac_head    = _SACActorHead(tanh_inside=USE_TANH_INSIDE)
_tqc_head    = _TQCActorHead(tanh_inside=USE_TANH_INSIDE)


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


# ── Action squashing ─────────────────────────────────────────────────────────
# Delegate to jax_network.scale_action_to_env so the mapping matches training /
# jax_eval_multi exactly (handles USE_TANH_INSIDE: heads emit already-squashed
# means, this only scales v by max_v and clips — no double tanh).
def _squash_action(mean, max_v):
    return scale_action_to_env(mean, max_v)


# ── Environment Wrappers ────────────────────────────────────────────────────
from legnav.core.jax_env import KIN_VEC_SIZE as _SVS

POSE_SIZE  = 2
STACK_DIM  = 3
# Full-resolution ring buffer length feeding the strided stack (o_t, o_{t-STRIDE}, ...).
STRIDE     = RobotConfig.LIDAR_STACK_STRIDE
BUFFER_LEN = (STACK_DIM - 1) * STRIDE + 1

@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    base_obs, base_state = reset_env(key, min_dist, scen_idx, 0.0)  # ghost_prob=0.0: pedestrians always avoid the robot (matches jax_eval_multi)
    goal      = base_obs[0:POSE_SIZE]
    kin_vec   = base_obs[POSE_SIZE : POSE_SIZE + _SVS]
    lidar     = base_obs[POSE_SIZE + _SVS:]

    base_state = base_state.replace(max_v=target_max_v)
    # kin_vec layout: [v, w, max_v_norm, goal_dist, goal_align]
    new_kin_vec = jnp.array([
        0.0, 0.0, (target_max_v - 0.2) / 1.8,
        kin_vec[3], kin_vec[4],
    ])

    lidar_stack = jnp.tile(lidar[None, :], (BUFFER_LEN, 1))
    goal_stack  = jnp.tile(goal[None, :],  (BUFFER_LEN, 1))
    pose        = jnp.array([base_state.x, base_state.y, base_state.theta])
    pose_stack  = jnp.tile(pose[None, :], (BUFFER_LEN, 1))
    stacked_state = StackedEnvState(
        env_state=base_state, lidar_stack=lidar_stack, goal_stack=goal_stack,
        pose_stack=pose_stack
    )
    flat_obs = assemble_stacked_obs(new_kin_vec, goal_stack, pose_stack, lidar_stack)
    return flat_obs, stacked_state


@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)
    new_goal      = base_obs[0:POSE_SIZE]
    new_kin_vec   = base_obs[POSE_SIZE : POSE_SIZE + _SVS]
    new_lidar     = base_obs[POSE_SIZE + _SVS:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_goal_stack  = jnp.concatenate([state.goal_stack[1:],  new_goal[None]],  axis=0)
    new_pose        = jnp.array([new_base_state.x, new_base_state.y, new_base_state.theta])
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:], new_pose[None]], axis=0)
    new_stacked_state = StackedEnvState(
        env_state=new_base_state, lidar_stack=new_lidar_stack, goal_stack=new_goal_stack,
        pose_stack=new_pose_stack
    )
    flat_obs = assemble_stacked_obs(new_kin_vec, new_goal_stack, new_pose_stack, new_lidar_stack)
    return flat_obs, new_stacked_state, reward, done, info


# ── Core Evaluation Kernel ───────────────────────────────────────────────────

YIELD_DIST = 1.5
YIELD_FOV  = 0.785  # rad = 45° → 90° total FOV (±45°), matches yielding reward in jax_env.py
SPACE_COMP_DIST = 0.5  # min surface distance (m) for a "space-compliant" timestep

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

    avg_jerk = jerks_w.sum(axis=0) / jnp.maximum(ep_lens, 1)   # angular jerk (rad/s^3)
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
        "step_ch":     step_dists,    # closest human (NaN-masked by active)
        "active_mask": active_mask,   # (MAX_STEPS, N_ENVS) bool
        "ep_lens":     ep_lens,       # (N_ENVS,)
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
    "PPO": paths.checkpoint("ppo", "ppo_legs_best.msgpack"),  # ppo_tanh_inside_final = old 662-obs arch
    "SAC": paths.checkpoint("sac", "sac_final.msgpack"),
    "TQC": paths.checkpoint("tqc", "tqc_final.msgpack"),
}

_LOADERS = {"PPO": load_ppo, "SAC": load_sac, "TQC": load_tqc}


def _check_arch(params):
    """Reject pre-ego-delta checkpoints (fused trunk was 206 = attn 192 + pose 9 + state 5)."""
    trunk = params.get("enc", params)
    in_dim = trunk["Dense_0"]["kernel"].shape[0]
    if in_dim != 203:
        raise ValueError(
            f"trunk input {in_dim} ≠ 203 — trained on the old 662-obs layout "
            "(no ego-motion deltas), incompatible with the new SE(2) network"
        )


# ── Dashboard plotting ──────────────────────────────────────────────────────

def _plot_dashboard(df, scatter_data):
    sns.set_theme(style="whitegrid", palette="muted")

    # Sized for full-page scientific figure (double-column); fonts sized for
    # readability when the PNG is embedded at typical paper widths.
    fig = plt.figure(figsize=(15.0, 10.5), constrained_layout=True)

    R, C = 3, 4
    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK  = 9
    FS_LEG   = 9
    LW = 1.6
    MS = 4.5

    # ── Row 1: outcome rates vs Max_V ────────────────────────────────────────
    _outcome_vs_speed(fig, R, C, 1, df, "SR",  "SR vs. Max Speed",  "o",
                      FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, LW, MS)
    _outcome_vs_speed(fig, R, C, 2, df, "ACR", "ACR vs. Max Speed", "X",
                      FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, LW, MS)
    _outcome_vs_speed(fig, R, C, 3, df, "PCR", "PCR vs. Max Speed", "o",
                      FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, LW, MS)
    _outcome_vs_speed(fig, R, C, 4, df, "TR",  "TR vs. Max Speed",  "s",
                      FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, LW, MS)

    # ── Row 2: quality metrics ───────────────────────────────────────────────
    suc = df[df["SR"] == 1.0]

    ax5 = plt.subplot(R, C, 5)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("SPL", fontsize=FS_TITLE)
    ax5.set_ylabel("SPL", fontsize=FS_LABEL)
    ax5.set_xlabel("")
    ax5.tick_params(labelsize=FS_TICK)

    ax6 = plt.subplot(R, C, 6)
    sns.boxplot(data=suc, x="Policy", y="TTG", hue="Policy",
                ax=ax6, showfliers=False)
    ax6.set_title("TTG", fontsize=FS_TITLE)
    ax6.set_ylabel("seconds", fontsize=FS_LABEL)
    ax6.set_xlabel("")
    ax6.tick_params(labelsize=FS_TICK)

    ax7 = plt.subplot(R, C, 7)
    sns.boxplot(data=df, x="Policy", y="MHD", hue="Policy", ax=ax7, showfliers=False)
    ax7.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax7.set_title("MHD", fontsize=FS_TITLE)
    ax7.set_ylabel("Min Human Distance (m)", fontsize=FS_LABEL)
    ax7.set_xlabel("")
    ax7.tick_params(labelsize=FS_TICK)

    ax8 = plt.subplot(R, C, 8)
    sns.boxplot(data=df, x="Policy", y="AJ", hue="Policy", ax=ax8, showfliers=False)
    ax8.set_title("AJ", fontsize=FS_TITLE)
    ax8.set_ylabel("Angular Jerk (rad/s^3)", fontsize=FS_LABEL)
    ax8.set_xlabel("")
    ax8.tick_params(labelsize=FS_TICK)

    # ── Row 3: SC, YS, scenario breakdown, overall ───────────────────────────
    sc_df = df.copy(); sc_df["SC"] = sc_df["SC"] * 100
    ax9 = plt.subplot(R, C, 9)
    sns.boxplot(data=sc_df, x="Policy", y="SC", hue="Policy", ax=ax9, showfliers=False)
    ax9.set_title("SC", fontsize=FS_TITLE)
    ax9.set_ylabel("Space Compliance (%)", fontsize=FS_LABEL)
    ax9.set_xlabel("")
    ax9.set_ylim(0, 100)
    ax9.tick_params(labelsize=FS_TICK)

    v_ys_df = df.groupby(["Max_V","Policy"])["YS"].mean().reset_index()
    v_ys_df["YS"] *= 100
    ax10 = plt.subplot(R, C, 10)
    sns.lineplot(data=v_ys_df, x="Max_V", y="YS", hue="Policy",
                 marker="D", linewidth=LW, markersize=MS, ax=ax10)
    ax10.set_title("YS vs. Max Speed", fontsize=FS_TITLE)
    ax10.set_xticks(MAX_V_TESTS)
    ax10.set_xticklabels([f"{v:.2f}" for v in MAX_V_TESTS], rotation=35, ha="right")
    ax10.set_ylim(0, 100)
    ax10.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax10.set_ylabel("Yield Compliance (%)", fontsize=FS_LABEL)
    ax10.tick_params(labelsize=FS_TICK)
    ax10.axhline(100.0, color="green", linestyle=":", alpha=0.4)
    ax10.axhline(50.0,  color="gray",  linestyle=":", alpha=0.4)
    ax10.legend(fontsize=FS_LEG)

    scen_df = df.groupby(["Scenario","Policy"])["SR"].mean().reset_index()
    scen_df["SR"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(TEST_SCENARIO_NAMES)
    ax11 = plt.subplot(R, C, 11)
    sns.barplot(data=scen_df, x="Scenario_Name", y="SR", hue="Policy", ax=ax11)
    ax11.set_title("SR by Scenario", fontsize=FS_TITLE)
    ax11.set_xticklabels(ax11.get_xticklabels(), rotation=35, ha="right", fontsize=FS_TICK)
    ax11.set_ylim(0, 100)
    ax11.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax11.set_xlabel("")
    ax11.tick_params(labelsize=FS_TICK)
    ax11.legend(fontsize=FS_LEG, loc="lower right")

    rate_df   = df.groupby("Policy")[["SR","ACR","PCR","TR"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax12 = plt.subplot(R, C, 12)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax12)
    ax12.set_title("Overall Outcomes", fontsize=FS_TITLE)
    ax12.set_ylim(0, 100)
    ax12.set_ylabel("Rate (%)", fontsize=FS_LABEL)
    ax12.set_xlabel("")
    ax12.tick_params(labelsize=FS_TICK)
    ax12.legend(fontsize=FS_LEG)

    plt.savefig(paths.figure("Evaluation_Dashboard.png"), dpi=300, bbox_inches="tight")
    print("Saved", paths.figure("Evaluation_Dashboard.png"))

    _plot_proximity_speed(scatter_data)


def _outcome_vs_speed(fig, R, C, pos, df, col, title, marker,
                      fs_title, fs_label, fs_tick, fs_leg, lw, ms):
    grp = df.groupby(["Max_V","Policy"])[col].mean().reset_index()
    grp[col] *= 100
    ax = plt.subplot(R, C, pos)
    sns.lineplot(data=grp, x="Max_V", y=col, hue="Policy",
                 marker=marker, linewidth=lw, markersize=ms, ax=ax)
    ax.set_title(title, fontsize=fs_title)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_xticklabels([f"{v:.2f}" for v in MAX_V_TESTS], rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=fs_label)
    ax.set_ylabel("Rate (%)", fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)
    ax.legend(fontsize=fs_leg)


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
from legnav.core.jax_scenarios import TEST_ROBOT_WAYPOINTS, TEST_SCENARIO_NAMES

N_TEST_ENVS   = 512
TEST_SCEN_IDS = sorted(TEST_SCENARIO_NAMES.keys())
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
        jnp.zeros(N_TEST_ENVS),           # jw_sum (accumulated angular jerk)
        jnp.ones(N_TEST_ENVS,  dtype=jnp.bool_),   # active
        jnp.zeros(N_TEST_ENVS, dtype=jnp.bool_),   # gr_flag (goal reached ever)
        init_state,                        # gr_state (pre-done state at first goal)
        init_obs,                          # gr_obs
        jnp.zeros(N_TEST_ENVS),           # yield_zone_steps
        jnp.zeros(N_TEST_ENVS),           # yield_comply_steps
    )

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p, jw_sum,
         active, gr_flag, gr_state, gr_obs, yz_steps, yc_steps) = carry

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
        jw = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)  # angular jerk (rad/s^3)
        new_jw_sum = jw_sum + jw

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

        # Yielding score (per-step accumulation)
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

        next_active = active & ~done
        step_data   = (active, g, c, pc, v, ch)
        return (
            next_state, next_obs, pl, mhd, v, w, av, aw, new_jw_sum,
            next_active, new_gr_flag, new_gr_state, new_gr_obs,
            new_yz_steps, new_yc_steps
        ), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    (_, _, final_pl, final_mhd, _, _, _, _, final_jw_sum,
     _, final_gr_flag, final_gr_state, final_gr_obs,
     final_yz, final_yc) = final_carry
    active_mask, goals, cols, pcols, step_vs, step_dists = step_data

    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)

    act_col  = ep_col  & ~ep_pcol & ~ep_goal
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    ep_lens = active_mask.sum(axis=0)
    spl = ep_goal * (init_dist / jnp.maximum(final_pl, init_dist))
    aj  = final_jw_sum / jnp.maximum(ep_lens, 1.0)          # angular jerk (rad/s^3)
    ttg = jnp.where(ep_goal, ep_lens * DT, jnp.nan)         # time-to-goal (s)

    # Space compliance: fraction of active timesteps where the robot keeps
    # a distance > 0.5 m from the closest human (surface-to-surface).
    compliant_steps  = jnp.where(active_mask & (step_dists > SPACE_COMP_DIST), 1.0, 0.0)
    n_active         = jnp.maximum(active_mask.sum(axis=0), 1.0)
    space_compliance = compliant_steps.sum(axis=0) / n_active

    metrics = {
        "goal_reached": ep_goal,
        "act_col":      act_col.astype(jnp.float32),
        "pass_col":     pass_col.astype(jnp.float32),
        "timeout":      tmo.astype(jnp.float32),
        "spl":          spl,
        "ttg":          ttg,
        "min_dist":     final_mhd,
        "aj":           aj,
        "yield_score":  jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan),
        "space_compliance": space_compliance,
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


@jax.jit
def _advance_waypoint(stacked_state, next_gx, next_gy, gr_flag, v_max, rng_key):
    """Batched WP hand-off mirroring test_scenarios_eval.py:advance_waypoint.

    Updates goal_x/goal_y in env_state (only where the previous goal was
    reached), resets time_step=0 so each segment gets a fresh MAX_STEPS budget,
    then recomputes obs via get_obs and refreshes the last frame of
    goal_stack / lidar_stack. Without this the next segment starts with stale
    goal-relative goal/kin_vec and an exhausted time budget, so multi-WP
    success rate is artificially low.
    """
    new_env_state = stacked_state.env_state.replace(
        goal_x=jnp.where(gr_flag, next_gx, stacked_state.env_state.goal_x),
        goal_y=jnp.where(gr_flag, next_gy, stacked_state.env_state.goal_y),
        max_v=jnp.full(N_TEST_ENVS, v_max),
        time_step=jnp.zeros(N_TEST_ENVS, dtype=jnp.int32),
    )
    obs_keys = jax.random.split(rng_key, N_TEST_ENVS)
    new_base_obs, sp_mask = jax.vmap(get_obs)(new_env_state, obs_keys)
    new_env_state = new_env_state.replace(sp_mask=sp_mask)

    new_goal      = new_base_obs[:, :POSE_SIZE]
    new_kin_vec   = new_base_obs[:, POSE_SIZE:POSE_SIZE + _SVS]
    new_lidar     = new_base_obs[:, POSE_SIZE + _SVS:]

    new_goal_stack  = stacked_state.goal_stack.at[:, -1, :].set(new_goal)
    new_lidar_stack = stacked_state.lidar_stack.at[:, -1, :].set(new_lidar)

    new_state = StackedEnvState(
        env_state=new_env_state,
        lidar_stack=new_lidar_stack,
        goal_stack=new_goal_stack,
        pose_stack=stacked_state.pose_stack,
    )
    new_obs = jax.vmap(assemble_stacked_obs)(
        new_kin_vec, new_goal_stack, stacked_state.pose_stack, new_lidar_stack
    )
    return new_obs, new_state


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
            # Scenario 9 ("Sequential Rooms") encodes per-env door positions
            # via sentinels: -1.0 → door1_y (obs_boxes[0,5]+1.0), -2.0 →
            # door2_y (obs_boxes[2,5]+1.0). Mirrors test_scenarios_eval.py.
            if scen_id == 9 and next_gy in (-1.0, -2.0):
                wall_idx = 0 if next_gy == -1.0 else 2
                next_gy = jnp.asarray(
                    gr_state.env_state.obs_boxes[:, wall_idx, 5] + 1.0,
                    dtype=jnp.float32,
                )
                next_gx = jnp.float32(next_gx)
            else:
                next_gx = jnp.float32(next_gx)
                next_gy = jnp.float32(next_gy)
            adv_rng = jax.random.fold_in(cell_rng, wp_idx + 1)
            obs, state = _advance_waypoint(
                gr_state, next_gx, next_gy,
                gr_flag, jnp.float32(v_max), adv_rng,
            )
            still_alive = still_alive & m_goal

    final_success = still_alive & metrics["goal_reached"]
    tmo = ~final_success & ~overall_col & ~overall_pcol
    return {
        "goal_reached": final_success,
        "act_col": overall_col,
        "pass_col": overall_pcol,
        "timeout": tmo,
        "spl": metrics["spl"],
        "ttg": metrics["ttg"],
        "min_dist": metrics["min_dist"],
        "aj": metrics["aj"],
        "yield_score": metrics["yield_score"],
        "space_compliance": metrics["space_compliance"],
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
                        "SR":            np.array(m["goal_reached"]).astype(float),
                        "ACR":           np.array(m["act_col"]),
                        "PCR":           np.array(m["pass_col"]),
                        "TR":            np.array(m["timeout"]),
                        "SPL":           np.array(m["spl"]),
                        "TTG":           np.array(m["ttg"]),
                        "MHD":           np.array(m["min_dist"]),
                        "AJ":            np.array(m["aj"]),
                        "YS":            np.array(m["yield_score"]),
                        "SC":            np.array(m["space_compliance"]),
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
                        "SR":            np.array(m["goal_reached"]).astype(float),
                        "ACR":           np.array(m["act_col"]).astype(float),
                        "PCR":           np.array(m["pass_col"]).astype(float),
                        "TR":            np.array(m["timeout"]).astype(float),
                        "SPL":           np.array(m["spl"]),
                        "TTG":           np.array(m["ttg"]),
                        "MHD":           np.array(m["min_dist"]),
                        "AJ":            np.array(m["aj"]),
                        "YS":            np.array(m["yield_score"]),
                        "SC":            np.array(m["space_compliance"]),
                    }))

            suc_pct = np.mean(
                [f["SR"].mean() for f in all_frames[-N_SPEEDS:]]
            ) * 100
            print(f"    {p_name} | {TEST_SCENARIO_NAMES[scen_id]:<22s} "
                  f"suc={suc_pct:5.1f}%  "
                  f"{time.time() - t_scen:.1f}s")

        print(f"  {p_name} test total: {time.time() - t_policy:.1f}s\n")

    return pd.concat(all_frames, ignore_index=True)


def _plot_test_dashboard(test_df):
    """Compact dashboard for the 6 test scenarios.

    NOTE on distance semantics:
    'Min Dist' = closest_shoe_surface from jax_env_multi.step_env, which is
    the surface-to-surface gap between the robot circle boundary (ROBOT_RADIUS)
    and the nearest point on the closest shoe AABB of any pedestrian
    (USE_LEGS=True), or the body-circle edge (USE_LEGS=False).
    This definition is identical to paper_comparison_eval.py and is used
    consistently in both the Safety Margin line-plot and the Min Human Distance
    box-plot below.
    All data come from the randomly-initialised test episodes produced by
    run_test_scenarios() (scenarios chosen at the start of each episode via
    dynamic_reset_stacked with random RNG keys).
    """
    sns.set_theme(style="whitegrid", palette="muted")

    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7
    FS_LEG   = 6
    LW = 2
    MS = 5

    # Expand to 2×4 to accommodate the new Min Human Distance box-plot.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # ── ROW 0: BAR PLOTS + SUCCESS LINE ─────────────────────────────────────
    
    # (0,0) Success rate per scenario (BAR)
    scen_suc = (test_df.groupby(["Scenario_Name", "Policy"])["SR"]
                .mean().reset_index())
    scen_suc["SR"] *= 100
    ax = axes[0, 0]
    sns.barplot(data=scen_suc, x="Scenario_Name", y="SR", hue="Policy", ax=ax)
    ax.set_title("Success Rate by Test Scenario", fontsize=FS_TITLE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS_TICK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # (0,1) Overall outcome breakdown (BAR)
    rate_df = (test_df.groupby("Policy")[
        ["SR", "ACR", "PCR", "TR"]
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

    # (0,2) Yielding score vs Max Speed (LINE)
    v_ys_df = test_df.groupby(["Max_V", "Policy"])["YS"].mean().reset_index()
    ax = axes[0, 2]
    sns.lineplot(data=v_ys_df, x="Max_V", y="YS", hue="Policy",
                 marker="D", linewidth=LW, markersize=MS, ax=ax)
    ax.set_title("Yielding Score vs. Max Speed", fontsize=FS_TITLE)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax.set_ylabel("Yield Compliance (0-1)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # (0,3) Success vs max speed (LINE)
    spd_suc = (test_df.groupby(["Max_V", "Policy"])["SR"]
               .mean().reset_index())
    spd_suc["SR"] *= 100
    ax = axes[0, 3]
    sns.lineplot(data=spd_suc, x="Max_V", y="SR", hue="Policy",
                 marker="o", linewidth=LW, markersize=MS, ax=ax)
    ax.set_title("Success Rate vs. Max Speed", fontsize=FS_TITLE)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max Linear Speed (m/s)", fontsize=FS_LABEL)
    ax.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # ── ROW 1: BOX PLOTS + SAFETY LINE ──────────────────────────────────────
    
    # (1,0) Min Human Distance box-plot (BOX)
    ax = axes[1, 0]
    palette = {p: POLICY_COLORS.get(p, "#888888") for p in test_df["Policy"].unique()}
    sns.boxplot(data=test_df, x="Policy", y="MHD", hue="Policy",
                palette=palette, showfliers=False, ax=ax)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Collision")
    ax.set_title("Min Human Distance (Global)", fontsize=FS_TITLE)
    ax.set_ylabel("Surface distance (m)", fontsize=FS_LABEL)
    ax.set_xlabel("")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    # (1,1) SPL (successful episodes only) (BOX)
    suc_only = test_df[test_df["SR"] == 1.0]
    ax = axes[1, 1]
    if not suc_only.empty:
        sns.boxplot(data=suc_only, x="Policy", y="SPL", hue="Policy", ax=ax, showfliers=False)
    ax.set_title("SPL (Successful Eps)", fontsize=FS_TITLE)
    ax.set_ylabel("SPL", fontsize=FS_LABEL)
    ax.set_xlabel("")
    ax.tick_params(labelsize=FS_TICK)

    # (1,2) Min Human Distance by Scenario (BOX)
    ax = axes[1, 2]
    sns.boxplot(data=test_df, x="Scenario_Name", y="MHD", hue="Policy",
                palette=palette, showfliers=False, ax=ax)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Min Human Distance by Scenario", fontsize=FS_TITLE)
    ax.set_ylabel("Surface distance (m)", fontsize=FS_LABEL)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK)
    ax.get_legend().remove()

    # (1,3) Safety Margin vs speed (LINE)
    ax = axes[1, 3]
    sns.lineplot(data=test_df, x="Max_V", y="MHD", hue="Policy",
                 marker="^", linewidth=LW, markersize=MS, ax=ax)
    ax.set_title("Safety Margin vs. Max Speed", fontsize=FS_TITLE)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(MAX_V_TESTS)
    ax.set_xlabel("Max Speed (m/s)", fontsize=FS_LABEL)
    ax.set_ylabel("Min Distance (m)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(paths.figure("Test_Scenario_Dashboard.png"), dpi=300)
    print("Saved", paths.figure("Test_Scenario_Dashboard.png"))
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
            _check_arch(params)
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

    # Training-scenario evaluation removed.
    # This script is intended to run *only* test-scenario evaluations and
    # gather test data. Skipping the training-scenario grid and its plots.
    print("Skipping training-scenario evaluation; proceeding to test scenarios only.\n")

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

    test_df.to_csv(paths.data("test_evaluation_raw_data.csv"), index=False)
    print("Saved test_evaluation_raw_data.csv\n")

    print("Generating evaluation dashboard...")
    _plot_dashboard(test_df, {})

    # ── Timing summary ─────────────────────────────────────────────────────
    t_total = time.time() - t_total
    print("\n" + "="*60)
    print("GPU BENCHMARK TIMING SUMMARY")
    print("="*60)
    print(f"  Compilation:           {t_compile:8.1f}s")
    print(f"  Test-scenario eval:    {t_test_eval:8.1f}s")
    print(f"  Total (incl. plots):   {t_total:8.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()