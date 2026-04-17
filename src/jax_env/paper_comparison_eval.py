"""
paper_comparison_eval.py — Comparison Table for Paper
=====================================================
Evaluates all navigation policies on the 6 test scenarios (7-12) with v_max
sampled uniformly in [0.2, 2.0] per episode (matching the training distribution).
Metrics are aggregated across all scenarios and speed conditions.

Policies:
  Model-Based : DWA, MPPI
  RL          : MLP, NavRep, PPO (circles), PPO (legs), SAC, TQC, TAGD

Metrics:
  Success (%)  |  Obst. Col. (%)  |  Act. Col. (%)  |  Pass. Col. (%)  |  Timeout (%)
  Yield Score  |  Jerk  |  Time to Goal (s)  |  Min Dist to Humans (m)

Usage:
  cd src/jax_env
  python3 paper_comparison_eval.py [--envs 512] [--seed 42]

Output:
  paper_comparison_table.csv          — one aggregated row per policy
  paper_comparison_table_per_scen.csv — per (policy, scenario) breakdown
"""

import argparse
import os
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import flax.serialization
import flax.linen as nn

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--envs",      type=int,  default=512,  help="Parallel envs per scenario (default 512)")
    p.add_argument("--envs-mppi", type=int,  default=128,  help="Parallel envs for MPPI (default 128)")
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--no-legs",   action="store_true",     help="Disable leg model for the environment")
    return p.parse_args()

args = _parse()

# ── Environment setup (must happen before any jax_env import) ─────────────────
import jax_env as _jax_env
_jax_env.USE_LEGS     = not args.no_legs
_jax_env.SENSOR_NOISE = True   # realistic eval

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, STATE_VEC_SIZE as _SVS)
from jax_legs import LEG_RADIUS
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState
from jax_network import (SharedEncoder, EndToEndActorCritic,
                         scale_action_to_env)

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_SIZE   = 662
ACTION_DIM = 2
POSE_SIZE  = 3
STACK_DIM  = 3

# 6 held-out test scenarios — same as test_scenarios_eval.py
TEST_SCENARIOS  = list(range(7, 13))   # [7, 8, 9, 10, 11, 12]

# v_max sampled uniformly per episode, matching the training distribution
V_MAX_MIN = 0.2
V_MAX_MAX = 2.0

N_ENVS      = args.envs
N_ENVS_MPPI = args.envs_mppi

YIELD_DIST = 1.5   # m — yield-zone radius (same as benchmark_eval.py)
YIELD_FOV  = 1.57  # rad ≈ 90°

HUMAN_R = LEG_RADIUS if _jax_env.USE_LEGS else PEOPLE_RADIUS

# ── Stacked-env helpers (same as benchmark_eval.py) ──────────────────────────

@jax.jit
def _reset_stacked(key, v_max, scenario_idx):
    """Reset one env for the given scenario with the given v_max."""
    base_obs, base_state = reset_env(key, 9.0, scenario_idx, 0.0)
    pose      = base_obs[:POSE_SIZE]
    state_vec = base_obs[POSE_SIZE: POSE_SIZE + _SVS]
    lidar     = base_obs[POSE_SIZE + _SVS:]

    base_state = base_state.replace(max_v=v_max)
    new_sv = jnp.array([0.0, 0.0, (v_max - 0.2) / 1.8, state_vec[3], state_vec[4]])
    lidar_stack = jnp.tile(lidar[None, :], (STACK_DIM, 1))
    pose_stack  = jnp.tile(pose[None, :],  (STACK_DIM, 1))
    stacked = StackedEnvState(env_state=base_state,
                              lidar_stack=lidar_stack, pose_stack=pose_stack)
    flat = jnp.concatenate([pose_stack.flatten(), new_sv, lidar_stack.flatten()])
    return flat, stacked


@jax.jit
def _step_stacked(key, state: StackedEnvState, action):
    base_obs, new_env, reward, done, info = step_env(key, state.env_state, action)
    new_pose  = base_obs[:POSE_SIZE]
    new_sv    = base_obs[POSE_SIZE: POSE_SIZE + _SVS]
    new_lidar = base_obs[POSE_SIZE + _SVS:]

    new_ls = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_ps = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)
    new_st = StackedEnvState(env_state=new_env, lidar_stack=new_ls, pose_stack=new_ps)
    flat   = jnp.concatenate([new_ps.flatten(), new_sv, new_ls.flatten()])
    return flat, new_st, reward, done, info


# ── Core rollout (stateless policies: RL + DWA) ───────────────────────────────

def _rollout_stateless(act_vmap, n_envs, rng_key, scenario_idx):
    """
    Run n_envs episodes in parallel using lax.scan.
    act_vmap: (obs_batch N×662) -> (actions_batch N×2)   [deterministic / RL]
    v_max is sampled uniformly in [V_MAX_MIN, V_MAX_MAX] per environment.
    Returns a metrics dict with arrays of shape (n_envs,).
    """
    rng_key, rng_v = jax.random.split(rng_key)
    v_max_batch = jax.random.uniform(rng_v, (n_envs,), minval=V_MAX_MIN, maxval=V_MAX_MAX)
    reset_keys  = jax.random.split(rng_key, n_envs)
    obs, state  = jax.vmap(_reset_stacked, in_axes=(0, 0, None))(
        reset_keys, v_max_batch, scenario_idx)

    init_dist = jnp.sqrt(
        (state.env_state.goal_x - state.env_state.x) ** 2 +
        (state.env_state.goal_y - state.env_state.y) ** 2
    )

    carry0 = (
        state, obs,
        jnp.zeros(n_envs),            # path_len
        jnp.full(n_envs, 100.0),      # min_human_dist (surface-to-surface)
        jnp.zeros(n_envs),            # v_prev
        jnp.zeros(n_envs),            # w_prev
        jnp.zeros(n_envs),            # av_prev
        jnp.zeros(n_envs),            # aw_prev
        jnp.ones(n_envs, dtype=jnp.bool_),   # active (still in first episode)
        jnp.zeros(n_envs),            # yield_zone_steps
        jnp.zeros(n_envs),            # yield_comply_steps
    )

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p,
         active, yz, yc) = carry

        k = jax.random.fold_in(rng_key, step_idx)
        actions = act_vmap(obs)

        step_keys = jax.random.split(k, n_envs)
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(
            step_keys, state, actions)

        v  = next_state.env_state.v
        w  = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT

        pl  = pl + jnp.where(active, v * DT, 0.0)
        ch  = info["closest_human"] - ROBOT_RADIUS - HUMAN_R
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        # Yield score (same logic as benchmark_eval.py)
        ppl     = next_state.env_state.people
        dp_x    = ppl[:, :, 0] - next_state.env_state.x[:, None]
        dp_y    = ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x ** 2 + dp_y ** 2 + 1e-8)
        rel_ang = jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None]
        rel_ang = (rel_ang + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        act_p   = ppl[:, :, 10] >= 0.0
        in_yz   = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & act_p
        any_yz  = jnp.any(in_yz, axis=1)
        stopped = v <= 0.1
        yz  = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc  = yc + jnp.where(active & any_yz & stopped, 1.0, 0.0)

        g   = info["goal_reached"] & active
        c   = info["collision"]    & active
        ac  = info["active_col"]   & active
        pc  = info["passive_col"]  & active

        # Jerk: |Δa_v / Δt| + |Δa_w / Δt|
        jv  = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0)
        jw  = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)

        step_data  = (active, g, c, ac, pc, jv + jw, v)
        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, v, w, av, aw,
                next_active, yz, yc), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))

    (_, _, final_pl, final_mhd, _, _, _, _,
     _, final_yz, final_yc) = final_carry
    active_mask, goals, cols, act_cols, pass_cols, jerks, step_vs = step_data

    ep_lens   = active_mask.sum(axis=0)
    ep_goal   = goals.any(axis=0)
    ep_col    = cols.any(axis=0)
    ep_actcol = act_cols.any(axis=0)
    ep_pscol  = pass_cols.any(axis=0)

    # obs_col = any collision that is neither active-human nor passive-human
    ep_obscol = ep_col & ~ep_actcol & ~ep_pscol
    ep_tmo    = ~ep_goal & ~ep_col   # ran out of MAX_STEPS

    avg_jerk   = jerks.sum(axis=0) / jnp.maximum(ep_lens, 1)
    time_goal  = jnp.where(ep_goal, ep_lens * DT, jnp.nan)
    yield_sc   = jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan)

    return {
        "success":    ep_goal.astype(jnp.float32),
        "obs_col":    ep_obscol.astype(jnp.float32),
        "act_col":    ep_actcol.astype(jnp.float32),
        "pass_col":   ep_pscol.astype(jnp.float32),
        "timeout":    ep_tmo.astype(jnp.float32),
        "jerk":       avg_jerk,
        "time_goal":  time_goal,
        "min_dist":   final_mhd,
        "yield_score":yield_sc,
        "spl":        ep_goal * (init_dist / jnp.maximum(final_pl, init_dist)),
    }


# ── MPPI rollout (carries u_mean per env) ─────────────────────────────────────

def _rollout_mppi(mppi, n_envs, rng_key, scenario_idx):
    """Like _rollout_stateless but carries u_mean_batch for warm-starting."""
    rng_key, rng_v = jax.random.split(rng_key)
    v_max_batch = jax.random.uniform(rng_v, (n_envs,), minval=V_MAX_MIN, maxval=V_MAX_MAX)
    reset_keys  = jax.random.split(rng_key, n_envs)
    obs, state  = jax.vmap(_reset_stacked, in_axes=(0, 0, None))(
        reset_keys, v_max_batch, scenario_idx)

    init_dist = jnp.sqrt(
        (state.env_state.goal_x - state.env_state.x) ** 2 +
        (state.env_state.goal_y - state.env_state.y) ** 2
    )

    u_mean_init = jnp.zeros((n_envs, mppi.horizon, 2))

    carry0 = (
        state, obs,
        jnp.zeros(n_envs),
        jnp.full(n_envs, 100.0),
        jnp.zeros(n_envs),
        jnp.zeros(n_envs),
        jnp.zeros(n_envs),
        jnp.zeros(n_envs),
        jnp.ones(n_envs, dtype=jnp.bool_),
        jnp.zeros(n_envs),
        jnp.zeros(n_envs),
        u_mean_init,
    )

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p,
         active, yz, yc, u_mean) = carry

        k = jax.random.fold_in(rng_key, step_idx)
        rng_batch = jax.random.split(k, n_envs)
        actions, new_u_mean = jax.vmap(mppi.act)(obs, u_mean, rng_batch)

        step_keys = jax.random.split(k, n_envs)
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(
            step_keys, state, actions)

        v  = next_state.env_state.v
        w  = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT

        pl  = pl + jnp.where(active, v * DT, 0.0)
        ch  = info["closest_human"] - ROBOT_RADIUS - HUMAN_R
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        ppl     = next_state.env_state.people
        dp_x    = ppl[:, :, 0] - next_state.env_state.x[:, None]
        dp_y    = ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x ** 2 + dp_y ** 2 + 1e-8)
        rel_ang = jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None]
        rel_ang = (rel_ang + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        act_p   = ppl[:, :, 10] >= 0.0
        in_yz   = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & act_p
        any_yz  = jnp.any(in_yz, axis=1)
        stopped = v <= 0.1
        yz  = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc  = yc + jnp.where(active & any_yz & stopped, 1.0, 0.0)

        g   = info["goal_reached"] & active
        c   = info["collision"]    & active
        ac  = info["active_col"]   & active
        pc  = info["passive_col"]  & active
        jv  = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0)
        jw  = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)

        # Reset u_mean to zeros on done (new episode starts fresh)
        new_u_mean = jnp.where(
            done[:, None, None], jnp.zeros_like(new_u_mean), new_u_mean)

        step_data   = (active, g, c, ac, pc, jv + jw, v)
        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, v, w, av, aw,
                next_active, yz, yc, new_u_mean), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))

    (_, _, final_pl, final_mhd, _, _, _, _,
     _, final_yz, final_yc, _) = final_carry
    active_mask, goals, cols, act_cols, pass_cols, jerks, _ = step_data

    ep_lens   = active_mask.sum(axis=0)
    ep_goal   = goals.any(axis=0)
    ep_col    = cols.any(axis=0)
    ep_actcol = act_cols.any(axis=0)
    ep_pscol  = pass_cols.any(axis=0)
    ep_obscol = ep_col & ~ep_actcol & ~ep_pscol
    ep_tmo    = ~ep_goal & ~ep_col

    avg_jerk   = jerks.sum(axis=0) / jnp.maximum(ep_lens, 1)
    time_goal  = jnp.where(ep_goal, ep_lens * DT, jnp.nan)
    yield_sc   = jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan)

    return {
        "success":    ep_goal.astype(jnp.float32),
        "obs_col":    ep_obscol.astype(jnp.float32),
        "act_col":    ep_actcol.astype(jnp.float32),
        "pass_col":   ep_pscol.astype(jnp.float32),
        "timeout":    ep_tmo.astype(jnp.float32),
        "jerk":       avg_jerk,
        "time_goal":  time_goal,
        "min_dist":   final_mhd,
        "yield_score":yield_sc,
        "spl":        ep_goal * (init_dist / jnp.maximum(final_pl, init_dist)),
    }


# ── Network / policy builders ──────────────────────────────────────────────────

_MAX_V_OBS_IDX = 11   # obs[11] = (max_v − 0.2) / 1.8

def _obs_to_max_v(obs):
    """Decode per-episode v_max from the stacked observation vector."""
    return jnp.clip(obs[..., _MAX_V_OBS_IDX] * 1.8 + 0.2, 0.2, 2.0)


def _load_raw(path):
    with open(path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def _build_ppo_act_vmap(ckpt_path):
    """Returns a vmapped act function for PPO / MLP / NavRep (same checkpoint format)."""
    from jax_network import EndToEndActorCritic, scale_action_to_env
    net = EndToEndActorCritic(action_dim=ACTION_DIM)
    bundle = _load_raw(ckpt_path)
    params = bundle.get("params", bundle)

    @jax.jit
    def act_vmap(obs_batch):   # (N, 662) → (N, 2)
        def _single(obs):
            mean, _, _ = net.apply({"params": params}, obs[None])
            return scale_action_to_env(jnp.squeeze(mean, 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)

    return act_vmap


def _build_mlp_act_vmap(ckpt_path):
    from comparison_policies.vanilla_mlp_network import VanillaMLPActorCritic
    from jax_network import scale_action_to_env
    net = VanillaMLPActorCritic(action_dim=ACTION_DIM, hidden_dim=128)
    bundle = _load_raw(ckpt_path)
    params = bundle.get("params", bundle)

    @jax.jit
    def act_vmap(obs_batch):
        def _single(obs):
            mean, _, _ = net.apply({"params": params}, obs[None])
            return scale_action_to_env(jnp.squeeze(mean, 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)

    return act_vmap


def _build_navrep_act_vmap(ckpt_path):
    from comparison_policies.navrep_network import NavRepActorCritic
    from jax_network import scale_action_to_env
    net = NavRepActorCritic(action_dim=ACTION_DIM)
    bundle = _load_raw(ckpt_path)
    params = bundle.get("params", bundle)

    @jax.jit
    def act_vmap(obs_batch):
        def _single(obs):
            mean, _, _ = net.apply({"params": params}, obs[None])
            return scale_action_to_env(jnp.squeeze(mean, 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)

    return act_vmap


def _build_sac_act_vmap(ckpt_path):
    enc  = SharedEncoder()
    head_cls = type("H", (nn.Module,), {
        "__annotations__": {},
        "__call__": lambda s, f: (nn.Dense(ACTION_DIM, name="mean")(f),
                                  nn.Dense(ACTION_DIM, name="log_std")(f))
    })

    class SACHead(nn.Module):
        @nn.compact
        def __call__(self, feat):
            mean    = nn.Dense(ACTION_DIM, name="mean")(feat)
            log_std = nn.Dense(ACTION_DIM, name="log_std")(feat)
            return mean, jnp.clip(log_std, -5.0, 2.0)

    head   = SACHead()
    bundle = _load_raw(ckpt_path)
    enc_p  = bundle["enc_params"]
    head_p = bundle["actor_head_params"]

    @jax.jit
    def act_vmap(obs_batch):
        def _single(obs):
            feat = enc.apply({"params": enc_p}, obs[None])
            mean, _ = head.apply({"params": head_p}, feat)
            mean = jnp.squeeze(mean, 0)
            v = (jnp.tanh(mean[0]) + 1.0) * 0.5 * _obs_to_max_v(obs)
            w = jnp.tanh(mean[1])
            return jnp.stack([v, w])
        return jax.vmap(_single)(obs_batch)

    return act_vmap


def _build_tqc_act_vmap(ckpt_path):
    class TQCHead(nn.Module):
        @nn.compact
        def __call__(self, feat):
            mean    = nn.Dense(ACTION_DIM)(feat)
            log_std = nn.Dense(ACTION_DIM)(feat)
            return mean.astype(jnp.float32), jnp.clip(log_std.astype(jnp.float32), -5.0, 0.5)

    enc    = SharedEncoder()
    head   = TQCHead()
    bundle = _load_raw(ckpt_path)
    enc_p  = bundle["enc_params"]
    head_p = bundle["actor_params"]

    @jax.jit
    def act_vmap(obs_batch):
        def _single(obs):
            feat = enc.apply({"params": enc_p}, obs[None])
            mean, _ = head.apply({"params": head_p}, feat)
            mean = jnp.squeeze(mean, 0)
            v = (jnp.tanh(mean[0]) + 1.0) * 0.5 * _obs_to_max_v(obs)
            w = jnp.tanh(mean[1])
            return jnp.stack([v, w])
        return jax.vmap(_single)(obs_batch)

    return act_vmap


def _build_dwa_act_vmap():
    from comparison_policies.dwa_planner import DWA
    dwa = DWA()
    act_vmap = jax.jit(jax.vmap(dwa.act))
    return act_vmap


def _build_tagd_act_vmap(ckpt_path):
    from comparison_policies.tagd_network import TAGDActor, make_tagd_act_fn
    bundle = _load_raw(ckpt_path)
    actor_params = bundle.get("actor_params", bundle)
    return make_tagd_act_fn(actor_params)  # TAGDActor reads v_max from obs[11] internally


# ── Policy registry ────────────────────────────────────────────────────────────
# Each entry: (display_name, type, ckpt_path_or_None, category_for_table)
POLICY_REGISTRY = [
    ("DWA",          "dwa",    None,
     "Model-Based"),
    ("MPPI",         "mppi",   None,
     "Model-Based"),
    ("MLP",          "mlp",    "checkpoints_vanilla_ppo/ppo_mlp_best.msgpack",
     "RL"),
    ("NavRep",       "navrep", "checkpoints_navrep/navrep_best.msgpack",
     "Unsup. Learning"),
    ("PPO (circles)","ppo",    "checkpoints/ppo_circles_best.msgpack",
     "End-to-end RL"),
    ("PPO (legs)",   "ppo",    "checkpoints/ppo_attn_best.msgpack",
     "End-to-end RL"),
    ("SAC",          "sac",    "checkpoints_sac/sac_best.msgpack",
     "End-to-end RL"),
    ("TQC",          "tqc",    "checkpoints_tqc/tqc_best.msgpack",
     "End-to-end RL"),
    ("TAGD",         "tagd",   "checkpoints_tagd/tagd_best.msgpack",
     "End-to-end RL"),
]


# ── Aggregate helper ───────────────────────────────────────────────────────────

def _agg(raw):
    """Compute mean from a raw metrics dict (arrays of shape N_ENVS)."""
    n = len(raw["success"])
    return {
        "N":           n,
        "Success":     float(np.mean(raw["success"])) * 100,
        "Obst. Col.":  float(np.mean(raw["obs_col"])) * 100,
        "Act. Col.":   float(np.mean(raw["act_col"])) * 100,
        "Pass. Col.":  float(np.mean(raw["pass_col"])) * 100,
        "Timeout":     float(np.mean(raw["timeout"])) * 100,
        "Yield Score": float(np.nanmean(raw["yield_score"])),
        "Jerk":        float(np.nanmean(raw["jerk"])),
        "Time to Goal":float(np.nanmean(raw["time_goal"])),
        "Min Dist (m)":float(np.mean(raw["min_dist"])),
    }


# ── LaTeX table printer ────────────────────────────────────────────────────────

def _print_latex(rows):
    header = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Comparison of Navigation Methods — Test Scenarios 7--12, "
        r"$v_{\max} \sim \mathcal{U}[0.2,\,2.0]$~m/s}" "\n"
        r"\label{tab:comparison}" "\n"
        r"\footnotesize" "\n"
        r"\begin{tabular}{lcccccccc}" "\n"
        r"\toprule" "\n"
        r"Method & Type & Success (\%) & Obst. Col. (\%) & Act. Col. (\%) "
        r"& Pass. Col. (\%) & Timeout (\%) & Yield Score & Jerk & Time (s) & Min Dist (m) \\" "\n"
        r"\midrule"
    )
    print(header)
    for r in rows:
        name = r["Method"].replace("(", r"\textit{(").replace(")", r")}")
        print(
            f"{name} & {r['Type']} & "
            f"{r['Success']:.1f} & {r['Obst. Col.']:.1f} & "
            f"{r['Act. Col.']:.1f} & {r['Pass. Col.']:.1f} & "
            f"{r['Timeout']:.1f} & "
            f"{r['Yield Score']:.2f} & {r['Jerk']:.1f} & "
            f"{r['Time to Goal']:.1f} & {r['Min Dist (m)']:.2f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


# ── Raw-metrics concatenation helper ──────────────────────────────────────────

def _concat_raw(raw_list):
    """Concatenate a list of per-scenario raw metric dicts along axis 0."""
    return {k: np.concatenate([np.array(r[k]) for r in raw_list]) for k in raw_list[0]}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = jax.random.PRNGKey(args.seed)
    rows      = []   # one row per policy (aggregated over all scenarios)
    rows_scen = []   # one row per (policy, scenario)

    n_scen = len(TEST_SCENARIOS)
    n_ep_total = N_ENVS * n_scen

    print(f"\n{'='*70}")
    print(f"Paper Comparison Eval — scenarios={TEST_SCENARIOS}")
    print(f"  v_max ~ U[{V_MAX_MIN}, {V_MAX_MAX}]  |  "
          f"envs/scenario={N_ENVS}  |  total_eps/policy={n_ep_total}")
    print(f"  USE_LEGS={_jax_env.USE_LEGS}")
    print(f"{'='*70}\n")

    for name, ptype, ckpt, category in POLICY_REGISTRY:
        # Skip if checkpoint missing
        if ckpt is not None and not os.path.exists(ckpt):
            print(f"  [{name}] checkpoint not found at {ckpt!r}, skipping.")
            continue

        t0 = time.time()

        # ── Build policy function ──────────────────────────────────────────
        try:
            if ptype == "dwa":
                act_fn = _build_dwa_act_vmap()
            elif ptype == "mppi":
                from comparison_policies.mppi_planner import MPPI
                _mppi = MPPI()
            elif ptype == "ppo":
                act_fn = _build_ppo_act_vmap(ckpt)
            elif ptype == "mlp":
                act_fn = _build_mlp_act_vmap(ckpt)
            elif ptype == "navrep":
                act_fn = _build_navrep_act_vmap(ckpt)
            elif ptype == "sac":
                act_fn = _build_sac_act_vmap(ckpt)
            elif ptype == "tqc":
                act_fn = _build_tqc_act_vmap(ckpt)
            elif ptype == "tagd":
                act_fn = _build_tagd_act_vmap(ckpt)
            else:
                raise ValueError(f"Unknown policy type: {ptype!r}")
        except Exception as e:
            print(f"\n  [{name}] BUILD ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        # ── Iterate over all 6 test scenarios ─────────────────────────────
        all_raw = []
        n_ep = N_ENVS_MPPI if ptype == "mppi" else N_ENVS
        first_scen = True

        for scen in TEST_SCENARIOS:
            scen_jax = jnp.int32(scen)
            rng, k = jax.random.split(rng)
            label = f"scen{scen}"
            if first_scen:
                print(f"  [{name}] compiling on scen {scen} ...", end="", flush=True)
            else:
                print(f"    scen {scen} ...", end="", flush=True)

            try:
                if ptype == "mppi":
                    raw_s = jax.device_get(jax.block_until_ready(
                        _rollout_mppi(_mppi, N_ENVS_MPPI, k, scen_jax)))
                else:
                    raw_s = jax.device_get(jax.block_until_ready(
                        _rollout_stateless(act_fn, N_ENVS, k, scen_jax)))
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback; traceback.print_exc()
                break

            scen_metrics = _agg(raw_s)
            rows_scen.append({"Method": name, "Type": category,
                              "Scenario": scen, **scen_metrics})
            all_raw.append(raw_s)

            tmo  = scen_metrics['Timeout']
            _sum = (scen_metrics['Success'] + scen_metrics['Obst. Col.'] +
                    scen_metrics['Act. Col.'] + scen_metrics['Pass. Col.'] + tmo)
            print(f" Suc={scen_metrics['Success']:.1f}% "
                  f"Col={scen_metrics['Obst. Col.']:.1f}+{scen_metrics['Act. Col.']:.1f}"
                  f"+{scen_metrics['Pass. Col.']:.1f} "
                  f"Tmo={tmo:.1f}% (Σ={_sum:.1f}%)")
            first_scen = False

        if not all_raw or len(all_raw) < n_scen:
            print(f"  [{name}] incomplete — skipping aggregation.")
            continue

        # ── Aggregate across all scenarios ─────────────────────────────────
        merged = _concat_raw(all_raw)
        metrics = _agg(merged)
        elapsed = time.time() - t0

        tmo  = metrics['Timeout']
        _sum = (metrics['Success'] + metrics['Obst. Col.'] +
                metrics['Act. Col.'] + metrics['Pass. Col.'] + tmo)
        print(f"  [{name}] TOTAL ({elapsed:.0f}s, N={n_ep * n_scen}): "
              f"Suc={metrics['Success']:.1f}%  "
              f"ObsCol={metrics['Obst. Col.']:.1f}%  "
              f"ActCol={metrics['Act. Col.']:.1f}%  "
              f"PasCol={metrics['Pass. Col.']:.1f}%  "
              f"Tmo={tmo:.1f}%  (Σ={_sum:.1f}%)  "
              f"Yield={metrics['Yield Score']:.2f}  "
              f"Jerk={metrics['Jerk']:.1f}  "
              f"T={metrics['Time to Goal']:.1f}s  "
              f"MinD={metrics['Min Dist (m)']:.2f}m\n")

        rows.append({"Method": name, "Type": category, **metrics})

    if not rows:
        print("No policies evaluated. Check checkpoint paths.")
        return

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df_scen = pd.DataFrame(rows_scen)
    df.to_csv("paper_comparison_table.csv", index=False, float_format="%.3f")
    df_scen.to_csv("paper_comparison_table_per_scen.csv", index=False, float_format="%.3f")
    print(f"Saved paper_comparison_table.csv  +  paper_comparison_table_per_scen.csv")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESULTS TABLE (aggregated over scenarios 7-12, random v_max)")
    print("="*70)
    cols_show = ["Method", "Type", "Success", "Obst. Col.", "Act. Col.",
                 "Pass. Col.", "Timeout", "Yield Score", "Jerk", "Time to Goal", "Min Dist (m)"]
    print(df[cols_show].to_string(index=False, float_format="%.2f"))

    print("\n" + "="*70)
    print("LaTeX")
    print("="*70)
    _print_latex(rows)


if __name__ == "__main__":
    main()
