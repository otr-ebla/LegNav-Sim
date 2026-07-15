"""
paper_comparison_eval.py — Comparison Table for Paper

Evaluates each policy on test scenarios 7-12 with multi-waypoint chaining:
multi-waypoint scenarios (7, 9, 11, 12) count as successful only if the robot
reaches the LAST waypoint. The metric definitions and waypoint handling mirror
benchmark_eval.py's dashboard so the table and the Evaluation_Dashboard figure
are directly comparable (the only intended difference is v_max sampling — this
script samples v_max ~ U[0.2, 2.0] per env, the dashboard sweeps a fixed grid).
"""

import argparse
import os
import time
import warnings
from functools import partial
from legnav import paths

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
warnings.filterwarnings("ignore")

import jax
jax.config.update("jax_default_device", jax.devices("cuda")[0])
import jax.numpy as jnp
import numpy as np
import pandas as pd
import flax.serialization
import flax.linen as nn

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--envs",      type=int,  default=512)
    p.add_argument("--envs-mppi", type=int,  default=128)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--no-legs",   action="store_true")
    return p.parse_args()

args = _parse()

import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS     = not args.no_legs
_jax_env.SENSOR_NOISE = True

from legnav.core.jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, KIN_VEC_SIZE as _SVS, get_obs)
from legnav.core.jax_legs import LEG_RADIUS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import StackedEnvState
from legnav.core.jax_network import SharedEncoder
from legnav.core.jax_scenarios import TEST_ROBOT_WAYPOINTS, TEST_SCENARIO_NAMES

OBS_SIZE   = 659
ACTION_DIM = 2
POSE_SIZE  = 2
STACK_DIM  = 3

TEST_SCENARIOS  = [7, 8, 9, 10, 11, 12]
V_MAX_MIN = 0.2
V_MAX_MAX = 2.0
N_ENVS      = args.envs
N_ENVS_MPPI = args.envs_mppi
YIELD_DIST = 1.5
YIELD_FOV  = 0.785  # ±45° (90° cone), matches yielding reward in jax_env.py
SPACE_COMP_DIST = 0.5  # min surface distance (m) for a "space-compliant" timestep

@jax.jit
def _reset_stacked(key, v_max, scenario_idx):
    base_obs, base_state = reset_env(key, 9.0, scenario_idx, 0.0)
    goal      = base_obs[:POSE_SIZE]
    kin_vec   = base_obs[POSE_SIZE: POSE_SIZE + _SVS]
    lidar     = base_obs[POSE_SIZE + _SVS:]
    base_state = base_state.replace(max_v=v_max)
    new_sv = jnp.array([0.0, 0.0, (v_max - 0.2) / 1.8, kin_vec[3], kin_vec[4]])
    lidar_stack = jnp.tile(lidar[None, :], (STACK_DIM, 1))
    goal_stack  = jnp.tile(goal[None, :],  (STACK_DIM, 1))
    stacked = StackedEnvState(env_state=base_state, lidar_stack=lidar_stack, goal_stack=goal_stack)
    flat = jnp.concatenate([goal_stack.flatten(), new_sv, lidar_stack.flatten()])
    return flat, stacked

@jax.jit
def _step_stacked(key, state: StackedEnvState, action):
    base_obs, new_env, reward, done, info = step_env(key, state.env_state, action)
    new_goal  = base_obs[:POSE_SIZE]
    new_sv    = base_obs[POSE_SIZE: POSE_SIZE + _SVS]
    new_lidar = base_obs[POSE_SIZE + _SVS:]
    new_ls = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_ps = jnp.concatenate([state.goal_stack[1:],  new_goal[None]],  axis=0)
    new_st = StackedEnvState(env_state=new_env, lidar_stack=new_ls, goal_stack=new_ps)
    flat   = jnp.concatenate([new_ps.flatten(), new_sv, new_ls.flatten()])
    return flat, new_st, reward, done, info

# ── Waypoint-chained rollout (mirrors benchmark_eval.py test machinery) ───────
# Each scenario is evaluated by running one MAX_STEPS *segment* per waypoint.
# A segment never resets; it saves the arrival state at the first goal step so
# the next segment can continue from there with the next goal. Multi-waypoint
# scenarios (7, 9, 11, 12) are therefore successful only if the robot reaches
# the LAST waypoint — identical definition to benchmark_eval.py's dashboard.


def _reset_scenario(rng_key, n_envs, scenario_idx):
    """Reset n_envs of a single scenario with per-env v_max ~ U[0.2, 2.0]."""
    rng_key, rng_v = jax.random.split(rng_key)
    v_max_batch = jax.random.uniform(rng_v, (n_envs,), minval=V_MAX_MIN, maxval=V_MAX_MAX)
    reset_keys  = jax.random.split(rng_key, n_envs)
    return jax.vmap(_reset_stacked, in_axes=(0, 0, None))(
        reset_keys, v_max_batch, jnp.int32(scenario_idx))


@partial(jax.jit, static_argnums=(0,))
def _advance_waypoint(n_envs, stacked_state, next_gx, next_gy, gr_flag, rng_key):
    """Set the next goal (only where the previous goal was reached), reset the
    time budget, recompute obs. Per-env max_v is preserved (unlike the dashboard,
    this eval uses a continuous per-env v_max)."""
    es = stacked_state.env_state.replace(
        goal_x=jnp.where(gr_flag, next_gx, stacked_state.env_state.goal_x),
        goal_y=jnp.where(gr_flag, next_gy, stacked_state.env_state.goal_y),
        time_step=jnp.zeros(n_envs, dtype=jnp.int32),
    )
    obs_keys = jax.random.split(rng_key, n_envs)
    new_base_obs, sp_mask = jax.vmap(get_obs)(es, obs_keys)
    es = es.replace(sp_mask=sp_mask)

    new_goal      = new_base_obs[:, :POSE_SIZE]
    new_kin_vec   = new_base_obs[:, POSE_SIZE:POSE_SIZE + _SVS]
    new_lidar     = new_base_obs[:, POSE_SIZE + _SVS:]
    new_goal_stack  = stacked_state.goal_stack.at[:, -1, :].set(new_goal)
    new_lidar_stack = stacked_state.lidar_stack.at[:, -1, :].set(new_lidar)
    new_state = StackedEnvState(env_state=es, lidar_stack=new_lidar_stack, goal_stack=new_goal_stack)
    new_obs = jnp.concatenate([
        new_goal_stack.reshape(n_envs, -1), new_kin_vec, new_lidar_stack.reshape(n_envs, -1)
    ], axis=1)
    return new_obs, new_state


def _seg_metrics(active_mask, goals, cols, pcols, step_ch,
                 final_pl, final_mhd, final_jw, final_yz, final_yc, init_dist):
    ep_lens = active_mask.sum(axis=0)
    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)
    act_col  = ep_col & ~ep_pcol & ~ep_goal          # active-human OR static-obstacle
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    compliant = jnp.where(active_mask & (step_ch > SPACE_COMP_DIST), 1.0, 0.0)
    sc = compliant.sum(axis=0) / jnp.maximum(active_mask.sum(axis=0), 1.0)
    return {
        "goal_reached": ep_goal,
        "act_col":  act_col.astype(jnp.float32),
        "pass_col": pass_col.astype(jnp.float32),
        "timeout":  tmo.astype(jnp.float32),
        "spl":      ep_goal * (init_dist / jnp.maximum(final_pl, init_dist)),
        "ttg":      jnp.where(ep_goal, ep_lens * DT, jnp.nan),
        "min_dist": final_mhd,
        "aj":       final_jw / jnp.maximum(ep_lens, 1.0),
        "yield_score": jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan),
        "space_compliance": sc,
    }


@partial(jax.jit, static_argnums=(0, 1))
def _segment_stateless(act_vmap, n_envs, init_obs, init_state, rng_key, step_offset):
    init_dist = jnp.sqrt((init_state.env_state.goal_x - init_state.env_state.x)**2 +
                         (init_state.env_state.goal_y - init_state.env_state.y)**2)
    carry0 = (init_state, init_obs,
              jnp.zeros(n_envs), jnp.full(n_envs, 100.0),       # pl, mhd
              jnp.zeros(n_envs), jnp.zeros(n_envs),             # w_p, aw_p
              jnp.ones(n_envs, dtype=jnp.bool_),                # active
              jnp.zeros(n_envs, dtype=jnp.bool_),               # gr_flag
              init_state, init_obs,                             # gr_state, gr_obs
              jnp.zeros(n_envs), jnp.zeros(n_envs))             # yz, yc

    def _step(carry, step_idx):
        (state, obs, pl, mhd, w_p, aw_p, active,
         gr_flag, gr_state, gr_obs, yz, yc) = carry
        k = jax.random.fold_in(rng_key, step_offset + step_idx)
        actions = act_vmap(obs)
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(
            jax.random.split(k, n_envs), state, actions)

        v, w = next_state.env_state.v, next_state.env_state.w
        aw = (w - w_p) / DT
        jw = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)  # angular jerk (rad/s^3)
        pl = pl + jnp.where(active, v * DT, 0.0)

        if _jax_env.USE_LEGS:
            ch = info["closest_shoe_surface"]
        else:
            ch = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"] & active
        c  = info["collision"]    & active
        pc = info["passive_col"]  & active

        first_goal = g & ~gr_flag
        def _sel(new_a, old_a):
            if new_a.ndim == 1:
                return jnp.where(first_goal, new_a, old_a)
            return jnp.where(first_goal.reshape([-1] + [1] * (new_a.ndim - 1)), new_a, old_a)
        new_gr_state = jax.tree_util.tree_map(_sel, state, gr_state)
        new_gr_obs   = _sel(obs, gr_obs)
        new_gr_flag  = gr_flag | g

        ppl = next_state.env_state.people
        dp_x, dp_y = ppl[:, :, 0] - next_state.env_state.x[:, None], ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang = (jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None] + jnp.pi) % (2*jnp.pi) - jnp.pi
        in_yz = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & (ppl[:, :, 10] >= 0.0)
        any_yz = jnp.any(in_yz, axis=1)
        yz = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc = yc + jnp.where(active & any_yz & (v <= 0.1), 1.0, 0.0)

        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, w, aw, next_active,
                new_gr_flag, new_gr_state, new_gr_obs, yz, yc), (active, g, c, pc, ch, jw)

    final_carry, step_data = jax.lax.scan(_step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))
    (_, _, final_pl, final_mhd, _, _, _,
     final_grflag, final_grstate, final_grobs, final_yz, final_yc) = final_carry
    active_mask, goals, cols, pcols, step_ch, jerks = step_data
    final_jw = jerks.sum(axis=0)

    metrics = _seg_metrics(active_mask, goals, cols, pcols, step_ch,
                           final_pl, final_mhd, final_jw, final_yz, final_yc, init_dist)
    return metrics, final_grstate, final_grobs, final_grflag


@partial(jax.jit, static_argnums=(0, 1))
def _segment_mppi(mppi, n_envs, init_obs, init_state, rng_key, step_offset):
    init_dist = jnp.sqrt((init_state.env_state.goal_x - init_state.env_state.x)**2 +
                         (init_state.env_state.goal_y - init_state.env_state.y)**2)
    u_mean_init = jnp.broadcast_to(mppi.init_u_mean(), (n_envs, mppi.horizon, 2))
    carry0 = (init_state, init_obs,
              jnp.zeros(n_envs), jnp.full(n_envs, 100.0),
              jnp.zeros(n_envs), jnp.zeros(n_envs),
              jnp.ones(n_envs, dtype=jnp.bool_),
              jnp.zeros(n_envs, dtype=jnp.bool_),
              init_state, init_obs,
              jnp.zeros(n_envs), jnp.zeros(n_envs),
              u_mean_init)

    def _step(carry, step_idx):
        (state, obs, pl, mhd, w_p, aw_p, active,
         gr_flag, gr_state, gr_obs, yz, yc, u_mean) = carry
        k = jax.random.fold_in(rng_key, step_offset + step_idx)
        actions, new_u_mean = jax.vmap(mppi.act)(obs, u_mean, jax.random.split(k, n_envs))
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(
            jax.random.split(k, n_envs), state, actions)

        v, w = next_state.env_state.v, next_state.env_state.w
        aw = (w - w_p) / DT
        jw = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)  # angular jerk (rad/s^3)
        pl = pl + jnp.where(active, v * DT, 0.0)

        if _jax_env.USE_LEGS:
            ch = info["closest_shoe_surface"]
        else:
            ch = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"] & active
        c  = info["collision"]    & active
        pc = info["passive_col"]  & active

        first_goal = g & ~gr_flag
        def _sel(new_a, old_a):
            if new_a.ndim == 1:
                return jnp.where(first_goal, new_a, old_a)
            return jnp.where(first_goal.reshape([-1] + [1] * (new_a.ndim - 1)), new_a, old_a)
        new_gr_state = jax.tree_util.tree_map(_sel, state, gr_state)
        new_gr_obs   = _sel(obs, gr_obs)
        new_gr_flag  = gr_flag | g

        ppl = next_state.env_state.people
        dp_x, dp_y = ppl[:, :, 0] - next_state.env_state.x[:, None], ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang = (jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None] + jnp.pi) % (2*jnp.pi) - jnp.pi
        in_yz = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & (ppl[:, :, 10] >= 0.0)
        any_yz = jnp.any(in_yz, axis=1)
        yz = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc = yc + jnp.where(active & any_yz & (v <= 0.1), 1.0, 0.0)

        new_u_mean = jnp.where(done[:, None, None],
                               jnp.broadcast_to(mppi.init_u_mean(), new_u_mean.shape), new_u_mean)
        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, w, aw, next_active,
                new_gr_flag, new_gr_state, new_gr_obs, yz, yc, new_u_mean), \
               (active, g, c, pc, ch, jw)

    final_carry, step_data = jax.lax.scan(_step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))
    (_, _, final_pl, final_mhd, _, _, _,
     final_grflag, final_grstate, final_grobs, final_yz, final_yc, _) = final_carry
    active_mask, goals, cols, pcols, step_ch, jerks = step_data
    final_jw = jerks.sum(axis=0)

    metrics = _seg_metrics(active_mask, goals, cols, pcols, step_ch,
                           final_pl, final_mhd, final_jw, final_yz, final_yc, init_dist)
    return metrics, final_grstate, final_grobs, final_grflag

# ── Python driver: chain waypoint segments for one scenario ───────────────────
# Single-waypoint scenarios (8, 10) run one segment; multi-waypoint scenarios
# (7, 9, 11, 12) chain segments, advancing the goal only for envs that reached
# the previous waypoint. Success requires reaching the LAST waypoint. Collisions
# accumulate across segments (while the env is still alive); secondary metrics
# (SPL/TTG/MHD/AJ/SC/YS) come from the final segment — identical to the dashboard
# in benchmark_eval.py so the table and figure stay directly comparable.

def _run_scenario(seg_call, n_envs, scenario, waypoints, rng_key):
    n_wp = len(waypoints)
    obs, state = _reset_scenario(rng_key, n_envs, scenario)

    still_alive  = np.ones(n_envs,  dtype=bool)
    overall_col  = np.zeros(n_envs, dtype=bool)
    overall_pcol = np.zeros(n_envs, dtype=bool)
    metrics = None

    for wp_idx in range(n_wp):
        step_off = jnp.int32(wp_idx * MAX_STEPS)
        metrics, gr_state, gr_obs, gr_flag = jax.device_get(
            seg_call(obs, state, rng_key, step_off))

        overall_col  |= np.asarray(metrics["act_col"]).astype(bool)  & still_alive
        overall_pcol |= np.asarray(metrics["pass_col"]).astype(bool) & still_alive

        if wp_idx < n_wp - 1:
            next_gx, next_gy = waypoints[wp_idx + 1]
            # Scenario 9 ("Sequential Rooms") encodes per-env door y-positions via
            # sentinels: -1.0 -> door1_y, -2.0 -> door2_y (obs_boxes[w, 5] + 1.0).
            if scenario == 9 and next_gy in (-1.0, -2.0):
                wall_idx = 0 if next_gy == -1.0 else 2
                next_gy = jnp.asarray(gr_state.env_state.obs_boxes[:, wall_idx, 5] + 1.0,
                                      dtype=jnp.float32)
                next_gx = jnp.float32(next_gx)
            else:
                next_gx = jnp.float32(next_gx)
                next_gy = jnp.float32(next_gy)
            adv_rng = jax.random.fold_in(rng_key, wp_idx + 1)
            obs, state = _advance_waypoint(n_envs, gr_state, next_gx, next_gy, gr_flag, adv_rng)
            still_alive = still_alive & np.asarray(metrics["goal_reached"]).astype(bool)

    final_success = still_alive & np.asarray(metrics["goal_reached"]).astype(bool)
    tmo = ~final_success & ~overall_col & ~overall_pcol
    return {
        "success":          final_success.astype(np.float32),
        "act_col":          overall_col.astype(np.float32),
        "pass_col":         overall_pcol.astype(np.float32),
        "timeout":          tmo.astype(np.float32),
        "spl":              np.asarray(metrics["spl"]),
        "time_goal":        np.where(final_success, np.asarray(metrics["ttg"]), np.nan),
        "min_dist":         np.asarray(metrics["min_dist"]),
        "jerk":             np.asarray(metrics["aj"]),
        "yield_score":      np.asarray(metrics["yield_score"]),
        "space_compliance": np.asarray(metrics["space_compliance"]),
    }

def _obs_to_max_v(obs): return jnp.clip(obs[..., 11] * 1.8 + 0.2, 0.2, 2.0)
def _load_raw(path):
    with open(path, "rb") as f: return flax.serialization.msgpack_restore(f.read())

def _build_ppo_act_vmap(ckpt_path):
    from legnav.core.jax_network import EndToEndActorCritic, scale_action_to_env
    net, bundle = EndToEndActorCritic(action_dim=ACTION_DIM), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_mlp_act_vmap(ckpt_path):
    from legnav.baselines.vanilla_mlp_network import VanillaMLPActorCritic
    from legnav.core.jax_network import scale_action_to_env
    net, bundle = VanillaMLPActorCritic(action_dim=ACTION_DIM, hidden_dim=128), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_navrep_act_vmap(ckpt_path):
    from legnav.baselines.navrep_network import NavRepActorCritic
    from legnav.core.jax_network import scale_action_to_env
    net, bundle = NavRepActorCritic(action_dim=ACTION_DIM), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

# SAC/TQC actor heads must match SACjax/TQCjac/jax_eval_multi exactly: a 'mean'
# Dense (named for SAC, unnamed for TQC) and tanh-inside squashing
# (USE_TANH_INSIDE). log_std differs by algo: SAC uses a state-dependent
# 'log_std' Dense, TQC a global 'log_std' param vector — the eval only needs the
# mean, but the log_std structure must match so the checkpoint restores.
# Squashing goes through scale_action_to_env, identical to training.
def _build_sac_act_vmap(ckpt_path):
    from legnav.core.jax_network import USE_TANH_INSIDE, scale_action_to_env
    enc, bundle = SharedEncoder(), _load_raw(ckpt_path)
    class SACHead(nn.Module):
        tanh_inside: bool = USE_TANH_INSIDE
        @nn.compact
        def __call__(self, feat):
            raw_mean = nn.Dense(ACTION_DIM, name="mean")(feat)
            if self.tanh_inside:
                raw_mean = jnp.stack([jnp.tanh(raw_mean[..., 0]) * 0.5 + 0.5,
                                      jnp.tanh(raw_mean[..., 1])], axis=-1)
            # SAC's log_std is a STATE-DEPENDENT Dense (see SACjax.SACActorHead),
            # so the checkpoint stores log_std/{kernel,bias}. Declare the layer so
            # the param tree matches on restore (its output is unused at eval).
            nn.Dense(ACTION_DIM, name="log_std")(feat)
            return raw_mean
    head, enc_p, head_p = SACHead(), bundle["enc_params"], bundle["actor_head_params"]
    def act_vmap(obs_batch):
        def _single(obs):
            mean = jnp.squeeze(head.apply({"params": head_p}, enc.apply({"params": enc_p}, obs[None])), 0)
            return scale_action_to_env(mean, _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_tqc_act_vmap(ckpt_path):
    from legnav.core.jax_network import USE_TANH_INSIDE, scale_action_to_env
    enc, bundle = SharedEncoder(), _load_raw(ckpt_path)
    class TQCHead(nn.Module):
        tanh_inside: bool = USE_TANH_INSIDE
        @nn.compact
        def __call__(self, feat):
            raw_mean = nn.Dense(ACTION_DIM)(feat)
            if self.tanh_inside:
                raw_mean = jnp.stack([jnp.tanh(raw_mean[..., 0]) * 0.5 + 0.5,
                                      jnp.tanh(raw_mean[..., 1])], axis=-1)
            self.param('log_std', nn.initializers.constant(1.0), (ACTION_DIM,))
            return raw_mean.astype(jnp.float32)
    head, enc_p, head_p = TQCHead(), bundle["enc_params"], bundle["actor_params"]
    def act_vmap(obs_batch):
        def _single(obs):
            mean = jnp.squeeze(head.apply({"params": head_p}, enc.apply({"params": enc_p}, obs[None])), 0)
            return scale_action_to_env(mean, _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_dwa_act_vmap():
    from legnav.baselines.dwa_planner import DWA
    dwa = DWA()
    return jax.vmap(dwa.act)

def _build_tagd_act_vmap(ckpt_path):
    from legnav.baselines.tagd_network import make_tagd_act_fn
    return make_tagd_act_fn(_load_raw(ckpt_path).get("actor_params", _load_raw(ckpt_path)))

POLICY_REGISTRY = [
    ("DWA",          "dwa",    None,                                           "Model-Based"),
    ("MPPI",         "mppi",   None,                                           "Model-Based"),
    ("NavRep",       "navrep", paths.checkpoint("navrep", "navrep_best.msgpack"),       "Unsup. Learning"),
    ("TAGD",         "tagd",   paths.checkpoint("tagd", "tagd_best.msgpack"),           "E2E RL"),
    ("VanillaE2E",   "mlp",    paths.checkpoint("vanilla_ppo", "ppo_mlp_best.msgpack"), "E2E RL"),
    ("PPO (circles)","ppo",    paths.checkpoint("ppo", "ppo_circles_best.msgpack"),         "E2E RL"),
    ("PPO (legs)",   "ppo",    paths.checkpoint("ppo", "ppo_tanh_inside_final.msgpack"),    "E2E RL"),
    ("SAC",          "sac",    paths.checkpoint("sac", "sac_final.msgpack"),             "E2E RL"),
    ("TQC",          "tqc",    paths.checkpoint("tqc", "tqc_final.msgpack"),             "E2E RL"),
]

def _agg(r):
    # Per-env scalar metrics from the waypoint-chained driver. ACR already lumps
    # active-human and static-obstacle collisions. SPL/TTG are over successful
    # episodes only (matches the dashboard's successful-only box-plots); MHD/AJ/
    # SC/YS are over all episodes.
    succ = r["success"] > 0.5
    spl_succ = np.asarray(r["spl"])[succ]
    return {"N": len(r["success"]),
            "SR":  float(np.mean(r["success"])) * 100,
            "ACR": float(np.mean(r["act_col"])) * 100,
            "PCR": float(np.mean(r["pass_col"])) * 100,
            "TR":  float(np.mean(r["timeout"])) * 100,
            "SPL": float(np.mean(spl_succ)) if spl_succ.size else float("nan"),
            "TTG": float(np.nanmean(r["time_goal"])) if succ.any() else float("nan"),
            "MHD": float(np.mean(r["min_dist"])),
            "AJ":  float(np.nanmean(r["jerk"])),       # angular jerk (rad/s^3)
            "SC":  float(np.mean(r["space_compliance"])) * 100,
            "YS":  float(np.nanmean(r["yield_score"])) * 100}

_TABLE_COLS = ["Method", "Type", "SR", "ACR", "PCR", "TR", "SPL", "TTG", "MHD", "AJ", "SC", "YS"]

_CITES = {"DWA": r"\cite{fox1997dynamic}", "MPPI": r"\cite{williams2017model}",
          "NavRep": r"\cite{dugas2021navrep}", "TAGD": r"\cite{de2024spatiotemporal}"}

def _latex_name(method):
    for algo in ("PPO", "SAC", "TQC"):
        if method == algo or method.startswith(algo + " "):
            method = method.replace(algo, f"CALF$_{{{algo}}}$", 1)
            break
    return method.replace('(', r'\textit{(').replace(')', r')}')

def _print_latex(rows):
    print(r"\begin{table*}[t]"+"\n"+r"\centering"
          +"\n"+rf"\caption{{Comparison of Navigation Methods — Test Scenarios {TEST_SCENARIOS[0]}--{TEST_SCENARIOS[-1]}, $v_{{\max}} \sim \mathcal{{U}}[{V_MAX_MIN},\,{V_MAX_MAX}]$~m/s. The symbols $\uparrow$ and $\downarrow$ indicate metrics respectively to be maximized and minimized.}}"
          +"\n"+r"\label{tab:comparison}"+"\n"+r"\footnotesize"
          +"\n"+r"\resizebox{\textwidth}{!}{"
          +"\n"+r"\begin{tabular}{lccccccccccc}"+"\n"+r"\toprule"
          +"\n"+r"Method & Type & SR (\%) $\uparrow$ & ACR (\%) $\downarrow$ & PCR (\%) $\downarrow$ & TR (\%) $\downarrow$ & SPL $\uparrow$ & TTG (s) $\downarrow$ & MHD (m) $\uparrow$ & AJ (rad/s$^3$) $\downarrow$ & SC (\%) $\uparrow$ & YS (\%) $\uparrow$ \\"
          +"\n"+r"\midrule")
    for r in rows:
        name = _latex_name(r['Method']) + _CITES.get(r['Method'], "")
        print(f"{name} & {r['Type']} & {r['SR']:.1f} & {r['ACR']:.1f} & {r['PCR']:.1f} & {r['TR']:.1f} & {r['SPL']:.2f} & {r['TTG']:.1f} & {r['MHD']:.2f} & {r['AJ']:.1f} & {r['SC']:.1f} & {r['YS']:.1f} \\\\")
    print(r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"+r"}"+"\n"+r"\end{table*}")

def main():
    rng = jax.random.PRNGKey(args.seed)
    rows, rows_scen = [], []

    print(f"\n{'='*70}\nWaypoint-Chained Eval — scenarios={TEST_SCENARIOS} | "
          f"envs/scen={N_ENVS} | v_max~U[{V_MAX_MIN},{V_MAX_MAX}]\n{'='*70}\n")

    for name, ptype, ckpt, cat in POLICY_REGISTRY:
        if ckpt and not os.path.exists(ckpt): continue
        t0 = time.time()
        print(f"  [{name}] Running {len(TEST_SCENARIOS)} scenarios (waypoint-chained)...")

        try:
            _mppi = None
            if ptype == "dwa": act_fn = _build_dwa_act_vmap()
            elif ptype == "mppi":
                from legnav.baselines.mppi_planner import MPPI
                _mppi = MPPI()
            elif ptype == "ppo": act_fn = _build_ppo_act_vmap(ckpt)
            elif ptype == "mlp": act_fn = _build_mlp_act_vmap(ckpt)
            elif ptype == "navrep": act_fn = _build_navrep_act_vmap(ckpt)
            elif ptype == "sac": act_fn = _build_sac_act_vmap(ckpt)
            elif ptype == "tqc": act_fn = _build_tqc_act_vmap(ckpt)
            elif ptype == "tagd": act_fn = _build_tagd_act_vmap(ckpt)

            if ptype == "mppi":
                n_envs = N_ENVS_MPPI
                seg_call = lambda obs, st, rk, off, _m=_mppi, _n=n_envs: \
                    _segment_mppi(_m, _n, obs, st, rk, off)
            else:
                n_envs = N_ENVS
                seg_call = lambda obs, st, rk, off, _a=act_fn, _n=n_envs: \
                    _segment_stateless(_a, _n, obs, st, rk, off)

            raw_scens = []
            for scen in TEST_SCENARIOS:
                rng, k = jax.random.split(rng)
                r = _run_scenario(seg_call, n_envs, scen, TEST_ROBOT_WAYPOINTS[scen], k)
                raw_scens.append(r)
                rows_scen.append({"Method": name, "Type": cat, "Scenario": scen, **_agg(r)})
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        raw_all = {key: np.concatenate([rs[key] for rs in raw_scens]) for key in raw_scens[0]}
        met = _agg(raw_all)
        rows.append({"Method": name, "Type": cat, **met})
        print(f"  [{name}] DONE ({time.time() - t0:.0f}s): SR={met['SR']:.1f}% ACR={met['ACR']:.1f}% PCR={met['PCR']:.1f}% TR={met['TR']:.1f}%\n")

    pd.DataFrame(rows).to_csv(paths.data("paper_comparison_table.csv"), index=False, float_format="%.3f")
    pd.DataFrame(rows_scen).to_csv(paths.data("paper_comparison_table_per_scen.csv"), index=False, float_format="%.3f")
    
    print("\n" + "="*70 + "\nRESULTS TABLE\n" + "="*70)
    print(pd.DataFrame(rows)[_TABLE_COLS].to_string(index=False, float_format="%.2f"))
    print("\n" + "="*70 + "\nLaTeX\n" + "="*70)
    _print_latex(rows)

if __name__ == "__main__": main()