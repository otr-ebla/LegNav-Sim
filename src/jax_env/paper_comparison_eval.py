"""
paper_comparison_eval.py — Comparison Table for Paper (Maximal Parallelization)
"""

import argparse
import os
import time
import warnings
from functools import partial

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

import jax_env as _jax_env
_jax_env.USE_LEGS     = not args.no_legs
_jax_env.SENSOR_NOISE = True

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, STATE_VEC_SIZE as _SVS)
from jax_legs import LEG_RADIUS
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState
from jax_network import SharedEncoder

OBS_SIZE   = 662
ACTION_DIM = 2
POSE_SIZE  = 3
STACK_DIM  = 3

TEST_SCENARIOS  = [7, 8, 9, 10, 11, 12]
V_MAX_MIN = 0.2
V_MAX_MAX = 2.0
N_ENVS      = args.envs
N_ENVS_MPPI = args.envs_mppi
YIELD_DIST = 1.5
YIELD_FOV  = 0.785

@jax.jit
def _reset_stacked(key, v_max, scenario_idx):
    base_obs, base_state = reset_env(key, 9.0, scenario_idx, 0.0)
    pose      = base_obs[:POSE_SIZE]
    state_vec = base_obs[POSE_SIZE: POSE_SIZE + _SVS]
    lidar     = base_obs[POSE_SIZE + _SVS:]
    base_state = base_state.replace(max_v=v_max)
    new_sv = jnp.array([0.0, 0.0, (v_max - 0.2) / 1.8, state_vec[3], state_vec[4]])
    lidar_stack = jnp.tile(lidar[None, :], (STACK_DIM, 1))
    pose_stack  = jnp.tile(pose[None, :],  (STACK_DIM, 1))
    stacked = StackedEnvState(env_state=base_state, lidar_stack=lidar_stack, pose_stack=pose_stack)
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

@partial(jax.jit, static_argnums=(0, 1))
def _rollout_stateless(act_vmap, total_envs, rng_key, scenario_batch):
    rng_key, rng_v = jax.random.split(rng_key)
    v_max_batch = jax.random.uniform(rng_v, (total_envs,), minval=V_MAX_MIN, maxval=V_MAX_MAX)
    reset_keys  = jax.random.split(rng_key, total_envs)
    # vmap dynamically across keys, v_maxes AND scenarios
    obs, state  = jax.vmap(_reset_stacked, in_axes=(0, 0, 0))(reset_keys, v_max_batch, scenario_batch)

    init_dist = jnp.sqrt((state.env_state.goal_x - state.env_state.x)**2 + (state.env_state.goal_y - state.env_state.y)**2)
    carry0 = (state, obs, jnp.zeros(total_envs), jnp.full(total_envs, 100.0), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.ones(total_envs, dtype=jnp.bool_), jnp.zeros(total_envs), jnp.zeros(total_envs))

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p, active, yz, yc) = carry
        k = jax.random.fold_in(rng_key, step_idx)
        actions = act_vmap(obs)
        step_keys = jax.random.split(k, total_envs)
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(step_keys, state, actions)

        v, w = next_state.env_state.v, next_state.env_state.w
        av, aw = (v - v_p) / DT, (w - w_p) / DT
        pl = pl + jnp.where(active, v * DT, 0.0)
        ch = info["closest_shoe_surface"]
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        ppl = next_state.env_state.people
        dp_x, dp_y = ppl[:, :, 0] - next_state.env_state.x[:, None], ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang = (jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None] + jnp.pi) % (2*jnp.pi) - jnp.pi
        in_yz = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & (ppl[:, :, 10] >= 0.0)
        any_yz = jnp.any(in_yz, axis=1)
        yz = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc = yc + jnp.where(active & any_yz & (v <= 0.1), 1.0, 0.0)

        g, c = info["goal_reached"] & active, info["collision"] & active
        ac, pc = info["active_col"] & active, info["passive_col"] & active
        jv, jw = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0), jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)

        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, v, w, av, aw, next_active, yz, yc), (active, g, c, ac, pc, jv + jw, v, ch)

    final_carry, step_data = jax.lax.scan(_step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))
    (_, _, final_pl, final_mhd, _, _, _, _, _, final_yz, final_yc) = final_carry
    active_mask, goals, cols, act_cols, pass_cols, jerks, step_vs, step_ch = step_data

    ep_lens = active_mask.sum(axis=0)
    has_any_col = cols.any(axis=0)
    ep_actcol = act_cols.any(axis=0)
    ep_pscol = pass_cols.any(axis=0) & ~ep_actcol
    ep_obscol = has_any_col & ~ep_actcol & ~ep_pscol
    ep_goal = goals.any(axis=0) & ~has_any_col
    ep_tmo = ~(ep_goal | has_any_col)

    return {
        "success": ep_goal.astype(jnp.float32), "obs_col": ep_obscol.astype(jnp.float32),
        "act_col": ep_actcol.astype(jnp.float32), "pass_col": ep_pscol.astype(jnp.float32),
        "timeout": ep_tmo.astype(jnp.float32), "jerk": jerks.sum(axis=0) / jnp.maximum(ep_lens, 1),
        "time_goal": jnp.where(ep_goal, ep_lens * DT, jnp.nan), "min_dist": final_mhd,
        "yield_score": jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan),
        "spl": ep_goal * (init_dist / jnp.maximum(final_pl, init_dist)),
        "step_ch": step_ch, "active_mask": active_mask, "ep_lens": ep_lens,
    }

@partial(jax.jit, static_argnums=(0, 1))
def _rollout_mppi(mppi, total_envs, rng_key, scenario_batch):
    rng_key, rng_v = jax.random.split(rng_key)
    v_max_batch = jax.random.uniform(rng_v, (total_envs,), minval=V_MAX_MIN, maxval=V_MAX_MAX)
    reset_keys = jax.random.split(rng_key, total_envs)
    obs, state = jax.vmap(_reset_stacked, in_axes=(0, 0, 0))(reset_keys, v_max_batch, scenario_batch)
    init_dist = jnp.sqrt((state.env_state.goal_x - state.env_state.x)**2 + (state.env_state.goal_y - state.env_state.y)**2)
    u_mean_init = jnp.broadcast_to(mppi.init_u_mean(), (total_envs, mppi.horizon, 2))

    carry0 = (state, obs, jnp.zeros(total_envs), jnp.full(total_envs, 100.0), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.ones(total_envs, dtype=jnp.bool_), jnp.zeros(total_envs), jnp.zeros(total_envs), u_mean_init)

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p, active, yz, yc, u_mean) = carry
        k = jax.random.fold_in(rng_key, step_idx)
        actions, new_u_mean = jax.vmap(mppi.act)(obs, u_mean, jax.random.split(k, total_envs))
        next_obs, next_state, _, done, info = jax.vmap(_step_stacked)(jax.random.split(k, total_envs), state, actions)

        v, w = next_state.env_state.v, next_state.env_state.w
        av, aw = (v - v_p) / DT, (w - w_p) / DT
        pl = pl + jnp.where(active, v * DT, 0.0)
        ch = info["closest_shoe_surface"]
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        ppl = next_state.env_state.people
        dp_x, dp_y = ppl[:, :, 0] - next_state.env_state.x[:, None], ppl[:, :, 1] - next_state.env_state.y[:, None]
        dists_p = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang = (jnp.arctan2(dp_y, dp_x) - next_state.env_state.theta[:, None] + jnp.pi) % (2*jnp.pi) - jnp.pi
        in_yz = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & (ppl[:, :, 10] >= 0.0)
        any_yz = jnp.any(in_yz, axis=1)
        yz = yz + jnp.where(active & any_yz, 1.0, 0.0)
        yc = yc + jnp.where(active & any_yz & (v <= 0.1), 1.0, 0.0)

        g, c = info["goal_reached"] & active, info["collision"] & active
        ac, pc = info["active_col"] & active, info["passive_col"] & active
        jv, jw = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0), jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)
        
        new_u_mean = jnp.where(done[:, None, None], jnp.broadcast_to(mppi.init_u_mean(), new_u_mean.shape), new_u_mean)
        next_active = active & ~done
        return (next_state, next_obs, pl, mhd, v, w, av, aw, next_active, yz, yc, new_u_mean), (active, g, c, ac, pc, jv + jw, v, ch)

    final_carry, step_data = jax.lax.scan(_step, carry0, jnp.arange(MAX_STEPS, dtype=jnp.uint32))
    (_, _, final_pl, final_mhd, _, _, _, _, _, final_yz, final_yc, _) = final_carry
    active_mask, goals, cols, act_cols, pass_cols, jerks, _, step_ch = step_data

    ep_lens = active_mask.sum(axis=0)
    has_any_col = cols.any(axis=0)
    ep_actcol = act_cols.any(axis=0)
    ep_pscol = pass_cols.any(axis=0) & ~ep_actcol
    ep_obscol = has_any_col & ~ep_actcol & ~ep_pscol
    ep_goal = goals.any(axis=0) & ~has_any_col
    ep_tmo = ~(ep_goal | has_any_col)

    return {
        "success": ep_goal.astype(jnp.float32), "obs_col": ep_obscol.astype(jnp.float32),
        "act_col": ep_actcol.astype(jnp.float32), "pass_col": ep_pscol.astype(jnp.float32),
        "timeout": ep_tmo.astype(jnp.float32), "jerk": jerks.sum(axis=0) / jnp.maximum(ep_lens, 1),
        "time_goal": jnp.where(ep_goal, ep_lens * DT, jnp.nan), "min_dist": final_mhd,
        "yield_score": jnp.where(final_yz > 0, final_yc / final_yz, jnp.nan),
        "spl": ep_goal * (init_dist / jnp.maximum(final_pl, init_dist)),
        "step_ch": step_ch, "active_mask": active_mask, "ep_lens": ep_lens,
    }

def _obs_to_max_v(obs): return jnp.clip(obs[..., 11] * 1.8 + 0.2, 0.2, 2.0)
def _load_raw(path):
    with open(path, "rb") as f: return flax.serialization.msgpack_restore(f.read())

def _build_ppo_act_vmap(ckpt_path):
    from jax_network import EndToEndActorCritic, scale_action_to_env
    net, bundle = EndToEndActorCritic(action_dim=ACTION_DIM), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_mlp_act_vmap(ckpt_path):
    from comparison_policies.vanilla_mlp_network import VanillaMLPActorCritic
    from jax_network import scale_action_to_env
    net, bundle = VanillaMLPActorCritic(action_dim=ACTION_DIM, hidden_dim=128), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_navrep_act_vmap(ckpt_path):
    from comparison_policies.navrep_network import NavRepActorCritic
    from jax_network import scale_action_to_env
    net, bundle = NavRepActorCritic(action_dim=ACTION_DIM), _load_raw(ckpt_path)
    params = bundle.get("params", bundle)
    def act_vmap(obs_batch):
        def _single(obs): return scale_action_to_env(jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0), _obs_to_max_v(obs))
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_sac_act_vmap(ckpt_path):
    enc, bundle = SharedEncoder(), _load_raw(ckpt_path)
    class SACHead(nn.Module):
        @nn.compact
        def __call__(self, feat): return nn.Dense(ACTION_DIM)(feat), jnp.clip(nn.Dense(ACTION_DIM)(feat), -5.0, 2.0)
    head, enc_p, head_p = SACHead(), bundle["enc_params"], bundle["actor_head_params"]
    def act_vmap(obs_batch):
        def _single(obs):
            mean = jnp.squeeze(head.apply({"params": head_p}, enc.apply({"params": enc_p}, obs[None]))[0], 0)
            return jnp.stack([(jnp.tanh(mean[0]) + 1.0) * 0.5 * _obs_to_max_v(obs), jnp.tanh(mean[1])])
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_tqc_act_vmap(ckpt_path):
    enc, bundle = SharedEncoder(), _load_raw(ckpt_path)
    class TQCHead(nn.Module):
        @nn.compact
        def __call__(self, feat): return nn.Dense(ACTION_DIM)(feat).astype(jnp.float32), jnp.clip(nn.Dense(ACTION_DIM)(feat).astype(jnp.float32), -5.0, 0.5)
    head, enc_p, head_p = TQCHead(), bundle["enc_params"], bundle["actor_params"]
    def act_vmap(obs_batch):
        def _single(obs):
            mean = jnp.squeeze(head.apply({"params": head_p}, enc.apply({"params": enc_p}, obs[None]))[0], 0)
            return jnp.stack([(jnp.tanh(mean[0]) + 1.0) * 0.5 * _obs_to_max_v(obs), jnp.tanh(mean[1])])
        return jax.vmap(_single)(obs_batch)
    return act_vmap

def _build_dwa_act_vmap():
    from comparison_policies.dwa_planner import DWA
    dwa = DWA()
    return jax.vmap(dwa.act)

def _build_tagd_act_vmap(ckpt_path):
    from comparison_policies.tagd_network import make_tagd_act_fn
    return make_tagd_act_fn(_load_raw(ckpt_path).get("actor_params", _load_raw(ckpt_path)))

POLICY_REGISTRY = [
    ("DWA",          "dwa",    None,                                           "Model-Based"),
    ("MPPI",         "mppi",   None,                                           "Model-Based"),
    ("MLP",          "mlp",    "checkpoints_vanilla_ppo/ppo_mlp_best.msgpack", "RL"),
    ("NavRep",       "navrep", "checkpoints_navrep/navrep_best.msgpack",       "Unsup. Learning"),
    ("PPO (circles)","ppo",    "checkpoints/ppo_circles_best.msgpack",         "End-to-end RL"),
    ("PPO (legs)",   "ppo",    "checkpoints/ppo_attn_best.msgpack",            "End-to-end RL"),
    ("SAC",          "sac",    "checkpoints_sac/sac_best.msgpack",             "End-to-end RL"),
    ("TQC",          "tqc",    "checkpoints_tqc/tqc_best.msgpack",             "End-to-end RL"),
    ("TAGD",         "tagd",   "checkpoints_tagd/tagd_best.msgpack",           "End-to-end RL"),
]

def _compute_social_score(raw, nu=0.35, nu_prime=0.25, d_u=0.45):
    suc, t_goal, act, epl = np.array(raw["success"], dtype=bool), np.array(raw["time_goal"]), np.array(raw["active_mask"]), np.array(raw["ep_lens"])
    n_tot, K1 = len(suc), int(suc.sum())
    F_time = max(0.0, min(1.0, 1.0 - float(np.mean((t_goal[suc] - float((epl * DT).min())) / max(float(t_goal[suc].max() if K1 else 1.0) - float((epl * DT).min()), 1.0))))) if K1 else 0.0
    
    ch_act = np.where(act, np.array(raw["step_ch"]), np.inf)
    has_unc = (ch_act < d_u).any(axis=0)
    K2 = int(has_unc.sum())
    
    if K2 == 0: F_scc = 1.0
    else:
        rat = d_u / np.maximum(np.where(act, np.array(raw["step_ch"]), 0.0).sum(axis=0), 1e-8)
        F_scc = max(0.0, min(1.0, 1.0 - 0.5 * (float(np.mean(1.0 / (1.0 + np.exp(-(rat[has_unc] - 1.0))))) + (K2 / max(K1, 1)))))

    F_F = -(int(np.array(raw["obs_col"]).sum() + np.array(raw["act_col"]).sum() + np.array(raw["pass_col"]).sum() + np.array(raw["timeout"]).sum())) / max(n_tot, 1)
    return float(100.0 * (nu * F_time + (1.0 - nu) * F_scc + nu_prime * F_F))

def _slice_raw(raw, start, end):
    return {k: v[start:end] if v.ndim == 1 else v[:, start:end] for k, v in raw.items()}

def _agg(r):
    return {"N": len(r["success"]), "Success": float(np.mean(r["success"]))*100, "Obst. Col.": float(np.mean(r["obs_col"]))*100,
            "Act. Col.": float(np.mean(r["act_col"]))*100, "Pass. Col.": float(np.mean(r["pass_col"]))*100, "Timeout": float(np.mean(r["timeout"]))*100,
            "Yield Score": float(np.nanmean(r["yield_score"])), "Jerk": float(np.nanmean(r["jerk"])), "Time to Goal": float(np.nanmean(r["time_goal"])),
            "Min Dist (m)": float(np.mean(r["min_dist"])), "Social Score": _compute_social_score(r)}

def _print_latex(rows):
    print(r"\begin{table*}[t]"+"\n"+r"\centering"+"\n"+r"\caption{Comparison of Navigation Methods}"+"\n"+r"\label{tab:comp}"+"\n"+r"\footnotesize"+"\n"+r"\begin{tabular}{lccccccccccc}"+"\n"+r"\toprule"+"\n"+r"Method & Type & Success (\%) & Obst. Col. (\%) & Act. Col. (\%) & Pass. Col. (\%) & Timeout (\%) & Yield Score & Jerk & Time (s) & Min Dist (m) & $F_{SC}$ \\"+"\n"+r"\midrule")
    for r in rows: print(f"{r['Method'].replace('(', r'\\textit{(').replace(')', r')}')} & {r['Type']} & {r['Success']:.1f} & {r['Obst. Col.']:.1f} & {r['Act. Col.']:.1f} & {r['Pass. Col.']:.1f} & {r['Timeout']:.1f} & {r['Yield Score']:.2f} & {r['Jerk']:.1f} & {r['Time to Goal']:.1f} & {r['Min Dist (m)']:.2f} & {r['Social Score']:.1f} \\\\")
    print(r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"+r"\end{table*}")

def main():
    rng = jax.random.PRNGKey(args.seed)
    rows, rows_scen = [], []
    n_scen = len(TEST_SCENARIOS)
    tot_envs, tot_envs_mppi = n_scen * N_ENVS, n_scen * N_ENVS_MPPI
    
    # ── Single pre-computed scenario batch for vmap ──
    scen_batch      = jnp.repeat(jnp.array(TEST_SCENARIOS, dtype=jnp.int32), N_ENVS)
    scen_batch_mppi = jnp.repeat(jnp.array(TEST_SCENARIOS, dtype=jnp.int32), N_ENVS_MPPI)

    print(f"\n{'='*70}\nMax Parallelization Eval — scenarios={TEST_SCENARIOS} | envs/scen={N_ENVS}\n{'='*70}\n")

    for name, ptype, ckpt, cat in POLICY_REGISTRY:
        if ckpt and not os.path.exists(ckpt): continue
        t0 = time.time()
        print(f"  [{name}] Dispatched completely in parallel...")

        try:
            if ptype == "dwa": act_fn = _build_dwa_act_vmap()
            elif ptype == "mppi":
                from comparison_policies.mppi_planner import MPPI
                _mppi = MPPI()
            elif ptype == "ppo": act_fn = _build_ppo_act_vmap(ckpt)
            elif ptype == "mlp": act_fn = _build_mlp_act_vmap(ckpt)
            elif ptype == "navrep": act_fn = _build_navrep_act_vmap(ckpt)
            elif ptype == "sac": act_fn = _build_sac_act_vmap(ckpt)
            elif ptype == "tqc": act_fn = _build_tqc_act_vmap(ckpt)
            elif ptype == "tagd": act_fn = _build_tagd_act_vmap(ckpt)

            rng, k = jax.random.split(rng)
            if ptype == "mppi":
                raw_s = jax.device_get(jax.block_until_ready(_rollout_mppi(_mppi, tot_envs_mppi, k, scen_batch_mppi)))
                chk = N_ENVS_MPPI
            else:
                raw_s = jax.device_get(jax.block_until_ready(_rollout_stateless(act_fn, tot_envs, k, scen_batch)))
                chk = N_ENVS
        except Exception as e:
            print(f" ERROR: {e}"); continue

        # Extract per-scenario stats from the monolithic batch
        for i, scen in enumerate(TEST_SCENARIOS):
            scen_raw = _slice_raw(raw_s, i * chk, (i + 1) * chk)
            met = _agg(scen_raw)
            rows_scen.append({"Method": name, "Type": cat, "Scenario": scen, **met})

        met = _agg(raw_s)
        rows.append({"Method": name, "Type": cat, **met})
        print(f"  [{name}] DONE ({time.time() - t0:.0f}s): Suc={met['Success']:.1f}% ObsCol={met['Obst. Col.']:.1f}% ActCol={met['Act. Col.']:.1f}% PasCol={met['Pass. Col.']:.1f}% Tmo={met['Timeout']:.1f}%\n")

    pd.DataFrame(rows).to_csv("paper_comparison_table.csv", index=False, float_format="%.3f")
    pd.DataFrame(rows_scen).to_csv("paper_comparison_table_per_scen.csv", index=False, float_format="%.3f")
    
    print("\n" + "="*70 + "\nRESULTS TABLE\n" + "="*70)
    print(pd.DataFrame(rows)[["Method", "Type", "Success", "Obst. Col.", "Act. Col.", "Pass. Col.", "Timeout", "Yield Score", "Jerk", "Time to Goal", "Min Dist (m)", "Social Score"]].to_string(index=False, float_format="%.2f"))
    print("\n" + "="*70 + "\nLaTeX\n" + "="*70)
    _print_latex(rows)

if __name__ == "__main__": main()