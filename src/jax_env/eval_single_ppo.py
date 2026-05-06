"""
eval_single_ppo.py — Fast Parallel Evaluation for a Single PPO Checkpoint
"""

import argparse
import os
import time
import warnings
from functools import partial

# Force GPU allocation
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
warnings.filterwarnings("ignore")

import jax
try:
    jax.config.update("jax_default_device", jax.devices("cuda")[0])
    print(f"✅ GPU CUDA rilevata e forzata: {jax.devices('cuda')[0]}")
except RuntimeError:
    print("❌ ERRORE CRITICO: JAX non vede la GPU CUDA e sta ripiegando su CPU!")

import jax.numpy as jnp
import numpy as np
import pandas as pd
import flax.serialization

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str,  default="checkpoints/ppo_tanh_fix_final.msgpack")
    p.add_argument("--envs",      type=int,  default=512, help="Numero di ambienti PARALLELI per scenario")
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--no-legs",   action="store_true")
    p.add_argument("--tanh-outside", action="store_true", help="Usa questo flag per i vecchi modelli (es. ppo_attn_final.msgpack)")
    return p.parse_args()

args = _parse()

# Configurazione Ambiente
import jax_env as _jax_env
_jax_env.USE_LEGS     = not args.no_legs
_jax_env.SENSOR_NOISE = True

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, STATE_VEC_SIZE as _SVS)
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState

OBS_SIZE   = 662
ACTION_DIM = 2
POSE_SIZE  = 3
STACK_DIM  = 3

TEST_SCENARIOS  = [7, 8, 9, 10, 11, 12]
V_MAX_MIN = 0.2
V_MAX_MAX = 2.0
N_ENVS    = args.envs
YIELD_DIST = 1.5
YIELD_FOV  = 0.785


# ==============================================================================
# WRAPPERS & ROLLOUT JIT
# ==============================================================================

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
    
    obs, state  = jax.vmap(_reset_stacked, in_axes=(0, 0, 0))(reset_keys, v_max_batch, scenario_batch)
    init_dist = jnp.sqrt((state.env_state.goal_x - state.env_state.x)**2 + (state.env_state.goal_y - state.env_state.y)**2)
    carry0 = (state, obs, jnp.zeros(total_envs), jnp.full(total_envs, 100.0), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.zeros(total_envs), jnp.ones(total_envs, dtype=jnp.bool_), jnp.zeros(total_envs), jnp.zeros(total_envs))

    def _step(carry, step_idx):
        (state, obs, pl, mhd, v_p, w_p, av_p, aw_p, active, yz, yc) = carry
        k = jax.random.fold_in(rng_key, step_idx)
        
        # Policy Forward
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


# ==============================================================================
# AGGREGATION & BUILDERS
# ==============================================================================

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
    return {
        "N": len(r["success"]), 
        "Success": float(np.mean(r["success"]))*100, 
        "Obst. Col.": float(np.mean(r["obs_col"]))*100,
        "Act. Col.": float(np.mean(r["act_col"]))*100, 
        "Pass. Col.": float(np.mean(r["pass_col"]))*100, 
        "Timeout": float(np.mean(r["timeout"]))*100,
        "Yield Score": float(np.nanmean(r["yield_score"])), 
        "Jerk": float(np.nanmean(r["jerk"])), 
        "Time to Goal": float(np.nanmean(r["time_goal"])),
        "Min Dist (m)": float(np.mean(r["min_dist"])), 
        "Social Score": _compute_social_score(r)
    }

def _obs_to_max_v(obs): 
    return jnp.clip(obs[..., 11] * 1.8 + 0.2, 0.2, 2.0)

def _build_ppo_act_vmap(ckpt_path, tanh_outside):
    from jax_network import EndToEndActorCritic
    
    # Inizializziamo la rete dicendole se il Tanh è dentro o fuori
    net = EndToEndActorCritic(action_dim=ACTION_DIM, tanh_inside=not tanh_outside)
    
    with open(ckpt_path, "rb") as f: 
        bundle = flax.serialization.msgpack_restore(f.read())
    params = bundle.get("params", bundle)
    
    def act_vmap(obs_batch):
        def _single(obs): 
            mean = jnp.squeeze(net.apply({"params": params}, obs[None])[0], 0)
            max_v = _obs_to_max_v(obs)
            
            if tanh_outside:
                # Variante 2 (Originale): Il modello ci dà numeri grezzi, applichiamo il Tanh qui
                v = (jnp.tanh(mean[0]) * 0.5 + 0.5) * max_v
                w = jnp.tanh(mean[1])
            else:
                # Variante 1 (Nuova): Il modello ha già fatto il Tanh, facciamo solo un clip di sicurezza
                v = jnp.clip(mean[0], 0.0, 1.0) * max_v
                w = jnp.clip(mean[1], -1.0, 1.0)
                
            return jnp.stack([v, w])
        return jax.vmap(_single)(obs_batch)
    return act_vmap

# ==============================================================================
# MAIN 
# ==============================================================================

def main():
    if not os.path.exists(args.ckpt):
        print(f"❌ ERRORE: Il checkpoint '{args.ckpt}' non esiste!")
        return

    rng = jax.random.PRNGKey(args.seed)
    n_scen = len(TEST_SCENARIOS)
    tot_envs = n_scen * N_ENVS
    
    # Pre-compute monolithic array of scenarios per env
    scen_batch = jnp.repeat(jnp.array(TEST_SCENARIOS, dtype=jnp.int32), N_ENVS)

    print(f"\n{'='*75}")
    print(f"🚀 PPO Tanh-Fix Evaluation")
    print(f"   Checkpoint : {args.ckpt}")
    print(f"   Scenarios  : {TEST_SCENARIOS}")
    print(f"   Envs/Scen  : {N_ENVS}")
    print(f"   Total Envs : {tot_envs}")
    print(f"{'='*75}\n")

    t0 = time.time()
    
    # 1. Carica Rete
    print(f"🔄 PPO Network compilation...")
    act_fn = _build_ppo_act_vmap(args.ckpt, args.tanh_outside)

    # 2. Run Rollout
    rng, k = jax.random.split(rng)
    print(f"⚡ Esecuzione Rollout VMAP massivo su GPU (attendi JIT...)")
    raw_s = jax.device_get(jax.block_until_ready(_rollout_stateless(act_fn, tot_envs, k, scen_batch)))
    
    calc_time = time.time() - t0
    print(f"✅ Rollout completato in {calc_time:.1f}s!\n")

    # 3. Aggregazione Risultati
    rows_scen = []
    for i, scen in enumerate(TEST_SCENARIOS):
        scen_raw = _slice_raw(raw_s, i * N_ENVS, (i + 1) * N_ENVS)
        met = _agg(scen_raw)
        rows_scen.append({"Scenario": str(scen), **met})

    # Aggregato totale
    met_tot = _agg(raw_s)
    tot_row = {"Scenario": "ALL", **met_tot}

    # Creazione DataFrame
    df_scen = pd.DataFrame(rows_scen)
    df_tot = pd.DataFrame([tot_row])
    df_final = pd.concat([df_scen, df_tot], ignore_index=True)

    # 4. Print & Save
    cols_to_print = ["Scenario", "Success", "Obst. Col.", "Act. Col.", "Pass. Col.", "Timeout", "Yield Score", "Jerk", "Time to Goal", "Min Dist (m)", "Social Score"]
    
    print("="*105)
    print(" 📊 RISULTATI ")
    print("="*105)
    print(df_final[cols_to_print].to_string(index=False, float_format="%.2f"))
    print("="*105)

    out_file = "ppo_tanh_fix_eval.csv"
    df_final.to_csv(out_file, index=False, float_format="%.3f")
    print(f"\n📁 Risultati dettagliati salvati in: {out_file}")

if __name__ == "__main__": 
    main()