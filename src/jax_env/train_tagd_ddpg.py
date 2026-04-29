"""
train_tagd_ddpg.py — DDPG training for the TAGD navigation policy

References: Spatiotemporal Attention Enhances Lidar-Based
Robot Navigation in Dynamic Environments (De Heuvel et al. IEEE RA-L 2024)

=================================================================
Trains the TAGD actor-critic (comparison_policies/tagd_network.py) using the
standard Deep Deterministic Policy Gradient algorithm (Lillicrap et al. 2015).

Hyper-parameters follow de Heuvel et al. IEEE RA-L 2024 where specified:
  γ = 0.98,  τ = 0.005,  replay buffer ≥ 1M transitions,
  exploration noise: Gaussian σ=0.1 added to deterministic action.

Usage:
    cd src/jax_env
    python3 train_tagd_ddpg.py [--gpu 0] [--envs 512] [--steps 20000000]
"""

import os
import sys
import csv
import argparse as _ap

# ── GPU / platform selection (must happen BEFORE import jax) ──────────────────
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

os.environ.setdefault("JAX_PLATFORMS",               "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR",            "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")

import time
import warnings
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Verify GPU ────────────────────────────────────────────────────────────────
def _check_gpu():
    try:
        devs = jax.devices("cuda")
    except RuntimeError:
        devs = []
    if not devs:
        raise RuntimeError("No CUDA devices found. Run with --gpu or set JAX_PLATFORMS.")
    dev = devs[0]
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"TAGD DDPG pinned to: JAX device {dev}  →  physical GPU {phys}")
    return dev

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)

# ── Environment ───────────────────────────────────────────────────────────────
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_ppo import get_continuous_curriculum
from comparison_policies.tagd_network import TAGDActor, TAGDCritic

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = _ap.ArgumentParser()
    p.add_argument("--gpu",   type=int,   default=None)
    p.add_argument("--envs",  type=int,   default=512,       help="Parallel envs")
    p.add_argument("--steps", type=int,   default=20_000_000,help="Total env steps")
    p.add_argument("--seed",  type=int,   default=0)
    p.add_argument("--resume",action="store_true",           help="Resume from latest ckpt")
    return p.parse_args()

args = _parse()

# ── Hyper-parameters ──────────────────────────────────────────────────────────
OBS_SIZE       = 662
ACTION_DIM     = 2
N_ENVS         = args.envs
TOTAL_STEPS    = args.steps

BUFFER_CAP     = 600_000      # transitions (≈ 1.2 GB float32 obs × 2)
BATCH_SIZE     = 256          # paper: 256
G_UPDATES      = 10           # gradient updates per env step
WARMUP_STEPS   = 20_000       # collect transitions before first update

GAMMA          = 0.98         # paper
TAU            = 0.005        # soft target update rate
LR_ACTOR       = 3e-4
LR_CRITIC      = 3e-4
MAX_GRAD_NORM  = 10.0

EXPL_NOISE_STD = 0.10         # Gaussian exploration noise σ
EXPL_NOISE_MIN = 0.01         # noise annealing floor
EXPL_NOISE_DECAY = 0.9999     # multiply std each env step (very slow decay)

LOG_EVERY      = 100          # env-step batches
SAVE_EVERY     = 2000

CKPT_DIR       = "checkpoints_tagd"
CKPT_BEST      = f"{CKPT_DIR}/tagd_best.msgpack"
CKPT_FINAL     = f"{CKPT_DIR}/tagd_final.msgpack"
LOG_CSV        = f"{CKPT_DIR}/tagd_training_log.csv"

os.makedirs(CKPT_DIR, exist_ok=True)

# ── Environment setup ─────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
_step_auto = make_autoreset_env(reset_stacked, step_stacked)

vmap_step = jax.jit(
    jax.vmap(_step_auto, in_axes=(0, 0, 0, None, None, None, None))
)

@jax.jit
def _vmap_reset(keys, dist, ghost, scenario):
    def _s(k):
        return reset_stacked(k, max_goal_dist=dist, ghost_prob=ghost,
                             scenario_idx=scenario)
    return jax.vmap(_s)(keys)

def init_env(rng, dist=1.5, ghost=0.0, scenario=-1):
    keys = jax.random.split(rng, N_ENVS)
    obs, state = _vmap_reset(keys, jnp.float32(dist), jnp.float32(ghost),
                              jnp.int32(scenario))
    return obs, state

# ── Replay buffer (same pattern as TQCjac.py) ─────────────────────────────────
_BUF_DTYPE = jnp.bfloat16  # halves memory; cast to float32 at sample time

def make_buffer(cap):
    return {
        "obs":      jnp.zeros((cap, OBS_SIZE),    _BUF_DTYPE),
        "act":      jnp.zeros((cap, ACTION_DIM),  jnp.float32),
        "rew":      jnp.zeros((cap,),             jnp.float32),
        "next_obs": jnp.zeros((cap, OBS_SIZE),    _BUF_DTYPE),
        "done":     jnp.zeros((cap,),             jnp.float32),
        "ptr":  jnp.int32(0),
        "size": jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, act, rew, next_obs, done):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs":      buf["obs"].at[idxs].set(obs.astype(_BUF_DTYPE)),
        "act":      buf["act"].at[idxs].set(act),
        "rew":      buf["rew"].at[idxs].set(rew),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs.astype(_BUF_DTYPE)),
        "done":     buf["done"].at[idxs].set(done),
        "ptr":  jnp.int32((buf["ptr"] + N) % cap),
        "size": jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }

@jax.jit(static_argnames=["bs"])
def buf_sample(buf, rng, bs: int):
    max_i = jnp.maximum(1, buf["size"])
    idx   = jax.random.randint(rng, (bs,), 0, max_i)
    return (
        buf["obs"][idx].astype(jnp.float32),
        buf["act"][idx],
        buf["rew"][idx],
        buf["next_obs"][idx].astype(jnp.float32),
        buf["done"][idx],
    )

# ── Networks & optimisers ─────────────────────────────────────────────────────
actor  = TAGDActor()
critic = TAGDCritic()

actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR_ACTOR))
critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR_CRITIC))

# ── Parameter initialisation ──────────────────────────────────────────────────
def init_params(rng):
    rng_a, rng_c = jax.random.split(rng, 2)
    dummy_obs = jnp.zeros((OBS_SIZE,))
    dummy_act = jnp.zeros((ACTION_DIM,))
    ap = actor.init(rng_a, dummy_obs)["params"]
    cp = critic.init(rng_c, dummy_obs, dummy_act)["params"]
    return ap, cp

# ── Soft target update ─────────────────────────────────────────────────────────
@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(
        lambda t, o: TAU * o + (1.0 - TAU) * t, target, online
    )

# ── Critic update step ────────────────────────────────────────────────────────
@jax.jit
def critic_step(cp, cp_opt_state, ap_tgt, cp_tgt,
                obs, act, rew, next_obs, done):
    def loss_fn(cp_):
        # Compute target actions using target actor
        target_act = jax.vmap(lambda o: actor.apply({"params": ap_tgt}, o))(next_obs)
        target_act = jax.lax.stop_gradient(target_act)

        # Target Q
        q_tgt = jax.vmap(lambda o, a: critic.apply({"params": cp_tgt}, o, a))(
            next_obs, target_act)
        q_tgt = jax.lax.stop_gradient(q_tgt)
        backup = rew + GAMMA * (1.0 - done) * q_tgt  # (B,)

        # Online Q
        q_online = jax.vmap(lambda o, a: critic.apply({"params": cp_}, o, a))(obs, act)

        return jnp.mean((q_online - backup) ** 2), jnp.mean(q_online)

    (c_loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(cp)
    updates, new_opt = critic_opt.update(grads, cp_opt_state, cp)
    new_cp = optax.apply_updates(cp, updates)
    return new_cp, new_opt, c_loss, q_mean

# ── Actor update step ─────────────────────────────────────────────────────────
@jax.jit
def actor_step(ap, ap_opt_state, cp, obs):
    def loss_fn(ap_):
        acts = jax.vmap(lambda o: actor.apply({"params": ap_}, o))(obs)
        q    = jax.vmap(lambda o, a: critic.apply({"params": cp}, o, a))(obs, acts)
        return -jnp.mean(q)   # maximise Q

    a_loss, grads = jax.value_and_grad(loss_fn)(ap)
    updates, new_opt = actor_opt.update(grads, ap_opt_state, ap)
    new_ap = optax.apply_updates(ap, updates)
    return new_ap, new_opt, a_loss

# ── Exploration: deterministic action + Gaussian noise ───────────────────────
@jax.jit
def explore(ap, obs_batch, rng, noise_std):
    def _single(obs):
        return actor.apply({"params": ap}, obs)
    acts  = jax.vmap(_single)(obs_batch)                  # (N, 2)
    noise = jax.random.normal(rng, acts.shape) * noise_std
    noisy = acts + noise
    # Clip to valid action ranges (v ∈ [0, max_v_i], w ∈ [-1, 1])
    # We can't easily clamp v per-env without knowing max_v; clamp w only
    noisy = noisy.at[:, 1].set(jnp.clip(noisy[:, 1], -1.0, 1.0))
    noisy = noisy.at[:, 0].set(jnp.clip(noisy[:, 0], 0.0,  2.0))
    return noisy

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _save(path, ap, cp, ap_tgt, cp_tgt, meta):
    bundle = {
        "actor_params":        ap,
        "critic_params":       cp,
        "actor_target_params": ap_tgt,
        "critic_target_params":cp_tgt,
        "meta":                meta,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Saved {path}")

def _load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return flax.serialization.msgpack_restore(raw)

# ── Main training loop ────────────────────────────────────────────────────────
def train():
    rng = jax.random.PRNGKey(args.seed)

    # ── Initialise parameters ────────────────────────────────────────────────
    rng, k_init = jax.random.split(rng)
    ap, cp = init_params(k_init)
    ap_tgt, cp_tgt = ap, cp

    ap_opt_state = actor_opt.init(ap)
    cp_opt_state = critic_opt.init(cp)

    # Optionally resume
    global_steps = 0
    best_suc     = 0.0
    if args.resume and os.path.exists(CKPT_FINAL):
        bundle = _load(CKPT_FINAL)
        ap        = bundle["actor_params"]
        cp        = bundle["critic_params"]
        ap_tgt    = bundle["actor_target_params"]
        cp_tgt    = bundle["critic_target_params"]
        meta      = bundle.get("meta", {})
        global_steps = int(meta.get("steps", 0))
        best_suc     = float(meta.get("best_suc", 0.0))
        ap_opt_state = actor_opt.init(ap)
        cp_opt_state = critic_opt.init(cp)
        print(f"Resumed from {CKPT_FINAL} at step {global_steps}")

    # ── Curriculum (same anchors as PPO/TQC) ─────────────────────────────────
    suc_pct = best_suc
    dist, ghost, _, max_scen = get_continuous_curriculum(suc_pct)

    rng, k_env = jax.random.split(rng)
    obs_buf, env_state = init_env(k_env, dist=dist, ghost=ghost, scenario=max_scen)

    buf = make_buffer(BUFFER_CAP)

    noise_std   = EXPL_NOISE_STD
    log_rows     = []
    ep_rew_acc   = np.zeros(N_ENVS, dtype=np.float32)
    ep_suc_acc   = np.zeros(N_ENVS, dtype=np.float32)
    ep_done_cnt  = 0
    ep_suc_total = 0
    recent_suc   = []

    # ── CSV log header ────────────────────────────────────────────────────────
    csv_exists = os.path.exists(LOG_CSV)
    csv_f = open(LOG_CSV, "a", newline="")
    csv_w = csv.writer(csv_f)
    if not csv_exists:
        csv_w.writerow(["step", "suc_pct", "ep_rew", "c_loss", "a_loss",
                        "q_mean", "noise_std", "dist", "ghost"])

    print(f"\n{'='*70}")
    print(f"TAGD DDPG training  |  envs={N_ENVS}  steps={TOTAL_STEPS:,}")
    print(f"buffer={BUFFER_CAP:,}  batch={BATCH_SIZE}  "
          f"G_updates/step={G_UPDATES}  warmup={WARMUP_STEPS:,}")
    print(f"{'='*70}\n")

    t0 = time.time()
    update_counter = 0
    c_loss_avg = a_loss_avg = q_mean_avg = 0.0

    while global_steps < TOTAL_STEPS:
        # ── Collect one step for all envs ────────────────────────────────────
        rng, k_act, k_step, k_noise = jax.random.split(rng, 4)

        if global_steps < WARMUP_STEPS:
            # Pure random actions during warmup
            actions = jax.random.uniform(
                k_act, (N_ENVS, ACTION_DIM),
                minval=jnp.array([0.0, -1.0]),
                maxval=jnp.array([1.0,  1.0]),
            )
        else:
            actions = explore(ap, obs_buf, k_noise, noise_std)

        step_keys = jax.random.split(k_step, N_ENVS)
        next_obs, env_state, rewards, dones, info = vmap_step(
            step_keys, env_state, actions,
            jnp.float32(9.0), jnp.int32(-1),
            jnp.float32(ghost), jnp.int32(max_scen),
        )

        # ── Store transition (keep arrays on device — no device_get) ─────────
        buf = buf_add(buf,
                      obs_buf, actions,
                      rewards, next_obs,
                      dones.astype(jnp.float32))

        # ── Track episode stats ───────────────────────────────────────────────
        np_rew  = np.array(rewards)
        np_done = np.array(dones)
        np_goal = np.array(info.get("goal_reached", np.zeros(N_ENVS)))
        ep_rew_acc += np_rew
        ep_suc_acc  = np.where(np_goal, 1.0, ep_suc_acc)

        finished     = np_done > 0
        ep_done_cnt += int(np.sum(finished))
        ep_suc_total+= int(np.sum(ep_suc_acc[finished]))
        recent_suc.extend(ep_suc_acc[finished].tolist())
        if len(recent_suc) > 2000:
            recent_suc = recent_suc[-2000:]
        ep_rew_acc[finished] = 0.0
        ep_suc_acc[finished] = 0.0

        obs_buf    = next_obs
        global_steps += N_ENVS

        # ── Gradient updates ─────────────────────────────────────────────────
        buf_ready = global_steps >= WARMUP_STEPS and int(jax.device_get(buf["size"])) >= BATCH_SIZE
        if buf_ready:
            for _ in range(G_UPDATES):
                rng, k_s = jax.random.split(rng)
                obs_b, act_b, rew_b, nobs_b, done_b = buf_sample(buf, k_s, BATCH_SIZE)

                # Critic update
                cp, cp_opt_state, c_loss, q_mean = critic_step(
                    cp, cp_opt_state, ap_tgt, cp_tgt,
                    obs_b, act_b, rew_b, nobs_b, done_b)

                # Actor update (every step — standard DDPG)
                ap, ap_opt_state, a_loss = actor_step(ap, ap_opt_state, cp, obs_b)

                # Soft target updates
                ap_tgt = soft_update(ap_tgt, ap)
                cp_tgt = soft_update(cp_tgt, cp)

                c_loss_avg += float(c_loss)
                a_loss_avg += float(a_loss)
                q_mean_avg += float(q_mean)
                update_counter += 1

            # Anneal exploration noise
            noise_std = max(EXPL_NOISE_MIN, noise_std * EXPL_NOISE_DECAY)

        # ── Curriculum update ────────────────────────────────────────────────
        if recent_suc:
            suc_pct = float(np.mean(recent_suc) * 100.0)
            dist_new, ghost_new, _, ms_new = get_continuous_curriculum(suc_pct)
            if abs(dist_new - dist) > 0.1 or ms_new != max_scen:
                dist, ghost, max_scen = dist_new, ghost_new, ms_new

        # ── Logging ──────────────────────────────────────────────────────────
        batch_num = global_steps // N_ENVS
        if batch_num % LOG_EVERY == 0 and update_counter > 0:
            c_l = c_loss_avg / (update_counter * G_UPDATES + 1e-8)
            a_l = a_loss_avg / (update_counter * G_UPDATES + 1e-8)
            q_m = q_mean_avg / (update_counter * G_UPDATES + 1e-8)
            suc_r = float(np.mean(recent_suc) * 100.0) if recent_suc else 0.0
            elapsed = time.time() - t0
            fps    = global_steps / elapsed

            print(
                f"  step={global_steps:>10,}  suc={suc_r:5.1f}%  "
                f"c_loss={c_l:.4f}  a_loss={a_l:.4f}  q={q_m:.3f}  "
                f"σ={noise_std:.4f}  dist={dist:.1f}  "
                f"fps={fps:.0f}"
            )
            csv_w.writerow([global_steps, f"{suc_r:.2f}", f"{0.0:.4f}",
                            f"{c_l:.4f}", f"{a_l:.4f}", f"{q_m:.4f}",
                            f"{noise_std:.4f}", f"{dist:.2f}", f"{ghost:.3f}"])
            csv_f.flush()

            c_loss_avg = a_loss_avg = q_mean_avg = 0.0
            update_counter = 0

            # Save best
            suc_float = suc_r
            if suc_float > best_suc:
                best_suc = suc_float
                _save(CKPT_BEST, ap, cp, ap_tgt, cp_tgt,
                      {"steps": global_steps, "best_suc": best_suc})

        if batch_num % SAVE_EVERY == 0:
            suc_r = float(np.mean(recent_suc) * 100.0) if recent_suc else 0.0
            _save(CKPT_FINAL, ap, cp, ap_tgt, cp_tgt,
                  {"steps": global_steps, "best_suc": max(best_suc, suc_r)})

    # ── Final save ────────────────────────────────────────────────────────────
    suc_r = float(np.mean(recent_suc) * 100.0) if recent_suc else 0.0
    _save(CKPT_FINAL, ap, cp, ap_tgt, cp_tgt,
          {"steps": global_steps, "best_suc": max(best_suc, suc_r)})
    csv_f.close()
    print(f"\nTraining complete.  Final suc={suc_r:.1f}%  best={best_suc:.1f}%")


if __name__ == "__main__":
    train()
