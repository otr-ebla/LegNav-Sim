"""
jax_ppo.py — Single-GPU PPO Training (GPU 0) — STATELESS / MAX SPEED

CHANGES vs GRU version:

  REMOVED GRU — FLAT PPO:
    - hidden removed from train_state, collect_rollouts, ppo_loss_fn.
    - ppo_loss_fn now does a single forward pass on (T*N, OBS_SIZE) instead
      of a sequential lax.scan over T steps. This is critical for speed:
      the loss is now fully parallelizable on the GPU.
    - hiddens_seq and dones_seq removed from run_ppo_updates / ppo_update_epoch.
    - collect_rollouts no longer returns hidden or last_hidden.

  BUGFIX — entropy coefficient:
    ENTROPY_COEF raised from 0.02 to 0.05 to counter premature policy collapse.
    With the flat loss, the entropy gradient is now correct.

  MINIBATCH GEOMETRY:
    With flat loss, the minibatch is over (T*N) independent samples.
    N_MINIBATCHES = 8 -> MINI_BATCH_SIZE = T*N/8 = 8192 samples per minibatch.
    Large batch + parallel forward = maximum GPU occupancy.

  UNCHANGED:
    - Curriculum, ghost-prob, reward normalization, GAE, checkpointing.
    - All hyperparameters not mentioned above.
"""

import os
import csv
import argparse

parser = argparse.ArgumentParser(description="JAX PPO Training")
parser.add_argument("--gpu", type=str, default="0", choices=["0", "1"])
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import warnings
import random as _random
import jax
import jax.numpy as jnp
import functools

jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_network import EndToEndActorCritic, squash_corrected_log_prob
from jax_train import (
    collect_rollouts, init_env_state, rebuild_vmap_step,
    NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)


# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25
ENTROPY_COEF   = 0.015   # initial value (suc=0) — continuously interpolated by curriculum
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 2.5e-4
LR_END         = 1e-5
LR_MIN         = 1e-5
WARMUP_UPDATES = 5

TOTAL_UPDATES  = 800

# ── Minibatch geometry ────────────────────────────────────────────────────────
# Flat loss over (T*N) samples. Shuffle full batch then split into minibatches.
BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 65_536
N_MINIBATCHES   = 8
assert BATCH_SIZE % N_MINIBATCHES == 0
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES       # 8_192

_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE
_TOTAL_OPT_STEPS      = TOTAL_UPDATES  * _OPT_STEPS_PER_UPDATE

network = EndToEndActorCritic(action_dim=2)

# ── Continuous Curriculum ─────────────────────────────────────────────────────
# Instead of discrete jumps causing domain shock, we continuously interpolate
# environment parameters based on the highest rolling success rate.
_SUC_ANCHORS  = np.array([0.0, 25.0, 38.0, 50.0, 60.0, 70.0, 82.0, 95.0, 100.0])
_DIST_ANCHORS = np.array([1.5,  1.5,  2.5,  4.0,  5.0,  6.5,  8.0,  9.0,   9.0])
_GHOST_ANCHORS= np.array([0.0,  0.0,  0.0,  0.0,  0.15, 0.3,  0.6,  0.9,   1.0])
_ENT_ANCHORS  = np.array([
    0.02, 0.02, 0.018, 0.015, 0.012,
    0.01, 0.008, 0.006, 0.005
])
# Progressively unlock scenarios [0: Random, 1: Parallel, 2: Perp, 3: Circle, 4: Bottleneck, 5: Intersect, 6: Static]
_SCEN_ANCHORS = np.array([0,    0,    1,    2,    4,    5,    6,    6,     6])

def get_continuous_curriculum(suc_pct: float):
    dist  = float(np.interp(suc_pct, _SUC_ANCHORS, _DIST_ANCHORS))
    ghost = float(np.interp(suc_pct, _SUC_ANCHORS, _GHOST_ANCHORS))
    ent   = float(np.interp(suc_pct, _SUC_ANCHORS, _ENT_ANCHORS))
    max_s = int(np.interp(suc_pct, _SUC_ANCHORS, _SCEN_ANCHORS))
    return dist, ghost, ent, max_s

from flax import struct

@struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray
    var:  jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls):
        return cls(mean=jnp.array(0.0), var=jnp.array(1.0), count=jnp.array(1e-4))

    def update(self, x: jnp.ndarray):
        batch_mean  = jnp.mean(x)
        batch_var   = jnp.var(x)
        batch_count = x.size
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        new_mean    = self.mean + delta * batch_count / tot_count
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        M2          = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var     = M2 / tot_count
        return self.replace(mean=new_mean, var=new_var, count=tot_count)


@jax.jit
def normalize_batch_rewards(rewards, dones, running_ret, rms_state, gamma):
    # FIX Bug#2: prima si stimava la varianza dei return cumulativi (scala ~1/(1-γ)≈100)
    # e poi si dividevano i reward istantanei per quella std → reward sottoscalati di ~10x,
    # segnale quasi piatto, policy stagnante.
    # Fix: si normalizzano i reward istantanei con la running std dei reward istantanei.
    # running_ret viene mantenuto per compatibilità firma ma non usato.
    new_rms_state = rms_state.update(rewards.flatten())
    normalized_rewards = rewards / jnp.sqrt(new_rms_state.var + 1e-8)
    normalized_rewards = jnp.clip(normalized_rewards, -10.0, 10.0)
    return normalized_rewards, running_ret, new_rms_state


_warmup_schedule = optax.linear_schedule(
    init_value=LR_MIN, end_value=LR_START, transition_steps=_WARMUP_OPT_STEPS,
)
_decay_schedule = optax.linear_schedule(
    init_value=LR_START, end_value=LR_END,
    transition_steps=_TOTAL_OPT_STEPS - _WARMUP_OPT_STEPS,
)
scheduler = optax.join_schedules(
    schedules=[_warmup_schedule, _decay_schedule],
    boundaries=[_WARMUP_OPT_STEPS],
)

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(learning_rate=scheduler, eps=1e-5),
)


# ── Episode-outcome helpers (invariati) ───────────────────────────────────────

@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col, active_col):
    N = rewards.shape[1]

    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p, a = t
        ep_ret  = ep_ret + r
        is_suc  = g
        is_acol = a & ~is_suc
        is_pcol = p & ~is_suc
        is_obs  = c & ~a & ~p & ~is_suc
        is_tmo  = d & ~is_suc & ~c
        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_obs  = jnp.where(d, is_obs.astype(jnp.float32),  0.0)
        out_acol = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        out_msk  = d.astype(jnp.float32)
        ep_ret   = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_obs, out_acol, out_pcol, out_tmo, out_msk)

    _, (ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N),
        (rewards, dones, goal_reached, collision, passive_col, active_col)
    )
    return (ep_rets.ravel(), ep_suc.ravel(), ep_obs.ravel(),
            ep_acol.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel())


@jax.jit
def compute_gae(rewards, values, dones, last_val):
    def _step(carry, t):
        gae, nv = carry
        r, v, d = t
        nd    = 1.0 - d
        delta = r + GAMMA * nv * nd - v
        gae   = delta + GAMMA * GAE_LAMBDA * nd * gae
        return (gae, v), gae

    _, adv = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_val), last_val),
        (rewards, values, dones.astype(jnp.float32)),
        reverse=True,
    )
    returns = adv + values
    return adv, returns


# ── PPO loss — forward pass PIATTO su (T*N) sample ────────────────────────────

@jax.jit
def ppo_loss_fn(
    params,
    obs_mb,         # (MB, OBS_SIZE)
    actions_mb,     # (MB, 2)
    advantages_mb,  # (MB,)
    returns_mb,     # (MB,)
    old_log_probs,  # (MB,)
    max_v_mb,       # (MB,)
    entropy_coef,   # () — JAX scalar, varies continuously with curriculum
):
    """
    Parallel forward pass over MB = MINI_BATCH_SIZE samples.
    No lax.scan: fully vectorized in a single GPU kernel.
    """
    mean, logstd, values = network.apply({"params": params}, obs_mb)

    log_prob    = squash_corrected_log_prob(actions_mb, mean, logstd, max_v_mb)
    ratio       = jnp.exp(log_prob - old_log_probs)
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages_mb,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_mb,
    ))
    value_loss   = VF_COEF * jnp.mean((returns_mb - values) ** 2)
    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -entropy_coef * entropy
    total_loss   = policy_loss + value_loss + entropy_loss

    kl_div    = jnp.mean(old_log_probs - log_prob)   # approx reverse KL
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32))

    return total_loss, (policy_loss, value_loss, entropy, kl_div, clip_frac)


# ── Minibatch update — shuffle over (T*N), split into N_MINIBATCHES chunks ────

@jax.jit
def ppo_update_epoch(carry, perm):
    """
    perm: (BATCH_SIZE,) — permutation over all T*N samples.
    """
    params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat, max_v_flat, entropy_coef = carry

    # Apply permutation
    obs_p     = obs_flat[perm]
    actions_p = actions_flat[perm]
    adv_p     = adv_flat[perm]
    ret_p     = ret_flat[perm]
    old_lp_p  = old_lp_flat[perm]
    max_v_p   = max_v_flat[perm]

    def _mb_step(mb_carry, mb_i):
        p, os_ = mb_carry
        s = mb_i * MINI_BATCH_SIZE
        mb_obs     = jax.lax.dynamic_slice_in_dim(obs_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_actions = jax.lax.dynamic_slice_in_dim(actions_p, s, MINI_BATCH_SIZE, axis=0)
        mb_adv     = jax.lax.dynamic_slice_in_dim(adv_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_ret     = jax.lax.dynamic_slice_in_dim(ret_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_old_lp  = jax.lax.dynamic_slice_in_dim(old_lp_p,  s, MINI_BATCH_SIZE, axis=0)
        mb_max_v   = jax.lax.dynamic_slice_in_dim(max_v_p,   s, MINI_BATCH_SIZE, axis=0)

        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, mb_obs, mb_actions, mb_adv, mb_ret, mb_old_lp, mb_max_v, entropy_coef
        )
        updates, new_os = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step,
        (params, opt_state),
        jnp.arange(N_MINIBATCHES),
    )

    new_carry = (new_p, new_os, obs_p, actions_p, adv_p, ret_p, old_lp_p, max_v_p, entropy_coef)
    return new_carry, (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_seq, actions_seq, adv_seq, ret_seq,
                    old_lp_seq, max_v_seq, rng_key, entropy_coef):
    """
    obs_seq: (T, N, OBS_SIZE) — reshaped to (T*N, OBS_SIZE) for flat loss.
    entropy_coef: JAX scalar with the current curriculum entropy value.
    """
    params, opt_state = train_state

    # Flatten time × envs
    TN = BATCH_SIZE
    obs_flat     = obs_seq.reshape(TN, OBS_SIZE)
    actions_flat = actions_seq.reshape(TN, -1)
    max_v_flat   = max_v_seq.reshape(TN)
    old_lp_flat  = old_lp_seq.reshape(TN)

    # Normalizza advantages sull'intero batch
    adv_flat = adv_seq.reshape(TN)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = ret_seq.reshape(TN)

    # One permutation per epoch over all T*N samples
    perms = jax.vmap(lambda k: jax.random.permutation(k, TN))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )

    carry = (params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat, max_v_flat, entropy_coef)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


# ── Checkpoint helpers (invariati) ────────────────────────────────────────────

def save_checkpoint(params, opt_state, filepath="checkpoints/ppo_model_best.msgpack"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {"params": jax.device_get(params), "opt_state": jax.device_get(opt_state)}
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Checkpoint -> {filepath}")


def load_checkpoint(dummy_params, dummy_opt_state,
                    filepath="checkpoints/ppo_model_best.msgpack"):
    with open(filepath, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes(
        {"params": dummy_params, "opt_state": dummy_opt_state}, raw
    )
    return bundle["params"], bundle["opt_state"]


# ── Main training loop ────────────────────────────────────────────────────────

LOG_EVERY = 1

if __name__ == "__main__":
    print(f"PPO Training — GPU {args.gpu}  [Stateless / Flat Loss / Frame-Stack Attention]")
    print(f"  Envs        : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches : {N_MINIBATCHES} x {MINI_BATCH_SIZE} sample  (flat T*N)")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}")
    print(f"  Continuous curriculum anchors: suc={list(_SUC_ANCHORS)}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # Network init stateless
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    ckpt_path       = "checkpoints/ppo_attn_best.msgpack"
    final_ckpt_path = "checkpoints/ppo_attn_final.msgpack"

    # Curriculum state
    cur_max_dist, cur_ghost, cur_ent, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc  = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    print(f"Curriculum: max_goal_dist={cur_max_dist:.1f} m, "
          f"ghost_prob={cur_ghost:.1f}, max_scenario={cur_max_scen}")

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(env_rng, ghost_prob=cur_ghost)

    rms_state   = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)

    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 55.0  # NEVER TOUCH THIS LINE

    _LOG_PATH = "checkpoints/ppo_training_log.csv"
    _log_file = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "acol_pct", "pcol_pct", "tmo_pct"])

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} {'KL':>6} {'ClpF':>5} | {'FPS':>7} {'#Ep':>6} {'LR':>8} | "
           f"{'MaxDist':>7} {'Ghost':>6} {'Ent':>7} {'ScenMax':>7} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        rng, rollout_rng, update_rng = jax.random.split(rng, 3)

        # collect_rollouts stateless: no hidden return
        # collect_rollouts stateless: no hidden return
        rollout_history, env_state, env_obs, last_val = collect_rollouts(
            rollout_rng, train_state[0], network.apply, vmap_step,
            env_state, env_obs, cur_max_dist, cur_scenario, cur_ghost
        )

        raw_rewards = rollout_history["rewards"]
        values      = rollout_history["values"]
        dones       = rollout_history["dones"]

        rewards, running_ret, rms_state = normalize_batch_rewards(
            raw_rewards, dones, running_ret, rms_state, GAMMA
        )

        obs_seq   = rollout_history["obs"]
        acts_seq  = rollout_history["actions"]
        lp_seq    = rollout_history["log_probs"]
        max_v_seq = rollout_history["max_v"]
        goal_reached = rollout_history["goal_reached"]
        collision    = rollout_history["collision"]
        passive_col  = rollout_history["passive_col"]
        active_col   = rollout_history["active_col"]

        ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(
            raw_rewards, dones, goal_reached, collision, passive_col, active_col
        )

        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            obs_pct  = float((ep_obs  * ep_msk).sum() / n_ep) * 100.0
            acol_pct = float((ep_acol * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret, suc_pct, obs_pct, acol_pct, pcol_pct, tmo_pct = 0., 0., 0., 0., 0., 0.

        # Curriculum
        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            highest_rolling_suc = 0.99 * highest_rolling_suc + 0.01 * rolling_suc

        new_max_dist, new_ghost, new_ent, new_max_scen = get_continuous_curriculum(highest_rolling_suc)

        # Progressively sample from the unlocked scenarios to avoid layout shock
        new_scenario = _random.randint(0, new_max_scen)

        if new_max_dist > cur_max_dist or new_ghost > cur_ghost or new_max_scen > cur_max_scen:
            max_step = 0.2  # metri per update — impedisce shock da salto di distanza
            cur_max_dist = min(cur_max_dist + max_step, new_max_dist)
            cur_ghost    = new_ghost
            cur_max_scen = new_max_scen
            print(f"  -> Curriculum smoothly advanced: dist={cur_max_dist:.1f}m, "
                  f"ghost_prob={cur_ghost:.2f}, unlocked_scenarios=0-{cur_max_scen}")

        cur_scenario = new_scenario
        entropy_coef = jnp.array(new_ent)

        advantages, returns = compute_gae(rewards, values, dones, last_val)

        # PPO update — stateless (no hidden state)
        train_state, mean_loss, aux = run_ppo_updates(
            train_state,
            obs_seq,
            acts_seq,
            advantages,
            returns,
            lp_seq,
            max_v_seq,
            update_rng,
            entropy_coef,
        )

        fps = BATCH_SIZE / (time.time() - t0)

        if update % 5 == 0:
            p_loss, v_loss, entropy, kl_div, clip_frac = aux
            lr_now       = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
            ent_coef_now = float(entropy_coef)
            elapsedtime  = (time.time() - t_start) / 60.0
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
                f"{float(v_loss):>6.2f} {float(entropy):>6.2f} "
                f"{float(kl_div):>6.4f} {float(clip_frac):>4.2f} | "
                f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                f"{cur_max_dist:>6.1f}m {cur_ghost:>5.2f}g "
                f"{ent_coef_now:>6.4f}e scen<=={cur_max_scen} {elapsedtime:>5.1f}min"
            )

        if n_ep > 0:
            _log_writer.writerow([update * BATCH_SIZE, round(mean_ret, 4),
                                   round(suc_pct, 2), round(acol_pct, 2),
                                   round(pcol_pct, 2), round(tmo_pct, 2)])
            _log_file.flush()

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    save_checkpoint(train_state[0], train_state[1], final_ckpt_path)