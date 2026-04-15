"""
ppo_mlp_baseline.py — PPO with a Vanilla MLP Backbone (2 × 128)
================================================================

Drop-in alternative to ``jax_ppo.py`` that replaces the complex
LidarCNN + FrameStackAttention architecture with a **plain 2-hidden-layer
MLP** (128 units each, ReLU activations).

Purpose
-------
Serves as an ablation / baseline for benchmarking:

  "How much of PPO's performance comes from the network architecture
   vs. the RL algorithm itself?"

Differences from ``jax_ppo.py``
--------------------------------
* **Network**: ``VanillaMLPActorCritic`` instead of ``EndToEndActorCritic``.
  The MLP receives the raw 662-dim stacked obs directly — no CNN, no
  frame-stack attention, no LiDAR-specific feature extraction.
* **Checkpoints**: saved under ``checkpoints_vanilla_ppo/``.
* **Log file**: ``checkpoints_vanilla_ppo/ppo_mlp_training_log.csv``.
* Everything else — curriculum, rollout collection, GAE, PPO clipping,
  minibatch shuffling, reward normalisation — is **identical** to the
  main PPO script.

Architecture
------------
::

    obs (662,)
      ↓ Dense(128) → ReLU
      ↓ Dense(128) → ReLU
      ├─ Dense(2)          → action_mean      (actor head)
      ├─ learnable log_std  → action_log_std   (actor head)
      └─ Dense(1)          → value            (critic head)

Usage
-----
Train::

    python ppo_mlp_baseline.py

Or from another script::

    from comparison_policies.ppo_mlp_baseline import train
    train(total_env_steps=50_000_000)

Evaluate (using the standard eval script, pointing to the new checkpoint)::

    python jax_eval.py --ckpt checkpoints_vanilla_ppo/ppo_mlp_best.msgpack
"""

import os
import csv
import sys

# ---------------------------------------------------------------------------
# Path setup (same logic used by dwa_planner.py / mppi_planner.py)
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# GPU / XLA settings (identical to jax_ppo.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import time
import warnings
import jax
import jax.numpy as jnp
import functools

# NOTE: jax.config.update("jax_default_device", cuda) is intentionally NOT
# called at module level so that this file can be safely imported from eval
# scripts that run on CPU (JAX_PLATFORMS=cpu). GPU pinning is done inside
# train() only.

import optax
import flax.linen as nn
import flax.serialization
import numpy as np
from flax.linen.initializers import orthogonal, constant
from flax import struct

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Shared infrastructure (unchanged from jax_ppo.py)
# ---------------------------------------------------------------------------
from jax_network import squash_corrected_log_prob, LOG_STD_MIN, LOG_STD_MAX
from jax_train import (
    collect_rollouts, init_env_state, rebuild_vmap_step,
    NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)
# Re-use curriculum helpers and reward normalisation directly from jax_ppo
from jax_ppo import (
    get_continuous_curriculum,
    RunningMeanStd,
    normalize_batch_rewards,
    collect_episode_outcomes,
    compute_gae,
    _SUC_ANCHORS, _DIST_ANCHORS, _GHOST_ANCHORS, _ENT_ANCHORS, _SCEN_ANCHORS,
)
# Network definition lives in its own lightweight module (safe to import on CPU)
from comparison_policies.vanilla_mlp_network import VanillaMLPActorCritic


# ── Hyperparameters (identical to jax_ppo.py) ──────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25
ENTROPY_COEF   = 0.015
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 2.5e-4
LR_END         = 1e-5
LR_MIN         = 1e-5
WARMUP_UPDATES = 5

DEFAULT_TOTAL_ENV_STEPS = 100_000_000

# ── Minibatch geometry (identical) ─────────────────────────────────────────
BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 65_536
N_MINIBATCHES   = 8
assert BATCH_SIZE % N_MINIBATCHES == 0
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES       # 8_192

_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE

# ── Checkpoint paths ────────────────────────────────────────────────────────
CKPT_DIR        = os.path.join(_JAX_ENV_DIR, "checkpoints_vanilla_ppo")
CKPT_BEST       = os.path.join(CKPT_DIR, "ppo_mlp_best.msgpack")
CKPT_FINAL      = os.path.join(CKPT_DIR, "ppo_mlp_final.msgpack")
LOG_PATH        = os.path.join(CKPT_DIR, "ppo_mlp_training_log.csv")


# Module-level singleton — referenced by the @jax.jit-compiled loss/update fns
network = VanillaMLPActorCritic(action_dim=2, hidden_dim=128)

# Mutable globals rebuilt inside train()
scheduler = None
optimizer = None


# ===========================================================================
# PPO loss (identical logic to jax_ppo.py — only `network` differs)
# ===========================================================================

@jax.jit
def ppo_loss_fn(
    params,
    obs_mb,         # (MB, OBS_SIZE)
    actions_mb,     # (MB, 2)
    advantages_mb,  # (MB,)
    returns_mb,     # (MB,)
    old_log_probs,  # (MB,)
    max_v_mb,       # (MB,)
    entropy_coef,   # () scalar
):
    """
    PPO clipped surrogate loss, evaluated on one minibatch.

    Uses the **vanilla MLP** instead of the CNN+Attention encoder.
    Everything else is identical to ``jax_ppo.ppo_loss_fn``.
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

    kl_div    = jnp.mean(old_log_probs - log_prob)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32))

    return total_loss, (policy_loss, value_loss, entropy, kl_div, clip_frac)


# ===========================================================================
# Minibatch update (identical to jax_ppo.py — references local `optimizer`)
# ===========================================================================

@jax.jit
def ppo_update_epoch(carry, perm):
    """One PPO epoch: shuffle full batch, split into minibatches, update."""
    params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, \
        old_lp_flat, max_v_flat, entropy_coef = carry

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

    new_carry = (new_p, new_os, obs_p, actions_p, adv_p, ret_p,
                 old_lp_p, max_v_p, entropy_coef)
    return new_carry, (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_seq, actions_seq, adv_seq, ret_seq,
                    old_lp_seq, max_v_seq, rng_key, entropy_coef):
    """Flatten T×N batch, run PPO_EPOCHS shuffle-and-update passes."""
    params, opt_state = train_state

    TN           = BATCH_SIZE
    obs_flat     = obs_seq.reshape(TN, OBS_SIZE)
    actions_flat = actions_seq.reshape(TN, -1)
    max_v_flat   = max_v_seq.reshape(TN)
    old_lp_flat  = old_lp_seq.reshape(TN)

    adv_flat = adv_seq.reshape(TN)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = ret_seq.reshape(TN)

    perms = jax.vmap(lambda k: jax.random.permutation(k, TN))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )

    carry = (params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat,
             old_lp_flat, max_v_flat, entropy_coef)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(params, opt_state, filepath=CKPT_BEST):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {"params": jax.device_get(params), "opt_state": jax.device_get(opt_state)}
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Checkpoint → {filepath}")


def load_checkpoint(dummy_params, dummy_opt_state, filepath=CKPT_BEST):
    with open(filepath, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes(
        {"params": dummy_params, "opt_state": dummy_opt_state}, raw
    )
    return bundle["params"], bundle["opt_state"]


# ===========================================================================
# Main training loop
# ===========================================================================

LOG_EVERY = 1


def train(total_env_steps: int = DEFAULT_TOTAL_ENV_STEPS):
    """
    PPO training with a vanilla 2×128 MLP backbone.

    Identical training loop to ``jax_ppo.train()``.  The only algorithmic
    difference is the neural network used for the actor-critic.

    Parameters
    ----------
    total_env_steps : int
        Total environment steps budget.  Default 100 M.

    Checkpoints
    -----------
    Best model  : ``checkpoints_vanilla_ppo/ppo_mlp_best.msgpack``
    Final model : ``checkpoints_vanilla_ppo/ppo_mlp_final.msgpack``
    CSV log     : ``checkpoints_vanilla_ppo/ppo_mlp_training_log.csv``
    """
    global optimizer, scheduler

    # Pin to GPU when running training (not needed during eval/import)
    cuda_devs = jax.devices("cuda")
    if cuda_devs:
        jax.config.update("jax_default_device", cuda_devs[0])

    total_updates   = max(1, int(total_env_steps) // BATCH_SIZE)
    total_opt_steps = total_updates * _OPT_STEPS_PER_UPDATE

    warmup_sched = optax.linear_schedule(
        init_value=LR_MIN, end_value=LR_START,
        transition_steps=_WARMUP_OPT_STEPS,
    )
    decay_sched = optax.linear_schedule(
        init_value=LR_START, end_value=LR_END,
        transition_steps=max(1, total_opt_steps - _WARMUP_OPT_STEPS),
    )
    scheduler = optax.join_schedules(
        schedules=[warmup_sched, decay_sched],
        boundaries=[_WARMUP_OPT_STEPS],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(learning_rate=scheduler, eps=1e-5),
    )

    print("PPO Training  [Vanilla MLP 2×128 — comparison baseline]")
    print(f"  Network     : Input({OBS_SIZE}) → Dense(128,ReLU) → Dense(128,ReLU) → Actor + Critic")
    print(f"  Envs        : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches : {N_MINIBATCHES} x {MINI_BATCH_SIZE} sample  (flat T*N)")
    print(f"  Budget      : {total_env_steps:,} env steps  →  {total_updates} updates")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}")
    print(f"  Continuous curriculum anchors: suc={list(_SUC_ANCHORS)}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # Initialise network parameters
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    # Log parameter count
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Parameters  : {n_params:,}\n")

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    # Curriculum state
    cur_max_dist, cur_ghost, cur_ent, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc         = 0.0
    highest_rolling_suc = 0.0

    print(f"Curriculum: max_goal_dist={cur_max_dist:.1f} m, "
          f"ghost_prob={cur_ghost:.1f}, max_scenario={cur_max_scen}")
    print("Initialising environments...")

    env_obs, env_state, vmap_step = init_env_state(env_rng, ghost_prob=cur_ghost)

    rms_state   = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)

    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 55.0   # mirror jax_ppo.py threshold — never touch

    os.makedirs(CKPT_DIR, exist_ok=True)
    _log_file   = open(LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct",
                           "acol_pct", "pcol_pct", "tmo_pct"])

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} "
           f"{'Pcol%':>5} {'Tmo%':>5} | {'Loss':>7} {'pi':>6} {'V':>6} "
           f"{'H':>6} {'KL':>6} {'ClpF':>5} | {'FPS':>7} {'#Ep':>6} {'LR':>8} | "
           f"{'MaxDist':>7} {'Ghost':>6} {'Ent':>7} {'ScenMax':>7} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    try:
        for update in range(total_updates):
            t0 = time.time()

            rng, rollout_rng, update_rng = jax.random.split(rng, 3)

            rollout_history, env_state, env_obs, last_val = collect_rollouts(
                rollout_rng, train_state[0], network.apply, vmap_step,
                env_state, env_obs, cur_max_dist, jnp.int32(-1), cur_ghost,
                jnp.int32(cur_max_scen),
            )

            raw_rewards = rollout_history["rewards"]
            values      = rollout_history["values"]
            dones       = rollout_history["dones"]

            rewards, running_ret, rms_state = normalize_batch_rewards(
                raw_rewards, dones, running_ret, rms_state, GAMMA
            )

            obs_seq      = rollout_history["obs"]
            acts_seq     = rollout_history["actions"]
            lp_seq       = rollout_history["log_probs"]
            max_v_seq    = rollout_history["max_v"]
            goal_reached = rollout_history["goal_reached"]
            collision    = rollout_history["collision"]
            passive_col  = rollout_history["passive_col"]
            active_col   = rollout_history["active_col"]

            ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk = \
                collect_episode_outcomes(
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
                mean_ret = suc_pct = obs_pct = acol_pct = pcol_pct = tmo_pct = 0.0

            # ── Curriculum (strictly monotonic) ──────────────────────────
            if n_ep > 0:
                rolling_suc         = 0.9 * rolling_suc + 0.1 * suc_pct
                highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

            new_max_dist, new_ghost, new_ent, new_max_scen = \
                get_continuous_curriculum(highest_rolling_suc)

            old_print_dist  = round(cur_max_dist, 1)
            old_print_ghost = round(cur_ghost, 2)
            old_print_scen  = cur_max_scen

            if new_max_dist > cur_max_dist:
                cur_max_dist = min(cur_max_dist + 0.2, new_max_dist)
            if new_ghost > cur_ghost:
                cur_ghost = new_ghost
            if new_max_scen > cur_max_scen:
                cur_max_scen = new_max_scen

            if (round(cur_max_dist, 1) > old_print_dist or
                    round(cur_ghost, 2) > old_print_ghost or
                    cur_max_scen > old_print_scen):
                print(f"  -> Curriculum advanced: dist={cur_max_dist:.1f}m, "
                      f"ghost_prob={cur_ghost:.2f}, unlocked_scenarios=0-{cur_max_scen}")

            cur_scenario = -1            # let env sample randomly
            entropy_coef = jnp.array(new_ent)

            advantages, returns = compute_gae(rewards, values, dones, last_val)

            train_state, mean_loss, aux = run_ppo_updates(
                train_state, obs_seq, acts_seq, advantages, returns,
                lp_seq, max_v_seq, update_rng, entropy_coef,
            )

            fps = BATCH_SIZE / (time.time() - t0)

            if update % LOG_EVERY == 0:
                p_loss, v_loss, entropy, kl_div, clip_frac = aux
                lr_now       = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
                ent_coef_now = float(entropy_coef)
                elapsed      = (time.time() - t_start) / 60.0
                print(
                    f"{update:>5d} | {mean_ret:>7.1f} | "
                    f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% "
                    f"{pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                    f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
                    f"{float(v_loss):>6.2f} {float(entropy):>6.2f} "
                    f"{float(kl_div):>6.4f} {float(clip_frac):>4.2f} | "
                    f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                    f"{cur_max_dist:>6.1f}m {cur_ghost:>5.2f}g "
                    f"{ent_coef_now:>6.4f}e scen<={cur_max_scen} {elapsed:>5.1f}min"
                )

            if n_ep > 0:
                _log_writer.writerow([
                    update * BATCH_SIZE, round(mean_ret, 4),
                    round(suc_pct, 2), round(acol_pct, 2),
                    round(pcol_pct, 2), round(tmo_pct, 2),
                ])
                _log_file.flush()

            # Save best checkpoint (same threshold as jax_ppo.py)
            curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
            if suc_pct > best_suc and n_ep > 0 and curriculum_mature:
                best_suc = suc_pct
                save_checkpoint(train_state[0], train_state[1], CKPT_BEST)

    finally:
        _log_file.close()

    print(f"Training log → {LOG_PATH}")
    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    save_checkpoint(train_state[0], train_state[1], CKPT_FINAL)


if __name__ == "__main__":
    train()
