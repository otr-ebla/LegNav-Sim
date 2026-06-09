"""
train_navrep.py — PPO-train NavRep's Controller (C) with V + M FROZEN.

Architecture (faithful NavRep, Dugas et al. ICRA 2021):
  V : LidarEncoder (VAE)      — loaded from JOINT pretraining, frozen
  M : causal Transformer       — loaded from JOINT pretraining, frozen
  C : 2-layer MLP [64, 64]    — trained here with PPO

Only the controller is optimised. Each PPO update:
  1. The frozen V+M run once over the rollout observations to produce the
     feature vector feat = [z_t, h_t, s_r] (navrep_extract_features applies
     stop_gradient to z_t / h_t).
  2. PPO updates the controller (NavRepControllerOnly) on those features.
V and M parameters never receive gradients, exactly as in canonical NavRep
(the "unsupervised representations are frozen, only the controller learns" setup
that defines the method). The full {V, M, C} params are saved so eval scripts
can reconstruct NavRepActorCritic and run obs → action end-to-end.

Pretraining step (must be run first):
    python comparison_policies/pretrain_navrep.py
    → checkpoints_navrep/navrep_vm.msgpack   (encoder + decoder + M)

Saves:
    checkpoints_navrep/navrep_best.msgpack
    checkpoints_navrep/navrep_final.msgpack
    checkpoints_navrep/navrep_training_log.csv

Usage:
    cd src/jax_env
    python comparison_policies/train_navrep.py [--steps N] \\
        [--vm-ckpt checkpoints_navrep/navrep_vm.msgpack]
"""

import os
import csv
import sys
from legnav import paths

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import time
import warnings
import jax
import jax.numpy as jnp
import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Use TF32 (tensor-float32) for matmuls — free ~2× throughput on Ampere+.
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# Persistent XLA compilation cache — avoids full recompile on every restart.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_navrep_cache")

# ── Shared infrastructure (identical to ppo_mlp_baseline.py) ──────────────────
from legnav.core.jax_network import squash_corrected_log_prob, LOG_STD_MIN, LOG_STD_MAX
from legnav.algorithms.jax_train import (
    collect_rollouts, init_env_state, rebuild_vmap_step,
    NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)
from legnav.algorithms.jax_ppo import (
    get_continuous_curriculum,
    RunningMeanStd,
    normalize_batch_rewards,
    collect_episode_outcomes,
    compute_gae,
    _SUC_ANCHORS, _DIST_ANCHORS, _GHOST_ANCHORS, _ENT_ANCHORS, _SCEN_ANCHORS,
)
from legnav.baselines.navrep_network import (
    NavRepActorCritic,
    NavRepControllerOnly,
    navrep_extract_features,
    FEAT_DIM,
    _VM_KEYS,
)

# ── Hyperparameters (identical to ppo_mlp_baseline.py) ────────────────────────
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

BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS
N_MINIBATCHES   = 8
assert BATCH_SIZE % N_MINIBATCHES == 0
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES

_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE

CKPT_DIR   = os.path.join(_JAX_ENV_DIR, paths.checkpoint("navrep"))
CKPT_BEST  = os.path.join(CKPT_DIR, "navrep_best.msgpack")
CKPT_FINAL = os.path.join(CKPT_DIR, "navrep_final.msgpack")
VM_CKPT    = os.path.join(CKPT_DIR, "navrep_vm.msgpack")
LOG_PATH   = os.path.join(CKPT_DIR, "navrep_training_log.csv")

# Full actor-critic (freeze_vm=True → V+M frozen via stop_gradient) is used for
# rollout action selection. The controller-only module is used in the PPO loss,
# operating on the frozen [z, h, s_r] features.
network    = NavRepActorCritic(action_dim=2, hidden_dim=64, freeze_vm=True)
controller = NavRepControllerOnly(action_dim=2, hidden_dim=64)

scheduler = None
optimizer = None


def _split_ctrl(params):
    """Controller subset of the full params dict (everything except V + M)."""
    return {k: v for k, v in params.items() if k not in _VM_KEYS}


def _merge_ctrl(full_params, ctrl_params):
    """Reassemble full params from frozen V+M plus updated controller."""
    merged = dict(full_params)
    merged.update(ctrl_params)
    return merged


def _load_vm_into(params, vm_path=VM_CKPT):
    """Overwrite params['encoder'] and params['M'] with the pretrained bundle.

    The JOINT checkpoint holds {encoder, decoder, M}; only encoder and M are
    needed by the actor-critic (the decoder is a pretraining-only artefact).
    """
    with open(vm_path, "rb") as f:
        raw = f.read()
    loaded = flax.serialization.msgpack_restore(raw)
    new_params = dict(params)
    new_params["encoder"] = flax.serialization.from_state_dict(
        params["encoder"], loaded["encoder"])
    new_params["M"] = flax.serialization.from_state_dict(
        params["M"], loaded["M"])
    return new_params


# ── Frozen feature extraction (V + M run once per rollout) ────────────────────

@jax.jit
def extract_features(params, obs_seq):
    """feat = [z_t, h_t, s_r] for every obs; stop_gradient on z_t / h_t.

    Mapped over the ROLLOUT_STEPS axis (one NUM_ENVS-sized slice at a time) so
    the frozen V+M never run over the whole flattened batch at once — running
    the 4-conv VAE + 8-block Transformer on all ROLLOUT_STEPS*NUM_ENVS obs in a
    single call exhausts GPU memory.
    """
    return jax.lax.map(lambda o: navrep_extract_features(params, o), obs_seq)


# ── PPO loss (controller-only, operates on frozen features) ───────────────────

@jax.jit
def ppo_loss_fn(ctrl_params, feat_mb, actions_mb, advantages_mb, returns_mb,
                old_log_probs, max_v_mb, entropy_coef):
    """PPO loss for the controller C on pre-extracted V+M features."""
    mean, logstd, values = controller.apply({"params": ctrl_params}, feat_mb)

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


@jax.jit
def run_ppo_updates(train_state, feat_seq, actions_seq, adv_seq, ret_seq,
                    old_lp_seq, max_v_seq, rng_key, entropy_coef):
    """Run PPO_EPOCHS × N_MINIBATCHES controller gradient steps.

    Only the controller params are optimised; the frozen V+M live in
    full_params and are passed through unchanged.
    """
    full_params, opt_state = train_state
    ctrl_params = _split_ctrl(full_params)
    TN = BATCH_SIZE

    feat_flat    = feat_seq.reshape(TN, FEAT_DIM)
    actions_flat = actions_seq.reshape(TN, -1)
    max_v_flat   = max_v_seq.reshape(TN)
    old_lp_flat  = old_lp_seq.reshape(TN)
    adv_flat     = adv_seq.reshape(TN)
    adv_flat     = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat     = ret_seq.reshape(TN)

    # ── One permutation per epoch ────────────────────────────────────────────
    perms = jax.vmap(lambda k: jax.random.permutation(k, TN))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )

    def _epoch(epoch_carry, perm):
        p, os_ = epoch_carry
        feat_p    = feat_flat[perm]
        actions_p = actions_flat[perm]
        adv_p     = adv_flat[perm]
        ret_p     = ret_flat[perm]
        old_lp_p  = old_lp_flat[perm]
        max_v_p   = max_v_flat[perm]

        def _mb(mb_carry, mb_i):
            p2, os2 = mb_carry
            s          = mb_i * MINI_BATCH_SIZE
            mb_feat    = jax.lax.dynamic_slice_in_dim(feat_p,    s, MINI_BATCH_SIZE, 0)
            mb_actions = jax.lax.dynamic_slice_in_dim(actions_p, s, MINI_BATCH_SIZE, 0)
            mb_adv     = jax.lax.dynamic_slice_in_dim(adv_p,     s, MINI_BATCH_SIZE, 0)
            mb_ret     = jax.lax.dynamic_slice_in_dim(ret_p,     s, MINI_BATCH_SIZE, 0)
            mb_old_lp  = jax.lax.dynamic_slice_in_dim(old_lp_p,  s, MINI_BATCH_SIZE, 0)
            mb_max_v   = jax.lax.dynamic_slice_in_dim(max_v_p,   s, MINI_BATCH_SIZE, 0)

            (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
                p2, mb_feat, mb_actions, mb_adv, mb_ret, mb_old_lp, mb_max_v, entropy_coef
            )
            updates, new_os2 = optimizer.update(grads, os2, p2)
            return (optax.apply_updates(p2, updates), new_os2), (loss, aux)

        (new_p, new_os), (losses, auxes) = jax.lax.scan(
            _mb, (p, os_), jnp.arange(N_MINIBATCHES)
        )
        return (new_p, new_os), (losses, auxes)

    (new_ctrl, new_opt_state), (all_losses, all_auxes) = jax.lax.scan(
        _epoch, (ctrl_params, opt_state), perms
    )

    new_full = _merge_ctrl(full_params, new_ctrl)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (new_full, new_opt_state), all_losses.mean(), last_aux


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

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


# ── Main training loop ─────────────────────────────────────────────────────────

LOG_EVERY = 1


def train(total_env_steps: int = DEFAULT_TOTAL_ENV_STEPS,
          vm_ckpt_path: str = VM_CKPT):
    """
    NavRep PPO training with vectorised JAX envs.

    Loads V + M weights from `vm_ckpt_path` and trains ONLY the controller (C);
    V and M are frozen via stop_gradient in the forward pass.

    Checkpoints:   checkpoints_navrep/navrep_best.msgpack
                   checkpoints_navrep/navrep_final.msgpack
    CSV log:       checkpoints_navrep/navrep_training_log.csv
    """
    global optimizer, scheduler

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
        schedules=[warmup_sched, decay_sched], boundaries=[_WARMUP_OPT_STEPS],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(learning_rate=scheduler, eps=1e-5),
    )

    print("PPO Training  [NavRep: V(VAE) + M(Transformer) FROZEN, C(MLP[64,64]) trained]")
    print(f"  V+M ckpt    : {vm_ckpt_path}")
    print(f"  Envs        : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches : {N_MINIBATCHES} x {MINI_BATCH_SIZE}  (flat T*N)")
    print(f"  Budget      : {total_env_steps:,} env steps  →  {total_updates} updates\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    if not os.path.isfile(vm_ckpt_path):
        raise FileNotFoundError(
            f"V+M pretraining checkpoint not found at '{vm_ckpt_path}'. "
            f"Run pretrain_navrep.py first."
        )
    params = _load_vm_into(params, vm_ckpt_path)
    print(f"  V + M       : loaded from {vm_ckpt_path} (FROZEN)")

    n_total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    ctrl_params = _split_ctrl(params)
    n_c = sum(x.size for x in jax.tree_util.tree_leaves(ctrl_params))
    print(f"  Parameters  : total={n_total:,}  controller(trained)={n_c:,}\n")

    # Optimizer tracks ONLY the controller params.
    opt_state   = optimizer.init(ctrl_params)
    train_state = (params, opt_state)

    cur_max_dist, cur_ghost, cur_ent, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc         = 0.0
    highest_rolling_suc = 0.0

    print(f"Curriculum: max_goal_dist={cur_max_dist:.1f} m, ghost_prob={cur_ghost:.1f}, "
          f"max_scenario={cur_max_scen}")
    print("Initialising environments...")

    env_obs, env_state, vmap_step = init_env_state(env_rng, ghost_prob=cur_ghost)

    rms_state   = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)

    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 55.0
    os.makedirs(CKPT_DIR, exist_ok=True)
    _log_file   = open(LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "acol_pct", "pcol_pct", "tmo_pct"])

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} "
           f"{'Pcol%':>5} {'Tmo%':>5} | {'Loss':>7} {'pi':>6} {'V':>6} "
           f"{'H':>6} {'KL':>6} {'ClpF':>5} | {'FPS':>7} {'#Ep':>6} {'LR':>8} | "
           f"{'MaxDist':>7} {'Ghost':>6} {'Ent':>7} {'ScenMax':>7} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    # ── Pre-dispatch the first rollout before the loop ────────────────────────
    rng, _init_rng = jax.random.split(rng)
    _pending_rollout = collect_rollouts(
        _init_rng, train_state[0], network.apply, vmap_step,
        env_state, env_obs, cur_max_dist, jnp.int32(-1), cur_ghost,
        jnp.int32(cur_max_scen),
    )

    try:
        for update in range(total_updates):
            t0 = time.time()

            rng, next_rollout_rng, update_rng = jax.random.split(rng, 3)

            # ── Unwrap pre-dispatched rollout (JAX futures, no sync yet) ─────
            rollout_history, env_state, env_obs, last_val = _pending_rollout

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

            advantages, returns = compute_gae(rewards, values, dones, last_val)

            # ── Frozen V+M → features, then controller-only PPO ──────────────
            feat_seq = extract_features(train_state[0], obs_seq)
            train_state, loss, aux = run_ppo_updates(
                train_state, feat_seq, acts_seq, advantages, returns,
                lp_seq, max_v_seq, update_rng, jnp.array(cur_ent),
            )

            # ── Pre-dispatch NEXT rollout immediately (GPU pipeline stays full)
            _pending_rollout = collect_rollouts(
                next_rollout_rng, train_state[0], network.apply, vmap_step,
                env_state, env_obs, cur_max_dist, jnp.int32(-1), cur_ghost,
                jnp.int32(cur_max_scen),
            )

            # ── Host sync — GPU is now executing the next rollout ─────────────
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

            pi_loss, v_loss, entropy, kl_div, clip_frac = aux
            elapsed = time.time() - t0
            fps     = int(BATCH_SIZE / elapsed)
            lr_now  = float(scheduler(update * _OPT_STEPS_PER_UPDATE))

            total_steps = (update + 1) * BATCH_SIZE
            total_hrs   = (time.time() - t_start) / 3600.0

            # ── Rolling success (EMA) ────────────────────────────────────────
            if n_ep > 0:
                rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            new_max_dist, new_ghost, new_ent, new_max_scen = \
                get_continuous_curriculum(rolling_suc)

            if new_ghost != cur_ghost:
                vmap_step = rebuild_vmap_step(new_ghost)
            cur_max_dist, cur_ghost, cur_ent, cur_max_scen = \
                new_max_dist, new_ghost, new_ent, new_max_scen

            # ── Checkpoint on best ───────────────────────────────────────────
            if suc_pct > best_suc and n_ep >= 10:
                best_suc = suc_pct
                highest_rolling_suc = max(highest_rolling_suc, rolling_suc)
                save_checkpoint(train_state[0], train_state[1], CKPT_BEST)

            # ── Logging ──────────────────────────────────────────────────────
            if update % LOG_EVERY == 0:
                _log_writer.writerow([total_steps, f"{mean_ret:.2f}", f"{suc_pct:.1f}",
                                      f"{acol_pct:.1f}", f"{pcol_pct:.1f}", f"{tmo_pct:.1f}"])
                _log_file.flush()

                print(
                    f"{update+1:>5} | {mean_ret:>7.2f} | {suc_pct:>5.1f} {obs_pct:>5.1f} "
                    f"{acol_pct:>5.1f} {pcol_pct:>5.1f} {tmo_pct:>5.1f} | "
                    f"{float(loss):>7.4f} {float(pi_loss):>6.4f} {float(v_loss):>6.4f} "
                    f"{float(entropy):>6.3f} {float(kl_div):>6.4f} {float(clip_frac):>5.3f} | "
                    f"{fps:>7} {n_ep:>6} {lr_now:>8.2e} | "
                    f"{cur_max_dist:>7.1f} {cur_ghost:>6.2f} {cur_ent:>7.4f} "
                    f"{cur_max_scen:>7} {total_hrs:>7.2f}h"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        _log_file.close()

    save_checkpoint(train_state[0], train_state[1], CKPT_FINAL)
    print(f"Done. Best suc%={best_suc:.1f}  →  {CKPT_BEST}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_ENV_STEPS,
                        help="Total environment steps (default 100M)")
    parser.add_argument("--vm-ckpt", type=str, default=VM_CKPT,
                        help="Pretrained V+M checkpoint (navrep_vm.msgpack)")
    args = parser.parse_args()
    train(args.steps, args.vm_ckpt)
