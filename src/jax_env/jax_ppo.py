"""
jax_ppo.py — Single-GPU PPO Training  (GPU 0), must train always with GPU, no CPU, never CPU

"""

import os
import csv
import argparse

# Parse arguments BEFORE setting environment variables and importing JAX
parser = argparse.ArgumentParser(description="JAX PPO Training")
# Keep type=str because os.environ requires string values
parser.add_argument("--gpu", type=str, default="0", choices=["0", "1"], help="Target GPU ID (0 or 1)")
args, _ = parser.parse_known_args()

# ── GPU memory configuration ─────────────────────────────────────────────────
# Use the BFC allocator (default JAX behaviour) — NOT "platform".
# "platform" allocates per-buffer from the OS, bypasses JAX's memory pool,
# and causes fragmentation + CUDA_ILLEGAL_ADDRESS when the pool is exhausted.
# BFC pre-reserves a fraction of VRAM and manages it internally.
#
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 reserves 88% of VRAM (~8.8 GB on a
# 10 GB card), leaving ~1.2 GB for the CUDA driver, cuDNN workspace, and OS.
# Do NOT set XLA_PYTHON_CLIENT_ALLOCATOR — absence means BFC (the safe default).
os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
# cuda_malloc_async reduces driver-level fragmentation without changing the
# JAX allocator — safe to keep.
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import warnings
import jax
import jax.numpy as jnp

# ALWAYS index 0, regardless of which physical GPU you chose
jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_network import EndToEndActorCritic
from jax_train import collect_rollouts, init_env_state, NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25
# FIX: ENTROPY_COEF abbassato 0.01→0.003.
# Con reward scale -50/-10 e valore assoluto episodio ~50, il value loss domina.
# ENTROPY_COEF=0.01 era troppo alto: il gradient dell'entropia spingeva sempre
# verso logstd_max, bloccando l'entropia a 2.838 (il massimo con LOG_STD_MAX=0.0).
# Con 0.003 l'entropia può scendere naturalmente quando la policy trova azioni
# che massimizzano il reward, senza essere soffocata dall'entropy bonus.
ENTROPY_COEF   = 0.001
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 5e-4
LR_END         = 1e-4
LR_MIN         = 1e-5
WARMUP_UPDATES = 5
# TOTAL_UPDATES: 128 × 16384 × 96 = 201,326,592 ≈ 200M env steps
TOTAL_UPDATES  = 400

BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 8192 × 64 = 524,288
# N_MINIBATCHES=64: MBS = 524288/64 = 8192 — matches GPU L2 cache working set
# well on 10 GB cards. Previous 128 minibatches × 16384 envs = 12288 MBS,
# but 16384 envs was already OOM before minibatching.
N_MINIBATCHES   = 64
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES       # 8192

assert BATCH_SIZE % N_MINIBATCHES == 0, (
    f"BATCH_SIZE={BATCH_SIZE} not divisible by N_MINIBATCHES={N_MINIBATCHES}."
)

# Each outer update runs PPO_EPOCHS × N_MINIBATCHES optimizer steps inside
# jax.lax.scan, so the optax step counter advances by that factor per update.
# The scheduler must be expressed in true optimizer-step units, otherwise
# warmup exhausts in update 0 and LR collapses to LR_END by update 1-2.
_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES   # 6 × 64 = 384
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE   # 1_920
_TOTAL_OPT_STEPS      = TOTAL_UPDATES  * _OPT_STEPS_PER_UPDATE   # 49_152

network = EndToEndActorCritic(action_dim=2)

# ── Curriculum ────────────────────────────────────────────────────────────────
# Each entry: (suc_pct_threshold, min_goal_dist)
# FIX #3: soglie alzate per evitare curriculum prematuro.
# Con le vecchie soglie (10%, 20%, 35%...) la policy saltava allo stage
# successivo con success rate ancora troppo bassa (~11% → stage 1).
# Il robot non aveva consolidato il comportamento base prima di dover
# navigare distanze più lunghe, causando regressione immediata post-jump.
# Nuove soglie: ogni stage richiede ~2x il successo precedente;
# il primo stage (1.5m→2.5m) ora richiede 25% invece di 10%.
CURRICULUM_STAGES = [
    (25.0, 1.5),
    (38.0, 2.5),
    (50.0, 4.0),
    (60.0, 5.0),   # intermediate stage — old 4.0m→6.5m jump caused stall
    (70.0, 6.5),
    (80.0, 8.0),
    (101., 9.0),
]

GHOST_PROB_STAGES = [
    # FIX #3: allineate alle nuove soglie curriculum.
    # Il ghost inizia a degradare solo dopo che la policy è stabile (≥50%),
    # non a 35% come prima quando era ancora in fase di consolidamento.
    (50.0, 1.0),
    (65.0, 0.8),
    (78.0, 0.6),
    (101., 0.4),
]

def curriculum_ghost_prob(suc_pct: float) -> float:
    for threshold, prob in GHOST_PROB_STAGES:
        if suc_pct < threshold:
            return prob
    return GHOST_PROB_STAGES[-1][1]

def curriculum_min_goal_dist(suc_pct: float) -> float:
    for threshold, dist in CURRICULUM_STAGES:
        if suc_pct < threshold:
            return dist
    return CURRICULUM_STAGES[-1][1]

def _curriculum_stage(suc_pct: float) -> int:
    for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
        if suc_pct < threshold:
            return i
    return len(CURRICULUM_STAGES) - 1

_warmup_schedule = optax.linear_schedule(
    init_value=LR_MIN,
    end_value=LR_START,
    transition_steps=_WARMUP_OPT_STEPS,
)
_decay_schedule = optax.linear_schedule(
    init_value=LR_START,
    end_value=LR_END,
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


@jax.jit
def ppo_loss_fn(params, obs, actions, advantages, returns, old_log_probs):
    mean, logstd, values = network.apply({"params": params}, obs)
    std = jnp.exp(logstd)

    z        = (actions - mean) / (std + 1e-8)
    log_prob = jnp.sum(-0.5 * (z ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)

    ratio       = jnp.exp(jnp.clip(log_prob - old_log_probs, -5.0, 5.0))
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages,
    ))

    value_loss   = VF_COEF * jnp.mean((returns - values) ** 2)
    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -ENTROPY_COEF * entropy

    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss, (policy_loss, value_loss, entropy)


@jax.jit
def ppo_update_epoch(carry, perm):
    params, opt_state, obs, actions, adv, ret, old_lp = carry

    def _mb_step(mb_carry, mb_idx):
        p, os_ = mb_carry
        idx = jax.lax.dynamic_slice(perm, (mb_idx * MINI_BATCH_SIZE,), (MINI_BATCH_SIZE,))
        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, obs[idx], actions[idx], adv[idx], ret[idx], old_lp[idx]
        )
        updates, new_os = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step, (params, opt_state), jnp.arange(N_MINIBATCHES)
    )
    return (new_p, new_os, obs, actions, adv, ret, old_lp), (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_flat, actions_flat, adv_flat, ret_flat,
                    old_lp_flat, rng_key):
    params, opt_state = train_state

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    perms = jax.vmap(lambda k: jax.random.permutation(k, BATCH_SIZE))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )
    carry = (params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    """
    Build mutually exclusive episode outcome categories that sum to 100%.

    From jax_env:
      collision   = human_col | obs_col | wall_col  (includes passive)
      passive_col = human_col & robot_stopped & human_in_fov  (subset of collision)

    So active_col = collision & ~passive_col  (wall/obs/active human — robot's fault)
       passive_col = passive_col              (human walked into stopped robot)

    Priority (applied in order, each exclusive):
      1. success     : goal_reached
      2. active_col  : collision & ~passive_col & ~goal_reached
      3. passive_col : passive_col & ~goal_reached
      4. timeout     : done & ~goal_reached & ~collision
    """
    N = rewards.shape[1]

    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p = t
        ep_ret = ep_ret + r

        # Split collision into mutually exclusive sub-types first.
        # p (passive_col) is always a subset of c (collision), so:
        #   active = c & ~p  (wall/obs/moving-robot human collision)
        #   passive = p      (human walked into stationary robot)
        act_col = c & ~p

        # Now build strict priority masks (each episode falls in exactly one).
        is_suc  = g
        is_acol = act_col & ~is_suc
        is_pcol = p       & ~is_suc
        is_tmo  = d & ~is_suc & ~act_col & ~p

        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_col  = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        out_msk  = d.astype(jnp.float32)

        ep_ret = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_col, out_pcol, out_tmo, out_msk)

    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N),
        (rewards, dones, goal_reached, collision, passive_col)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()

if __name__ == "__main__":

    print(f"PPO Training — GPU {args.gpu}  [30-min mode]") # <-- Updated to use args.gpu    print(f"  Envs       : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches: {N_MINIBATCHES} x {MINI_BATCH_SIZE} | epochs {PPO_EPOCHS}")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}  LR warmup {LR_MIN}->{LR_START} then decay ->{LR_END}")
    print(f"  OBS_SIZE={OBS_SIZE}  log_std: state-dependent, bias=-1.0, clamp [{-4.0},{0.0}]")
    print(f"  Curriculum stages: {CURRICULUM_STAGES}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    ckpt_path       = "checkpoints/ppo_model_best.msgpack"
    LOAD_CHECKPOINT = False
    if LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        try:
            params, opt_state = load_checkpoint(params, opt_state, ckpt_path)
            train_state = (params, opt_state)
            print("Resumed from checkpoint.")
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh.")
    else:
        print("Starting fresh.")

    # ── Curriculum state ──────────────────────────────────────────────────────
    cur_min_dist = curriculum_min_goal_dist(0.0)
    cur_stage    = _curriculum_stage(0.0)
    cur_ghost    = curriculum_ghost_prob(0.0)   # ghost_prob for current stage
    rolling_suc  = 0.0   # FIX Issue #10: EMA alpha raised to 0.1 below

    print(f"Curriculum: starting stage {cur_stage}, min_goal_dist={cur_min_dist:.1f} m, ghost_prob={cur_ghost:.1f}")

    print("Initialising environments...")
    # FIX Bug #3: init_env_state returns vmap_step; thread it into collect_rollouts.
    env_obs, env_state, vmap_step = init_env_state(env_rng, min_goal_dist=cur_min_dist,
                                                    ghost_prob=cur_ghost)
    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 76.0   # FIX: was 65.0 — hardcoded floor meant no checkpoint was ever
                     # written when the run peaked at 61.8%. Now saves from the first
                     # improvement, then only on new highs.

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Col%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} | {'FPS':>7} {'#Ep':>6} {'LR':>6}| "
           f"{'Stage':>5} {'MinDist':>7} {'Ghost':>6} {'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    # ── Training log (CSV) — read by benchmark_eval.py for the curves panel ──
    _LOG_PATH = "checkpoints/ppo_training_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    t_start = time.time()

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        rng, rollout_rng, update_rng = jax.random.split(rng, 3)
        # FIX Bug #3: pass vmap_step explicitly as static arg.
        rollout_history, env_state, env_obs = collect_rollouts(
            rollout_rng, train_state[0], network.apply, vmap_step, env_state, env_obs
        )

        rewards      = rollout_history["rewards"]
        values       = rollout_history["values"]
        dones        = rollout_history["dones"]
        obs_all      = rollout_history["obs"]
        acts_all     = rollout_history["actions"]
        lp_all       = rollout_history["log_probs"]
        goal_reached = rollout_history["goal_reached"]
        collision    = rollout_history["collision"]
        passive_col  = rollout_history["passive_col"]  # <-- ADD THIS LINE

        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(
            rewards, dones, goal_reached, collision, passive_col
        )

        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0  # <-- CALCULATE PASSIVE COLLISION PCT
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret, suc_pct, col_pct, pcol_pct, tmo_pct = 0.0, 0.0, 0.0, 0.0, 0.0

        # ── Curriculum update ─────────────────────────────────────────────────
        # FIX Issue #10: alpha raised from 0.03 → 0.1 for faster stage transitions.
        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct

        new_min_dist = curriculum_min_goal_dist(rolling_suc)
        new_stage    = _curriculum_stage(rolling_suc)
        new_ghost    = curriculum_ghost_prob(rolling_suc)

        # Reinitialise envs if either goal distance OR ghost_prob changed.
        # ghost_prob change means make_stacked_env needs to be rebuilt with a new
        # closure — the vmap_step object changes, triggering JAX retrace.
        if new_min_dist > cur_min_dist or new_ghost < cur_ghost:
            cur_min_dist = new_min_dist
            cur_stage    = new_stage
            cur_ghost    = new_ghost

            rng, reinit_rng = jax.random.split(rng)
            # FIX Bug #3: capture new vmap_step; also passes updated ghost_prob.
            env_obs, env_state, vmap_step = init_env_state(reinit_rng,
                                                            min_goal_dist=cur_min_dist,
                                                            ghost_prob=cur_ghost)
            print(f"  → Curriculum reinit: stage={cur_stage}, dist={cur_min_dist:.1f}m, "
                  f"ghost_prob={cur_ghost:.1f}")

        _, _, last_val = network.apply({"params": train_state[0]}, env_obs)
        advantages, returns = compute_gae(rewards, values, dones, last_val)

        train_state, mean_loss, aux = run_ppo_updates(
            train_state,
            obs_all.reshape(-1, OBS_SIZE),
            acts_all.reshape(-1, 2),
            advantages.reshape(-1),
            returns.reshape(-1),
            lp_all.reshape(-1),
            update_rng
        )

        fps = BATCH_SIZE / (time.time() - t0)

        if update % 5 == 0:
            p_loss, v_loss, entropy = aux
            lr_now = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
            elapsedtime = (time.time() - t_start)/60.0
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.1f} {float(p_loss):>6.1f} "
                f"{float(v_loss):>6.1f} {float(entropy):>6.1f} | "
                f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                f"{cur_stage:>5d} {cur_min_dist:>5.1f}m {cur_ghost:>5.1f}g {elapsedtime:>5.1f}min"
            )
            # ── CSV log row — total_env_steps for aligned x-axis ───────────────
            total_env_steps = (update + 1) * NUM_ENVS * ROLLOUT_STEPS
            _log_writer.writerow([total_env_steps, round(mean_ret, 4),
                                   round(suc_pct, 4), round(col_pct, 4),
                                   round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
            _log_file.flush()

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")

    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")

    # Save the final model state regardless of its performance
    final_ckpt_path = "checkpoints/ppo_model_final.msgpack"
    save_checkpoint(train_state[0], train_state[1], final_ckpt_path)