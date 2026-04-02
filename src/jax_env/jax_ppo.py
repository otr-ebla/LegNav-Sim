"""
jax_ppo.py — Single-GPU PPO Training (GPU 0) — STATELESS / MAX SPEED

CHANGES vs GRU version:

  GRU RIMOSSO — PPO PIATTO:
    - hidden rimosso da train_state, collect_rollouts, ppo_loss_fn.
    - ppo_loss_fn ora fa un singolo forward pass su (T*N, OBS_SIZE) invece
      di un lax.scan sequenziale su T step. Questo è il cambiamento critico
      per la velocità: il loss è ora completamente parallelizzabile su GPU.
    - hiddens_seq e dones_seq rimossi da run_ppo_updates / ppo_update_epoch.
    - collect_rollouts non restituisce più hidden o last_hidden.

  BUGFIX — hidden mismatch permutazione:
    (rimosso con il GRU — il bug non esiste più)

  BUGFIX — entropy coefficient:
    ENTROPY_COEF alzato da 0.02 a 0.05 per contrastare il collasso prematuro
    della policy. Con il loss piatto il gradiente entropico è ora corretto.

  MINIBATCH GEOMETRY:
    Con loss piatto il minibatch è su (T*N) sample indipendenti.
    N_MINIBATCHES = 8 → MINI_BATCH_SIZE = T*N/8 = 8192 sample per minibatch.
    Batch grande + forward parallelo = massima occupazione GPU.

  INVARIATI:
    - Curriculum, ghost-prob, reward normalisation, GAE, checkpointing.
    - Tutti gli iperparametri non menzionati sopra.
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
ENTROPY_COEF   = 0.015   # valore stage 0 — usato solo per il print iniziale
# Decay per stage: esplorazione calibrata su ogni livello del curriculum
# Stage:                 0      1      2      3      4      5      6
ENTROPY_COEF_BY_STAGE = [0.015, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001]
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 2.5e-4
LR_END         = 1e-5
LR_MIN         = 1e-5
WARMUP_UPDATES = 5

TOTAL_UPDATES  = 10000

# ── Minibatch geometry ────────────────────────────────────────────────────────
# Loss piatto su (T*N) sample. Shuffle su tutto il batch poi split in minibatch.
BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 65_536
N_MINIBATCHES   = 8
assert BATCH_SIZE % N_MINIBATCHES == 0
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES       # 8_192

_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE
_TOTAL_OPT_STEPS      = TOTAL_UPDATES  * _OPT_STEPS_PER_UPDATE

network = EndToEndActorCritic(action_dim=2)

# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM_STAGES = [
    (25.0, 1.5),
    (38.0, 2.5),
    (50.0, 4.0),
    (60.0, 5.0),
    (70.0, 6.5),
    (80.0, 8.0),
    (101., 9.0),
]

GHOST_PROB_STAGES = [
    # ghost_prob basso → robot visibile agli umani → umani lo evitano → più facile
    # ghost_prob alto  → robot invisibile agli umani → umani non collaborano → più difficile
    # Il curriculum aumenta la difficoltà man mano che il robot migliora.
    # Stage:           suc <  threshold  → ghost_prob
    (50.0, 0.0),   # stage 0-2: robot sempre visibile (umani cedono il passo)
    (70.0, 0.3),   # stage 3:   robot visibile 70% degli episodi
    (82.0, 0.6),   # stage 4-5: mix bilanciato
    (101., 1.0),   # stage 6:   robot sempre invisibile (comportamento naturale umani)
]

# Probabilità di usare scenario random (-1) vs scenario 0 fisso, per stage.
# Stage 0-1: solo scenario 0 (robot impara a navigare senza struttura)
# Stage 2+:  mix crescente — il robot inizia a vedere scenari strutturati
# Stage:                  0     1     2     3     4     5     6
SCENARIO_RANDOM_PROB = [0.0,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0]
# Probabilità 0.0 → scenario_idx=0 fisso
# Probabilità 1.0 → scenario_idx=-1 (tutti i 7 scenari in modo uniforme)

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
    def _step(ret, t):
        r, d = t
        ret = r + gamma * ret * (1.0 - d)
        return ret, ret
    running_ret, returns = jax.lax.scan(_step, running_ret, (rewards, dones))
    new_rms_state = rms_state.update(returns.flatten())
    normalized_rewards = rewards / jnp.sqrt(new_rms_state.var + 1e-8)
    normalized_rewards = jnp.clip(normalized_rewards, -10.0, 10.0)
    return normalized_rewards, running_ret, new_rms_state


def curriculum_ghost_prob(suc_pct: float) -> float:
    for threshold, prob in GHOST_PROB_STAGES:
        if suc_pct < threshold:
            return prob
    return GHOST_PROB_STAGES[-1][1]

def curriculum_max_goal_dist(suc_pct: float) -> float:
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
    entropy_coef,   # () — scalare JAX, varia per stage curriculum
):
    """
    Forward pass parallelo su MB = MINI_BATCH_SIZE sample.
    Nessun lax.scan: tutto vectorizzato in un colpo solo sulla GPU.
    """
    mean, logstd, values = network.apply({"params": params}, obs_mb)

    log_prob    = squash_corrected_log_prob(actions_mb, mean, logstd, max_v_mb)
    ratio       = jnp.exp(jnp.clip(log_prob - old_log_probs, -5.0, 5.0))
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages_mb,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_mb,
    ))
    value_loss   = VF_COEF * jnp.mean((returns_mb - values) ** 2)
    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -entropy_coef * entropy
    total_loss   = policy_loss + value_loss + entropy_loss

    return total_loss, (policy_loss, value_loss, entropy)


# ── Minibatch update — shuffle su (T*N), split in N_MINIBATCHES chunk ─────────

@jax.jit
def ppo_update_epoch(carry, perm):
    """
    perm: (BATCH_SIZE,) — permutazione su tutti i T*N sample.
    """
    params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat, max_v_flat, entropy_coef = carry

    # Applica permutazione
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
    obs_seq: (T, N, OBS_SIZE) — reshapato a (T*N, OBS_SIZE) per il loss piatto.
    entropy_coef: scalare JAX con il valore dello stage corrente.
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

    # Una permutazione per epoca su tutti i T*N sample
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
    print(f"  Curriculum stages: {CURRICULUM_STAGES}\n")

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
    cur_max_dist = curriculum_max_goal_dist(0.0)
    cur_stage    = _curriculum_stage(0.0)
    cur_ghost    = curriculum_ghost_prob(0.0)
    rolling_suc  = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    print(f"Curriculum: starting stage {cur_stage}, max_goal_dist={cur_max_dist:.1f} m, "
          f"ghost_prob={cur_ghost:.1f}, scenario={cur_scenario}")

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(env_rng, ghost_prob=cur_ghost)

    rms_state   = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)

    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 55.0  # NEVER TOUCH THIS LINE

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} | {'FPS':>7} {'#Ep':>6} {'LR':>6}  | "
           f"{'Stage':>5} {'MaxDist':>7} {'Ghost':>6} {'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        rng, rollout_rng, update_rng = jax.random.split(rng, 3)

        # collect_rollouts stateless: non restituisce più hidden
        rollout_history, env_state, env_obs, last_val = collect_rollouts(
            rollout_rng, train_state[0], network.apply, vmap_step,
            env_state, env_obs, cur_max_dist, cur_scenario
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
            highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

        new_max_dist = curriculum_max_goal_dist(highest_rolling_suc)
        new_stage    = _curriculum_stage(highest_rolling_suc)
        new_ghost    = curriculum_ghost_prob(highest_rolling_suc)

        # Scenario per-update: campionato Python-side in base alla probabilità dello stage corrente.
        # SCENARIO_RANDOM_PROB[stage] = prob di usare -1 (random tra tutti i 7 scenari).
        # Prob complementare → scenario 0 fisso (random statico, più semplice).
        _scen_rand_prob = SCENARIO_RANDOM_PROB[new_stage]
        new_scenario = -1 if (_random.random() < _scen_rand_prob) else 0

        if new_max_dist > cur_max_dist or new_ghost > cur_ghost or new_stage != cur_stage:
            cur_max_dist = new_max_dist
            cur_stage    = new_stage

            if new_ghost > cur_ghost:
                cur_ghost = new_ghost
                vmap_step = rebuild_vmap_step(cur_ghost)
                print(f"  -> Ghost closure rebuilt: ghost_prob={cur_ghost:.1f}")
            else:
                print(f"  -> Curriculum advanced: stage={cur_stage}, dist={cur_max_dist:.1f}m, "
                      f"scen_rand_prob={_scen_rand_prob:.0%}")

        cur_scenario = new_scenario

        # Entropy coefficient dallo stage corrente
        entropy_coef = jnp.array(ENTROPY_COEF_BY_STAGE[cur_stage])

        advantages, returns = compute_gae(rewards, values, dones, last_val)

        # PPO update senza hidden
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
            p_loss, v_loss, entropy = aux
            lr_now       = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
            ent_coef_now = float(entropy_coef)
            elapsedtime  = (time.time() - t_start) / 60.0
            scen_prob    = SCENARIO_RANDOM_PROB[cur_stage]
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
                f"{float(v_loss):>6.2f} {float(entropy):>6.2f} | "
                f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                f"{cur_stage:>5d} {cur_max_dist:>5.1f}m {cur_ghost:>5.1f}g "
                f"{ent_coef_now:.4f}e {scen_prob:.0%}s {elapsedtime:>5.1f}min"
            )

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    save_checkpoint(train_state[0], train_state[1], final_ckpt_path)