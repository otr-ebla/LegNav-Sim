"""
jax_ppo.py — Single-GPU PPO Training  (GPU 0), must train always with GPU, no CPU, never CPU

CHANGES vs previous version:

  GRU MEMORY SUPPORT:
    - Network init now passes a dummy `hidden` carry.
    - collect_rollouts receives and returns `hidden`; the live carry persists
      across rollout boundaries (true TBPTT over chunks).
    - ppo_loss_fn re-runs the GRU over the full T=ROLLOUT_STEPS sequence for
      each minibatch using jax.lax.scan, restoring the stored initial hidden.
    - Minibatching is now over the ENVIRONMENT axis only (N_MINIBATCHES
      splits NUM_ENVS), while the time axis is preserved intact. The old
      flat-batch reshape that destroyed temporal order has been removed.
    - run_ppo_updates, ppo_update_epoch, and ppo_train_chunk updated accordingly.
    - train_state now includes `hidden` as a third element so it flows through
      the jax.lax.scan inside ppo_train_chunk without host round-trips.

  UNCHANGED:
    - Curriculum, ghost-prob logic, reward normalisation, GAE, checkpointing.
    - All hyperparameters except where noted.
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
import jax
import jax.numpy as jnp
import functools

jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_network import EndToEndActorCritic, squash_corrected_log_prob, GRU_HIDDEN_SIZE
from jax_train import (
    collect_rollouts, init_env_state, rebuild_vmap_step,
    NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)


# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25

ENTROPY_COEF   = 0.02
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 2.5e-4
LR_END         = 1e-5
LR_MIN         = 1e-5
WARMUP_UPDATES = 5

TOTAL_UPDATES  = 800

# ── Minibatch geometry ────────────────────────────────────────────────────────
# We split over the ENVIRONMENT axis only.  Time (ROLLOUT_STEPS) is kept intact
# so the GRU can be unrolled over the full sequence during the update.
N_MINIBATCHES   = 64                                # must divide NUM_ENVS
assert NUM_ENVS % N_MINIBATCHES == 0, (
    f"NUM_ENVS={NUM_ENVS} not divisible by N_MINIBATCHES={N_MINIBATCHES}."
)
MINI_BATCH_ENVS = NUM_ENVS // N_MINIBATCHES         # 128 envs per minibatch

BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 524 288  (kept for FPS reporting)

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
    (50.0, 1.0),
    (65.0, 0.8),
    (78.0, 0.6),
    (101., 0.4),
]

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
        batch_mean = jnp.mean(x)
        batch_var  = jnp.var(x)
        batch_count = x.size
        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean  = self.mean + delta * batch_count / tot_count
        m_a  = self.var * self.count
        m_b  = batch_var * batch_count
        M2   = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
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


# ── Episode-outcome helpers (unchanged) ───────────────────────────────────────

@jax.jit
def collect_episode_outcomes_chunked(rewards, dones, goal_reached, collision, passive_col, active_col):
    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p, a = t
        ep_ret = ep_ret + r
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
        ep_ret = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_obs, out_acol, out_pcol, out_tmo, out_msk)

    N_ENVS   = rewards.shape[-1]
    flat_len = rewards.shape[0] * rewards.shape[1]
    r_flat = rewards.reshape(flat_len, N_ENVS)
    d_flat = dones.reshape(flat_len, N_ENVS)
    g_flat = goal_reached.reshape(flat_len, N_ENVS)
    c_flat = collision.reshape(flat_len, N_ENVS)
    p_flat = passive_col.reshape(flat_len, N_ENVS)
    a_flat = active_col.reshape(flat_len, N_ENVS)
    _, (ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N_ENVS),
        (r_flat, d_flat, g_flat, c_flat, p_flat, a_flat)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_obs.ravel(), ep_acol.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()


@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col, active_col):
    N = rewards.shape[1]

    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p, a = t
        ep_ret = ep_ret + r
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
        ep_ret = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_obs, out_acol, out_pcol, out_tmo, out_msk)

    _, (ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N),
        (rewards, dones, goal_reached, collision, passive_col, active_col)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_obs.ravel(), ep_acol.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()


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


# ── PPO loss with GRU unrolling ───────────────────────────────────────────────

@jax.jit
def ppo_loss_fn(
    params,
    obs_seq,          # (T, E_mb, OBS_SIZE)
    actions_seq,      # (T, E_mb, 2)
    advantages_seq,   # (T, E_mb)
    returns_seq,      # (T, E_mb)
    old_log_probs_seq,# (T, E_mb)
    max_v_seq,        # (T, E_mb)
    init_hidden,      # (E_mb, GRU_HIDDEN_SIZE) — stored from rollout
    dones_seq,        # (T, E_mb) — TBPTT episode-boundary masking
):
    """
    Re-runs the GRU over T steps for the minibatch of E_mb environments.
    Using jax.lax.scan over the time dimension ensures gradients flow through
    the full sequence (TBPTT with horizon T = ROLLOUT_STEPS).
    dones_seq masks the GRU carry at episode boundaries so the training
    forward pass matches what the policy saw during rollout collection.
    """
    T, E_mb = obs_seq.shape[:2]

    def _step(hidden, t):
        obs_t, actions_t, adv_t, ret_t, old_lp_t, max_v_t, done_t = t
        mean, logstd, values, new_hidden = network.apply(
            {"params": params}, obs_t, hidden
        )
        log_prob = squash_corrected_log_prob(actions_t, mean, logstd, max_v_t)
        ratio       = jnp.exp(jnp.clip(log_prob - old_lp_t, -5.0, 5.0))
        policy_loss = -jnp.mean(jnp.minimum(
            ratio * adv_t,
            jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t,
        ))
        value_loss   = VF_COEF * jnp.mean((ret_t - values) ** 2)
        entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
        entropy_loss = -ENTROPY_COEF * entropy
        step_loss = policy_loss + value_loss + entropy_loss

        # Mask GRU carry so gradients don't flow across episode boundaries.
        # This mirrors the identical masking applied in collect_rollouts,
        # ensuring the training forward pass matches what the policy saw.
        mask = (1.0 - done_t.astype(jnp.float32))[:, None]
        masked_hidden = new_hidden * mask

        return masked_hidden, (step_loss, policy_loss, value_loss, entropy)

    _, (step_losses, policy_losses, value_losses, entropies) = jax.lax.scan(
        _step,
        init_hidden,
        (obs_seq, actions_seq, advantages_seq, returns_seq, old_log_probs_seq, max_v_seq, dones_seq),
    )

    total_loss   = jnp.mean(step_losses)
    policy_loss  = jnp.mean(policy_losses)
    value_loss   = jnp.mean(value_losses)
    entropy_mean = jnp.mean(entropies)
    return total_loss, (policy_loss, value_loss, entropy_mean)


# ── Minibatch update (env-axis only, time axis intact) ────────────────────────

@jax.jit
def ppo_update_epoch(carry, env_perm):
    """
    env_perm: (NUM_ENVS,) — permutation over the environment axis for this epoch.
    Splits into N_MINIBATCHES contiguous chunks of MINI_BATCH_ENVS environments.
    """
    params, opt_state, obs_seq, actions_seq, adv_seq, ret_seq, old_lp_seq, max_v_seq, hiddens_seq, dones_seq = carry
    # Shapes: obs_seq (T, N, D), adv_seq (T, N), hiddens_seq (T, N, H), dones_seq (T, N)

    def _mb_step(mb_carry, mb_i):
        p, os_ = mb_carry
        # Slice contiguous env chunk along axis=1 (env axis)
        e_start = mb_i * MINI_BATCH_ENVS
        mb_obs     = jax.lax.dynamic_slice_in_dim(obs_seq,    e_start, MINI_BATCH_ENVS, axis=1)
        mb_actions = jax.lax.dynamic_slice_in_dim(actions_seq, e_start, MINI_BATCH_ENVS, axis=1)
        mb_adv     = jax.lax.dynamic_slice_in_dim(adv_seq,     e_start, MINI_BATCH_ENVS, axis=1)
        mb_ret     = jax.lax.dynamic_slice_in_dim(ret_seq,     e_start, MINI_BATCH_ENVS, axis=1)
        mb_old_lp  = jax.lax.dynamic_slice_in_dim(old_lp_seq,  e_start, MINI_BATCH_ENVS, axis=1)
        mb_max_v   = jax.lax.dynamic_slice_in_dim(max_v_seq,   e_start, MINI_BATCH_ENVS, axis=1)
        # hiddens_seq[0] = initial hidden for each env at t=0
        mb_init_h  = jax.lax.dynamic_slice_in_dim(hiddens_seq[0], e_start, MINI_BATCH_ENVS, axis=0)
        mb_dones   = jax.lax.dynamic_slice_in_dim(dones_seq,   e_start, MINI_BATCH_ENVS, axis=1)

        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, mb_obs, mb_actions, mb_adv, mb_ret, mb_old_lp, mb_max_v, mb_init_h, mb_dones
        )
        updates, new_os = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    # NOTE: env_perm is used to shuffle the environment ORDER before slicing.
    # We permute along the environment axis (axis=1 for time-first tensors).
    obs_perm     = obs_seq[:, env_perm]
    actions_perm = actions_seq[:, env_perm]
    adv_perm     = adv_seq[:, env_perm]
    ret_perm     = ret_seq[:, env_perm]
    old_lp_perm  = old_lp_seq[:, env_perm]
    max_v_perm   = max_v_seq[:, env_perm]
    hiddens_perm = hiddens_seq[:, env_perm]
    dones_perm   = dones_seq[:, env_perm]

    new_carry_inner = (params, opt_state)
    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step,
        new_carry_inner,
        jnp.arange(N_MINIBATCHES),
    )

    new_carry = (new_p, new_os,
                 obs_perm, actions_perm, adv_perm, ret_perm,
                 old_lp_perm, max_v_perm, hiddens_perm, dones_perm)
    return new_carry, (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_seq, actions_seq, adv_seq, ret_seq,
                    old_lp_seq, max_v_seq, hiddens_seq, dones_seq, rng_key):
    """
    obs_seq:      (T, N, OBS_SIZE)  — time-first layout, environments second
    hiddens_seq:  (T, N, GRU_H)    — GRU input hidden at each step
    dones_seq:    (T, N)            — episode boundary mask for TBPTT
    All other tensors: (T, N, ...) or (T, N)
    """
    params, opt_state = train_state

    # Normalise advantages over the full (T, N) batch
    adv_flat = adv_seq.reshape(-1)
    adv_seq  = (adv_seq - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # One permutation per epoch, over the environment axis
    env_perms = jax.vmap(lambda k: jax.random.permutation(k, NUM_ENVS))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )

    carry = (params, opt_state,
             obs_seq, actions_seq, adv_seq, ret_seq,
             old_lp_seq, max_v_seq, hiddens_seq, dones_seq)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, env_perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


# ── Checkpoint helpers (unchanged) ────────────────────────────────────────────

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

LOG_EVERY = 1   # steps between console prints (kept at 1 for direct loop)

if __name__ == "__main__":
    print(f"PPO Training — GPU {args.gpu}  [GRU Memory Mode]")
    print(f"  Envs       : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches: {N_MINIBATCHES} x {MINI_BATCH_ENVS} envs  (time dim preserved)")
    print(f"  GRU hidden : {GRU_HIDDEN_SIZE}")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}")
    print(f"  Curriculum stages: {CURRICULUM_STAGES}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # Network init — must pass dummy hidden
    dummy_obs    = jnp.zeros((1, OBS_SIZE))
    dummy_hidden = jnp.zeros((1, GRU_HIDDEN_SIZE))
    params       = network.init(init_rng, dummy_obs, dummy_hidden)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    ckpt_path       = "checkpoints/ppo_gru_best.msgpack"
    final_ckpt_path = "checkpoints/ppo_gru_final.msgpack"

    # ── Curriculum state ──────────────────────────────────────────────────────
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

    # Live GRU carry — persists across rollout chunks (true TBPTT)
    hidden = EndToEndActorCritic.initialize_carry(NUM_ENVS)

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

        # collect_rollouts now returns new_hidden as well
        rollout_history, env_state, env_obs, hidden, last_val = collect_rollouts(
            rollout_rng, train_state[0], network.apply, vmap_step,
            env_state, env_obs, hidden, cur_max_dist, cur_scenario
        )

        raw_rewards  = rollout_history["rewards"]
        values       = rollout_history["values"]
        dones        = rollout_history["dones"]

        rewards, running_ret, rms_state = normalize_batch_rewards(
            raw_rewards, dones, running_ret, rms_state, GAMMA
        )

        # rollout tensors: shape (T, N, ...) — time-first from lax.scan
        obs_seq      = rollout_history["obs"]         # (T, N, OBS_SIZE)
        acts_seq     = rollout_history["actions"]     # (T, N, 2)
        lp_seq       = rollout_history["log_probs"]   # (T, N)
        max_v_seq    = rollout_history["max_v"]       # (T, N)
        hiddens_seq  = rollout_history["hiddens"]     # (T, N, GRU_H)
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

        # ── Curriculum update ─────────────────────────────────────────────────
        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

        new_max_dist = curriculum_max_goal_dist(highest_rolling_suc)
        new_stage    = _curriculum_stage(highest_rolling_suc)
        new_ghost    = curriculum_ghost_prob(highest_rolling_suc)

        # if highest_rolling_suc < 35.0:
        #     new_scenario = 0
        # elif highest_rolling_suc < 50.0:
        #     new_scenario = 1
        # elif highest_rolling_suc < 60.0:
        #     new_scenario = 2
        # else:
        #     new_scenario = -1

        # The agent must first master long-distance navigation (> 6.5m) in open space (scenario 0)
        # before being subjected to complex social structures with hardcoded 8m+ goals.
        if new_max_dist < 8.0:
            new_scenario = 0
        else:
            # Once it can travel far, train on ALL scenarios simultaneously (-1)
            # to force generalization and prevent catastrophic forgetting.
            new_scenario = -1

        if new_max_dist > cur_max_dist or new_ghost < cur_ghost or new_scenario != cur_scenario:
            cur_max_dist = new_max_dist
            cur_stage    = new_stage
            cur_scenario = new_scenario

            if new_ghost < cur_ghost:
                cur_ghost = new_ghost
                vmap_step = rebuild_vmap_step(cur_ghost)
                print(f"  -> Ghost closure rebuilt: ghost_prob={cur_ghost:.1f} (env_state preserved)")
            else:
                print(f"  -> Curriculum advanced: stage={cur_stage}, dist={cur_max_dist:.1f}m, "
                      f"scenario={cur_scenario}")

        advantages, returns = compute_gae(rewards, values, dones, last_val)

        # run_ppo_updates receives time-first (T, N, ...) tensors
        train_state, mean_loss, aux = run_ppo_updates(
            train_state,
            obs_seq,
            acts_seq,
            advantages,
            returns,
            lp_seq,
            max_v_seq,
            hiddens_seq,
            dones,          # episode-boundary mask for TBPTT
            update_rng,
        )

        fps = BATCH_SIZE / (time.time() - t0)

        if update % 5 == 0:
            p_loss, v_loss, entropy = aux
            lr_now      = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
            elapsedtime = (time.time() - t_start) / 60.0
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
                f"{float(v_loss):>6.2f} {float(entropy):>6.2f} | "
                f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                f"{cur_stage:>5d} {cur_max_dist:>5.1f}m {cur_ghost:>5.1f}g {elapsedtime:>5.1f}min"
            )

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    save_checkpoint(train_state[0], train_state[1], final_ckpt_path)