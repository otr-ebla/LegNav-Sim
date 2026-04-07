"""
SACjax.py — Soft Actor-Critic (shared-encoder, fused-JIT)

Architecture:
  - ONE shared encoder (ep) updated through critic loss + scaled actor gradient.
  - Actor gradients flow into LidarCNN scaled by ACTOR_ENC_GRAD_SCALE=0.1 via
    custom_vjp — single CNN pass, zero redundant forwards.
  - Target LidarCNN (tlcp) & target Q branches (tq1p, tq2p) — EMA of online params.
  - Buffer stores terminal (done & ~timeout) — bootstraps through timeouts.
  - obs/next_obs stored as float16 — halves VRAM footprint.
  - extract_max_v called ONCE per collection step; stored in buffer.
  - 2 encoder passes per update: enc(obs) [trainable] + enc_tgt(next_obs) [frozen].
  - Actor Q-eval reuses feat from single actor pass — zero redundant CNN forwards.

"""

import os
import csv
os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import time
import functools
import warnings
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import flax.serialization

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_SIZE       = 662        # 9 (pose) + 5 (state_vec) + 648 (lidar)
ACTION_DIM     = 2
N_ENVS         = 256        # FIX 1: reduced from 2048; large N_ENVS drowns the buffer
                             #        faster than G updates can replay — overwrite before reuse.
BUFFER_CAP     = 2_000_000  # 2M — with N_ENVS=256: ~7800 steps to fill → plenty of mixing
BATCH_SIZE     = 512        # FIX 1: kept proportional to N_ENVS; smaller is fine for SAC
G_UPDATES      = 20         # FIX 1: gradient updates per collect step (replay ratio = G/1 ≫ 1)
                             #        SAC's sample efficiency comes from here — not from more envs.
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005 / G_UPDATES   # Compensate for G_UPDATES EMA applications per env step.
                             # Without this, effective tau per env step = 1-(1-0.005)^20 ≈ 0.095,
                             # turning the "slow anchor" target into a fast-moving target.
                             # TAU=0.00025 restores the intended per-env-step drift.
LR             = 3e-4
TARGET_ENTROPY = -2.0       # FIX 2: correct value is -|A| = -dim(action_space) = -2.0
                             #        -1.0 was LESS exploratory (demanded less entropy),
                             #        causing premature alpha decay and policy collapse.
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 50         # chunks; each chunk = N_ENVS*LOG_EVERY env steps collected
SAVE_EVERY     = 5000
MAX_GRAD_NORM  = 10.0

# pose_stack indices 0-8; state_vec[2] = max_v/2 at flat index 9+2 = 11
MAX_V_OBS_IDX  = 11
LOG_STD_EPS    = 1e-6

# ── Perception normalization ───────────────────────────────────────────────────
# LiDAR rays are in meters. Normalize to [0,1] before the first Conv to prevent
# the CNN from wasting early training steps rescaling weights for meter-scale inputs.
# VERIFY this matches the MAX_RANGE constant in jax_env.py before running.
LIDAR_MAX_RANGE = 10.0   # metres — set to your env's actual max sensor range

# ── Actor-encoder gradient coupling ───────────────────────────────────────────
# Scale factor applied to actor gradients flowing back into LidarCNN.
# 0.0 = full stop_gradient (old behaviour, critic-only encoder)
# 0.0 = full stop_gradient (standard off-policy practice for stable critic)
ACTOR_ENC_GRAD_SCALE = 0.0

CKPT_DIR  = "checkpoints_sac"
CKPT_PATH = f"{CKPT_DIR}/sac_best.msgpack"


# ── GPU check ─────────────────────────────────────────────────────────────────
def _check_gpu():
    num_devices = jax.device_count(backend='gpu')
    if num_devices >= 2:
        target_gpu = jax.devices('gpu')[1]
        print(f"Found {num_devices} GPU(s). Using CudaDevice(1)")
    elif num_devices == 1:
        target_gpu = jax.devices('gpu')[0]
        print(f"Found {num_devices} GPU(s). Using CudaDevice(0)")
    else:
        raise RuntimeError("No CUDA devices found.")
    return target_gpu

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)


# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

def init_env_state(rng_key, max_goal_dist: float = 3.0, scenario_idx: int = -1):
    """Build vmapped step/reset, initialise all envs. Returns (obs, state, vmap_step).
    vmap_step is NOT jit-wrapped — the outer train_chunk JIT fuses it.
    """
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    vmap_step = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))

    def _reset(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx)
    vmap_reset = jax.jit(jax.vmap(_reset))

    env_obs, env_state = vmap_reset(jax.random.split(rng_key, N_ENVS))
    return env_obs, env_state, vmap_step


# ── Shared Encoder + Actor/Critic Heads ──────────────────────────────────────
# SharedEncoder is imported from jax_network — identical trunk to PPO's
# EndToEndActorCritic (LidarFrameCNN + FrameStackAttention + MLP).
# This guarantees SAC, TQC and PPO use the same observation feature extractor
# for a fair algorithm comparison.

from jax_network import SharedEncoder

_orth_relu = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
_orth_out  = nn.initializers.orthogonal(scale=0.01)


class SACActorHead(nn.Module):
    """Policy head: 128-dim shared feat → (mean, log_std)."""
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0

    @nn.compact
    def __call__(self, feat: jnp.ndarray):
        mean    = nn.Dense(self.action_dim, kernel_init=_orth_out, name='mean')(feat)
        log_std = nn.Dense(self.action_dim, kernel_init=_orth_out, name='log_std')(feat)
        return mean, jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)


class CriticBranch(nn.Module):
    """Single Q-head: (feat_128, action) → scalar."""
    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray):
        x = jnp.concatenate([feat, action], axis=-1)
        q = nn.LayerNorm()(nn.relu(nn.Dense(256, kernel_init=_orth_relu)(x)))
        q = nn.LayerNorm()(nn.relu(nn.Dense(128, kernel_init=_orth_relu)(q)))
        return jnp.squeeze(nn.Dense(1)(q), axis=-1)


# Module instances (stateless — params stored separately)
shared_enc = SharedEncoder()
actor_head  = SACActorHead()
critic_q1   = CriticBranch()
critic_q2   = CriticBranch()


# ── Optimisers ────────────────────────────────────────────────────────────────
enc_opt        = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_actor_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_q1_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_q2_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
alpha_opt      = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))

# ── Action squashing + exact log-prob ─────────────────────────────────────────

def _tanh_log_prob_correction(tanh_u, max_v):
    corr_v = (
        jnp.log(max_v * 0.5 + LOG_STD_EPS)
        + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )
    corr_w = jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    return -(corr_v + corr_w)


def sample_action_sac_batched(rng_key, mean, log_std, max_v):
    """Fully batched reparameterised sample.
    mean, log_std : (N, 2)   max_v : (N,)
    Returns: env_action (N, 2), log_pi (N,)
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise**2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)
    tanh_u = jnp.tanh(u)
    a_v = (tanh_u[:, 0] + 1.0) * 0.5 * max_v
    a_w = tanh_u[:, 1]
    return jnp.stack([a_v, a_w], axis=-1), lp_gauss + _tanh_log_prob_correction(tanh_u, max_v)


def extract_max_v(obs):
    """Decode max_v from observation vector. Call ONCE per step."""
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular) ───────────────────────────────────────────
# VRAM budget: obs/next_obs stored as float16 to halve memory.
#   2M * 662 * 2B * 2 arrays = 5.30 GB (vs 10.59 GB with float32).
# buf_sample casts back to float32 before returning — encoder sees full precision.
# Field 'terminal' = done & ~timeout — only true environmental endings stored.
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),   jnp.float16),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),             jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),   jnp.float16),
        "terminal": jnp.zeros((capacity,),             jnp.float32),
        "max_v":    jnp.zeros((capacity,),             jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }


@jax.jit
def buf_add(buf, obs, action, reward, next_obs, terminal, max_v):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs":      buf["obs"].at[idxs].set(obs.astype(jnp.float16)),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs.astype(jnp.float16)),
        "terminal": buf["terminal"].at[idxs].set(terminal),
        "max_v":    buf["max_v"].at[idxs].set(max_v),
        "ptr":      jnp.int32((buf["ptr"] + N) % cap),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    idxs = jax.random.randint(rng_key, (batch_size,), 0, buf["size"])
    return (buf["obs"][idxs].astype(jnp.float32),
            buf["action"][idxs],
            buf["reward"][idxs],
            buf["next_obs"][idxs].astype(jnp.float32),
            buf["terminal"][idxs],
            buf["max_v"][idxs])


# ── SAC update step ───────────────────────────────────────────────────────────
# Param layout:
#   lcp  / lcos   — shared LidarCNN params + opt state
#   tlcp          — target LidarCNN (EMA of lcp)
#   ahp  / ahos   — actor head params + opt state
#   q1p  / q1os   — Q1 branch params + opt state
#   q2p  / q2os   — Q2 branch params + opt state  (INDEPENDENT of q1p → decorrelated)
#   tq1p / tq2p   — target Q-branch params (EMA of q1p / q2p)
#   la   / alo    — log alpha + opt state
#
# CNN passes per update:
#   lidar_cnn(obs)       — online, trainable via critic loss              [1]
#   lidar_cnn_tgt(next)  — target, stop_grad, for target Q backup        [1]
#   (actor reuses lcp with scale_gradient — same 2 passes, no extra fwd) [0]
#   Total: 2 CNN passes  (down from 3; custom_vjp handles backward scaling)
#
# Actor→encoder gradient coupling:
#   Critic pass: full gradient into lcp (critic shapes encoder)
#   Actor pass:  gradient scaled by ACTOR_ENC_GRAD_SCALE=0.1 into lcp
#   Net encoder gradient = c_grad_cnn + 0.1 * a_grad_cnn
#   This lets the actor nudge perception toward navigation-relevant features
#   without destabilising the critic's value representation.
# ── Gradient scaling via custom_vjp ──────────────────────────────────────────
# Scales the backward-pass gradient by `scale` while leaving the forward pass
# identical. Used to give the actor a soft 0.1x nudge into the shared encoder
# without running a separate (redundant) CNN forward+backward pass.
@jax.custom_vjp
def scale_gradient(x, scale):
    return x

def _scale_gradient_fwd(x, scale):
    return x, scale

def _scale_gradient_bwd(scale, g):
    return g * scale, None

scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)


@jax.jit
def sac_update(sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os,
               tq1p, tq2p, la, alo,
               obs, action, reward, next_obs, terminal, max_v_obs, max_v_next, rng_key):


    rng_c, rng_a = jax.random.split(rng_key)

    # ── 1. Critic loss — Bellman backup ──────────────────────────────────────
    def _critic_loss(sep_, q1p_, q2p_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))

        feat_next = jax.lax.stop_gradient(shared_enc.apply({"params": tsep}, next_obs))
        mean_n, lgs_n = actor_head.apply({"params": ahp}, feat_next)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, max_v_next)

        q1_t = critic_q1.apply({"params": tq1p}, feat_next, next_act)
        q2_t = critic_q2.apply({"params": tq2p}, feat_next, next_act)
        v_next = jnp.minimum(q1_t, q2_t) - alpha * next_lp
        backup = jax.lax.stop_gradient(reward + GAMMA * (1.0 - terminal) * v_next)

        feat_obs = shared_enc.apply({"params": sep_}, obs)
        q1 = critic_q1.apply({"params": q1p_}, feat_obs, action)
        q2 = critic_q2.apply({"params": q2p_}, feat_obs, action)
        return jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2), jnp.mean(backup)

    (c_loss, q_mean), (c_grads_enc, c_grads_q1, c_grads_q2) = jax.value_and_grad(
        _critic_loss, argnums=(0, 1, 2), has_aux=True
    )(sep, q1p, q2p)

    q1_upd, new_q1os = head_q1_opt.update(c_grads_q1, q1os, q1p)
    q2_upd, new_q2os = head_q2_opt.update(c_grads_q2, q2os, q2p)
    new_q1p = optax.apply_updates(q1p, q1_upd)
    new_q2p = optax.apply_updates(q2p, q2_upd)

    # ── 2. Actor loss — scaled backward gradient into encoder ────────────────
    def _actor_loss(ahp_, sep_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        feat_raw = shared_enc.apply({"params": sep_}, obs)
        feat_f   = scale_gradient(feat_raw, ACTOR_ENC_GRAD_SCALE)
        mean, log_std = actor_head.apply({"params": ahp_}, feat_f)
        action_new, log_pi = sample_action_sac_batched(rng_a, mean, log_std, max_v_obs)
        q1 = critic_q1.apply({"params": jax.lax.stop_gradient(new_q1p)}, feat_f, action_new)
        q2 = critic_q2.apply({"params": jax.lax.stop_gradient(new_q2p)}, feat_f, action_new)
        return jnp.mean(alpha * log_pi - jnp.minimum(q1, q2)), jnp.mean(log_pi)

    (a_loss, log_pi_mean), (a_grads_head, a_grads_enc_actor) = jax.value_and_grad(
        _actor_loss, argnums=(0, 1), has_aux=True
    )(ahp, sep)

    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # ── 3. Encoder update — critic + actor gradients ─────────────────────────
    combined_enc_grads = jax.tree_util.tree_map(
        lambda cg, ag: cg + ag, c_grads_enc, a_grads_enc_actor
    )
    enc_upd, new_eos = enc_opt.update(combined_enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, enc_upd)

    # ── 4. Alpha update ───────────────────────────────────────────────────────
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    new_la = optax.apply_updates(la, al_upd)

    # ── 5. Soft target update ─────────────────────────────────────────────────
    new_tsep = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, tsep, new_sep)
    new_tq1p = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, tq1p, new_q1p)
    new_tq2p = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, tq2p, new_q2p)

    metrics = {
        "critic_loss": c_loss,
        "actor_loss":  a_loss,
        "alpha":       jnp.exp(new_la),
        "log_pi":      log_pi_mean,
        "q_mean":      q_mean,
    }
    return (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
            new_q1p, new_q1os, new_q2p, new_q2os,
            new_tq1p, new_tq2p, new_la, new_alo, metrics)


# ── Collection step ───────────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ahp, env_state, env_obs, rng_key, vmap_step, max_goal_dist, scenario_idx):
    """Compute max_v ONCE, sample action via shared encoder + actor head, step env."""
    max_v = extract_max_v(env_obs)
    k_act, k_step = jax.random.split(rng_key)
    feat = shared_enc.apply({"params": sep}, env_obs)
    mean, log_std = actor_head.apply({"params": ahp}, feat)
    env_action, _ = sample_action_sac_batched(k_act, mean, log_std, max_v)
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx
    )

    # Scale reward down to keep Q-values within reasonable bounds for entropy tuning
    reward = reward * 0.01

    return new_obs, new_state, env_obs, env_action, reward, done, info, max_v


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(13,))
def train_chunk(sep, eos, tsep, ahp, ahos,
                q1p, q1os, q2p, q2os, tq1p, tq2p,
                la, alo, vmap_step,
                buf, es, eo, key,
                max_goal_dist, scenario_idx):

    def _loop_body(carry, _):
        (sep, eos, tsep, ahp, ahos,
         q1p, q1os, q2p, q2os, tq1p, tq2p,
         la, alo, buf, es, eo, key) = carry
        key, k_col, k_upd_base = jax.random.split(key, 3)

        new_eo, new_es, obs_b, env_a, rew, done, info, max_v_cur = collect_step(
            sep, ahp, es, eo, k_col, vmap_step, max_goal_dist, scenario_idx
        )

        terminal = done & ~info["timeout"]
        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo,
                          terminal.astype(jnp.float32), max_v_cur)

        def _update_step(update_carry, upd_idx):
            (sep_, eos_, tsep_, ahp_, ahos_,
             q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_) = update_carry
            k_samp = jax.random.fold_in(k_upd_base, upd_idx * 2)
            k_upd  = jax.random.fold_in(k_upd_base, upd_idx * 2 + 1)
            b_obs, b_act, b_rew, b_next, b_terminal, b_max_v = buf_sample(
                new_buf, k_samp, BATCH_SIZE
            )
            b_max_v_next = extract_max_v(b_next)
            (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
             new_q1p_, new_q1os_, new_q2p_, new_q2os_,
             new_tq1p_, new_tq2p_, new_la_, new_alo_, metrics_) = sac_update(
                sep_, eos_, tsep_, ahp_, ahos_,
                q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_,
                b_obs, b_act, b_rew, b_next, b_terminal, b_max_v, b_max_v_next, k_upd
            )
            return (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
                    new_q1p_, new_q1os_, new_q2p_, new_q2os_,
                    new_tq1p_, new_tq2p_, new_la_, new_alo_), metrics_

        update_carry_init = (sep, eos, tsep, ahp, ahos,
                             q1p, q1os, q2p, q2os, tq1p, tq2p, la, alo)
        (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
         new_q1p, new_q1os, new_q2p, new_q2os,
         new_tq1p, new_tq2p, new_la, new_alo), all_update_metrics = jax.lax.scan(
            _update_step, update_carry_init, jnp.arange(G_UPDATES)
        )
        metrics = jax.tree_util.tree_map(lambda x: x[-1], all_update_metrics)

        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        new_carry = (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
                     new_q1p, new_q1os, new_q2p, new_q2os, new_tq1p, new_tq2p,
                     new_la, new_alo, new_buf, new_es, new_eo, key)
        return new_carry, (step_data, metrics)

    carry = (sep, eos, tsep, ahp, ahos,
             q1p, q1os, q2p, q2os, tq1p, tq2p,
             la, alo, buf, es, eo, key)
    new_carry, (all_step_data, all_metrics) = jax.lax.scan(
        _loop_body, carry, None, length=LOG_EVERY
    )
    return new_carry, all_step_data, all_metrics


# ── On-GPU episode stats ───────────────────────────────────────────────────────
@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p = t
        ep_ret  = ep_ret + r
        act_col = c & ~p
        is_suc  = g
        is_acol = act_col & ~g
        is_pcol = p & ~g
        is_tmo  = d & ~g & ~act_col & ~p
        out_ret  = jnp.where(d, ep_ret,                       0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_col  = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        return jnp.where(d, 0.0, ep_ret), (out_ret, out_suc, out_col, out_pcol, out_tmo,
                                            d.astype(jnp.float32))

    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan,
        jnp.zeros(rewards.shape[1]),
        (rewards, dones, goal_reached, collision, passive_col),
    )
    return (ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(),
            ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel())


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                    eos, ahos, q1os, q2os, la, alo, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "enc_params":        jax.device_get(sep),
        "target_enc_params": jax.device_get(tsep),
        "actor_head_params": jax.device_get(ahp),
        "q1_branch_params":  jax.device_get(q1p),
        "q2_branch_params":  jax.device_get(q2p),
        "target_q1_params":  jax.device_get(tq1p),
        "target_q2_params":  jax.device_get(tq2p),
        "enc_opt":           jax.device_get(eos),
        "actor_head_opt":    jax.device_get(ahos),
        "q1_opt":            jax.device_get(q1os),
        "q2_opt":            jax.device_get(q2os),
        "log_alpha":         jax.device_get(la),
        "alpha_opt_state":   jax.device_get(alo),
        "step":              int(step),
    }
    with open(CKPT_PATH, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("SAC Training  (shared LidarCNN + decoupled Q1/Q2 branches)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}  G_UPDATES={G_UPDATES}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}")
    print(f"  LOG_EVERY={LOG_EVERY}  transitions/chunk={N_ENVS*LOG_EVERY:,}  grad_updates/chunk={LOG_EVERY*G_UPDATES}\n")

    master_key = jax.random.PRNGKey(7)
    master_key, k_init, k_env, k_warmup = jax.random.split(master_key, 4)
    train_rng  = jax.random.PRNGKey(42)

    dummy_obs = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    k_se, k_ah, k_q1, k_q2 = jax.random.split(k_init, 4)

    # Init shared encoder (same architecture as PPO's EndToEndActorCritic trunk)
    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)   # target encoder (EMA copy)

    # Init actor head — needs encoder forward pass to get 128-dim feature shape
    dummy_feat = shared_enc.apply({"params": sep}, dummy_obs)   # (2, 128)
    ahp  = actor_head.init(k_ah, dummy_feat)["params"]

    # Init independent Q1 and Q2 branches
    q1p  = critic_q1.init(k_q1, dummy_feat, dummy_act)["params"]
    q2p  = critic_q2.init(k_q2, dummy_feat, dummy_act)["params"]
    tq1p = jax.tree_util.tree_map(jnp.array, q1p)
    tq2p = jax.tree_util.tree_map(jnp.array, q2p)

    eos  = enc_opt.init(sep)
    ahos = head_actor_opt.init(ahp)
    q1os = head_q1_opt.init(q1p)
    q2os = head_q2_opt.init(q2p)
    # Alpha init: after /10 env scaling, rewards are O(10), Q-values O(100).
    # alpha=1.0 gives entropy term ~1.0*log_pi ~ -2.0, i.e. ~2% of Q — sane ratio.
    la   = jnp.array(0.0, dtype=jnp.float32)  # log(1.0) = 0.0 → alpha = 1.0
    alo  = alpha_opt.init(la)

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(k_env)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    best_suc    = 49.5
    best_ret    = -1e9

    # Initialize curriculum state for fair comparison with PPO
    from jax_ppo import get_continuous_curriculum
    cur_max_dist, _, _, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    # ── Warmup: fill buffer with random actions ───────────────────────────────
    print("Warming up buffer with random actions...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_act, k_step = jax.random.split(k_warmup, 3)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)
        rand_v     = jax.random.uniform(k_act, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_act, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = jax.jit(vmap_step)(
            step_keys, env_state, env_action, cur_max_dist, cur_scenario
        )
        terminal   = done & ~info["timeout"].astype(jnp.bool_)
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             new_obs, terminal.astype(jnp.float32), max_v)
        env_obs     = new_obs
        total_steps += N_ENVS
    print("Warmup done. JIT compiling train_chunk (this may take ~5 min — nested scan)...")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | "
           f"{'FPS':>7} | {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    _LOG_PATH = "checkpoints_sac/sac_training_log.csv"
    os.makedirs("checkpoints_sac", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    # Print every 10th chunk
    PRINT_EVERY_CHUNKS = 10

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()

        new_carry, all_step_data, all_metrics = train_chunk(
            sep, eos, tsep, ahp, ahos,
            q1p, q1os, q2p, q2os, tq1p, tq2p,
            la, alo, vmap_step,
            replay_buf, env_state, env_obs, train_rng,
            cur_max_dist, cur_scenario
        )
        (sep, eos, tsep, ahp, ahos,
         q1p, q1os, q2p, q2os, tq1p, tq2p,
         la, alo, replay_buf, env_state, env_obs, train_rng) = new_carry

        n_updates   += LOG_EVERY * G_UPDATES
        total_steps += N_ENVS * LOG_EVERY

        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = \
            collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())

        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col  * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0

            # Advance continuous curriculum
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            highest_rolling_suc = 0.99 * highest_rolling_suc + 0.01 * rolling_suc

            new_max_dist, _, _, new_max_scen = get_continuous_curriculum(highest_rolling_suc)

            if new_max_dist > cur_max_dist or new_max_scen > cur_max_scen:
                cur_max_dist = min(cur_max_dist + 0.2, new_max_dist)
                cur_max_scen = new_max_scen

            cur_scenario = jax.random.randint(jax.random.PRNGKey(int(time.time())), (), 0, cur_max_scen + 1)
            cur_scenario = int(cur_scenario)
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = (N_ENVS * LOG_EVERY) / (time.time() - t0)
        elapsed = time.time() - t_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

        chunk_idx = n_updates // LOG_EVERY
        if chunk_idx == 1 or chunk_idx % PRINT_EVERY_CHUNKS == 0:
            print(
                f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(all_metrics['critic_loss'].mean()):>7.4f} "
                f"{float(all_metrics['actor_loss'].mean()):>7.4f} "
                f"{float(all_metrics['alpha'].mean()):>6.4f} "
                f"{float(all_metrics['log_pi'].mean()):>6.3f} "
                f"{float(all_metrics['q_mean'].mean()):>7.3f} | "
                f"{fps:>7,.0f} | "
                f"{elapsed_str:>8}"
            )

        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        is_better = (suc_pct > best_suc) or (suc_pct == best_suc and mean_ret > best_ret)
        if is_better and n_ep > 0:
            best_suc = suc_pct
            best_ret = mean_ret
            save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                            eos, ahos, q1os, q2os, la, alo, n_updates)

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")