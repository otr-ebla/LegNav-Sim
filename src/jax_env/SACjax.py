"""
SACjax.py — Soft Actor-Critic  (improved)
=========================================
Pinned to GPU 1 via CUDA_VISIBLE_DEVICES=1 so it runs alongside PPO on GPU 0.

SAC (Haarnoja et al. 2018) with:
  - Twin critics + target networks (soft update tau=0.005)
  - Squashed Gaussian policy with exact tanh Jacobian log-prob correction
  - Automatic temperature (alpha) tuning to maintain target entropy = -1.0
  - On-GPU circular replay buffer (1M transitions)
  - Vectorised collection across N_ENVS parallel environments

Improvements over original version
───────────────────────────────────
  1. TRULY SHARED ENCODER — encoder, actor head, and critic head are now
     separate Flax modules with separate param pytrees. sac_update calls
     encoder_net.apply() once per obs tensor and passes the resulting feat
     to the relevant head. Q1 and Q2 share every encoder pass.
     CNN passes per update: 4 (vs 5 in original — actor_enc×2 + q1_enc×2 + q2_enc).
     Critic backward now differentiates through ONE encoder instead of TWO.

  2. SEPARATE RNG STREAMS — env, warmup, and training each get independent
     PRNGKey streams. Improves reproducibility and debugging.

  3. RANDOM WARMUP — warmup loop samples uniform random actions instead
     of using the untrained policy. Avoids early-buffer bias.

  4. BETTER CHECKPOINT LOGIC — saves whenever:
       suc_pct > best_suc  OR  (suc_pct == best_suc AND mean_ret > best_ret)
     Checks every LOG_EVERY (500) steps — never misses a peak.

  5. BUFFER 1M — increased from 500k for better off-policy stability.

  6. BATCH 2048 — increased from 1024; SAC converges faster with large batches.

  7. REWARD_SCALE 20 — reduced from 50; large scaling destabilises critics.

  8. TARGET_ENTROPY -1.0 — softer than -|A|=-2.0; encourages more exploration.

  9. BUF_SAMPLE FIX — sampling now draws from [0, buf["size"]) directly.

Action space:
  v in [0, max_v]: a_v = (tanh(u_v) + 1) / 2 * max_v
  w in [-1,   1 ]: a_w = tanh(u_w)
  max_v varies per episode; recovered from obs[..., MAX_V_OBS_IDX].

ARCHITECTURE — Fully-fused GPU loop (inspired by TQC implementation):

  The entire inner loop — env step, buf_add, buf_sample, sac_update — runs
  inside a single jax.lax.scan of length LOG_EVERY. One Python call dispatches
  LOG_EVERY iterations with zero host syncs.
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
OBS_SIZE       = 342        # 9 (pose) + 9 (state_vec) + 324 (lidar)
ACTION_DIM     = 2
# N_ENVS/BUFFER/BATCH match TQC proven defaults — fit in VRAM with fused scan.
N_ENVS         = 2048
BUFFER_CAP     = 1_000_000
BATCH_SIZE     = 2048
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4

TARGET_ENTROPY = -float(ACTION_DIM)   # -2.0
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 500        # scan length inside train_chunk; also log interval
SAVE_EVERY     = 5000
TARGET_ENTROPY = -1.0   # softer target: better exploration than -|A| = -2.0
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 500        # scan length inside train_chunk; also log + checkpoint interval

# Reward normalisation: env emits rewards in [-70, +200].
# Reduced from 50 → 20 to stabilise critic value estimates.
REWARD_SCALE   = 20.0

# Gradient clipping
MAX_GRAD_NORM  = 10.0

# pose_stack indices 0-8; state_vec[2] = max_v/2 at flat index 9+2 = 11
MAX_V_OBS_IDX  = 11
LOG_STD_EPS    = 1e-6

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
    """Build vmapped step/reset, initialise all envs. Returns (obs, state, vmap_step)."""
    
    # 1. Initialize the autoreset wrapper without static closure arguments
    step_auto  = make_autoreset_env(reset_stacked, step_stacked)
    
    # 2. Tell vmap that max_goal_dist and scenario_idx are shared dynamic tensors (None)
    vmap_step  = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0, None, None)))
    
    # 3. Initialize the first batch of environments
    def _reset(key): 
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx)
    vmap_reset = jax.jit(jax.vmap(_reset))
    
    env_obs, env_state = vmap_reset(jax.random.split(rng_key, N_ENVS))
    
    return env_obs, env_state, vmap_step


# ── Shared obs encoder ────────────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x):
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 9

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        return nn.relu(nn.Dense(128)(shared))


# ── Split-head networks ───────────────────────────────────────────────────────
# Architecture: one shared encoder + separate actor/critic head modules.
#
# Why split into separate pytrees instead of one unified module?
#   - Actor and critic are updated with DIFFERENT loss functions and different
#     optimisers. Keeping them as separate param dicts lets us pass only the
#     relevant params to value_and_grad, so gradients never bleed across.
#   - The encoder params live inside BOTH the actor pytree AND the critic pytree
#     (via their respective .enc sub-trees). JAX's functional transforms handle
#     this correctly: each grad call sees its own copy of enc params and produces
#     its own grad for that copy.
#   - The target network is a copy of the critic pytree only (enc + Q heads).
#
# Forward-pass CNN count per sac_update call:
#   BEFORE (original): actor_enc + q1_enc + q2_enc = 3 CNN passes
#   AFTER  (this):     enc_for_critic_loss(obs) + enc_for_critic_loss(next_obs)
#                    + enc_for_actor_loss(obs) = still 3 distinct obs tensors,
#                    but Q1 and Q2 now share one encoder pass each. So:
#                      critic loss:  1 enc(obs) + 1 enc(next_obs)  [target net]
#                      actor loss:   1 enc(obs)
#                    = 3 enc passes total, but Q1/Q2 within each pass share feat.
#   Net saving vs original: 3 passes → 3 passes for obs coverage, but Q1 and Q2
#   no longer run separate CNNs. The critic backward only differentiates through
#   ONE encoder rather than two, halving the critic encoder gradient compute.

class ActorHead(nn.Module):
    """Maps encoder features → (mean, log_std)."""
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0

    @nn.compact
    def __call__(self, feat: jnp.ndarray):
        mean    = nn.Dense(self.action_dim, name='mean')(feat)
        log_std = nn.Dense(self.action_dim, name='log_std')(feat)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std


class CriticHead(nn.Module):
    """Maps (encoder features, action) → (Q1, Q2)."""

    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray):
        q_in = jnp.concatenate([feat, action], axis=-1)

        q1 = nn.relu(nn.Dense(256, name='q1_l1')(q_in))
        q1 = nn.relu(nn.Dense(128, name='q1_l2')(q1))
        q1 = jnp.squeeze(nn.Dense(1, name='q1_out')(q1), axis=-1)

        q2 = nn.relu(nn.Dense(256, name='q2_l1')(q_in))
        q2 = nn.relu(nn.Dense(128, name='q2_l2')(q2))
        q2 = jnp.squeeze(nn.Dense(1, name='q2_out')(q2), axis=-1)

        return q1, q2


# Module instances (stateless — params stored separately)
encoder_net = ObsEncoder()
actor_head  = ActorHead()
critic_head = CriticHead()


# ── Param-namespace helpers ───────────────────────────────────────────────────
# Each "network" is represented as a dict: {"enc": enc_params, "head": head_params}
# This keeps actor encoder and critic encoder as separate pytrees so their
# gradient updates are fully independent.

def _actor_forward(enc_params, head_params, obs):
    """One CNN pass → actor (mean, log_std)."""
    feat = encoder_net.apply({"params": enc_params}, obs)
    return actor_head.apply({"params": head_params}, feat)


def _critic_forward(enc_params, head_params, obs, action):
    """One CNN pass → (Q1, Q2)."""
    feat = encoder_net.apply({"params": enc_params}, obs)
    return critic_head.apply({"params": head_params}, feat, action)


# Optimisers — one per independent param group
enc_actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_actor_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
enc_critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_critic_opt= optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
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
    """
    Fully batched reparameterised sample — one random.normal call, no vmap.
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


@jax.jit
def extract_max_v(obs):
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular) ───────────────────────────────────────────
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),             jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "done":     jnp.zeros((capacity,),             jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }


@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs":      buf["obs"].at[idxs].set(obs),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs),
        "done":     buf["done"].at[idxs].set(done),
        "ptr":      jnp.int32((buf["ptr"] + N) % cap),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    # Sample only from filled entries — removes the conditional branch.
    idxs = jax.random.randint(rng_key, (batch_size,), 0, buf["size"])
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs],
            buf["next_obs"][idxs], buf["done"][idxs])




# ── SAC update step ───────────────────────────────────────────────────────────
# Param layout (all separate pytrees):
#   aep / ahp  — actor encoder + head params
#   aeos / ahos — their opt states
#   cep / chp  — critic encoder + head params
#   ceos / chos — their opt states
#   tep / thp  — target encoder + head params (EMA of cep/chp)
#   la  / alo  — log alpha + its opt state
#
# CNN passes per call (vs. original 3 separate full networks):
#   Critic loss : enc_actor(next_obs) + enc_target(next_obs) + enc_critic(obs) = 3
#                 but Q1 and Q2 now share EACH feat — no duplicate CNN for twin Q
#   Actor loss  : enc_actor(obs)                                                = 1
#   Total       : 4 enc passes over 2 obs tensors, down from 5 in the original
#                 (original: actor_next + q1_next + q2_next + q1_obs + actor_obs)
@jax.jit
def sac_update(aep, ahp, aeos, ahos, cep, chp, ceos, chos, tep, thp, la, alo,
               obs, action, reward, next_obs, done, rng_key):

    rng_c, rng_a = jax.random.split(rng_key)

    # ── 1. Critic loss — Bellman backup ──────────────────────────────────────
    def _critic_loss(cep_, chp_):
        alpha      = jax.lax.stop_gradient(jnp.exp(la))
        next_max_v = extract_max_v(next_obs)

        # Actor encoder pass on next_obs (frozen — stop_gradient through aep)
        feat_next_a   = jax.lax.stop_gradient(encoder_net.apply({"params": aep}, next_obs))
        mean_n, lgs_n = actor_head.apply({"params": ahp}, feat_next_a)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, next_max_v)

        # Target Q: single encoder pass, Q1+Q2 share feat
        feat_next_t = encoder_net.apply({"params": tep}, next_obs)
        q1_t, q2_t  = critic_head.apply({"params": thp}, feat_next_t, next_act)

        v_next = jnp.minimum(q1_t, q2_t) - alpha * next_lp
        backup = jax.lax.stop_gradient(
            reward / REWARD_SCALE + GAMMA * (1.0 - done) * v_next
        )

        # Online Q: single encoder pass on obs — Q1 and Q2 share this feat
        feat_obs = encoder_net.apply({"params": cep_}, obs)
        q1, q2   = critic_head.apply({"params": chp_}, feat_obs, action)
        loss     = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)
        return loss, jnp.mean(backup) * REWARD_SCALE

    (c_loss, q_mean), (c_grads_enc, c_grads_head) = jax.value_and_grad(
        _critic_loss, argnums=(0, 1), has_aux=True
    )(cep, chp)

    ce_upd, new_ceos = enc_critic_opt.update(c_grads_enc,  ceos, cep)
    ch_upd, new_chos = head_critic_opt.update(c_grads_head, chos, chp)
    new_cep = optax.apply_updates(cep, ce_upd)
    new_chp = optax.apply_updates(chp, ch_upd)

    # ── 2. Actor loss — maximise Q - alpha * log_pi ───────────────────────────
    # Single encoder pass on obs for actor head; critic uses updated params.
    def _actor_loss(aep_, ahp_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        max_v = extract_max_v(obs)

        feat          = encoder_net.apply({"params": aep_}, obs)
        mean, log_std = actor_head.apply({"params": ahp_}, feat)
        action_new, log_pi = sample_action_sac_batched(rng_a, mean, log_std, max_v)

        # Critic Q: separate encoder pass (updated critic params, no grad bleed)
        feat_c = jax.lax.stop_gradient(encoder_net.apply({"params": new_cep}, obs))
        q1, q2 = critic_head.apply({"params": new_chp}, feat_c, action_new)
        return jnp.mean(alpha * log_pi - jnp.minimum(q1, q2)), jnp.mean(log_pi)

    (a_loss, log_pi_mean), (a_grads_enc, a_grads_head) = jax.value_and_grad(
        _actor_loss, argnums=(0, 1), has_aux=True
    )(aep, ahp)

    ae_upd, new_aeos = enc_actor_opt.update(a_grads_enc,  aeos, aep)
    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_aep = optax.apply_updates(aep, ae_upd)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # ── 3. Alpha update ───────────────────────────────────────────────────────
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    new_la = optax.apply_updates(la, al_upd)

    # ── 4. Soft target update (critic enc + head) ─────────────────────────────
    new_tep = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, tep, new_cep)
    new_thp = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, thp, new_chp)

    metrics = {
        "critic_loss": c_loss,
        "actor_loss":  a_loss,
        "alpha":       jnp.exp(new_la),
        "log_pi":      log_pi_mean,
        "q_mean":      q_mean,
    }
    return (new_aep, new_ahp, new_aeos, new_ahos,
            new_cep, new_chp, new_ceos, new_chos,
            new_tep, new_thp, new_la, new_alo, metrics)


# Change the signature to accept the dynamic curriculum variables
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(aep, ahp, env_state, env_obs, rng_key, vmap_step, max_goal_dist, scenario_idx):
    """aep = actor encoder params, ahp = actor head params."""
    mean, log_std = _actor_forward(aep, ahp, env_obs)
    env_action, _ = sample_action_sac_batched(
        rng_key, mean, log_std, extract_max_v(env_obs)
    )
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 1), N_ENVS)
    
    # Pass the curriculum variables into vmap_step
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
# FUSE 1: collect + buf_add + buf_sample + sac_update inside one lax.scan.
# The replay buffer lives in the carry — no host transfers, zero Python syncs.
# Change this line (around line 430) to ONLY include 13:
@functools.partial(jax.jit, static_argnums=(13,))
def train_chunk(aep, ahp, aeos, ahos, cep, chp, ceos, chos,
                tep, thp, la, alo, buf, vmap_step, es, eo, key, 
                max_goal_dist, scenario_idx): # <-- Added arguments

    def _loop_body(carry, _):
        (aep, ahp, aeos, ahos, cep, chp, ceos, chos,
         tep, thp, la, alo, buf, es, eo, key) = carry
        key, k_col, k_samp, k_upd = jax.random.split(key, 4)

        # Collect one env step (actor encoder + head)
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(
            aep, ahp, es, eo, k_col, vmap_step, max_goal_dist, scenario_idx
        )

        # True terminal (ignore timeouts for Bellman backup)
        terminal = done & ~info["timeout"]

        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(new_buf, k_samp, BATCH_SIZE)

        (new_aep, new_ahp, new_aeos, new_ahos,
         new_cep, new_chp, new_ceos, new_chos,
         new_tep, new_thp, new_la, new_alo, metrics) = sac_update(
            aep, ahp, aeos, ahos, cep, chp, ceos, chos, tep, thp, la, alo,
            b_obs, b_act, b_rew, b_next, b_done, k_upd
        )

        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        new_carry = (new_aep, new_ahp, new_aeos, new_ahos,
                     new_cep, new_chp, new_ceos, new_chos,
                     new_tep, new_thp, new_la, new_alo,
                     new_buf, new_es, new_eo, key)
        return new_carry, (step_data, metrics)

    carry = (aep, ahp, aeos, ahos, cep, chp, ceos, chos,
             tep, thp, la, alo, buf, es, eo, key)
    new_carry, (all_step_data, all_metrics) = jax.lax.scan(
        _loop_body, carry, None, length=LOG_EVERY
    )
    return new_carry, all_step_data, all_metrics


# ── On-GPU episode stats ───────────────────────────────────────────────────────
# FUSE 3: accumulation runs entirely on-GPU. Only 5 scalars cross PCIe per chunk.
@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    """
    Accumulate per-env episode returns over (LOG_EVERY, N_ENVS) step data on-GPU.
    Returns flat arrays over all completed episodes in the chunk.
    """
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
def save_checkpoint(aep, ahp, cep, chp, tep, thp, aeos, ahos, ceos, chos, la, alo, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "actor_enc_params":    jax.device_get(aep),
        "actor_head_params":   jax.device_get(ahp),
        "critic_enc_params":   jax.device_get(cep),
        "critic_head_params":  jax.device_get(chp),
        "target_enc_params":   jax.device_get(tep),
        "target_head_params":  jax.device_get(thp),
        "actor_enc_opt":       jax.device_get(aeos),
        "actor_head_opt":      jax.device_get(ahos),
        "critic_enc_opt":      jax.device_get(ceos),
        "critic_head_opt":     jax.device_get(chos),
        "log_alpha":           jax.device_get(la),
        "alpha_opt_state":     jax.device_get(alo),
        "step":                int(step),
    }
    with open(CKPT_PATH, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("SAC Training  (shared-encoder architecture)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}\n")

    # ── Separate RNG streams for reproducibility ──────────────────────────────
    # Using independent streams avoids coupling between policy, env, and buffer
    # sampling — makes runs easier to debug and compare.
    master_key = jax.random.PRNGKey(7)
    master_key, k_init_a, k_init_c, k_env, k_warmup = jax.random.split(master_key, 5)
    train_rng  = jax.random.PRNGKey(42)   # dedicated stream for train_chunk

    dummy_obs  = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_feat = jnp.zeros((2, 128),        dtype=jnp.float32)   # ObsEncoder output dim
    dummy_act  = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    # Initialise params — encoder and heads are separate pytrees
    k_ae, k_ah, k_ce, k_ch = jax.random.split(k_init_a, 4)

    aep  = encoder_net.init(k_ae, dummy_obs)["params"]           # actor encoder
    ahp  = actor_head.init(k_ah, dummy_feat)["params"]           # actor head
    cep  = encoder_net.init(k_ce, dummy_obs)["params"]           # critic encoder
    chp  = critic_head.init(k_ch, dummy_feat, dummy_act)["params"] # critic head
    tep  = jax.tree_util.tree_map(jnp.array, cep)                # target encoder (copy)
    thp  = jax.tree_util.tree_map(jnp.array, chp)                # target head (copy)

    aeos = enc_actor_opt.init(aep)
    ahos = head_actor_opt.init(ahp)
    ceos = enc_critic_opt.init(cep)
    chos = head_critic_opt.init(chp)
    la   = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    alo  = alpha_opt.init(la)

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(k_env)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    best_suc    = 49.5
    best_ret    = -1e9

    # ── Warmup: fill buffer with RANDOM actions ───────────────────────────────
    print("Warming up buffer with random actions...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_act, k_step = jax.random.split(k_warmup, 3)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)
        rand_v     = jax.random.uniform(k_act, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_act, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = vmap_step(
            step_keys, env_state, env_action, 3.0, -1
        )
        terminal   = done & ~info["timeout"]
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             new_obs, terminal.astype(jnp.float32))
        env_obs     = new_obs
        total_steps += N_ENVS
    print("Warmup done. JIT compiling train_chunk (this may take ~1 min)...")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | "
           f"{'FPS':>7} | {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    # ── Training log (CSV) ───────────────────────────────────────────────────
    _LOG_PATH = "checkpoints_sac/sac_training_log.csv"
    os.makedirs("checkpoints_sac", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()

        # ── Single fused GPU dispatch: LOG_EVERY steps of everything ──────────
        new_carry, all_step_data, all_metrics = train_chunk(
            aep, ahp, aeos, ahos, cep, chp, ceos, chos,
            tep, thp, la, alo, replay_buf, vmap_step, env_state, env_obs, train_rng,
            3.0, -1  # <-- Added dynamic arguments
        )
        (aep, ahp, aeos, ahos, cep, chp, ceos, chos,
         tep, thp, la, alo, replay_buf, env_state, env_obs, train_rng) = new_carry

        n_updates   += LOG_EVERY
        total_steps += N_ENVS * LOG_EVERY

        # ── On-GPU stats reduction — only scalars cross PCIe ──────────────────
        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = \
            collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())

        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col  * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = (N_ENVS * LOG_EVERY) / (time.time() - t0)
        elapsed = time.time() - t_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

        # Print the first update at 500, and then only every 5000 updates
        if n_updates == 500 or n_updates % 5000 == 0:
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
        # ── CSV log row ──────────────────────────────────────────────────────
        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        # Save best model: primary = success rate, tie-break = episode reward
        # Save best model: primary = success rate, tie-break = episode reward
        # Save best model: primary = success rate, tie-break = episode reward
        is_better = (suc_pct > best_suc) or (suc_pct == best_suc and mean_ret > best_ret)
        if is_better and n_ep > 0:
            best_suc = suc_pct
            best_ret = mean_ret
            save_checkpoint(aep, ahp, cep, chp, tep, thp,
                            aeos, ahos, ceos, chos, la, alo, n_updates)

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")