"""
SACjax.py — Soft Actor-Critic (shared-encoder, fused-JIT)

Architecture:
  - ONE shared encoder (ep) updated only through critic loss.
  - Actor head receives stop_gradient(feat_obs) — no separate encoder params.
  - Target network: EMA of critic head only (thp); encoder is shared, no tep.
  - extract_max_v called ONCE per collection step; stored in buffer.
  - vmap_step is NOT jit-wrapped here — outer train_chunk JIT fuses naturally.
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
N_ENVS         = 2048
BUFFER_CAP     = 1_000_000
BATCH_SIZE     = 2048
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)   # -2.0
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 500
SAVE_EVERY     = 5000
REWARD_SCALE   = 20.0
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
    """Build vmapped step/reset, initialise all envs. Returns (obs, state, vmap_step).
    vmap_step is NOT jit-wrapped here — the outer train_chunk JIT fuses it.
    """
    step_auto  = make_autoreset_env(reset_stacked, step_stacked)
    # No jax.jit here: let train_chunk's top-level JIT own the full graph.
    vmap_step  = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))

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
# ONE shared encoder (ep) — updated only through critic loss.
# Actor head receives stop_gradient(feat) so encoder gradients come
# exclusively from the Bellman backup signal — the more stable of the two.
# Target network = EMA of critic HEAD only (thp); encoder is already shared.

class ActorHead(nn.Module):
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


# ── Optimisers ────────────────────────────────────────────────────────────────
# Three param groups: shared encoder (ep), actor head (ahp), critic head (chp).
# encoder is updated via critic loss only — one optimiser for ep.
enc_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
alpha_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))


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
# max_v stored per transition so the hot loop never re-slices obs.
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),             jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "done":     jnp.zeros((capacity,),             jnp.float32),
        "max_v":    jnp.zeros((capacity,),             jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }


@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done, max_v):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs":      buf["obs"].at[idxs].set(obs),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs),
        "done":     buf["done"].at[idxs].set(done),
        "max_v":    buf["max_v"].at[idxs].set(max_v),
        "ptr":      jnp.int32((buf["ptr"] + N) % cap),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    idxs = jax.random.randint(rng_key, (batch_size,), 0, buf["size"])
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs],
            buf["next_obs"][idxs], buf["done"][idxs], buf["max_v"][idxs])


# ── SAC update step ───────────────────────────────────────────────────────────
# Param layout:
#   ep            — shared encoder params (updated via critic loss only)
#   eos           — encoder opt state
#   ahp / ahos    — actor head params + opt state
#   chp / chos    — critic head params + opt state
#   thp           — target critic head (EMA of chp)
#   la  / alo     — log alpha + opt state
#
# Encoder passes per call:
#   Critic loss : enc(obs) [trainable] + enc(next_obs) [stop_grad, for actor next]
#   Actor loss  : stop_gradient(feat_obs) reused — ZERO extra enc passes
#   Total       : 2 enc passes over 2 obs tensors (down from 4 in split-encoder design)
@jax.jit
def sac_update(ep, eos, ahp, ahos, chp, chos, thp, la, alo,
               obs, action, reward, next_obs, done, max_v_obs, max_v_next, rng_key):

    rng_c, rng_a = jax.random.split(rng_key)

    # Extract features ONCE for obs — used by both critic and actor losses.
    # Gradients through feat_obs flow only into the critic loss (see below).
    feat_obs = encoder_net.apply({"params": ep}, obs)

    # ── 1. Critic loss — Bellman backup ──────────────────────────────────────
    def _critic_loss(ep_, chp_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))

        # Actor next: encoder frozen — gradients must not update ep via actor path
        feat_next_sg  = jax.lax.stop_gradient(encoder_net.apply({"params": ep_}, next_obs))
        mean_n, lgs_n = actor_head.apply({"params": ahp}, feat_next_sg)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, max_v_next)

        # Target Q: head-only EMA, encoder also stop_grad
        feat_next_t = jax.lax.stop_gradient(encoder_net.apply({"params": ep_}, next_obs))
        q1_t, q2_t  = critic_head.apply({"params": thp}, feat_next_t, next_act)

        v_next = jnp.minimum(q1_t, q2_t) - alpha * next_lp
        backup = jax.lax.stop_gradient(
            reward / REWARD_SCALE + GAMMA * (1.0 - done) * v_next
        )

        # Online Q: feat comes from ep_ so gradients update the encoder
        feat_o = encoder_net.apply({"params": ep_}, obs)
        q1, q2 = critic_head.apply({"params": chp_}, feat_o, action)
        loss   = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)
        return loss, jnp.mean(backup) * REWARD_SCALE

    (c_loss, q_mean), (c_grads_enc, c_grads_head) = jax.value_and_grad(
        _critic_loss, argnums=(0, 1), has_aux=True
    )(ep, chp)

    e_upd,  new_eos  = enc_opt.update(c_grads_enc,  eos, ep)
    ch_upd, new_chos = head_critic_opt.update(c_grads_head, chos, chp)
    new_ep  = optax.apply_updates(ep,  e_upd)
    new_chp = optax.apply_updates(chp, ch_upd)

    # ── 2. Actor loss — maximise Q - alpha * log_pi ───────────────────────────
    # Encoder is FROZEN for the actor: stop_gradient(feat_obs).
    # This ensures enc gradients come only from the critic Bellman signal.
    def _actor_loss(ahp_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        feat  = jax.lax.stop_gradient(feat_obs)   # reuse, zero extra enc pass
        mean, log_std = actor_head.apply({"params": ahp_}, feat)
        action_new, log_pi = sample_action_sac_batched(rng_a, mean, log_std, max_v_obs)

        # Critic Q with updated params, encoder also frozen
        feat_c = jax.lax.stop_gradient(encoder_net.apply({"params": new_ep}, obs))
        q1, q2 = critic_head.apply({"params": new_chp}, feat_c, action_new)
        return jnp.mean(alpha * log_pi - jnp.minimum(q1, q2)), jnp.mean(log_pi)

    (a_loss, log_pi_mean), a_grads_head = jax.value_and_grad(
        _actor_loss, has_aux=True
    )(ahp)

    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # ── 3. Alpha update ───────────────────────────────────────────────────────
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    new_la = optax.apply_updates(la, al_upd)

    # ── 4. Soft target update (critic head only — encoder is shared) ──────────
    new_thp = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, thp, new_chp)

    metrics = {
        "critic_loss": c_loss,
        "actor_loss":  a_loss,
        "alpha":       jnp.exp(new_la),
        "log_pi":      log_pi_mean,
        "q_mean":      q_mean,
    }
    return (new_ep, new_eos, new_ahp, new_ahos,
            new_chp, new_chos, new_thp, new_la, new_alo, metrics)


# ── Collection step (runs inside train_chunk lax.scan) ───────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(ep, ahp, env_state, env_obs, rng_key, vmap_step, max_goal_dist, scenario_idx):
    """Compute max_v ONCE, sample action, step env. rng_key is threaded by lax.scan carry."""
    max_v    = extract_max_v(env_obs)
    k_act, k_step = jax.random.split(rng_key)
    feat  = encoder_net.apply({"params": ep}, env_obs)
    mean, log_std = actor_head.apply({"params": ahp}, feat)
    env_action, _ = sample_action_sac_batched(k_act, mean, log_std, max_v)
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info, max_v


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(9,))
def train_chunk(ep, eos, ahp, ahos, chp, chos,
                thp, la, alo, vmap_step,
                buf, es, eo, key,
                max_goal_dist, scenario_idx):

    def _loop_body(carry, _):
        (ep, eos, ahp, ahos, chp, chos,
         thp, la, alo, buf, es, eo, key) = carry
        key, k_col, k_samp, k_upd = jax.random.split(key, 4)

        # Collect — max_v decoded ONCE inside collect_step from the pre-step obs
        new_eo, new_es, obs_b, env_a, rew, done, info, max_v_cur = collect_step(
            ep, ahp, es, eo, k_col, vmap_step, max_goal_dist, scenario_idx
        )

        terminal = done & ~info["timeout"]
        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo,
                          terminal.astype(jnp.float32), max_v_cur)

        b_obs, b_act, b_rew, b_next, b_done, b_max_v = buf_sample(new_buf, k_samp, BATCH_SIZE)
        b_max_v_next = extract_max_v(b_next)   # decoded once from sampled next_obs

        (new_ep, new_eos, new_ahp, new_ahos,
         new_chp, new_chos, new_thp, new_la, new_alo, metrics) = sac_update(
            ep, eos, ahp, ahos, chp, chos, thp, la, alo,
            b_obs, b_act, b_rew, b_next, b_done, b_max_v, b_max_v_next, k_upd
        )

        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        new_carry = (new_ep, new_eos, new_ahp, new_ahos,
                     new_chp, new_chos, new_thp, new_la, new_alo,
                     new_buf, new_es, new_eo, key)
        return new_carry, (step_data, metrics)

    carry = (ep, eos, ahp, ahos, chp, chos,
             thp, la, alo, buf, es, eo, key)
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
def save_checkpoint(ep, ahp, chp, thp, eos, ahos, chos, la, alo, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "encoder_params":      jax.device_get(ep),
        "actor_head_params":   jax.device_get(ahp),
        "critic_head_params":  jax.device_get(chp),
        "target_head_params":  jax.device_get(thp),
        "encoder_opt":         jax.device_get(eos),
        "actor_head_opt":      jax.device_get(ahos),
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

    master_key = jax.random.PRNGKey(7)
    master_key, k_init, k_env, k_warmup = jax.random.split(master_key, 4)
    train_rng  = jax.random.PRNGKey(42)

    dummy_obs  = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_feat = jnp.zeros((2, 128),        dtype=jnp.float32)
    dummy_act  = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    k_e, k_ah, k_ch = jax.random.split(k_init, 3)

    ep  = encoder_net.init(k_e,  dummy_obs)["params"]               # shared encoder
    ahp = actor_head.init(k_ah, dummy_feat)["params"]               # actor head
    chp = critic_head.init(k_ch, dummy_feat, dummy_act)["params"]   # critic head
    thp = jax.tree_util.tree_map(jnp.array, chp)                    # target head (copy)

    eos  = enc_opt.init(ep)
    ahos = head_actor_opt.init(ahp)
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

    # ── Warmup: fill buffer with random actions ───────────────────────────────
    print("Warming up buffer with random actions...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_act, k_step = jax.random.split(k_warmup, 3)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)          # once per warmup step
        rand_v     = jax.random.uniform(k_act, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_act, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = jax.jit(vmap_step)(
            step_keys, env_state, env_action, 3.0, -1
        )
        terminal   = done & ~info["timeout"]
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             new_obs, terminal.astype(jnp.float32), max_v)
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

    _LOG_PATH = "checkpoints_sac/sac_training_log.csv"
    os.makedirs("checkpoints_sac", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()

        new_carry, all_step_data, all_metrics = train_chunk(
            ep, eos, ahp, ahos, chp, chos,
            thp, la, alo, vmap_step,
            replay_buf, env_state, env_obs, train_rng,
            3.0, -1
        )
        (ep, eos, ahp, ahos, chp, chos,
         thp, la, alo, replay_buf, env_state, env_obs, train_rng) = new_carry

        n_updates   += LOG_EVERY
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
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = (N_ENVS * LOG_EVERY) / (time.time() - t0)
        elapsed = time.time() - t_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

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

        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        is_better = (suc_pct > best_suc) or (suc_pct == best_suc and mean_ret > best_ret)
        if is_better and n_ep > 0:
            best_suc = suc_pct
            best_ret = mean_ret
            save_checkpoint(ep, ahp, chp, thp, eos, ahos, chos, la, alo, n_updates)

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")