"""
SACjax.py — Soft Actor-Critic (shared-encoder, fused-JIT, curato con Reward Normalization)
"""

import os
import sys
import csv

# ── GPU selection — must happen BEFORE import jax ────────────────────────────
import argparse as _ap
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
OBS_SIZE       = 662
ACTION_DIM     = 2
N_ENVS         = 2048
BUFFER_CAP     = 1_000_000
BATCH_SIZE     = 512
G_UPDATES      = 50
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
                            
LR             = 3e-4
TARGET_ENTROPY = -1.0       
DEFAULT_TOTAL_ENV_STEPS = 12_000_000  
LOG_EVERY      = 50
SAVE_EVERY     = 5000
MAX_GRAD_NORM  = 10.0

MAX_V_OBS_IDX  = 11
LOG_STD_EPS    = 1e-6
LIDAR_MAX_RANGE = 12.0
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

def init_env_state(rng_key, max_goal_dist: float = 1.5, ghost_prob: float = 0.0, scenario_idx: int = -1):
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    vmap_step = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None, None))

    def _reset(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    vmap_reset = jax.jit(jax.vmap(_reset))

    env_obs, env_state = vmap_reset(jax.random.split(rng_key, N_ENVS))
    return env_obs, env_state, vmap_step


# ── Shared Encoder + Actor/Critic Heads ──────────────────────────────────────
from jax_network import SharedEncoder

_orth_relu = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
_orth_out  = nn.initializers.orthogonal(scale=0.01)

class SACActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat: jnp.ndarray):
        mean    = nn.Dense(self.action_dim, kernel_init=_orth_out, name='mean')(feat)
        log_std = nn.Dense(self.action_dim, kernel_init=_orth_out, name='log_std')(feat)
        return mean, jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

class CriticBranch(nn.Module):
    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray):
        x = jnp.concatenate([feat, action], axis=-1)
        q = nn.LayerNorm()(nn.relu(nn.Dense(256, kernel_init=_orth_relu)(x)))
        q = nn.LayerNorm()(nn.relu(nn.Dense(128, kernel_init=_orth_relu)(q)))
        return jnp.squeeze(nn.Dense(1)(q), axis=-1)

shared_enc = SharedEncoder()
actor_head  = SACActorHead()
critic_q1   = CriticBranch()
critic_q2   = CriticBranch()


# ── Optimisers ────────────────────────────────────────────────────────────────
ALPHA_LR = 1e-4
enc_opt        = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR,       eps=1e-5))
head_actor_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR,       eps=1e-5))
head_q1_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR,       eps=1e-5))
head_q2_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR,       eps=1e-5))
alpha_opt      = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(ALPHA_LR, eps=1e-5))

# ── Reward Normalization (Running RMS) ────────────────────────────────────────
@jax.jit
def update_reward_rms(mean, var, count, batch):
    """Aggiorna la varianza delle reward usando l'algoritmo di Welford in JAX."""
    batch_mean = jnp.mean(batch)
    batch_var = jnp.var(batch)
    batch_count = batch.shape[0]

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    
    new_var = jnp.where(tot_count > 0, M2 / tot_count, 1.0)
    return new_mean, new_var, tot_count


# ── Action squashing + exact log-prob ─────────────────────────────────────────
def _tanh_log_prob_correction(tanh_u, max_v):
    corr_v = (
        jnp.log(max_v * 0.5 + LOG_STD_EPS)
        + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )
    corr_w = jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    return -(corr_v + corr_w)

def sample_action_sac_batched(rng_key, mean, log_std, max_v):
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise**2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)
    tanh_u = jnp.tanh(u)
    
    # Lo scaling consistente avviene qui: usato sia in iterazione che nella loss
    a_v = (tanh_u[:, 0] + 1.0) * 0.5 * max_v
    a_w = tanh_u[:, 1]
    return jnp.stack([a_v, a_w], axis=-1), lp_gauss + _tanh_log_prob_correction(tanh_u, max_v)

def extract_max_v(obs):
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular) ───────────────────────────────────────────
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

    # 1. Critic loss
    def _critic_loss(sep_, q1p_, q2p_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        feat_next = jax.lax.stop_gradient(shared_enc.apply({"params": tsep}, next_obs))
        mean_n, lgs_n = actor_head.apply({"params": ahp}, feat_next)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, max_v_next)

        q1_t = critic_q1.apply({"params": tq1p}, feat_next, next_act)
        q2_t = critic_q2.apply({"params": tq2p}, feat_next, next_act)
        v_next = jnp.minimum(q1_t, q2_t) - alpha * next_lp
        
        # Reward è già normalizzata prima di essere passata qui
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

    # 2. Actor loss
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

    # 3. Encoder update
    combined_enc_grads = jax.tree_util.tree_map(
        lambda cg, ag: cg + ag, c_grads_enc, a_grads_enc_actor
    )
    enc_upd, new_eos = enc_opt.update(combined_enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, enc_upd)

    # 4. Alpha update
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    new_la = optax.apply_updates(la, al_upd)

    # 5. Soft target update
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
def collect_step(sep, ahp, env_state, env_obs, rng_key, vmap_step, max_goal_dist, scenario_idx, ghost_prob, max_scenario):
    max_v = extract_max_v(env_obs)
    k_act, k_step = jax.random.split(rng_key)
    feat = shared_enc.apply({"params": sep}, env_obs)
    mean, log_std = actor_head.apply({"params": ahp}, feat)
    env_action, _ = sample_action_sac_batched(k_act, mean, log_std, max_v)
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info, max_v


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(13,))
def train_chunk(sep, eos, tsep, ahp, ahos,
                q1p, q1os, q2p, q2os, tq1p, tq2p,
                la, alo, vmap_step,
                buf, es, eo, key,
                max_goal_dist, scenario_idx, ghost_prob,
                max_scenario, rmean, rvar, rcount):

    # ── Phase 1: collect LOG_EVERY env steps, write to buffer ─────────────────
    def _collect_body(carry, _):
        es_, eo_, buf_, key_, rm_, rv_, rc_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info, max_v_cur = collect_step(
            sep, ahp, es_, eo_, k_col, vmap_step, max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        terminal = done & ~info["timeout"]
        new_buf = buf_add(buf_, obs_b, env_a, rew, new_eo,
                          terminal.astype(jnp.float32), max_v_cur)
        
        # Aggiorniamo le running stats delle reward sulla base dei campioni freschi raccolti
        new_rm, new_rv, new_rc = update_reward_rms(rm_, rv_, rc_, rew)
        
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_, new_rm, new_rv, new_rc), step_data

    (new_es, new_eo, new_buf, key, new_rmean, new_rvar, new_rcount), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key, rmean, rvar, rcount), None, length=LOG_EVERY
    )

    # ── Phase 2: G_UPDATES * LOG_EVERY gradient steps, buffer as constant ─────
    def _update_step(update_carry, upd_idx):
        (sep_, eos_, tsep_, ahp_, ahos_,
         q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_, key_) = update_carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_terminal, b_max_v = buf_sample(
            new_buf, k_samp, BATCH_SIZE
        )
        b_max_v_next = extract_max_v(b_next)
        
        # Reward Normalization prima del calcolo della Bellman Loss
        norm_b_rew = b_rew / jnp.sqrt(new_rvar + 1e-8)
        
        (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
         new_q1p_, new_q1os_, new_q2p_, new_q2os_,
         new_tq1p_, new_tq2p_, new_la_, new_alo_, metrics_) = sac_update(
            sep_, eos_, tsep_, ahp_, ahos_,
            q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_,
            b_obs, b_act, norm_b_rew, b_next, b_terminal, b_max_v, b_max_v_next, k_upd
        )
        return (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
                new_q1p_, new_q1os_, new_q2p_, new_q2os_,
                new_tq1p_, new_tq2p_, new_la_, new_alo_, key_), metrics_

    update_carry_init = (sep, eos, tsep, ahp, ahos,
                         q1p, q1os, q2p, q2os, tq1p, tq2p, la, alo, key)
    (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
     new_q1p, new_q1os, new_q2p, new_q2os,
     new_tq1p, new_tq2p, new_la, new_alo, key), all_metrics = jax.lax.scan(
        _update_step, update_carry_init, jnp.arange(G_UPDATES * LOG_EVERY)
    )

    new_carry = (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
                 new_q1p, new_q1os, new_q2p, new_q2os, new_tq1p, new_tq2p,
                 new_la, new_alo, new_buf, new_es, new_eo, key,
                 new_rmean, new_rvar, new_rcount)
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
                    eos, ahos, q1os, q2os, la, alo, step,
                    filepath=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = filepath or CKPT_PATH
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
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {path}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(total_env_steps: int = DEFAULT_TOTAL_ENV_STEPS):
    print("SAC Training  (shared LidarCNN + decoupled Q1/Q2 branches)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}  G_UPDATES={G_UPDATES}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}")
    print(f"  LOG_EVERY={LOG_EVERY}  transitions/chunk={N_ENVS*LOG_EVERY:,}  grad_updates/chunk={LOG_EVERY*G_UPDATES}")
    print(f"  Budget: {int(total_env_steps):,} env steps\n")

    master_key = jax.random.PRNGKey(7)
    master_key, k_init, k_env, k_warmup = jax.random.split(master_key, 4)
    train_rng  = jax.random.PRNGKey(42)

    dummy_obs = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    k_se, k_ah, k_q1, k_q2 = jax.random.split(k_init, 4)

    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)

    dummy_feat = shared_enc.apply({"params": sep}, dummy_obs)
    ahp  = actor_head.init(k_ah, dummy_feat)["params"]

    q1p  = critic_q1.init(k_q1, dummy_feat, dummy_act)["params"]
    q2p  = critic_q2.init(k_q2, dummy_feat, dummy_act)["params"]
    tq1p = jax.tree_util.tree_map(jnp.array, q1p)
    tq2p = jax.tree_util.tree_map(jnp.array, q2p)

    eos  = enc_opt.init(sep)
    ahos = head_actor_opt.init(ahp)
    q1os = head_q1_opt.init(q1p)
    q2os = head_q2_opt.init(q2p)

    la   = jnp.array(0.0, dtype=jnp.float32)
    alo  = alpha_opt.init(la)

    from jax_ppo import get_continuous_curriculum
    # Forza la probabilità dei ghost a 0 all'inizio
    cur_max_dist, _, _, cur_max_scen = get_continuous_curriculum(0.0)
    cur_ghost = 0.0
    rolling_suc = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(k_env, max_goal_dist=cur_max_dist, ghost_prob=cur_ghost)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    best_suc    = 49.5
    best_ret    = -1e9
    
    # Init Reward stats
    r_mean  = jnp.zeros(())
    r_var   = jnp.ones(())
    r_count = jnp.zeros(())

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
            step_keys, env_state, env_action, cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen)
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
           f"{'FPS':>7} | {'Time':>8} | {'Dist':>5} {'Scen':>4} {'Gst':>4}")
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

    PRINT_EVERY_CHUNKS = 10

    while total_steps < total_env_steps:
        t0 = time.time()

        new_carry, all_step_data, all_metrics = train_chunk(
            sep, eos, tsep, ahp, ahos,
            q1p, q1os, q2p, q2os, tq1p, tq2p,
            la, alo, vmap_step,
            replay_buf, env_state, env_obs, train_rng,
            cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen), r_mean, r_var, r_count
        )
        (sep, eos, tsep, ahp, ahos,
         q1p, q1os, q2p, q2os, tq1p, tq2p,
         la, alo, replay_buf, env_state, env_obs, train_rng,
         r_mean, r_var, r_count) = new_carry

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

            # ── Curriculum Sbloccato ──────────────────────────────────────
            rolling_suc         = 0.90 * rolling_suc + 0.10 * suc_pct
            highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

            new_max_dist, new_ghost, _, new_max_scen = get_continuous_curriculum(highest_rolling_suc)

            if new_max_dist > cur_max_dist:
                cur_max_dist = min(cur_max_dist + 0.2, new_max_dist)
            if new_ghost > cur_ghost:
                cur_ghost = new_ghost
            if new_max_scen > cur_max_scen:
                cur_max_scen = new_max_scen

            cur_scenario = -1
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
                f"{elapsed_str:>8} | "
                f"{cur_max_dist:>5.1f} {cur_max_scen:>4d} {cur_ghost:>4.2f}"
            )

        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
        is_better = (suc_pct > best_suc) or (suc_pct == best_suc and mean_ret > best_ret)
        if is_better and n_ep > 0 and curriculum_mature:
            best_suc = suc_pct
            best_ret = mean_ret
            save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                            eos, ahos, q1os, q2os, la, alo, n_updates)

    save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                    eos, ahos, q1os, q2os, la, alo, n_updates,
                    filepath="checkpoints_sac/sac_final.msgpack")
    print(f"  Final checkpoint -> checkpoints_sac/sac_final.msgpack")

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")

if __name__ == "__main__":
    train()