"""SACjax.py — Soft Actor-Critic: shared LidarCNN encoder, decoupled Q1/Q2
branches, prioritized replay, fully fused JIT train chunk."""

import os
import csv
import socket
import argparse

from legnav import paths

# Training is pinned to the `fermi` box (16 GB GPU). The buffer/batch sizes below
# are tuned for its VRAM, so refuse to run anywhere else rather than OOM or run
# an untuned config. Override with LEGNAV_ALLOW_ANY_HOST=1 if you know what you
# are doing.
_HOST = socket.gethostname().split(".")[0].lower()
if _HOST != "fermi" and os.environ.get("LEGNAV_ALLOW_ANY_HOST") != "1":
    raise RuntimeError(
        f"SAC training is pinned to host 'fermi' but this is '{_HOST}'. "
        f"Set LEGNAV_ALLOW_ANY_HOST=1 to override (buffer sizes are tuned for "
        f"fermi's 16 GB GPU and may OOM elsewhere)."
    )

# GPU selection must happen before `import jax` (CUDA_VISIBLE_DEVICES is read
# at import time).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

os.environ.setdefault("JAX_PLATFORMS",               "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")  # no 75% prealloc on the shared card
os.environ.setdefault("TF_GPU_ALLOCATOR",            "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")

# jaxlib 0.9's fusion autotuner takes ~50 min and then hard-crashes on the
# bf16 replay-buffer transpose ("Autotuning failed"). Keep it disabled.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_gpu_experimental_enable_fusion_autotuner=false").strip()

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

jax.config.update("jax_compilation_cache_dir",
                  os.path.expanduser("~/.cache/jax_compilation_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

from legnav.core.jax_env import MAX_STEPS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env

# ══ Observation / action ══════════════════════════════════════════════════════
OBS_SIZE      = 668   # kin(5) + goal_stack(6) + ego_deltas(9) + lidar_stack(648)
ACTION_DIM    = 2
MAX_V_OBS_IDX = 2     # kin_vec[v_norm, w, max_v_norm, ...] → max_v at idx 2

# ══ Training length ═══════════════════════════════════════════════════════════
# Training runs in fixed chunks (one fused train_chunk call each): collect
# COLLECT_STEPS × N_ENVS transitions, then run GRAD_UPDATES_PER_CHUNK gradient
# updates. COLLECT_STEPS is kept small so the buffer holds ~3 chunks of history.
# Buffer config is kept IDENTICAL to TQCjac (300k cap, max-priority insertion):
# TQC trains stably on it, so it is not the cause of SAC's failure, and matching
# it keeps SAC-vs-TQC a clean single-variable comparison (critic head only).
TOTAL_ENV_STEPS        = 70_000_000
N_ENVS                 = 4096
COLLECT_STEPS          = 25
GRAD_UPDATES_PER_CHUNK = 1_000

# Fix 1: Ensure enough warmup steps so every environment completes multiple episodes
WARMUP_STEPS           = N_ENVS * MAX_STEPS * 2

STEPS_PER_CHUNK    = N_ENVS * COLLECT_STEPS                 # 102,400
TOTAL_CHUNKS       = TOTAL_ENV_STEPS // STEPS_PER_CHUNK     # ~683
TOTAL_GRAD_UPDATES = TOTAL_CHUNKS * GRAD_UPDATES_PER_CHUNK  # LR-schedule horizon

# ══ Replay buffer (prioritized: p = |TD|^α, IS weight ∝ (N·P)^-β) ═════════════
BUFFER_CAP     = 300_000   # == TQCjac; fits fermi's 16 GB with wide margin
BATCH_SIZE     = 512
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_EPS        = 1e-6
_BUF_OBS_DTYPE = jnp.bfloat16   # halves obs storage; cast to f32 at sample time

# ══ SAC hyperparameters ═══════════════════════════════════════════════════════
GAMMA         = 0.99
TAU           = 0.005
LR            = 3e-4    # decays linearly to LR*0.1 over the run
ALPHA_FIXED   = 0.005

# Fix 3: Scale down Huber delta to match the new scaled reward magnitude
HUBER_DELTA   = 0.8
MAX_GRAD_NORM = 10.0
LOG_STD_EPS   = 1e-6
ACTOR_ENC_GRAD_SCALE = 0.1   # fraction of actor gradient let into the shared encoder

# ══ Logging / eval / safety stop ══════════════════════════════════════════════
PRINT_EVERY_CHUNKS   = 5
EVAL_EVERY_CHUNKS    = 5
DIVERGENCE_CRIT_LOSS = 200.0   # stop if mean critic loss exceeds this

CKPT_DIR  = paths.checkpoint("sac")
CKPT_PATH = f"{CKPT_DIR}/sac_best.msgpack"


def _check_gpu():
    num_devices = jax.device_count(backend="gpu")
    if num_devices == 0:
        raise RuntimeError("No CUDA devices found.")
    idx = 1 if num_devices >= 2 else 0
    print(f"Found {num_devices} GPU(s). Using CudaDevice({idx})")
    return jax.devices("gpu")[idx]

jax.config.update("jax_default_device", _check_gpu())


# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

@jax.jit
def _vmap_reset(reset_keys, max_goal_dist, ghost_prob, scenario_idx):
    """Batched reset at the CURRENT curriculum — used by the deterministic eval."""
    def _single(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist,
                             scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    return jax.vmap(_single)(reset_keys)

def init_env_state(rng_key, max_goal_dist=1.5, ghost_prob=0.0, scenario_idx=-1):
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    vmap_step = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None, None))
    env_obs, env_state = _vmap_reset(jax.random.split(rng_key, N_ENVS),
                                     jnp.float32(max_goal_dist),
                                     jnp.float32(ghost_prob),
                                     jnp.int32(scenario_idx))
    return env_obs, env_state, vmap_step


# ── Networks: shared encoder + actor / Q1 / Q2 heads ──────────────────────────
from legnav.core.jax_network import SharedEncoder
from legnav.core.precision import bf16_apply, NET_DTYPE, PRECISION_STR

_orth_relu = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
_orth_out  = nn.initializers.orthogonal(scale=0.01)

class SACActorHead(nn.Module):
    """Emits the RAW (pre-tanh) mean and a state-independent log-std. The tanh
    squash happens after noise injection in sample_action, with the matching
    log-Jacobian correction."""
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat):
        raw_mean = nn.Dense(self.action_dim, kernel_init=_orth_out, name="mean")(feat)
        logstd_param = self.param("log_std", nn.initializers.constant(1.0), (self.action_dim,))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) \
                  * (jnp.tanh(jnp.broadcast_to(logstd_param, raw_mean.shape)) + 1.0)
        # float32 out: sample / log-prob math stays full precision.
        return raw_mean.astype(jnp.float32), log_std.astype(jnp.float32)

class CriticBranch(nn.Module):
    @nn.compact
    def __call__(self, feat, action):
        x = jnp.concatenate([feat, action], axis=-1)
        q = nn.LayerNorm()(nn.relu(nn.Dense(256, kernel_init=_orth_relu)(x)))
        q = nn.LayerNorm()(nn.relu(nn.Dense(128, kernel_init=_orth_relu)(q)))
        # float32 out: Bellman / TD math in full precision.
        return jnp.squeeze(nn.Dense(1)(q), axis=-1).astype(jnp.float32)

shared_enc = SharedEncoder()
actor_head = SACActorHead()
critic_q1  = CriticBranch()
critic_q2  = CriticBranch()

# bf16 forward passes (params + activations); outputs come back fp32.
enc_apply   = bf16_apply(shared_enc.apply)
actor_apply = bf16_apply(actor_head.apply)
q1_apply    = bf16_apply(critic_q1.apply)
q2_apply    = bf16_apply(critic_q2.apply)

_lr_sched      = optax.linear_schedule(LR, LR * 0.1, TOTAL_GRAD_UPDATES)
def _make_opt():
    return optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
enc_opt        = _make_opt()
head_actor_opt = _make_opt()
head_q1_opt    = _make_opt()
head_q2_opt    = _make_opt()


# ── Action squashing + exact log-prob ─────────────────────────────────────────
def _log1m_tanh2_stable(u):
    """log(1 - tanh²(u)) = 2·(log2 - u - softplus(-2u)); the naive form
    underflows to log(0) once |u| ≳ 8."""
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))

def _tanh_log_prob_correction(u, max_v):
    """-log|det J| of u → (v, w) with v = (tanh(u_v)+1)·max_v/2, w = tanh(u_w)."""
    log_dv = jnp.log(0.5 * max_v + LOG_STD_EPS) + _log1m_tanh2_stable(u[..., 0])
    log_dw = _log1m_tanh2_stable(u[..., 1])
    return -(log_dv + log_dw)

def sample_action(rng_key, mean, log_std, max_v):
    """Reparameterised tanh-squashed Gaussian sample with exact log-prob.
    Noise is injected in the unbounded latent space, THEN tanh bounds
    v ∈ [0, max_v] and w ∈ [-1, 1] — no clipping plateau. Fully batched."""
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)
    tanh_u = jnp.tanh(u)
    action = jnp.stack([(tanh_u[..., 0] + 1.0) * 0.5 * max_v, tanh_u[..., 1]], axis=-1)
    return action, lp_gauss + _tanh_log_prob_correction(u, max_v)

def deterministic_action(mean, max_v):
    """Policy mode: tanh(raw_mean) mapped to the env action box — identical to
    what the eval scripts reconstruct from a checkpoint."""
    tanh_m = jnp.tanh(mean)
    return jnp.stack([(tanh_m[..., 0] + 1.0) * 0.5 * max_v, tanh_m[..., 1]], axis=-1)

def extract_max_v(obs):
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular, prioritized) ──────────────────────────────
def make_buffer(capacity):
    return {
        "obs":          jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "action":       jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":       jnp.zeros((capacity,),            jnp.float32),
        "next_obs":     jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "terminal":     jnp.zeros((capacity,),            jnp.float32),
        "max_v":        jnp.zeros((capacity,),            jnp.float32),
        "priorities":   jnp.zeros((capacity,),            jnp.float32),
        "max_priority": jnp.float32(1.0),
        # (p+eps)^α per slot, maintained incrementally (0 for empty slots) so
        # buf_sample never re-exponentiates the whole array.
        "p_alpha":      jnp.zeros((capacity,), jnp.float32),
        "ptr":          jnp.int32(0),
        "size":         jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, action, reward, next_obs, terminal, max_v):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    # New transitions get max priority so each is sampled at least once (matches
    # TQCjac). This is what makes fresh collision/goal transitions propagate
    # promptly — mean-priority insertion starved them early.
    new_prio = jnp.broadcast_to(buf["max_priority"], (N,)).astype(jnp.float32)
    return {
        "obs":          buf["obs"].at[idxs].set(obs.astype(_BUF_OBS_DTYPE)),
        "action":       buf["action"].at[idxs].set(action),
        "reward":       buf["reward"].at[idxs].set(reward),
        "next_obs":     buf["next_obs"].at[idxs].set(next_obs.astype(_BUF_OBS_DTYPE)),
        "terminal":     buf["terminal"].at[idxs].set(terminal),
        "max_v":        buf["max_v"].at[idxs].set(max_v),
        "priorities":   buf["priorities"].at[idxs].set(new_prio),
        "max_priority": buf["max_priority"],
        "p_alpha":      buf["p_alpha"].at[idxs].set((new_prio + PER_EPS) ** PER_ALPHA),
        "ptr":          jnp.int32((buf["ptr"] + N) % cap),
        "size":         jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }

@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int, beta):
    # Inverse-CDF sampling (cumsum + searchsorted): same distribution as
    # jax.random.categorical without materializing a (batch, capacity) array.
    # Empty slots have p_alpha == 0 → zero-width intervals, never drawn.
    cdf   = jnp.cumsum(buf["p_alpha"])
    total = cdf[-1]
    u     = jax.random.uniform(rng_key, (batch_size,)) * total
    idxs  = jnp.clip(jnp.searchsorted(cdf, u, side="right"),
                     0, jnp.maximum(buf["size"] - 1, 0))
    sample_probs = buf["p_alpha"][idxs] / (total + 1e-10)
    N = jnp.maximum(buf["size"], 1).astype(jnp.float32)
    weights = (N * sample_probs + 1e-10) ** (-beta)
    weights = weights / (jnp.max(weights) + 1e-10)
    return (buf["obs"][idxs].astype(jnp.float32),
            buf["action"][idxs],
            buf["reward"][idxs],
            buf["next_obs"][idxs].astype(jnp.float32),
            buf["terminal"][idxs],
            buf["max_v"][idxs],
            idxs,
            weights)

@jax.jit
def buf_update_priorities(buf, idxs, td_errors):
    new_prio = jnp.abs(td_errors) + PER_EPS
    return {
        **buf,
        "priorities":   buf["priorities"].at[idxs].set(new_prio),
        "max_priority": jnp.maximum(buf["max_priority"], jnp.max(new_prio)),
        "p_alpha":      buf["p_alpha"].at[idxs].set((new_prio + PER_EPS) ** PER_ALPHA),
    }


# ── SAC update step ───────────────────────────────────────────────────────────
@jax.custom_vjp
def scale_gradient(x, scale):
    return x

def _scale_gradient_fwd(x, scale):
    return x, scale

def _scale_gradient_bwd(scale, g):
    return g * scale, None

scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)


def soft_update(target, online):
    return jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, target, online)


@jax.jit
def sac_update(sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os, tq1p, tq2p,
               obs, action, reward, next_obs, terminal, max_v_obs, max_v_next,
               is_weights, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    # Critic: IS-weighted Huber loss to the entropy-regularised Bellman backup;
    # per-sample |TD| is returned for the PER priority write-back.
    def _huber(err):
        abs_e = jnp.abs(err)
        quad  = jnp.minimum(abs_e, HUBER_DELTA)
        return 0.5 * quad ** 2 + HUBER_DELTA * (abs_e - quad)

    def _critic_loss(sep_, q1p_, q2p_):
        feat_next = jax.lax.stop_gradient(enc_apply({"params": tsep}, next_obs.astype(NET_DTYPE)))
        mean_n, lgs_n = actor_apply({"params": ahp}, feat_next)
        next_act, next_lp = sample_action(rng_c, mean_n, lgs_n, max_v_next)

        q1_t = q1_apply({"params": tq1p}, feat_next, next_act.astype(NET_DTYPE))
        q2_t = q2_apply({"params": tq2p}, feat_next, next_act.astype(NET_DTYPE))
        v_next = jnp.minimum(q1_t, q2_t) - ALPHA_FIXED * next_lp
        backup = jax.lax.stop_gradient(reward + GAMMA * (1.0 - terminal) * v_next)

        feat_obs = enc_apply({"params": sep_}, obs.astype(NET_DTYPE))
        q1 = q1_apply({"params": q1p_}, feat_obs, action.astype(NET_DTYPE))
        q2 = q2_apply({"params": q2p_}, feat_obs, action.astype(NET_DTYPE))
        loss = jnp.mean(is_weights * (_huber(q1 - backup) + _huber(q2 - backup)))
        td_abs = jax.lax.stop_gradient(0.5 * (jnp.abs(q1 - backup) + jnp.abs(q2 - backup)))
        return loss, (jnp.mean(backup), td_abs)

    (c_loss, (q_mean, td_error)), (c_grads_enc, c_grads_q1, c_grads_q2) = jax.value_and_grad(
        _critic_loss, argnums=(0, 1, 2), has_aux=True
    )(sep, q1p, q2p)

    q1_upd, new_q1os = head_q1_opt.update(c_grads_q1, q1os, q1p)
    q2_upd, new_q2os = head_q2_opt.update(c_grads_q2, q2os, q2p)
    new_q1p = optax.apply_updates(q1p, q1_upd)
    new_q2p = optax.apply_updates(q2p, q2_upd)

    def _actor_loss(ahp_, sep_):
        feat = scale_gradient(enc_apply({"params": sep_}, obs.astype(NET_DTYPE)),
                              ACTOR_ENC_GRAD_SCALE)
        mean, log_std = actor_apply({"params": ahp_}, feat)
        action_new, log_pi = sample_action(rng_a, mean, log_std, max_v_obs)
        q1 = q1_apply({"params": jax.lax.stop_gradient(new_q1p)}, feat, action_new.astype(NET_DTYPE))
        q2 = q2_apply({"params": jax.lax.stop_gradient(new_q2p)}, feat, action_new.astype(NET_DTYPE))
        
        action_reg = 1e-3 * jnp.mean(mean ** 2) # L2 reg of mean for avoiding drifting too far from zero
        
        return jnp.mean(ALPHA_FIXED * log_pi - jnp.minimum(q1, q2) + action_reg), jnp.mean(log_pi)

    (a_loss, log_pi_mean), (a_grads_head, a_grads_enc) = jax.value_and_grad(
        _actor_loss, argnums=(0, 1), has_aux=True
    )(ahp, sep)

    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # Encoder gets critic gradients + (scaled) actor gradients.
    enc_grads = jax.tree_util.tree_map(lambda cg, ag: cg + ag, c_grads_enc, a_grads_enc)
    enc_upd, new_eos = enc_opt.update(enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, enc_upd)

    metrics = {"critic_loss": c_loss, "actor_loss": a_loss,
               "log_pi": log_pi_mean, "q_mean": q_mean}
    return (new_sep, new_eos, soft_update(tsep, new_sep), new_ahp, new_ahos,
            new_q1p, new_q1os, new_q2p, new_q2os,
            soft_update(tq1p, new_q1p), soft_update(tq2p, new_q2p),
            td_error, metrics)


# ── Collection step ───────────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ahp, env_state, env_obs, rng_key, vmap_step,
                 max_goal_dist, scenario_idx, ghost_prob, max_scenario):
    max_v = extract_max_v(env_obs)
    k_act, k_step = jax.random.split(rng_key)
    feat = enc_apply({"params": sep}, env_obs.astype(NET_DTYPE))
    mean, log_std = actor_apply({"params": ahp}, feat)
    env_action, _ = sample_action(k_act, mean, log_std, max_v)
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info, max_v


# ── Deterministic evaluation (policy mode, no exploration noise) ──────────────
# 1024 envs × MAX_STEPS completes thousands of episodes — plenty for checkpoint
# selection at a fraction of the cost of evaluating all training envs.
EVAL_ENVS    = 1024
EVAL_HORIZON = MAX_STEPS

@functools.partial(jax.jit, static_argnums=(5, 10))
def deterministic_eval(sep, ahp, env_state, env_obs, rng_key, vmap_step,
                       max_goal_dist, scenario_idx, ghost_prob, max_scenario, horizon):
    """Roll out the policy MODE — measures exactly what the eval scripts see
    when they load the checkpoint."""
    def _body(carry, _):
        es_, eo_, key_ = carry
        feat = enc_apply({"params": sep}, eo_.astype(NET_DTYPE))
        mean, _ = actor_apply({"params": ahp}, feat)
        env_action = deterministic_action(mean, extract_max_v(eo_))
        key_, k_step = jax.random.split(key_)
        step_keys = jax.random.split(k_step, eo_.shape[0])
        new_eo, new_es, rew, done, info = vmap_step(
            step_keys, es_, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        return (new_es, new_eo, key_), (rew, done, info["goal_reached"],
                                        info["collision"], info["passive_col"])
    _, step_data = jax.lax.scan(_body, (env_state, env_obs, rng_key), None, length=horizon)
    return step_data


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
# Donate params, optimizer states, buffer and env state so XLA reuses their
# device memory in place.
@functools.partial(jax.jit, static_argnums=(11,),
                   donate_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14))
def train_chunk(sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os, tq1p, tq2p,
                vmap_step, buf, es, eo, key,
                max_goal_dist, scenario_idx, ghost_prob, max_scenario, beta_per):

    # Phase 1: collect COLLECT_STEPS env steps into the buffer.
    def _collect_body(carry, _):
        es_, eo_, buf_, key_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info, max_v = collect_step(
            sep, ahp, es_, eo_, k_col, vmap_step,
            max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        terminal = done & ~info["timeout"]
        
        # Fix 2: Scale the reward specifically prior to buffer insertion
        new_buf = buf_add(buf_, obs_b, env_a, rew * 0.01, info["final_obs"],
                          terminal.astype(jnp.float32), max_v)
                          
        # Original reward remains in `step_data` for correct training logging
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_), step_data

    (new_es, new_eo, new_buf, key), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key), None, length=COLLECT_STEPS
    )

    # Phase 2: GRAD_UPDATES_PER_CHUNK gradient steps with PER priority write-back.
    def _update_step(carry, upd_idx):
        (sep_, eos_, tsep_, ahp_, ahos_,
         q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, buf_, key_) = carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_term, b_max_v, idxs, weights = buf_sample(
            buf_, k_samp, BATCH_SIZE, beta_per
        )
        (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
         new_q1p_, new_q1os_, new_q2p_, new_q2os_,
         new_tq1p_, new_tq2p_, td_err_, metrics_) = sac_update(
            sep_, eos_, tsep_, ahp_, ahos_, q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_,
            b_obs, b_act, b_rew, b_next, b_term, b_max_v, extract_max_v(b_next),
            weights, k_upd
        )
        new_buf_ = buf_update_priorities(buf_, idxs, td_err_)
        return (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
                new_q1p_, new_q1os_, new_q2p_, new_q2os_,
                new_tq1p_, new_tq2p_, new_buf_, key_), metrics_

    carry_init = (sep, eos, tsep, ahp, ahos,
                  q1p, q1os, q2p, q2os, tq1p, tq2p, new_buf, key)
    (sep, eos, tsep, ahp, ahos,
     q1p, q1os, q2p, q2os, tq1p, tq2p, new_buf, key), all_metrics = jax.lax.scan(
        _update_step, carry_init, jnp.arange(GRAD_UPDATES_PER_CHUNK)
    )

    new_carry = (sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os, tq1p, tq2p,
                 new_buf, new_es, new_eo, key)
    return new_carry, all_step_data, all_metrics


# ── On-GPU episode stats ──────────────────────────────────────────────────────
@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    """Scan step-level data into per-episode outcome flags (emitted at each done)."""
    def _scan(ep_ret, t):
        r, d, g, c, p = t
        ep_ret  = ep_ret + r
        act_col = c & ~p
        is_acol = act_col & ~g
        is_pcol = p & ~g
        is_tmo  = d & ~g & ~act_col & ~p
        outs = (jnp.where(d, ep_ret, 0.0),
                jnp.where(d, g.astype(jnp.float32),       0.0),
                jnp.where(d, is_acol.astype(jnp.float32), 0.0),
                jnp.where(d, is_pcol.astype(jnp.float32), 0.0),
                jnp.where(d, is_tmo.astype(jnp.float32),  0.0),
                d.astype(jnp.float32))
        return jnp.where(d, 0.0, ep_ret), outs

    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(rewards.shape[1]),
        (rewards, dones, goal_reached, collision, passive_col),
    )
    return (ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(),
            ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel())


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                    eos, ahos, q1os, q2os, step, filepath=None):
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
        # α is a fixed constant; key kept for loader compatibility.
        "log_alpha":         float(jnp.log(ALPHA_FIXED)),
        "step":              int(step),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {path}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print("SAC Training  (shared LidarCNN + decoupled Q1/Q2 branches)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  alpha={ALPHA_FIXED} (fixed)")
    print(f"  Precision: {PRECISION_STR} (GPU compute)")
    print(f"  Chunk: {STEPS_PER_CHUNK:,} env steps -> {GRAD_UPDATES_PER_CHUNK} grad updates")
    print(f"  Budget: {TOTAL_ENV_STEPS:,} env steps  ->  ~{TOTAL_CHUNKS} chunks\n")

    master_key = jax.random.PRNGKey(7)
    master_key, k_init, k_env, k_warmup = jax.random.split(master_key, 4)
    train_rng  = jax.random.PRNGKey(42)

    # Init params: shared encoder, actor head, Q1/Q2 branches (+ target copies).
    dummy_obs = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    k_se, k_ah, k_q1, k_q2 = jax.random.split(k_init, 4)

    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)
    dummy_feat = enc_apply({"params": sep}, dummy_obs)
    ahp  = actor_head.init(k_ah, dummy_feat)["params"]
    q1p  = critic_q1.init(k_q1, dummy_feat, dummy_act)["params"]
    q2p  = critic_q2.init(k_q2, dummy_feat, dummy_act)["params"]
    tq1p = jax.tree_util.tree_map(jnp.array, q1p)
    tq2p = jax.tree_util.tree_map(jnp.array, q2p)

    eos  = enc_opt.init(sep)
    ahos = head_actor_opt.init(ahp)
    q1os = head_q1_opt.init(q1p)
    q2os = head_q2_opt.init(q2p)

    from legnav.algorithms.jax_ppo import get_continuous_curriculum
    cur_max_dist, _, _, cur_max_scen = get_continuous_curriculum(0.0)
    cur_ghost    = 0.0   # ghosts start disabled regardless of curriculum
    cur_scenario = 0
    rolling_suc  = 0.0

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(k_env, max_goal_dist=cur_max_dist,
                                                   ghost_prob=cur_ghost)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    chunk       = 0
    best_suc    = 0.0
    best_ret    = -1e9

    print("Warming up buffer with random actions...")
    vmap_step_jit = jax.jit(vmap_step)   # jit once — re-wrapping per iteration would re-trace
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_v, k_w, k_step = jax.random.split(k_warmup, 4)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)
        rand_v     = jax.random.uniform(k_v, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_w, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = vmap_step_jit(
            step_keys, env_state, env_action, cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen)
        )
        terminal   = done & ~info["timeout"].astype(jnp.bool_)
        
        # Fix 2b: Apply the exact same reward scaling in the warmup phase
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward * 0.01,
                             info["final_obs"], terminal.astype(jnp.float32), max_v)
                             
        env_obs     = new_obs
        total_steps += N_ENVS
    print("Warmup done. JIT compiling train_chunk (nested scan — can take minutes)...")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'LogPi':>6} {'Qmean':>7} | "
           f"{'FPS':>7} | {'Time':>8} | {'Dist':>5} {'Scen':>4} {'Gst':>4}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    log_path = paths.checkpoint("sac", "sac_training_log.csv")
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                         "pcol_pct", "tmo_pct", "n_ep"])
    log_file.flush()

    while total_steps < TOTAL_ENV_STEPS:
        t0 = time.time()

        # PER IS-weight exponent β anneals linearly to 1 over the run.
        frac = min(1.0, total_steps / TOTAL_ENV_STEPS)
        beta_per = PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

        new_carry, all_step_data, all_metrics = train_chunk(
            sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os, tq1p, tq2p,
            vmap_step, replay_buf, env_state, env_obs, train_rng,
            cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen), jnp.float32(beta_per)
        )
        (sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os, tq1p, tq2p,
         replay_buf, env_state, env_obs, train_rng) = new_carry

        chunk       += 1
        n_updates   += GRAD_UPDATES_PER_CHUNK
        total_steps += STEPS_PER_CHUNK

        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = \
            collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())

        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col  * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0

            # Curriculum: difficulty tracks rolling success in BOTH directions,
            # rate-limited per chunk, so a collapse eases difficulty back
            # instead of drowning the buffer in max-difficulty failures.
            rolling_suc = 0.80 * rolling_suc + 0.20 * suc_pct
            new_max_dist, new_ghost, _, new_max_scen = get_continuous_curriculum(rolling_suc)

            if new_max_dist > cur_max_dist:
                cur_max_dist = min(new_max_dist, cur_max_dist + 0.3)
            elif new_max_dist < cur_max_dist:
                cur_max_dist = max(new_max_dist, cur_max_dist - 0.3)
            if new_ghost > cur_ghost:
                cur_ghost = min(new_ghost, cur_ghost + 0.02)
            elif new_ghost < cur_ghost:
                cur_ghost = max(new_ghost, cur_ghost - 0.02)
            if new_max_scen > cur_max_scen:
                cur_max_scen = min(new_max_scen, cur_max_scen + 1)
            elif new_max_scen < cur_max_scen:
                cur_max_scen = max(new_max_scen, cur_max_scen - 1)

            cur_scenario = -1   # -1 → uniform draw in [0, cur_max_scen] at each reset
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = STEPS_PER_CHUNK / (time.time() - t0)
        elapsed = time.time() - t_start
        h, rem  = divmod(int(elapsed), 3600)
        m, s    = divmod(rem, 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

        m_crit = float(all_metrics["critic_loss"].mean())
        if chunk == 1 or chunk % PRINT_EVERY_CHUNKS == 0:
            print(
                f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{m_crit:>7.4f} "
                f"{float(all_metrics['actor_loss'].mean()):>7.4f} "
                f"{float(all_metrics['log_pi'].mean()):>6.3f} "
                f"{float(all_metrics['q_mean'].mean()):>7.3f} | "
                f"{fps:>7,.0f} | "
                f"{elapsed_str:>8} | "
                f"{cur_max_dist:>5.1f} {cur_max_scen:>4d} {cur_ghost:>4.2f}"
            )

        log_writer.writerow([total_steps, round(mean_ret, 4),
                             round(suc_pct, 4), round(col_pct, 4),
                             round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        log_file.flush()

        if m_crit > DIVERGENCE_CRIT_LOSS:
            print(f"  !! Critic loss {m_crit:.1f} > {DIVERGENCE_CRIT_LOSS:.0f} — training diverged, stopping early.")
            break

        # Best-checkpoint selection: training suc% is measured under exploration
        # noise and understates the mode policy — evaluate tanh(mean) on fresh
        # resets at the current curriculum instead.
        curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
        if curriculum_mature and (chunk == 1 or chunk % EVAL_EVERY_CHUNKS == 0):
            master_key, k_evr, k_ev = jax.random.split(master_key, 3)
            ev_obs, ev_state = _vmap_reset(jax.random.split(k_evr, EVAL_ENVS),
                                           jnp.float32(cur_max_dist),
                                           jnp.float32(cur_ghost),
                                           jnp.int32(cur_scenario))
            ev_data = deterministic_eval(
                sep, ahp, ev_state, ev_obs, k_ev, vmap_step,
                jnp.float32(cur_max_dist), jnp.int32(cur_scenario),
                jnp.float32(cur_ghost), jnp.int32(cur_max_scen), EVAL_HORIZON)
            _, ev_suc, _, _, _, ev_msk = collect_episode_outcomes(*ev_data)
            n_ev = float(ev_msk.sum())
            det_suc = float((ev_suc * ev_msk).sum() / n_ev) * 100.0 if n_ev > 0 else 0.0
            print(f"    [det-eval] mode success {det_suc:5.1f}%  (n={int(n_ev)} ep)  best={best_suc:.1f}%")
            if det_suc > best_suc:
                best_suc = det_suc
                best_ret = mean_ret
                save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                                eos, ahos, q1os, q2os, n_updates)

    # Final checkpoint == best checkpoint: the live params at the end of a SAC
    # run can be worse than the best on disk.
    import shutil
    final_path = paths.checkpoint("sac", "sac_final.msgpack")
    if os.path.exists(CKPT_PATH):
        shutil.copyfile(CKPT_PATH, final_path)
        print(f"  Final checkpoint -> {final_path} (copied from best, {best_suc:.1f}%)")
    else:
        save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                        eos, ahos, q1os, q2os, n_updates, filepath=final_path)
        print(f"  Final checkpoint -> {final_path} (no best found, saved current)")

    print(f"\nSAC done! {(time.time() - t_start)/3600:.2f}h | "
          f"Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    log_file.close()
    print(f"Training log saved -> {log_path}")


if __name__ == "__main__":
    train()