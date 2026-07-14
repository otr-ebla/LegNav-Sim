"""
SACjax.py — Soft Actor-Critic (shared encoder, fused-JIT train chunk, PER,
reward normalization)
"""

import os
import csv

# GPU selection — must happen BEFORE import jax (JAX reads CUDA_VISIBLE_DEVICES
# at import time).
import argparse as _ap
from legnav import paths
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

os.environ.setdefault("JAX_PLATFORMS",               "cuda")
# cuda_async allocator: the default preallocating BFC pool grabs 75% VRAM,
# too tight next to the ~7GB this run needs on a shared 10GB card.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("TF_GPU_ALLOCATOR",            "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")

# jaxlib 0.9's fusion autotuner spends ~50 min compiling the env-reset reduce
# fusion, then hard-crashes on the 1M-slot bf16 replay-buffer transpose
# ("Autotuning failed ... No valid config found!"). Keep it disabled.
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

# Persistent compilation cache — train_chunk compiles take minutes otherwise.
jax.config.update("jax_compilation_cache_dir",
                  os.path.expanduser("~/.cache/jax_compilation_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

from legnav.core.jax_env import MAX_STEPS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env

# ══ Observation / action ══════════════════════════════════════════════════════
OBS_SIZE      = 668   # kin(5) + goal_stack(6) + ego_deltas(9) + lidar_stack(648)
ACTION_DIM    = 2
MAX_V_OBS_IDX = 2     # kin_vec[v_norm, w, max_v_norm, ...] at obs head → idx 2

# ══ Training length ═══════════════════════════════════════════════════════════
# Training runs in fixed-size "chunks" (one fused train_chunk call). Each chunk:
#   1. collects COLLECT_STEPS steps in each of N_ENVS envs → STEPS_PER_CHUNK
#      new transitions into the replay buffer
#   2. runs GRAD_UPDATES_PER_CHUNK gradient updates on BATCH_SIZE replay batches
# The loop stops when TOTAL_ENV_STEPS collected env steps are reached (the
# WARMUP_STEPS random-action steps that seed the buffer count toward the budget).
TOTAL_ENV_STEPS        = 12_000_000
N_ENVS                 = 2048
COLLECT_STEPS          = 50
GRAD_UPDATES_PER_CHUNK = 2_500
WARMUP_STEPS           = 10_000

STEPS_PER_CHUNK    = N_ENVS * COLLECT_STEPS                 # 102,400
TOTAL_CHUNKS       = TOTAL_ENV_STEPS // STEPS_PER_CHUNK     # ~117
TOTAL_GRAD_UPDATES = TOTAL_CHUNKS * GRAD_UPDATES_PER_CHUNK  # LR-schedule horizon

# ══ Replay buffer ═════════════════════════════════════════════════════════════
BUFFER_CAP = 1_000_000
BATCH_SIZE = 512

# Prioritized Experience Replay: priorities are |TD-error|+ε; sampling
# prob ∝ p^α; IS weight ∝ (N·P)^(-β), β annealed linearly over the run.
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_EPS        = 1e-6

_BUF_OBS_DTYPE = jnp.bfloat16  # halves obs storage; cast back to float32 at sample time

# ══ SAC hyperparameters ═══════════════════════════════════════════════════════
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4       # decays linearly to LR*0.1 over TOTAL_GRAD_UPDATES
ALPHA_LR       = 1e-4
TARGET_ENTROPY = -1.0
MAX_GRAD_NORM  = 10.0
LOG_STD_EPS    = 1e-6
# Fraction of the actor's gradient allowed into the shared encoder (matches TQCjac).
ACTOR_ENC_GRAD_SCALE = 0.1

# ══ Logging / eval / safety stops ═════════════════════════════════════════════
PRINT_EVERY_CHUNKS   = 1     # progress line cadence (CSV logs every chunk)
EVAL_EVERY_CHUNKS    = 10    # deterministic-eval / best-checkpoint cadence
DIVERGENCE_CRIT_LOSS = 200.0 # stop if mean critic loss exceeds this
DIVERGENCE_SUC_DROP  = 20.0  # stop if rolling success sits this far below best for 2 chunks

CKPT_DIR  = paths.checkpoint("sac")
CKPT_PATH = f"{CKPT_DIR}/sac_best.msgpack"


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

@jax.jit
def _vmap_reset(reset_keys, max_goal_dist, ghost_prob, scenario_idx):
    """Batched reset with traced curriculum args — used by the deterministic
    eval so fresh episodes are drawn at the CURRENT curriculum settings."""
    def _single(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    return jax.vmap(_single)(reset_keys)

def init_env_state(rng_key, max_goal_dist: float = 1.5, ghost_prob: float = 0.0, scenario_idx: int = -1):
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    vmap_step = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None, None))

    def _reset(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    vmap_reset = jax.jit(jax.vmap(_reset))

    env_obs, env_state = vmap_reset(jax.random.split(rng_key, N_ENVS))
    return env_obs, env_state, vmap_step


# ── Shared Encoder + Actor/Critic Heads ──────────────────────────────────────
from legnav.core.jax_network import SharedEncoder
from legnav.core.precision import bf16_apply, NET_DTYPE, PRECISION_STR

_orth_relu = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
_orth_out  = nn.initializers.orthogonal(scale=0.01)

class SACActorHead(nn.Module):
    """Squashed-Gaussian actor: emits the RAW (pre-tanh) mean and log-std.
    The tanh squash is applied AFTER noise injection in
    `sample_action_sac_batched`, with the matching log-Jacobian correction."""
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat: jnp.ndarray):
        raw_mean = nn.Dense(self.action_dim, kernel_init=_orth_out, name='mean')(feat)

        logstd_param = self.param('log_std', nn.initializers.constant(1.0), (self.action_dim,))
        actor_logstd_raw = jnp.broadcast_to(logstd_param, raw_mean.shape)
        actor_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (jnp.tanh(actor_logstd_raw) + 1.0)

        # Emit float32 so downstream sample / log-prob math stays full precision.
        return raw_mean.astype(jnp.float32), actor_logstd.astype(jnp.float32)

class CriticBranch(nn.Module):
    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray):
        x = jnp.concatenate([feat, action], axis=-1)
        q = nn.LayerNorm()(nn.relu(nn.Dense(256, kernel_init=_orth_relu)(x)))
        q = nn.LayerNorm()(nn.relu(nn.Dense(128, kernel_init=_orth_relu)(q)))
        # float32 out: Bellman / TD-error math in full precision regardless of encoder dtype
        return jnp.squeeze(nn.Dense(1)(q), axis=-1).astype(jnp.float32)

shared_enc = SharedEncoder()
actor_head  = SACActorHead()
critic_q1   = CriticBranch()
critic_q2   = CriticBranch()

# bf16 forward passes (params + activations); outputs come back fp32.
enc_apply    = bf16_apply(shared_enc.apply)
actor_apply  = bf16_apply(actor_head.apply)
q1_apply     = bf16_apply(critic_q1.apply)
q2_apply     = bf16_apply(critic_q2.apply)


# ── Optimisers ────────────────────────────────────────────────────────────────
# LR decays linearly to 10% over the run: Adam runs too hot in the late phase —
# success peaks then drifts down as Q-estimates churn. Steps once per gradient
# update; horizon is TOTAL_GRAD_UPDATES (fixed at import from TOTAL_ENV_STEPS).
_lr_sched = optax.linear_schedule(LR, LR * 0.1, TOTAL_GRAD_UPDATES)
enc_opt        = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
head_actor_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
head_q1_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
head_q2_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
alpha_opt      = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(ALPHA_LR, eps=1e-5))

# ── Reward normalization (running RMS, Welford) ───────────────────────────────
@jax.jit
def update_reward_rms(mean, var, count, batch):
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
def _log1m_tanh2_stable(u):
    """log(1 - tanh^2(u)) = 2*(log2 - u - softplus(-2u)); the naive form
    underflows to log(0) once |u| ≳ 8."""
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


def _tanh_log_prob_correction(u, max_v):
    """-log|det J| for u → action:
        v = (tanh(u_v) + 1) * 0.5 * max_v,   w = tanh(u_w).
    Ready to be ADDED to the base Gaussian log-prob."""
    log_dv = jnp.log(0.5 * max_v + LOG_STD_EPS) + _log1m_tanh2_stable(u[..., 0])
    log_dw = _log1m_tanh2_stable(u[..., 1])
    return -(log_dv + log_dw)


def sample_action_sac_batched(rng_key, mean, log_std, max_v):
    """Reparameterised tanh-squashed Gaussian sample with exact log-prob.
    Noise is injected in the unbounded latent space (u = mean + noise * std),
    THEN tanh bounds v ∈ [0, max_v] and w ∈ [-1, 1] — no clipping plateau."""
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)

    tanh_u = jnp.tanh(u)
    a_v = (tanh_u[..., 0] + 1.0) * 0.5 * max_v
    a_w = tanh_u[..., 1]
    return jnp.stack([a_v, a_w], axis=-1), lp_gauss + _tanh_log_prob_correction(u, max_v)

def extract_max_v(obs):
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular, prioritized) ──────────────────────────────
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),             jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "terminal": jnp.zeros((capacity,),             jnp.float32),
        "max_v":    jnp.zeros((capacity,),             jnp.float32),
        "priorities":   jnp.zeros((capacity,),         jnp.float32),
        "max_priority": jnp.float32(1.0),
        # (priorities[i]+eps)^alpha per slot (0 for empty slots), maintained
        # incrementally so buf_sample never re-exponentiates the full array.
        "p_alpha":  jnp.zeros((capacity,), jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, action, reward, next_obs, terminal, max_v):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    # New transitions inherit current max priority so they are sampled at least once.
    new_prio = jnp.broadcast_to(buf["max_priority"], (N,)).astype(jnp.float32)
    new_pa = (new_prio + PER_EPS) ** PER_ALPHA
    return {
        "obs":      buf["obs"].at[idxs].set(obs.astype(_BUF_OBS_DTYPE)),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs.astype(_BUF_OBS_DTYPE)),
        "terminal": buf["terminal"].at[idxs].set(terminal),
        "max_v":    buf["max_v"].at[idxs].set(max_v),
        "priorities":   buf["priorities"].at[idxs].set(new_prio),
        "max_priority": buf["max_priority"],
        "p_alpha":  buf["p_alpha"].at[idxs].set(new_pa),
        "ptr":      jnp.int32((buf["ptr"] + N) % cap),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }

@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int, beta):
    # Inverse-CDF categorical sampling: jax.random.categorical over the full
    # buffer would materialize a (batch, capacity) Gumbel array (~2GB at 1M cap);
    # cumsum + searchsorted draws from the same distribution for a fraction of
    # the work. Empty slots have p_alpha == 0 → zero-width intervals, never drawn.
    cdf = jnp.cumsum(buf["p_alpha"])
    total = cdf[-1]
    u = jax.random.uniform(rng_key, (batch_size,)) * total
    idxs = jnp.searchsorted(cdf, u, side="right")
    idxs = jnp.clip(idxs, 0, jnp.maximum(buf["size"] - 1, 0))
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
    new_pa = (new_prio + PER_EPS) ** PER_ALPHA
    return {
        **buf,
        "priorities":   buf["priorities"].at[idxs].set(new_prio),
        "max_priority": jnp.maximum(buf["max_priority"], jnp.max(new_prio)),
        "p_alpha":      buf["p_alpha"].at[idxs].set(new_pa),
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


@jax.jit
def sac_update(sep, eos, tsep, ahp, ahos, q1p, q1os, q2p, q2os,
               tq1p, tq2p, la, alo,
               obs, action, reward, next_obs, terminal, max_v_obs, max_v_next, is_weights, rng_key):

    rng_c, rng_a = jax.random.split(rng_key)

    # 1. Critic loss (IS-weighted; per-sample TD error returned for PER priorities)
    def _critic_loss(sep_, q1p_, q2p_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        feat_next = jax.lax.stop_gradient(enc_apply({"params": tsep}, next_obs.astype(NET_DTYPE)))
        mean_n, lgs_n = actor_apply({"params": ahp}, feat_next)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, max_v_next)

        q1_t = q1_apply({"params": tq1p}, feat_next, next_act.astype(NET_DTYPE))
        q2_t = q2_apply({"params": tq2p}, feat_next, next_act.astype(NET_DTYPE))
        v_next = jnp.minimum(q1_t, q2_t) - alpha * next_lp

        backup = jax.lax.stop_gradient(reward + GAMMA * (1.0 - terminal) * v_next)

        feat_obs = enc_apply({"params": sep_}, obs.astype(NET_DTYPE))
        q1 = q1_apply({"params": q1p_}, feat_obs, action.astype(NET_DTYPE))
        q2 = q2_apply({"params": q2p_}, feat_obs, action.astype(NET_DTYPE))
        per_sample = (q1 - backup) ** 2 + (q2 - backup) ** 2
        weighted_loss = jnp.mean(is_weights * per_sample)
        td_abs = jax.lax.stop_gradient(0.5 * (jnp.abs(q1 - backup) + jnp.abs(q2 - backup)))
        return weighted_loss, (jnp.mean(backup), td_abs)

    (c_loss, (q_mean, td_error)), (c_grads_enc, c_grads_q1, c_grads_q2) = jax.value_and_grad(
        _critic_loss, argnums=(0, 1, 2), has_aux=True
    )(sep, q1p, q2p)

    q1_upd, new_q1os = head_q1_opt.update(c_grads_q1, q1os, q1p)
    q2_upd, new_q2os = head_q2_opt.update(c_grads_q2, q2os, q2p)
    new_q1p = optax.apply_updates(q1p, q1_upd)
    new_q2p = optax.apply_updates(q2p, q2_upd)

    # 2. Actor loss
    def _actor_loss(ahp_, sep_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        feat_raw = enc_apply({"params": sep_}, obs.astype(NET_DTYPE))
        feat_f   = scale_gradient(feat_raw, ACTOR_ENC_GRAD_SCALE)
        mean, log_std = actor_apply({"params": ahp_}, feat_f)
        action_new, log_pi = sample_action_sac_batched(rng_a, mean, log_std, max_v_obs)
        q1 = q1_apply({"params": jax.lax.stop_gradient(new_q1p)}, feat_f, action_new.astype(NET_DTYPE))
        q2 = q2_apply({"params": jax.lax.stop_gradient(new_q2p)}, feat_f, action_new.astype(NET_DTYPE))
        return jnp.mean(alpha * log_pi - jnp.minimum(q1, q2)), jnp.mean(log_pi)

    (a_loss, log_pi_mean), (a_grads_head, a_grads_enc_actor) = jax.value_and_grad(
        _actor_loss, argnums=(0, 1), has_aux=True
    )(ahp, sep)

    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # 3. Encoder update (critic + scaled actor gradients)
    combined_enc_grads = jax.tree_util.tree_map(
        lambda cg, ag: cg + ag, c_grads_enc, a_grads_enc_actor
    )
    enc_upd, new_eos = enc_opt.update(combined_enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, enc_upd)

    # 4. Alpha update
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    # Cap α ≤ 0.2 to kill the late-training runaway (α explodes → Q-targets blow
    # up → policy collapses).
    new_la = jnp.minimum(optax.apply_updates(la, al_upd), jnp.log(0.2))

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
            new_tq1p, new_tq2p, new_la, new_alo, td_error, metrics)


# ── Collection step ───────────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ahp, env_state, env_obs, rng_key, vmap_step, max_goal_dist, scenario_idx, ghost_prob, max_scenario):
    max_v = extract_max_v(env_obs)
    k_act, k_step = jax.random.split(rng_key)
    feat = enc_apply({"params": sep}, env_obs.astype(NET_DTYPE))
    mean, log_std = actor_apply({"params": ahp}, feat)
    env_action, _ = sample_action_sac_batched(k_act, mean, log_std, max_v)
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info, max_v


# ── Deterministic evaluation (policy mode, no exploration noise) ──────────────
def deterministic_action(mean, max_v):
    """Policy mode: tanh(raw_mean) mapped to the env action box — identical to
    the action the eval scripts reconstruct from the checkpoint."""
    tanh_m = jnp.tanh(mean)
    return jnp.stack([(tanh_m[..., 0] + 1.0) * 0.5 * max_v, tanh_m[..., 1]], axis=-1)

EVAL_HORIZON = 2 * MAX_STEPS   # steps per deterministic eval (autoreset → many episodes)

@functools.partial(jax.jit, static_argnums=(5, 10))
def deterministic_eval(sep, ahp, env_state, env_obs, rng_key, vmap_step,
                       max_goal_dist, scenario_idx, ghost_prob, max_scenario, horizon):
    """Roll out the policy MODE — measures exactly what jax_eval_multi /
    benchmark_eval see at checkpoint-load time."""
    def _body(carry, _):
        es_, eo_, key_ = carry
        feat = enc_apply({"params": sep}, eo_.astype(NET_DTYPE))
        mean, _ = actor_apply({"params": ahp}, feat)
        env_action = deterministic_action(mean, extract_max_v(eo_))
        key_, k_step = jax.random.split(key_)
        step_keys = jax.random.split(k_step, N_ENVS)
        new_eo, new_es, rew, done, info = vmap_step(
            step_keys, es_, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        return (new_es, new_eo, key_), (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
    _, step_data = jax.lax.scan(_body, (env_state, env_obs, rng_key), None, length=horizon)
    return step_data


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
# Donate params, optimizer states, buffer and env state so XLA reuses their
# device buffers in place — meaningful when the buffer alone is ~3GB at 1M cap.
@functools.partial(jax.jit, static_argnums=(13,),
                   donate_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 22, 23, 24))
def train_chunk(sep, eos, tsep, ahp, ahos,
                q1p, q1os, q2p, q2os, tq1p, tq2p,
                la, alo, vmap_step,
                buf, es, eo, key,
                max_goal_dist, scenario_idx, ghost_prob,
                max_scenario, rmean, rvar, rcount, beta_per):

    # Phase 1: collect COLLECT_STEPS env steps, write to buffer
    def _collect_body(carry, _):
        es_, eo_, buf_, key_, rm_, rv_, rc_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info, max_v_cur = collect_step(
            sep, ahp, es_, eo_, k_col, vmap_step, max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        terminal = done & ~info["timeout"]
        new_buf = buf_add(buf_, obs_b, env_a, rew, new_eo,
                          terminal.astype(jnp.float32), max_v_cur)

        new_rm, new_rv, new_rc = update_reward_rms(rm_, rv_, rc_, rew)

        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_, new_rm, new_rv, new_rc), step_data

    (new_es, new_eo, new_buf, key, new_rmean, new_rvar, new_rcount), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key, rmean, rvar, rcount), None, length=COLLECT_STEPS
    )

    # Phase 2: GRAD_UPDATES_PER_CHUNK gradient steps, PER sample + priority write-back
    def _update_step(update_carry, upd_idx):
        (sep_, eos_, tsep_, ahp_, ahos_,
         q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_, buf_, key_) = update_carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_terminal, b_max_v, idxs, weights = buf_sample(
            buf_, k_samp, BATCH_SIZE, beta_per
        )
        b_max_v_next = extract_max_v(b_next)

        norm_b_rew = b_rew / jnp.sqrt(new_rvar + 1e-8)

        (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
         new_q1p_, new_q1os_, new_q2p_, new_q2os_,
         new_tq1p_, new_tq2p_, new_la_, new_alo_, td_err_, metrics_) = sac_update(
            sep_, eos_, tsep_, ahp_, ahos_,
            q1p_, q1os_, q2p_, q2os_, tq1p_, tq2p_, la_, alo_,
            b_obs, b_act, norm_b_rew, b_next, b_terminal, b_max_v, b_max_v_next, weights, k_upd
        )
        new_buf_ = buf_update_priorities(buf_, idxs, td_err_)
        return (new_sep_, new_eos_, new_tsep_, new_ahp_, new_ahos_,
                new_q1p_, new_q1os_, new_q2p_, new_q2os_,
                new_tq1p_, new_tq2p_, new_la_, new_alo_, new_buf_, key_), metrics_

    update_carry_init = (sep, eos, tsep, ahp, ahos,
                         q1p, q1os, q2p, q2os, tq1p, tq2p, la, alo, new_buf, key)
    (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
     new_q1p, new_q1os, new_q2p, new_q2os,
     new_tq1p, new_tq2p, new_la, new_alo, new_buf2, key), all_metrics = jax.lax.scan(
        _update_step, update_carry_init, jnp.arange(GRAD_UPDATES_PER_CHUNK)
    )

    new_carry = (new_sep, new_eos, new_tsep, new_ahp, new_ahos,
                 new_q1p, new_q1os, new_q2p, new_q2os, new_tq1p, new_tq2p,
                 new_la, new_alo, new_buf2, new_es, new_eo, key,
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

def train():
    """Run SAC training for the TOTAL_ENV_STEPS budget (see Training length
    block at the top of the file — the LR schedule horizon is derived from the
    same constants at import time, so change them there rather than here)."""
    print("SAC Training  (shared LidarCNN + decoupled Q1/Q2 branches)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}")
    print(f"  Precision: {PRECISION_STR} (GPU compute)")
    print(f"  Chunk: {STEPS_PER_CHUNK:,} env steps -> {GRAD_UPDATES_PER_CHUNK} grad updates")
    print(f"  Budget: {TOTAL_ENV_STEPS:,} env steps  ->  ~{TOTAL_CHUNKS} chunks\n")

    master_key = jax.random.PRNGKey(7)
    master_key, k_init, k_env, k_warmup = jax.random.split(master_key, 4)
    train_rng  = jax.random.PRNGKey(42)

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

    la   = jnp.array(0.0, dtype=jnp.float32)
    alo  = alpha_opt.init(la)

    from legnav.algorithms.jax_ppo import get_continuous_curriculum
    cur_max_dist, _, _, cur_max_scen = get_continuous_curriculum(0.0)
    cur_ghost = 0.0   # ghosts start disabled regardless of curriculum
    rolling_suc = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(k_env, max_goal_dist=cur_max_dist, ghost_prob=cur_ghost)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    chunk       = 0
    best_suc    = 49.5
    best_ret    = -1e9

    r_mean  = jnp.zeros(())
    r_var   = jnp.ones(())
    r_count = jnp.zeros(())

    print("Warming up buffer with random actions...")
    # jit once outside the loop — jax.jit(vmap_step) inside the loop would
    # create a fresh wrapper (and jit cache) per iteration, re-tracing each time.
    vmap_step_jit = jax.jit(vmap_step)
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_act, k_step = jax.random.split(k_warmup, 3)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)
        rand_v     = jax.random.uniform(k_act, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_act, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = vmap_step_jit(
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

    _LOG_PATH = paths.checkpoint("sac", "sac_training_log.csv")
    os.makedirs(paths.checkpoint("sac"), exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    # rolling_suc (training rollouts) is the smoothed early-stop signal;
    # best-checkpoint selection uses the deterministic eval below.
    bad_chunks = 0

    while total_steps < TOTAL_ENV_STEPS:
        t0 = time.time()

        # Anneal PER IS-weight exponent β from PER_BETA_START → PER_BETA_END.
        frac = min(1.0, total_steps / TOTAL_ENV_STEPS)
        beta_per = PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

        new_carry, all_step_data, all_metrics = train_chunk(
            sep, eos, tsep, ahp, ahos,
            q1p, q1os, q2p, q2os, tq1p, tq2p,
            la, alo, vmap_step,
            replay_buf, env_state, env_obs, train_rng,
            cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen), r_mean, r_var, r_count,
            jnp.float32(beta_per)
        )
        (sep, eos, tsep, ahp, ahos,
         q1p, q1os, q2p, q2os, tq1p, tq2p,
         la, alo, replay_buf, env_state, env_obs, train_rng,
         r_mean, r_var, r_count) = new_carry

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

            # Curriculum: gate on CURRENT rolling success (not the all-time high)
            # so advancement pauses after each difficulty bump until the policy
            # consolidates, and cap the per-chunk delta on each axis so a success
            # spike can't slam dist+scen+ghost up at once. Difficulty itself
            # never decreases — cur_* only ratchets up.
            rolling_suc         = 0.80 * rolling_suc + 0.20 * suc_pct
            highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

            new_max_dist, new_ghost, _, new_max_scen = get_continuous_curriculum(rolling_suc)

            if new_max_dist > cur_max_dist:
                cur_max_dist = min(new_max_dist, cur_max_dist + 0.3)
            if new_ghost > cur_ghost:
                cur_ghost = min(new_ghost, cur_ghost + 0.02)
            if new_max_scen > cur_max_scen:
                cur_max_scen = min(new_max_scen, cur_max_scen + 1)

            cur_scenario = -1
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = STEPS_PER_CHUNK / (time.time() - t0)
        elapsed = time.time() - t_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

        m_crit = float(all_metrics['critic_loss'].mean())
        if chunk == 1 or chunk % PRINT_EVERY_CHUNKS == 0:
            print(
                f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{m_crit:>7.4f} "
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

        if m_crit > DIVERGENCE_CRIT_LOSS:
            print(f"  !! Critic loss {m_crit:.1f} > {DIVERGENCE_CRIT_LOSS:.0f} — training diverged, stopping early.")
            break
        if n_ep > 0 and rolling_suc < highest_rolling_suc - DIVERGENCE_SUC_DROP:
            bad_chunks += 1
            if bad_chunks >= 2:
                print(f"  !! Rolling success {rolling_suc:.1f}% is ≥{DIVERGENCE_SUC_DROP:.0f}pts below best ({highest_rolling_suc:.1f}%) for 2 chunks — stopping early.")
                break
        else:
            bad_chunks = 0

        # Best-checkpoint selection: training suc% is measured under exploration
        # noise and understates the mode policy — evaluate tanh(mean) on fresh
        # resets at the current curriculum and select checkpoints on that.
        curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
        if curriculum_mature and (chunk == 1 or chunk % EVAL_EVERY_CHUNKS == 0):
            master_key, k_evr, k_ev = jax.random.split(master_key, 3)
            ev_obs, ev_state = _vmap_reset(jax.random.split(k_evr, N_ENVS),
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
                                eos, ahos, q1os, q2os, la, alo, n_updates)

    # Final checkpoint == best checkpoint: the tail of SAC training often
    # diverges, so the live params at the end can be worse than the best on disk.
    import shutil
    if os.path.exists(CKPT_PATH):
        shutil.copyfile(CKPT_PATH, paths.checkpoint("sac", "sac_final.msgpack"))
        print(f"  Final checkpoint -> checkpoints_sac/sac_final.msgpack (copied from best, {best_suc:.1f}%)")
    else:
        save_checkpoint(sep, tsep, ahp, q1p, q2p, tq1p, tq2p,
                        eos, ahos, q1os, q2os, la, alo, n_updates,
                        filepath=paths.checkpoint("sac", "sac_final.msgpack"))
        print(f"  Final checkpoint -> checkpoints_sac/sac_final.msgpack (no best found, saved current)")

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%  Best reward: {best_ret:.1f}")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")

if __name__ == "__main__":
    train()
