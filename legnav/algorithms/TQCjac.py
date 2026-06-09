import os
import sys
import csv

# ── GPU selection — must happen BEFORE import jax ────────────────────────────
# JAX reads CUDA_VISIBLE_DEVICES at import time, so we parse --gpu here
# with a minimal parser rather than relying on argparse later.
import argparse as _ap
from legnav import paths
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

# Use setdefault so an orchestrator script (or the user's shell) can pre-set
# these before importing this module without being overwritten.
os.environ.setdefault("JAX_PLATFORMS",               "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR",            "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")

# Default to bfloat16: engages Ampere Tensor Cores (RTX 3080) for ~2x throughput
# on encoder/critic matmuls and halves activation memory. Set TQC_BFLOAT16=0 to
# fall back to fp32 if numerical issues arise.
_USE_BFLOAT16 = os.environ.get("TQC_BFLOAT16", "1") == "1"

import time
import warnings
import functools
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import flax.serialization

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Human model / sensor config — MUST be set before importing jax_env_multi ──
# jax_env_multi does `from jax_env import USE_LEGS`, binding its own name at import
# time, so we flip the flag on jax_env *first*. Training now matches the eval
# default (leg-pair human model + salt&pepper LiDAR noise).
import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS    = True
_jax_env.SENSOR_NOISE = True

from legnav.core.jax_env import MAX_STEPS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env

OBS_SIZE       = 662
ACTION_DIM     = 2
N_ENVS         = 4096
BUFFER_CAP     = 300_000    # 300K — ~1.5 GB for obs+next_obs (fits 8GB VRAM)
BATCH_SIZE     = 512
G_UPDATES      = 20
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4
MAX_GRAD_NORM  = 10.0
TARGET_ENTROPY = -2.0
ACTOR_ENC_GRAD_SCALE = 0.1
DEFAULT_TOTAL_ENV_STEPS = 70_000_000   # default budget; override via train(total_env_steps=...)
LOG_EVERY      = 50
SAVE_EVERY     = 5000

N_CRITICS        = 5
N_ATOMS          = 25
N_TOP_ATOMS_DROP = 3
N_TARGET_ATOMS   = N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP
HUBER_KAPPA      = 1.0

# ── Prioritized Experience Replay ──────────────────────────────────────────
# Priorities are |TD-error|+ε; sampling prob ∝ p^α; IS weight ∝ (N·P)^(-β).
PER_ALPHA      = 0.6   # priority exponent (0 = uniform, 1 = fully prioritized)
PER_BETA_START = 0.4   # IS exponent at training start
PER_BETA_END   = 1.0   # IS exponent at training end (linearly annealed)
PER_EPS        = 1e-6  # priority floor

MAX_V_OBS_IDX = 11  # stacked obs: pose_stack(9) + state_vec[v_norm, w, max_v_norm, ...] → idx 9+2=11
LOG_STD_EPS   = 1e-6

CKPT_DIR  = paths.checkpoint("tqc")
CKPT_PATH = f"{CKPT_DIR}/tqc_best.msgpack"

NET_DTYPE = jnp.bfloat16 if _USE_BFLOAT16 else jnp.float32

from legnav.algorithms.jax_ppo import get_continuous_curriculum

def _check_gpu():
    try: devs = jax.devices("cuda")
    except RuntimeError: devs = []
    if not devs:
        raise RuntimeError("No CUDA devices found for TQC training.")
    target_device = devs[0]
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"TQC pinned to: JAX device {target_device}  →  physical GPU {physical}")
    return target_device

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)

reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

_step_auto = make_autoreset_env(reset_stacked, step_stacked)
# in_axes: (key, state, action, max_goal_dist, scenario_idx, ghost_prob, max_scenario)
vmap_step  = jax.jit(jax.vmap(_step_auto, in_axes=(0, 0, 0, None, None, None, None)))

@jax.jit
def _vmap_reset(reset_keys, min_goal_dist, ghost_prob, scenario_idx):
    def _single(key):
        return reset_stacked(key, max_goal_dist=min_goal_dist, ghost_prob=ghost_prob, scenario_idx=scenario_idx)
    return jax.vmap(_single)(reset_keys)

def init_env_state(rng_key, min_goal_dist: float = 1.5, ghost_prob: float = 0.0, scenario_idx: int = -1):
    reset_keys = jax.random.split(rng_key, N_ENVS)
    env_obs, env_state = _vmap_reset(reset_keys, jnp.float32(min_goal_dist), jnp.float32(ghost_prob), jnp.int32(scenario_idx))
    return env_obs, env_state, vmap_step

from legnav.core.jax_network import SharedEncoder

@jax.custom_vjp
def scale_gradient(x, scale): return x
def _scale_gradient_fwd(x, scale): return x, scale
def _scale_gradient_bwd(scale, g): return g * scale, None
scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)

class TQCActorHead(nn.Module):
    """Proper SAC squashed-Gaussian actor head.

    Outputs the RAW (pre-tanh) Gaussian mean. The tanh squashing is applied in
    `sample_action` *after* the noise, with the matching log-prob Jacobian
    correction — this is the only correct SAC parameterisation.

    Loadability note: the eval heads (jax_eval_multi._TQCActorHead,
    benchmark_eval._TQCActorHead, …) run with `tanh_inside=True`, i.e. they apply
    `tanh` to this exact same `Dense_0` output. So the deterministic action they
    reconstruct, `tanh(raw_mean)` mapped to the env, is identical to this policy's
    mode. No eval-side change is needed; the checkpoint param tree is unchanged
    ({Dense_0/kernel, Dense_0/bias, log_std}).
    """
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat):
        raw_mean = nn.Dense(self.action_dim)(feat)   # pre-tanh mean (unbounded)

        logstd_param = self.param('log_std', nn.initializers.constant(1.0), (self.action_dim,))
        actor_logstd_raw = jnp.broadcast_to(logstd_param, raw_mean.shape)
        actor_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (jnp.tanh(actor_logstd_raw) + 1.0)

        return raw_mean.astype(jnp.float32), actor_logstd.astype(jnp.float32)

class QuantileCriticBranch(nn.Module):
    n_atoms: int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        x = nn.relu(nn.Dense(256)(jnp.concatenate([feat, action], axis=-1)))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(self.n_atoms)(x).astype(jnp.float32)

class TQCCriticEnsemble(nn.Module):
    n_critics: int = N_CRITICS
    n_atoms:   int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        vmap_critic = nn.vmap(
            QuantileCriticBranch,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_critics
        )
        return vmap_critic(n_atoms=self.n_atoms, name='critic')(feat, action)

_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)

shared_enc = SharedEncoder()
actor_head = TQCActorHead()
critic_net = TQCCriticEnsemble()

# Linearly decay LR to 10% over the run. Adam runs too hot in the late phase here:
# success peaks then drifts down (Qmean bleeds, collisions spike). The schedule
# steps once per gradient update; total updates = chunks × G_UPDATES × LOG_EVERY.
_TOTAL_UPDATES = (DEFAULT_TOTAL_ENV_STEPS // (N_ENVS * LOG_EVERY)) * (G_UPDATES * LOG_EVERY)
_lr_sched  = optax.linear_schedule(LR, LR * 0.1, _TOTAL_UPDATES)
enc_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
ALPHA_LR   = 1e-4
alpha_opt  = optax.adam(ALPHA_LR, eps=1e-5)

def _log1m_tanh2_stable(u):
    """Numerically stable log(1 - tanh^2(u)).

    Derivation: 1 - tanh^2(u) = sech^2(u) = 4 / (e^u + e^-u)^2, so
    log(1 - tanh^2(u)) = 2 * log(2) - 2 * log(e^u + e^-u)
                      = 2 * (log(2) - u - softplus(-2u)).

    The naive form `log(1 - tanh(u)**2 + eps)` underflows to log(eps) once
    |u| ≳ 8; the softplus form stays exact across the whole range.
    """
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


def _tanh_log_prob_correction(u, max_v):
    """log|det J| for u → action, where
        v = (tanh(u_v) + 1) * 0.5 * max_v,  dv/du_v = 0.5 * max_v * sech^2(u_v)
        w =  tanh(u_w),                     dw/du_w =          sech^2(u_w)

    Sign convention: returned value is the NEGATIVE of log|J|, ready to be
    added to the base Gaussian log-prob.
    """
    log_dv = jnp.log(max_v * 0.5 + LOG_STD_EPS) + _log1m_tanh2_stable(u[..., 0])
    log_dw = _log1m_tanh2_stable(u[..., 1])
    return -(log_dv + log_dw)


def sample_action(rng_key, mean, log_std, max_v):
    """Reparameterised tanh-squashed Gaussian sample with corrected log-prob.

    `mean` is the RAW (pre-tanh) mean from TQCActorHead. Noise is injected in
    the unbounded latent space (u = mean + noise * std), THEN tanh is applied
    as the final step of the action pipeline — so the policy is natively
    bounded in (-1, 1) (and v in (0, max_v)) and the squash log-Jacobian is a
    valid change-of-variables correction. The entropy term properly
    constrains the mean: no clip dead-zone, no saturation collapse.
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)

    tanh_u = jnp.tanh(u)
    env_action = jnp.stack([(tanh_u[..., 0] + 1.0) * 0.5 * max_v, tanh_u[..., 1]], axis=-1)
    return env_action, lp_gauss + _tanh_log_prob_correction(u, max_v)


def deterministic_action(mean, max_v):
    """Policy mode used for evaluation / best-checkpoint selection.

    Identical to what the eval heads reconstruct: tanh(raw_mean) mapped to env.
    """
    tanh_m = jnp.tanh(mean)
    return jnp.stack([(tanh_m[..., 0] + 1.0) * 0.5 * max_v, tanh_m[..., 1]], axis=-1)

@jax.jit
def extract_max_v(obs):
    # state_vec[2] = (max_v - 0.2) / 1.8  →  inverse: * 1.8 + 0.2
    # Clamp to [0.2, 2.0] matching env range for safety
    return jnp.clip(obs[..., MAX_V_OBS_IDX] * 1.8 + 0.2, 0.2, 2.0)

def quantile_huber_loss_per_sample(atoms, targets, taus, kappa=HUBER_KAPPA):
    # atoms:   (B, N_ATOMS)    — online quantile predictions
    # targets: (B, N_TARGET)   — truncated target atoms
    # taus:    (N_ATOMS,)
    # Returns: (B,) per-sample loss — used for both IS-weighted training and TD priorities.
    u = targets[:, None, :] - atoms[:, :, None]   # (B, N_ATOMS, N_TARGET)
    abs_u = jnp.abs(u)
    huber = jnp.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    indicator = (u < 0.0).astype(jnp.float32)
    rho = jnp.abs(taus[None, :, None] - indicator) * huber / kappa
    return jnp.mean(jnp.sum(rho, axis=2), axis=1)

_BUF_OBS_DTYPE = jnp.bfloat16  # halves obs storage; cast back to float32 at sample time

def make_buffer(capacity):
    return {
        "obs": jnp.zeros((capacity, OBS_SIZE), _BUF_OBS_DTYPE),
        "action": jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward": jnp.zeros((capacity,), jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE), _BUF_OBS_DTYPE),
        "done": jnp.zeros((capacity,), jnp.float32),
        "priorities": jnp.zeros((capacity,), jnp.float32),
        "max_priority": jnp.float32(1.0),
        # Running sum of (priorities[i]+eps)^alpha over valid slots, maintained
        # incrementally by buf_add / buf_update_priorities so buf_sample doesn't
        # need to re-reduce the full priority array on every sample call.
        "sum_p_alpha": jnp.float32(0.0),
        "ptr": jnp.int32(0), "size": jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done):
    cap = buf["obs"].shape[0]; N = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    # New transitions inherit current max priority so they are sampled at least once.
    new_prio = jnp.broadcast_to(buf["max_priority"], (N,)).astype(jnp.float32)
    # Incremental sum update: slot was previously valid iff idx < old size,
    # in which case its prior contribution was (old_prio+eps)^alpha; otherwise 0.
    was_valid = idxs < buf["size"]
    old_pa = jnp.where(was_valid, (buf["priorities"][idxs] + PER_EPS) ** PER_ALPHA, 0.0)
    new_pa = (new_prio + PER_EPS) ** PER_ALPHA
    sum_p_alpha = buf["sum_p_alpha"] + jnp.sum(new_pa - old_pa)
    return {
        "obs": buf["obs"].at[idxs].set(obs.astype(_BUF_OBS_DTYPE)),
        "action": buf["action"].at[idxs].set(action),
        "reward": buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs.astype(_BUF_OBS_DTYPE)),
        "done": buf["done"].at[idxs].set(done),
        "priorities": buf["priorities"].at[idxs].set(new_prio),
        "max_priority": buf["max_priority"],
        "sum_p_alpha": sum_p_alpha,
        "ptr": jnp.int32((buf["ptr"] + N) % cap),
        "size": jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int, beta):
    cap = buf["priorities"].shape[0]
    valid_mask = jnp.arange(cap) < buf["size"]
    p_alpha = (buf["priorities"] + PER_EPS) ** PER_ALPHA
    p_alpha = jnp.where(valid_mask, p_alpha, 0.0)
    # Use the incrementally maintained sum instead of reducing the 300K array.
    # Drift from scatter-with-duplicates is a constant scaling factor on probs,
    # which cancels in both the categorical softmax and the max-normalized
    # IS weights below — so the sampling distribution and IS weights are exact.
    probs = p_alpha / (buf["sum_p_alpha"] + 1e-10)
    log_probs = jnp.log(probs + 1e-30)
    idxs = jax.random.categorical(rng_key, log_probs, shape=(batch_size,))
    sample_probs = probs[idxs]
    N = jnp.maximum(buf["size"], 1).astype(jnp.float32)
    weights = (N * sample_probs + 1e-10) ** (-beta)
    weights = weights / (jnp.max(weights) + 1e-10)
    return (
        buf["obs"][idxs].astype(jnp.float32),
        buf["action"][idxs],
        buf["reward"][idxs],
        buf["next_obs"][idxs].astype(jnp.float32),
        buf["done"][idxs],
        idxs,
        weights,
    )

@jax.jit
def buf_update_priorities(buf, idxs, td_errors):
    new_prio = jnp.abs(td_errors) + PER_EPS
    # Idxs come from sampling, so the slots are always valid (within [0, size)).
    old_pa = (buf["priorities"][idxs] + PER_EPS) ** PER_ALPHA
    new_pa = (new_prio + PER_EPS) ** PER_ALPHA
    sum_p_alpha = buf["sum_p_alpha"] + jnp.sum(new_pa - old_pa)
    return {
        **buf,
        "priorities": buf["priorities"].at[idxs].set(new_prio),
        "max_priority": jnp.maximum(buf["max_priority"], jnp.max(new_prio)),
        "sum_p_alpha": sum_p_alpha,
    }

@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, target, online)

@jax.jit
def critic_loss_fn(cp, tp, ap, sep, tsep, log_alpha, obs, action, reward, next_obs, done, is_weights, rng_key):
    alpha = jnp.exp(log_alpha)

    feat_next_critic = jax.lax.stop_gradient(shared_enc.apply({"params": tsep}, next_obs.astype(NET_DTYPE)))
    feat_next_actor = jax.lax.stop_gradient(shared_enc.apply({"params": sep}, next_obs.astype(NET_DTYPE)))

    mean_n, lgs_n = actor_head.apply({"params": ap}, feat_next_actor)
    next_act, next_lp = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean_n, lgs_n, extract_max_v(next_obs))

    target_atoms = critic_net.apply({"params": tp}, feat_next_critic, next_act.astype(NET_DTYPE))
    target_kept = jnp.sort(target_atoms.reshape(obs.shape[0], N_CRITICS * N_ATOMS), axis=1)[:, :N_TARGET_ATOMS]
    backup = jax.lax.stop_gradient(reward[:, None] + GAMMA * (1.0 - done[:, None]) * (target_kept - alpha * next_lp[:, None]))

    feat_obs = shared_enc.apply({"params": sep}, obs.astype(NET_DTYPE))
    online_atoms = critic_net.apply({"params": cp}, feat_obs, action.astype(NET_DTYPE))

    per_sample = jnp.zeros((obs.shape[0],), jnp.float32)
    q_sum = jnp.float32(0.0)
    for i in range(N_CRITICS):
        per_sample = per_sample + quantile_huber_loss_per_sample(online_atoms[:, i, :], backup, _TAUS)
        q_sum += jnp.mean(online_atoms[:, i, :])
    per_sample = per_sample / N_CRITICS
    weighted_loss = jnp.mean(is_weights * per_sample)
    td_error = jax.lax.stop_gradient(per_sample)
    return weighted_loss, (q_sum / N_CRITICS, td_error)

@jax.jit
def actor_loss_fn(ap, cp, sep, log_alpha, obs, rng_key):
    feat_raw = shared_enc.apply({"params": sep}, obs.astype(NET_DTYPE))
    feat_f   = scale_gradient(feat_raw, ACTOR_ENC_GRAD_SCALE)

    mean, log_std = actor_head.apply({"params": ap}, feat_f)
    action_new, log_pi = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean, log_std, extract_max_v(obs))

    q_atoms = critic_net.apply({"params": jax.lax.stop_gradient(cp)}, feat_f, action_new.astype(NET_DTYPE))
    return jnp.mean(jnp.exp(log_alpha) * log_pi) - jnp.mean(q_atoms), jnp.mean(log_pi)

@jax.jit
def tqc_update(ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, obs, action, reward, next_obs, done, is_weights, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    (c_loss, (q_mean, td_error)), (c_grads_cp, c_grads_sep) = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 3), has_aux=True
    )(cp, tp, ap, sep, tsep, la, obs, action, reward, next_obs, done, is_weights, rng_c)
    c_upd, new_cos = critic_opt.update(c_grads_cp, cos, cp)
    new_cp = optax.apply_updates(cp, c_upd)

    (a_loss, log_pi_mean), (a_grads_ap, a_grads_sep) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 2), has_aux=True
    )(ap, new_cp, sep, la, obs, rng_a)
    a_upd, new_aos = actor_opt.update(a_grads_ap, aos, ap)
    new_ap = optax.apply_updates(ap, a_upd)

    combined_enc_grads = jax.tree_util.tree_map(lambda cg, ag: cg + ag, c_grads_sep, a_grads_sep)
    e_upd, new_eos = enc_opt.update(combined_enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, e_upd)

    al_loss, al_grad = jax.value_and_grad(lambda a, lp: -jnp.exp(a) * (lp + TARGET_ENTROPY))(la, jax.lax.stop_gradient(log_pi_mean))
    al_upd, new_laos = alpha_opt.update(al_grad, laos)
    # Cap log_alpha to prevent late-training entropy-coefficient runaway. In the
    # healthy regime α sits around 0.03–0.09; capping at 0.2 leaves plenty of
    # headroom but kills the runaway feedback loop where α explodes, Q-targets
    # blow up, and the policy collapses.
    new_la = jnp.minimum(optax.apply_updates(la, al_upd), jnp.log(0.2))

    metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": jnp.exp(new_la), "log_pi": log_pi_mean, "q_mean": q_mean}
    return new_ap, new_aos, new_cp, new_cos, soft_update(tp, new_cp), new_sep, new_eos, soft_update(tsep, new_sep), new_la, new_laos, td_error, metrics

@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ap, env_state, env_obs, rng_key, vmap_step, min_goal_dist, scenario_idx, ghost_prob, max_scenario):
    feat = shared_enc.apply({"params": sep}, env_obs.astype(NET_DTYPE))
    mean, log_std = actor_head.apply({"params": ap}, feat)
    env_action, _ = jax.vmap(sample_action)(jax.random.split(rng_key, N_ENVS), mean, log_std, extract_max_v(env_obs))
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 99), N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(step_keys, env_state, env_action, min_goal_dist, scenario_idx, ghost_prob, max_scenario)
    return new_obs, new_state, env_obs, env_action, reward, done, info

EVAL_HORIZON = 2 * MAX_STEPS   # steps per deterministic eval (autoreset → many episodes)

@functools.partial(jax.jit, static_argnums=(5, 10))
def deterministic_eval(sep, ap, env_state, env_obs, rng_key, vmap_step,
                       min_goal_dist, scenario_idx, ghost_prob, max_scenario, horizon):
    """Roll out the policy MODE (no exploration noise) and return per-step outcome
    arrays — the same quantities collect_episode_outcomes consumes. This measures
    exactly what jax_eval_multi / benchmark_eval see at load time."""
    def _body(carry, _):
        es_, eo_, key_ = carry
        feat = shared_enc.apply({"params": sep}, eo_.astype(NET_DTYPE))
        mean, _ = actor_head.apply({"params": ap}, feat)
        env_action = jax.vmap(deterministic_action)(mean, extract_max_v(eo_))
        key_, k_step = jax.random.split(key_)
        step_keys = jax.random.split(k_step, N_ENVS)
        new_eo, new_es, rew, done, info = vmap_step(step_keys, es_, env_action, min_goal_dist, scenario_idx, ghost_prob, max_scenario)
        return (new_es, new_eo, key_), (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
    _, step_data = jax.lax.scan(_body, (env_state, env_obs, rng_key), None, length=horizon)
    return step_data

# Donate network params (ap, cp, tp, sep, tsep), optimizer states (aos, cos, eos, laos),
# log_alpha (la), the replay buffer, and env state/obs. All are rebound from new_carry
# after the call, so XLA can reuse their device buffers in place instead of allocating
# fresh copies on every chunk — meaningful in a 10GB VRAM budget.
@functools.partial(jax.jit, static_argnums=(11,), donate_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16))
def train_chunk(ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, buf, vmap_step, min_goal_dist, scenario_idx, ghost_prob, es, eo, key, max_scenario, beta_per):
    def _collect_body(carry, _):
        es_, eo_, buf_, key_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(sep, ap, es_, eo_, k_col, vmap_step, min_goal_dist, scenario_idx, ghost_prob, max_scenario)
        terminal = done & ~info["timeout"]
        new_buf = buf_add(buf_, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_), step_data

    (new_es, new_eo, new_buf, key), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key), None, length=LOG_EVERY
    )

    def _update_step(update_carry, upd_idx):
        ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_, la_, laos_, buf_, key_ = update_carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_done, idxs, weights = buf_sample(buf_, k_samp, BATCH_SIZE, beta_per)
        new_ap_, new_aos_, new_cp_, new_cos_, new_tp_, new_sep_, new_eos_, new_tsep_, new_la_, new_laos_, td_err_, metrics_ = \
            tqc_update(ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_, la_, laos_, b_obs, b_act, b_rew, b_next, b_done, weights, k_upd)
        new_buf_ = buf_update_priorities(buf_, idxs, td_err_)
        return (new_ap_, new_aos_, new_cp_, new_cos_, new_tp_, new_sep_, new_eos_, new_tsep_, new_la_, new_laos_, new_buf_, key_), metrics_

    update_carry_init = (ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, new_buf, key)
    (new_ap, new_aos, new_cp, new_cos, new_tp, new_sep, new_eos, new_tsep, new_la, new_laos, new_buf2, key), all_metrics = jax.lax.scan(
        _update_step, update_carry_init, jnp.arange(G_UPDATES * LOG_EVERY)
    )

    new_carry = (new_ap, new_aos, new_cp, new_cos, new_tp, new_sep, new_eos, new_tsep, new_la, new_laos, new_buf2, new_es, new_eo, key)
    return new_carry, all_step_data, all_metrics

@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    def _scan(carry, t):
        ep_ret = carry; r, d, g, c, p = t; ep_ret = ep_ret + r
        act_col = c & ~p
        is_suc, is_acol, is_pcol, is_tmo = g, act_col & ~g, p & ~g, d & ~g & ~act_col & ~p
        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_col  = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        return jnp.where(d, 0.0, ep_ret), (out_ret, out_suc, out_col, out_pcol, out_tmo, d.astype(jnp.float32))
    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(rewards.shape[1]), (rewards, dones, goal_reached, collision, passive_col)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()

def save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, la, laos, step,
                    filepath=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = filepath or CKPT_PATH
    bundle = {
        "enc_params":        jax.device_get(sep),
        "target_enc_params": jax.device_get(tsep),
        "actor_params":      jax.device_get(ap),
        "critic_params":     jax.device_get(cp),
        "target_params":     jax.device_get(tp),
        "enc_opt_state":     jax.device_get(eos),
        "actor_opt_state":   jax.device_get(aos),
        "critic_opt_state":  jax.device_get(cos),
        "log_alpha":         jax.device_get(la),
        "alpha_opt_state":   jax.device_get(laos),
        "step":              int(step),
    }
    with open(path, "wb") as f: f.write(flax.serialization.to_bytes(bundle))
    print(f"  TQC checkpoint -> {path}  (step {step})")

def train(total_env_steps: int = DEFAULT_TOTAL_ENV_STEPS):
    """Run TQC training for a fixed env-step budget.

    Loop exits when `total_steps >= total_env_steps`. Each chunk advances
    `total_steps` by `N_ENVS * LOG_EVERY` env steps.
    """
    precision_str = "bfloat16" if _USE_BFLOAT16 else "float32"
    print(f"TQC Training — CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} | Precision: {precision_str}")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")
    print(f"  Budget: {int(total_env_steps):,} env steps")

    rng = jax.random.PRNGKey(7)
    rng, k_se, ka, kc = jax.random.split(rng, 4)
    dummy_obs = jnp.zeros((2, OBS_SIZE), dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)

    dummy_feat = shared_enc.apply({"params": sep}, dummy_obs)
    ap  = actor_head.init(ka, dummy_feat)["params"]
    cp  = critic_net.init(kc, dummy_feat, dummy_act)["params"]
    tp  = jax.tree_util.tree_map(jnp.array, cp)

    eos  = enc_opt.init(sep)
    aos  = actor_opt.init(ap)
    cos  = critic_opt.init(cp)
    la   = jnp.array(0.0, dtype=jnp.float32)
    laos = alpha_opt.init(la)

    cur_max_dist, cur_ghost, _, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc  = 0.0
    highest_rolling_suc = 0.0
    cur_scenario = 0

    print(f"Curriculum: max_dist={cur_max_dist:.1f} ghost={cur_ghost:.2f} max_scen={cur_max_scen}")
    print(f"Initialising environments...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state, vmap_step = init_env_state(env_rng, min_goal_dist=cur_max_dist, ghost_prob=cur_ghost)
    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps, n_updates = 0, 0
    best_suc = 44.0

    print("Warming up buffer with random actions...")
    rng, k_warmup = jax.random.split(rng)
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
        terminal   = done & ~info["timeout"]
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward, new_obs, terminal.astype(jnp.float32))
        env_obs = new_obs
        total_steps += N_ENVS

    print("Warmup done. JIT compiling train chunk (this may take up to a minute)...")
    _c, _sd, _m = train_chunk(
        ap, aos, cp, cos, tp, sep, eos, tsep, la, laos,
        replay_buf, vmap_step, jnp.float32(cur_max_dist), jnp.int32(cur_scenario), jnp.float32(cur_ghost),
        env_state, env_obs, rng, jnp.int32(cur_max_scen), jnp.float32(PER_BETA_START)
    )
    jax.block_until_ready(_c)
    ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, replay_buf, env_state, env_obs, rng = _c
    n_updates   += LOG_EVERY
    total_steps += N_ENVS * LOG_EVERY
    print("Compilation done.")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | {'FPS':>7} | "
           f"{'Time':>8} | {'Dist':>5} {'Ghost':>5} {'Scen':>4}")
    print(hdr); print("─" * len(hdr))

    PRINT_EVERY_CHUNKS = 5   # print every 5th chunk (1/5 of previous frequency)
    chunk_idx = 0

    # Divergence early-stop: kill training if the critic blows up, or if two
    # deterministic evals in a row come in ≥20 points below best.
    DIVERGENCE_CRIT_LOSS = 200.0
    DIVERGENCE_SUC_DROP  = 20.0
    bad_evals = 0

    t_start = time.time()

    _LOG_PATH = paths.checkpoint("tqc", "tqc_training_log.csv")
    os.makedirs(paths.checkpoint("tqc"), exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while total_steps < total_env_steps:
        t0 = time.time()

        # Linearly anneal PER IS-weight exponent β from PER_BETA_START → PER_BETA_END.
        frac = min(1.0, total_steps / max(1, total_env_steps))
        beta_per = PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

        new_carry, all_step_data, all_metrics = train_chunk(
            ap, aos, cp, cos, tp, sep, eos, tsep, la, laos,
            replay_buf, vmap_step, jnp.float32(cur_max_dist), jnp.int32(cur_scenario), jnp.float32(cur_ghost),
            env_state, env_obs, rng, jnp.int32(cur_max_scen), jnp.float32(beta_per)
        )

        ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, replay_buf, env_state, env_obs, rng = new_carry

        n_updates += LOG_EVERY
        total_steps += N_ENVS * LOG_EVERY

        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())

        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else: mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        m_crit = float(all_metrics["critic_loss"].mean())
        m_act  = float(all_metrics["actor_loss"].mean())
        m_alph = float(all_metrics["alpha"].mean())
        m_lpi  = float(all_metrics["log_pi"].mean())
        m_qm   = float(all_metrics["q_mean"].mean())

        fps = (N_ENVS * LOG_EVERY) / (time.time() - t0)

        elapsed = time.time() - t_start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        chunk_idx += 1
        if chunk_idx == 1 or chunk_idx % PRINT_EVERY_CHUNKS == 0:
            print(f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                  f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                  f"{m_crit:>7.4f} {m_act:>7.4f} {m_alph:>6.4f} {m_lpi:>6.3f} {m_qm:>7.3f} | "
                  f"{fps:>7,.0f} | {time_str:>8} | {cur_max_dist:>4.1f}m {cur_ghost:>4.2f} scen<={cur_max_scen}")
        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        if m_crit > DIVERGENCE_CRIT_LOSS:
            print(f"  !! Critic loss {m_crit:.1f} > {DIVERGENCE_CRIT_LOSS:.0f} — training diverged, stopping early.")
            break

        if n_ep > 0:
            # ── Monotonic curriculum ──────────────────────────────────────
            rolling_suc         = 0.80 * rolling_suc + 0.20 * suc_pct
            highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

            new_max_dist, new_ghost, _, new_max_scen = get_continuous_curriculum(highest_rolling_suc)

            if new_max_dist > cur_max_dist:
                cur_max_dist = new_max_dist
            if new_ghost > cur_ghost:
                cur_ghost = new_ghost
            if new_max_scen > cur_max_scen:
                cur_max_scen = new_max_scen

            # Always sample scenario uniformly from [0, cur_max_scen] at each reset.
            cur_scenario = -1

        # ── Deterministic best-checkpoint selection ───────────────────────────
        
        curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
        if curriculum_mature and (chunk_idx == 1 or chunk_idx % PRINT_EVERY_CHUNKS == 0):
            rng, k_evr, k_ev = jax.random.split(rng, 3)
            eval_keys = jax.random.split(k_evr, N_ENVS)
            eo_obs, eo_state = _vmap_reset(eval_keys, jnp.float32(cur_max_dist),
                                           jnp.float32(cur_ghost), jnp.int32(cur_scenario))
            ev_data = deterministic_eval(
                sep, ap, eo_state, eo_obs, k_ev, vmap_step,
                jnp.float32(cur_max_dist), jnp.int32(cur_scenario), jnp.float32(cur_ghost),
                jnp.int32(cur_max_scen), EVAL_HORIZON)
            _, ev_suc, _, _, _, ev_msk = collect_episode_outcomes(*ev_data)
            n_ev = float(ev_msk.sum())
            det_suc = float((ev_suc * ev_msk).sum() / n_ev) * 100.0 if n_ev > 0 else 0.0
            print(f"    [det-eval] mode success {det_suc:5.1f}%  (n={int(n_ev)} ep)  best={best_suc:.1f}%")
            if det_suc > best_suc:
                best_suc = det_suc
                save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, la, laos, n_updates)
                bad_evals = 0
            elif det_suc < best_suc - DIVERGENCE_SUC_DROP:
                bad_evals += 1
                if bad_evals >= 2:
                    print(f"  !! Two consecutive evals ≥{DIVERGENCE_SUC_DROP:.0f}% below best ({best_suc:.1f}%) — stopping early.")
                    break
            else:
                bad_evals = 0

    # Final checkpoint == best checkpoint. 
    import shutil
    if os.path.exists(CKPT_PATH):
        shutil.copyfile(CKPT_PATH, paths.checkpoint("tqc", "tqc_final.msgpack"))
        print(f"  Final checkpoint -> checkpoints_tqc/tqc_final.msgpack (copied from best, {best_suc:.1f}%)")
    else:
        save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, la, laos, n_updates,
                        filepath=paths.checkpoint("tqc", "tqc_final.msgpack"))
        print(f"  Final checkpoint -> checkpoints_tqc/tqc_final.msgpack (no best found, saved current)")

    print(f"\nTQC done! {(time.time() - t_start)/3600:.2f}h | Best success: {best_suc:.1f}%")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")


if __name__ == "__main__":
    train()