"""TQCjac.py — Truncated Quantile Critics: shared LidarCNN encoder, 5-critic
quantile ensemble, prioritized replay, fully fused JIT train chunk."""

import os
import csv
import argparse

from legnav import paths

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

# jaxlib 0.9's fusion autotuner takes tens of minutes and can hard-crash on the
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

# Human/sensor config MUST be set before importing jax_env_multi (it binds
# `from jax_env import USE_LEGS` at import time). Matches the eval default:
# leg-pair human model + salt&pepper LiDAR noise.
import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS     = True
_jax_env.SENSOR_NOISE = True

from legnav.core.jax_env import MAX_STEPS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env
from legnav.algorithms.jax_ppo import get_continuous_curriculum

# ══ Observation / action ══════════════════════════════════════════════════════
OBS_SIZE      = 668   # kin(5) + goal_stack(6) + ego_deltas(9) + lidar_stack(648)
ACTION_DIM    = 2
MAX_V_OBS_IDX = 2     # kin_vec[v_norm, w, max_v_norm, ...] → max_v at idx 2

# ══ Training length ═══════════════════════════════════════════════════════════
# Training runs in fixed chunks (one fused train_chunk call each): collect
# COLLECT_STEPS × N_ENVS transitions, then run GRAD_UPDATES_PER_CHUNK gradient
# updates. COLLECT_STEPS is kept small so the buffer holds ~3 chunks of history
# (at 50 it held only ~1.5 — effectively on-policy data).
TOTAL_ENV_STEPS        = 70_000_000
N_ENVS                 = 4096
COLLECT_STEPS          = 25
GRAD_UPDATES_PER_CHUNK = 1_000
WARMUP_STEPS           = 10_000

STEPS_PER_CHUNK    = N_ENVS * COLLECT_STEPS                 # 102,400
TOTAL_CHUNKS       = TOTAL_ENV_STEPS // STEPS_PER_CHUNK     # ~683
TOTAL_GRAD_UPDATES = TOTAL_CHUNKS * GRAD_UPDATES_PER_CHUNK  # LR-schedule horizon

# ══ Replay buffer (prioritized: p = |TD|^α, IS weight ∝ (N·P)^-β) ═════════════
BUFFER_CAP     = 300_000
BATCH_SIZE     = 512
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_EPS        = 1e-6
_BUF_OBS_DTYPE = jnp.bfloat16   # halves obs storage; cast to f32 at sample time

# ══ TQC hyperparameters ═══════════════════════════════════════════════════════
GAMMA         = 0.99
TAU           = 0.005
LR            = 3e-4    # decays linearly to LR*0.1 over the run
# Fixed entropy coefficient. Auto-tuning was removed: the -2.0 target was
# unreachable (log_pi sits at +4..5 for a good policy), so α railed against its
# cap and the -α·log_pi term in the backup dominated Q; α swings then collapsed
# training. A small constant keeps the entropy bonus a mild regulariser.
ALPHA_FIXED   = 0.05
MAX_GRAD_NORM = 10.0
LOG_STD_EPS   = 1e-6
ACTOR_ENC_GRAD_SCALE = 0.1   # fraction of actor gradient let into the shared encoder

N_CRITICS        = 5
N_ATOMS          = 25
N_TOP_ATOMS_DROP = 3    # truncation: drop the top atoms of the pooled target
N_TARGET_ATOMS   = N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP
HUBER_KAPPA      = 1.0

# ══ Logging / eval / safety stop ══════════════════════════════════════════════
PRINT_EVERY_CHUNKS   = 5
EVAL_EVERY_CHUNKS    = 5
DIVERGENCE_CRIT_LOSS = 200.0   # stop if mean critic loss exceeds this

CKPT_DIR  = paths.checkpoint("tqc")
CKPT_PATH = f"{CKPT_DIR}/tqc_best.msgpack"


def _check_gpu():
    try:
        devs = jax.devices("cuda")
    except RuntimeError:
        devs = []
    if not devs:
        raise RuntimeError("No CUDA devices found for TQC training.")
    print(f"TQC pinned to: JAX device {devs[0]}  →  physical GPU "
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    return devs[0]

jax.config.update("jax_default_device", _check_gpu())


# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

_step_auto = make_autoreset_env(reset_stacked, step_stacked)
# in_axes: (key, state, action, max_goal_dist, scenario_idx, ghost_prob, max_scenario)
vmap_step  = jax.jit(jax.vmap(_step_auto, in_axes=(0, 0, 0, None, None, None, None)))

@jax.jit
def _vmap_reset(reset_keys, max_goal_dist, ghost_prob, scenario_idx):
    """Batched reset at the CURRENT curriculum — used by the deterministic eval."""
    def _single(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist,
                             ghost_prob=ghost_prob, scenario_idx=scenario_idx)
    return jax.vmap(_single)(reset_keys)

def init_env_state(rng_key, max_goal_dist=1.5, ghost_prob=0.0, scenario_idx=-1):
    env_obs, env_state = _vmap_reset(jax.random.split(rng_key, N_ENVS),
                                     jnp.float32(max_goal_dist),
                                     jnp.float32(ghost_prob),
                                     jnp.int32(scenario_idx))
    return env_obs, env_state, vmap_step


# ── Networks: shared encoder + actor head + quantile critic ensemble ──────────
from legnav.core.jax_network import SharedEncoder
from legnav.core.precision import bf16_apply, NET_DTYPE, PRECISION_STR

class TQCActorHead(nn.Module):
    """Emits the RAW (pre-tanh) mean and a state-independent log-std; tanh is
    applied after noise injection in sample_action. The param tree
    ({Dense_0/kernel, Dense_0/bias, log_std}) must stay unchanged — the eval
    heads rebuild the deterministic action from exactly these params."""
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  0.5

    @nn.compact
    def __call__(self, feat):
        raw_mean = nn.Dense(self.action_dim)(feat)
        logstd_param = self.param("log_std", nn.initializers.constant(1.0), (self.action_dim,))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) \
                  * (jnp.tanh(jnp.broadcast_to(logstd_param, raw_mean.shape)) + 1.0)
        # float32 out: sample / log-prob math stays full precision.
        return raw_mean.astype(jnp.float32), log_std.astype(jnp.float32)

class QuantileCriticBranch(nn.Module):
    n_atoms: int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        x = nn.relu(nn.Dense(256)(jnp.concatenate([feat, action], axis=-1)))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(self.n_atoms)(x).astype(jnp.float32)

class TQCCriticEnsemble(nn.Module):
    """N_CRITICS independent quantile branches → (batch, n_critics, n_atoms)."""
    n_critics: int = N_CRITICS
    n_atoms:   int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        vmap_critic = nn.vmap(
            QuantileCriticBranch,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_critics,
        )
        return vmap_critic(n_atoms=self.n_atoms, name="critic")(feat, action)

_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)

shared_enc = SharedEncoder()
actor_head = TQCActorHead()
critic_net = TQCCriticEnsemble()

# bf16 forward passes (params + activations); outputs come back fp32.
enc_apply    = bf16_apply(shared_enc.apply)
actor_apply  = bf16_apply(actor_head.apply)
critic_apply = bf16_apply(critic_net.apply)

_lr_sched  = optax.linear_schedule(LR, LR * 0.1, TOTAL_GRAD_UPDATES)
def _make_opt():
    return optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(_lr_sched, eps=1e-5))
enc_opt    = _make_opt()
actor_opt  = _make_opt()
critic_opt = _make_opt()


# ── Action squashing + exact log-prob ─────────────────────────────────────────
def _log1m_tanh2_stable(u):
    """log(1 - tanh²(u)) = 2·(log2 - u - softplus(-2u)); the naive form
    underflows to log(0) once |u| ≳ 8."""
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))

def _tanh_log_prob_correction(u, max_v):
    """-log|det J| of u → (v, w) with v = (tanh(u_v)+1)·max_v/2, w = tanh(u_w)."""
    log_dv = jnp.log(max_v * 0.5 + LOG_STD_EPS) + _log1m_tanh2_stable(u[..., 0])
    log_dw = _log1m_tanh2_stable(u[..., 1])
    return -(log_dv + log_dw)

def sample_action(rng_key, mean, log_std, max_v):
    """Reparameterised tanh-squashed Gaussian sample with exact log-prob.
    Noise is injected in the unbounded latent space, THEN tanh bounds
    v ∈ (0, max_v) and w ∈ (-1, 1) — no clipping plateau. Fully batched."""
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

@jax.jit
def extract_max_v(obs):
    # kin_vec[2] = (max_v - 0.2) / 1.8  →  invert and clamp to env range.
    return jnp.clip(obs[..., MAX_V_OBS_IDX] * 1.8 + 0.2, 0.2, 2.0)


# ── Quantile Huber loss ───────────────────────────────────────────────────────
def quantile_huber_loss_per_sample(atoms, targets, taus, kappa=HUBER_KAPPA):
    """atoms (B, N_CRITICS, N_ATOMS), targets (B, N_TARGET) → per-sample loss
    (B,) averaged over critics, in one broadcasted kernel."""
    u = targets[:, None, None, :] - atoms[..., None]   # (B, N_CRITICS, N_ATOMS, N_TARGET)
    abs_u = jnp.abs(u)
    huber = jnp.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    rho = jnp.abs(taus[None, None, :, None] - (u < 0.0).astype(jnp.float32)) * huber / kappa
    return jnp.mean(jnp.sum(rho, axis=3), axis=(1, 2))


# ── Replay buffer (on-GPU circular, prioritized) ──────────────────────────────
def make_buffer(capacity):
    return {
        "obs":          jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "action":       jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":       jnp.zeros((capacity,),            jnp.float32),
        "next_obs":     jnp.zeros((capacity, OBS_SIZE),   _BUF_OBS_DTYPE),
        "done":         jnp.zeros((capacity,),            jnp.float32),
        "priorities":   jnp.zeros((capacity,),            jnp.float32),
        "max_priority": jnp.float32(1.0),
        # (p+eps)^α per slot, maintained incrementally (0 for empty slots) so
        # buf_sample never re-exponentiates the whole array.
        "p_alpha":      jnp.zeros((capacity,), jnp.float32),
        "ptr":          jnp.int32(0),
        "size":         jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    # New transitions get max priority so each is sampled at least once.
    new_prio = jnp.broadcast_to(buf["max_priority"], (N,)).astype(jnp.float32)
    return {
        "obs":          buf["obs"].at[idxs].set(obs.astype(_BUF_OBS_DTYPE)),
        "action":       buf["action"].at[idxs].set(action),
        "reward":       buf["reward"].at[idxs].set(reward),
        "next_obs":     buf["next_obs"].at[idxs].set(next_obs.astype(_BUF_OBS_DTYPE)),
        "done":         buf["done"].at[idxs].set(done),
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
            buf["done"][idxs],
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


# ── TQC update step ───────────────────────────────────────────────────────────
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
def critic_loss_fn(cp, tp, ap, sep, tsep, obs, action, reward, next_obs, done,
                   is_weights, rng_key):
    # Next actions are sampled from the TARGET-encoder features (as in SACjax):
    # one encoder forward on next_obs instead of two — the encoder is the most
    # expensive module.
    feat_next = jax.lax.stop_gradient(enc_apply({"params": tsep}, next_obs.astype(NET_DTYPE)))
    mean_n, lgs_n = actor_apply({"params": ap}, feat_next)
    next_act, next_lp = sample_action(rng_key, mean_n, lgs_n, extract_max_v(next_obs))

    # Truncated target: pool all critics' atoms, sort, drop the top ones.
    target_atoms = critic_apply({"params": tp}, feat_next, next_act.astype(NET_DTYPE))
    target_kept = jnp.sort(target_atoms.reshape(obs.shape[0], N_CRITICS * N_ATOMS),
                           axis=1)[:, :N_TARGET_ATOMS]
    backup = jax.lax.stop_gradient(
        reward[:, None] + GAMMA * (1.0 - done[:, None]) * (target_kept - ALPHA_FIXED * next_lp[:, None])
    )

    feat_obs = enc_apply({"params": sep}, obs.astype(NET_DTYPE))
    online_atoms = critic_apply({"params": cp}, feat_obs, action.astype(NET_DTYPE))

    per_sample = quantile_huber_loss_per_sample(online_atoms, backup, _TAUS)
    weighted_loss = jnp.mean(is_weights * per_sample)
    return weighted_loss, (jnp.mean(online_atoms), jax.lax.stop_gradient(per_sample))

@jax.jit
def actor_loss_fn(ap, cp, sep, obs, rng_key):
    feat = scale_gradient(enc_apply({"params": sep}, obs.astype(NET_DTYPE)),
                          ACTOR_ENC_GRAD_SCALE)
    mean, log_std = actor_apply({"params": ap}, feat)
    action_new, log_pi = sample_action(rng_key, mean, log_std, extract_max_v(obs))
    q_atoms = critic_apply({"params": jax.lax.stop_gradient(cp)}, feat, action_new.astype(NET_DTYPE))
    return jnp.mean(ALPHA_FIXED * log_pi) - jnp.mean(q_atoms), jnp.mean(log_pi)

@jax.jit
def tqc_update(ap, aos, cp, cos, tp, sep, eos, tsep,
               obs, action, reward, next_obs, done, is_weights, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    (c_loss, (q_mean, td_error)), (c_grads_cp, c_grads_sep) = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 3), has_aux=True
    )(cp, tp, ap, sep, tsep, obs, action, reward, next_obs, done, is_weights, rng_c)
    c_upd, new_cos = critic_opt.update(c_grads_cp, cos, cp)
    new_cp = optax.apply_updates(cp, c_upd)

    (a_loss, log_pi_mean), (a_grads_ap, a_grads_sep) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 2), has_aux=True
    )(ap, new_cp, sep, obs, rng_a)
    a_upd, new_aos = actor_opt.update(a_grads_ap, aos, ap)
    new_ap = optax.apply_updates(ap, a_upd)

    # Encoder gets critic gradients + (scaled) actor gradients.
    enc_grads = jax.tree_util.tree_map(lambda cg, ag: cg + ag, c_grads_sep, a_grads_sep)
    e_upd, new_eos = enc_opt.update(enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, e_upd)

    metrics = {"critic_loss": c_loss, "actor_loss": a_loss,
               "log_pi": log_pi_mean, "q_mean": q_mean}
    return (new_ap, new_aos, new_cp, new_cos, soft_update(tp, new_cp),
            new_sep, new_eos, soft_update(tsep, new_sep), td_error, metrics)


# ── Collection step ───────────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ap, env_state, env_obs, rng_key, vmap_step,
                 max_goal_dist, scenario_idx, ghost_prob, max_scenario):
    k_act, k_step = jax.random.split(rng_key)
    feat = enc_apply({"params": sep}, env_obs.astype(NET_DTYPE))
    mean, log_std = actor_apply({"params": ap}, feat)
    env_action, _ = sample_action(k_act, mean, log_std, extract_max_v(env_obs))
    step_keys = jax.random.split(k_step, N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(
        step_keys, env_state, env_action, max_goal_dist, scenario_idx, ghost_prob, max_scenario
    )
    return new_obs, new_state, env_obs, env_action, reward, done, info


# ── Deterministic evaluation (policy mode, no exploration noise) ──────────────
# 1024 envs × MAX_STEPS completes thousands of episodes — plenty for checkpoint
# selection at a fraction of the cost of evaluating all training envs.
EVAL_ENVS    = 1024
EVAL_HORIZON = MAX_STEPS

@functools.partial(jax.jit, static_argnums=(5, 10))
def deterministic_eval(sep, ap, env_state, env_obs, rng_key, vmap_step,
                       max_goal_dist, scenario_idx, ghost_prob, max_scenario, horizon):
    """Roll out the policy MODE — measures exactly what the eval scripts see
    when they load the checkpoint."""
    def _body(carry, _):
        es_, eo_, key_ = carry
        feat = enc_apply({"params": sep}, eo_.astype(NET_DTYPE))
        mean, _ = actor_apply({"params": ap}, feat)
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
@functools.partial(jax.jit, static_argnums=(9,),
                   donate_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14))
def train_chunk(ap, aos, cp, cos, tp, sep, eos, tsep, buf,
                vmap_step, max_goal_dist, scenario_idx, ghost_prob,
                es, eo, key, max_scenario, beta_per):

    # Phase 1: collect COLLECT_STEPS env steps into the buffer.
    def _collect_body(carry, _):
        es_, eo_, buf_, key_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(
            sep, ap, es_, eo_, k_col, vmap_step,
            max_goal_dist, scenario_idx, ghost_prob, max_scenario
        )
        terminal = done & ~info["timeout"]
        # Store info["final_obs"] (the pre-autoreset successor obs): on done,
        # new_eo is already the NEXT episode's first obs, which would corrupt
        # the bootstrap of timeout transitions.
        new_buf = buf_add(buf_, obs_b, env_a, rew, info["final_obs"],
                          terminal.astype(jnp.float32))
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_), step_data

    (new_es, new_eo, new_buf, key), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key), None, length=COLLECT_STEPS
    )

    # Phase 2: GRAD_UPDATES_PER_CHUNK gradient steps with PER priority write-back.
    def _update_step(carry, upd_idx):
        ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_, buf_, key_ = carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_done, idxs, weights = buf_sample(
            buf_, k_samp, BATCH_SIZE, beta_per
        )
        (new_ap_, new_aos_, new_cp_, new_cos_, new_tp_,
         new_sep_, new_eos_, new_tsep_, td_err_, metrics_) = tqc_update(
            ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_,
            b_obs, b_act, b_rew, b_next, b_done, weights, k_upd
        )
        new_buf_ = buf_update_priorities(buf_, idxs, td_err_)
        return (new_ap_, new_aos_, new_cp_, new_cos_, new_tp_,
                new_sep_, new_eos_, new_tsep_, new_buf_, key_), metrics_

    carry_init = (ap, aos, cp, cos, tp, sep, eos, tsep, new_buf, key)
    (ap, aos, cp, cos, tp, sep, eos, tsep, new_buf, key), all_metrics = jax.lax.scan(
        _update_step, carry_init, jnp.arange(GRAD_UPDATES_PER_CHUNK)
    )

    new_carry = (ap, aos, cp, cos, tp, sep, eos, tsep, new_buf, new_es, new_eo, key)
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
def save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, step, filepath=None):
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
        # α is a fixed constant; key kept for loader compatibility.
        "log_alpha":         float(jnp.log(ALPHA_FIXED)),
        "step":              int(step),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  TQC checkpoint -> {path}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print(f"TQC Training — CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} | Precision: {PRECISION_STR}")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}  alpha={ALPHA_FIXED} (fixed)")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")
    print(f"  Chunk: {STEPS_PER_CHUNK:,} env steps -> {GRAD_UPDATES_PER_CHUNK} grad updates")
    print(f"  Budget: {TOTAL_ENV_STEPS:,} env steps  ->  ~{TOTAL_CHUNKS} chunks")

    rng = jax.random.PRNGKey(7)
    rng, k_se, ka, kc = jax.random.split(rng, 4)

    # Init params: shared encoder, actor head, critic ensemble (+ target copies).
    dummy_obs = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)
    dummy_feat = enc_apply({"params": sep}, dummy_obs)
    ap = actor_head.init(ka, dummy_feat)["params"]
    cp = critic_net.init(kc, dummy_feat, dummy_act)["params"]
    tp = jax.tree_util.tree_map(jnp.array, cp)

    eos = enc_opt.init(sep)
    aos = actor_opt.init(ap)
    cos = critic_opt.init(cp)

    cur_max_dist, cur_ghost, _, cur_max_scen = get_continuous_curriculum(0.0)
    cur_scenario = 0
    rolling_suc  = 0.0

    print(f"Curriculum: max_dist={cur_max_dist:.1f} ghost={cur_ghost:.2f} max_scen={cur_max_scen}")
    print("Initialising environments...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state, vmap_step = init_env_state(env_rng, max_goal_dist=cur_max_dist,
                                                   ghost_prob=cur_ghost)
    replay_buf = make_buffer(BUFFER_CAP)
    total_steps, n_updates = 0, 0
    best_suc = 0.0

    print("Warming up buffer with random actions...")
    rng, k_warmup = jax.random.split(rng)
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_v, k_w, k_step = jax.random.split(k_warmup, 4)
        obs_before = env_obs
        rand_v     = jax.random.uniform(k_v, (N_ENVS,)) * extract_max_v(env_obs)
        rand_w     = jax.random.uniform(k_w, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = vmap_step(
            step_keys, env_state, env_action, cur_max_dist, cur_scenario, cur_ghost,
            jnp.int32(cur_max_scen)
        )
        terminal   = done & ~info["timeout"]
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             info["final_obs"], terminal.astype(jnp.float32))
        env_obs = new_obs
        total_steps += N_ENVS

    print("Warmup done. JIT compiling train chunk (this may take a minute)...")
    _c, _sd, _m = train_chunk(
        ap, aos, cp, cos, tp, sep, eos, tsep, replay_buf,
        vmap_step, jnp.float32(cur_max_dist), jnp.int32(cur_scenario), jnp.float32(cur_ghost),
        env_state, env_obs, rng, jnp.int32(cur_max_scen), jnp.float32(PER_BETA_START)
    )
    jax.block_until_ready(_c)
    ap, aos, cp, cos, tp, sep, eos, tsep, replay_buf, env_state, env_obs, rng = _c
    n_updates   += GRAD_UPDATES_PER_CHUNK
    total_steps += STEPS_PER_CHUNK
    print("Compilation done.")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'LogPi':>6} {'Qmean':>7} | {'FPS':>7} | "
           f"{'Time':>8} | {'Dist':>5} {'Ghost':>5} {'Scen':>4}")
    print(hdr)
    print("─" * len(hdr))

    chunk   = 0
    t_start = time.time()

    log_path = paths.checkpoint("tqc", "tqc_training_log.csv")
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
            ap, aos, cp, cos, tp, sep, eos, tsep, replay_buf,
            vmap_step, jnp.float32(cur_max_dist), jnp.int32(cur_scenario), jnp.float32(cur_ghost),
            env_state, env_obs, rng, jnp.int32(cur_max_scen), jnp.float32(beta_per)
        )
        ap, aos, cp, cos, tp, sep, eos, tsep, replay_buf, env_state, env_obs, rng = new_carry

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
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        m_crit  = float(all_metrics["critic_loss"].mean())
        fps     = STEPS_PER_CHUNK / (time.time() - t0)
        elapsed = time.time() - t_start
        h, rem  = divmod(int(elapsed), 3600)
        m, s    = divmod(rem, 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"

        if chunk == 1 or chunk % PRINT_EVERY_CHUNKS == 0:
            print(f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                  f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                  f"{m_crit:>7.4f} "
                  f"{float(all_metrics['actor_loss'].mean()):>7.4f} "
                  f"{float(all_metrics['log_pi'].mean()):>6.3f} "
                  f"{float(all_metrics['q_mean'].mean()):>7.3f} | "
                  f"{fps:>7,.0f} | {time_str:>8} | "
                  f"{cur_max_dist:>4.1f}m {cur_ghost:>4.2f} scen<={cur_max_scen}")

        log_writer.writerow([total_steps, round(mean_ret, 4),
                             round(suc_pct, 4), round(col_pct, 4),
                             round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        log_file.flush()

        if m_crit > DIVERGENCE_CRIT_LOSS:
            print(f"  !! Critic loss {m_crit:.1f} > {DIVERGENCE_CRIT_LOSS:.0f} — training diverged, stopping early.")
            break

        if n_ep > 0:
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

        # Best-checkpoint selection: training suc% is measured under exploration
        # noise and understates the mode policy — evaluate tanh(mean) on fresh
        # resets at the current curriculum instead.
        curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
        if curriculum_mature and (chunk == 1 or chunk % EVAL_EVERY_CHUNKS == 0):
            rng, k_evr, k_ev = jax.random.split(rng, 3)
            eo_obs, eo_state = _vmap_reset(jax.random.split(k_evr, EVAL_ENVS),
                                           jnp.float32(cur_max_dist),
                                           jnp.float32(cur_ghost),
                                           jnp.int32(cur_scenario))
            ev_data = deterministic_eval(
                sep, ap, eo_state, eo_obs, k_ev, vmap_step,
                jnp.float32(cur_max_dist), jnp.int32(cur_scenario),
                jnp.float32(cur_ghost), jnp.int32(cur_max_scen), EVAL_HORIZON)
            _, ev_suc, _, _, _, ev_msk = collect_episode_outcomes(*ev_data)
            n_ev = float(ev_msk.sum())
            det_suc = float((ev_suc * ev_msk).sum() / n_ev) * 100.0 if n_ev > 0 else 0.0
            print(f"    [det-eval] mode success {det_suc:5.1f}%  (n={int(n_ev)} ep)  best={best_suc:.1f}%")
            if det_suc > best_suc:
                best_suc = det_suc
                save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, n_updates)

    # Final checkpoint == best checkpoint: the live params at the end of a run
    # can be worse than the best on disk.
    import shutil
    final_path = paths.checkpoint("tqc", "tqc_final.msgpack")
    if os.path.exists(CKPT_PATH):
        shutil.copyfile(CKPT_PATH, final_path)
        print(f"  Final checkpoint -> {final_path} (copied from best, {best_suc:.1f}%)")
    else:
        save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, n_updates, filepath=final_path)
        print(f"  Final checkpoint -> {final_path} (no best found, saved current)")

    print(f"\nTQC done! {(time.time() - t_start)/3600:.2f}h | Best success: {best_suc:.1f}%")
    log_file.close()
    print(f"Training log saved -> {log_path}")


if __name__ == "__main__":
    train()
