"""
jax_train.py — Rollout Collection
===================================
FIXES vs previous version:

  FIX — _vmap_step stale-closure bug (Bug #3): init_env_state() now returns
    the freshly built vmap_step. collect_rollouts receives it as a static arg.

  FIX — @jax.jit removed from reset_env / step_env (Issue #8).

  FIX — ghost_prob curriculum + vmap_step cache:
    init_env_state accepts ghost_prob. Resolves to bool (>= 0.5).
    Compiled vmap_step objects are cached in _VMAP_STEP_CACHE keyed by the
    ghost_robot bool. Curriculum transitions that keep the same ghost value
    reuse the cached kernel — zero retrace cost. Only a true ghost flip
    (True→False) triggers a one-time recompile. No separate precompile step
    or warm-up rollout needed — avoids the CUDA_ILLEGAL_ADDRESS OOM that
    a full-rollout precompile caused on 10GB VRAM.

  FIX — ROLLOUT_STEPS 150→96: frees ~1.2GB rollout buffer.

  All previous changes (OBS_SIZE=342, curriculum) are carried.
"""

import os
# NOTE: Do NOT set XLA_PYTHON_CLIENT_ALLOCATOR here — jax_ppo.py sets it
# before importing JAX, and setting it again after JAX is imported has no
# effect and risks conflicting with the already-initialised allocator.
# The TF_GPU_ALLOCATOR env var is similarly set in jax_ppo.py only.

import functools
import jax
import jax.numpy as jnp
#from jax_env import reset_env, step_env
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_network import scale_actions_batched


def _verify_gpu():
    try:
        cuda_devices = jax.devices("cuda")
    except RuntimeError:
        cuda_devices = []
    if not cuda_devices:
        raise RuntimeError(
            f"No CUDA GPU found. All devices: {jax.devices()}."
        )
    print(f"GPU verified: {cuda_devices}")
    return cuda_devices[0]

GPU_DEVICE = _verify_gpu()

# Config
# Memory budget (10 GB VRAM):
#   rollout obs buffer : NUM_ENVS × ROLLOUT_STEPS × OBS_SIZE × 4 bytes
#     8192 × 64 × 342 × 4 ≈ 713 MB
#   + values/actions/log_probs/rewards/dones × same dims ≈ 200 MB
#   + PPO minibatch working set (MINI_BATCH_SIZE=8192×64/64=8192) ≈ 200 MB
#   + network params + gradients + Adam moments ≈ 500 MB
#   + compiled vmap kernel static buffers ≈ 1-2 GB
#   Total ≈ 3-4 GB, well within 10 GB.
#
# Previous values (16384 × 96) produced a ~2.1 GB obs buffer alone;
# combined with PPO scan buffers and Adam moments this exceeded 10 GB.
NUM_ENVS      = 8192
ROLLOUT_STEPS = 64

# OBS_SIZE: 9 (pose×3) + 9 (state_vec) + 324 (lidar×3) = 342
OBS_SIZE = 3 * 3 + 9 + 108 * 3    # 342

# NOTE: make_stacked_env is now called inside init_env_state with the correct
# ghost_prob for each curriculum stage. No module-level instance needed.

# ── Compiled vmap_step cache ──────────────────────────────────────────────────
# ghost_robot is a Python bool baked into the JAX closure → any change triggers
# a full XLA retrace (~1-2 min on 3080). To avoid retracing at curriculum
# transitions, we cache compiled vmap_step objects keyed by ghost_robot bool.
# The cache is populated lazily on first use (no extra VRAM for warm-up buffers).
# Curriculum transitions that don't change the ghost_robot value reuse the
# cached kernel at zero cost. Only true ghost_robot flips (True→False) cause
# a one-time retrace, amortized over the remaining curriculum stages.
_VMAP_STEP_CACHE: dict = {}   # {ghost_robot_bool: vmap_step_fn}


def init_env_state(rng_key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0):
    """
    Initialise all NUM_ENVS environments and return the vmap_step closure.

    ghost_prob : float in [0,1] — resolved to bool (ghost_prob >= 0.5).
        Uses cached vmap_step when available — no XLA retrace on curriculum
        transitions that keep the same ghost_robot value.
        A ghost_robot value change causes a one-time retrace and caches result.

    Returns: (env_obs, env_state, vmap_step)
    """
    ghost_robot = (ghost_prob >= 0.5)

    # Build reset/step closures (pure Python, no JAX tracing here)
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_robot=ghost_robot
    )

    # vmap_step: reuse cached compiled kernel if ghost_robot hasn't changed.
    # Building a new jax.jit object when ghost_robot IS the same would still
    # cause a retrace because it's a new Python object (different id()).
    if ghost_robot not in _VMAP_STEP_CACHE:
        # Clear any stale entries first: each compiled vmap kernel holds
        # significant VRAM for XLA scratch buffers. Keeping old kernels alive
        # while new ones are compiled causes a peak-VRAM spike that triggers
        # CUDA_ILLEGAL_ADDRESS on 10 GB cards.
        _VMAP_STEP_CACHE.clear()
        step_auto = make_autoreset_env(reset_stacked, step_stacked,
                                       max_goal_dist=max_goal_dist)
        _VMAP_STEP_CACHE[ghost_robot] = jax.jit(
            jax.vmap(step_auto, in_axes=(0, 0, 0))
        )
    vmap_step = _VMAP_STEP_CACHE[ghost_robot]

    # vmap_reset is always rebuilt (cheap — just Python closure over max_goal_dist)
    def _reset_with_dist(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)
    return env_obs, env_state, vmap_step


def batched_sample_action(rng_key, mean, logstd):
    std       = jnp.exp(logstd)
    noise     = jax.random.normal(rng_key, shape=mean.shape)
    actions   = mean + noise * std
    log_probs = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return actions, log_probs


# static_argnums:
#   2 = apply_fn   (unchanged across curriculum)
#   3 = vmap_step  (NEW — changes on curriculum stage, triggers retrace)
@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts(rng_key, params, apply_fn, vmap_step, env_state, env_obs):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.

    vmap_step is a static arg: when init_env_state() returns a new vmap_step
    object (after a curriculum change), passing it here causes JAX to retrace
    and compile a new kernel that uses the updated autoreset closure.
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        mean, logstd, values = apply_fn({"params": params}, current_obs)
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd)

        max_v       = current_state.env_state.max_v
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions
        )

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
            "passive_col":  infos["passive_col"],
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )
    _, _, last_val = apply_fn({"params": params}, new_obs)
    return rollout_history, new_state, new_obs, last_val