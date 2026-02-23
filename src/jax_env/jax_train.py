"""
jax_train.py — Rollout Collection
===================================
CHANGES vs previous version:

  OBS_SIZE: 339 → 342
    Matches jax_env.py new layout: 9 + 9 + 324 = 342
    (pose_stack=9, state_vec=9, lidar_stack=324)

  FIX (carried) — _vmap_reset JIT applied once at module level.

  ROLLOUT_STEPS: 256 → 64
    Shorter rollouts = more frequent PPO updates = faster convergence.
    Part of 30-min training overhaul (see jax_ppo.py).

  CURRICULUM (NEW) — init_env_state accepts min_goal_dist:
    jax_ppo.py passes the current curriculum distance at each stage change.
    Because min_goal_dist is a Python float (not a JAX array), a change in
    its value triggers a retrace of _vmap_reset — but this happens at most
    3 times across the full run (one per curriculum stage boundary).
    collect_rollouts is unchanged: curriculum only affects resets, which
    happen inside make_autoreset_env → step_autoreset → reset_fn.
    The autoreset reset_fn is bound at make_stacked_env time, so to change
    curriculum distance we reinitialise env_obs/env_state from jax_ppo.py.
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import functools
import jax
import jax.numpy as jnp
from jax_env import reset_env, step_env
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
NUM_ENVS      = 4096
ROLLOUT_STEPS = 64     # was 256 — shorter rollouts for more frequent PPO updates

# OBS_SIZE updated: 9 (pose×3) + 9 (state_vec) + 324 (lidar×3) = 342
OBS_SIZE = 3 * 3 + 9 + 108 * 3    # 342

# Environment setup — built with default min_goal_dist; re-bound at curriculum changes
# via init_env_state(rng, min_goal_dist=X) which rebuilds _vmap_reset with the new dist.
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto   = make_autoreset_env(reset_stacked, step_stacked)

# JIT applied at module level — one compilation only (retraces on min_goal_dist change, max 3×)
_vmap_step  = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))


def init_env_state(rng_key, min_goal_dist: float = 3.0):
    """
    Initialise all NUM_ENVS environments once before the training loop.
    min_goal_dist is a Python float — changing it triggers a retrace of
    _vmap_reset (cheap: happens at most 3 times across the full run).
    """
    # Build a specialised reset that closes over min_goal_dist
    def _reset_with_dist(key):
        return reset_stacked(key, min_goal_dist=min_goal_dist)

    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))
    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    obs, state = vmap_reset(reset_keys)
    return obs, state


def batched_sample_action(rng_key, mean, logstd):
    std       = jnp.exp(logstd)
    noise     = jax.random.normal(rng_key, shape=mean.shape)
    actions   = mean + noise * std
    log_probs = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return actions, log_probs


@functools.partial(jax.jit, static_argnums=(2,))
def collect_rollouts(rng_key, params, apply_fn, env_state, env_obs):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.
    Persistent env state across calls; autoreset handles episode boundaries.
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        mean, logstd, values = apply_fn({"params": params}, current_obs)
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd)

        max_v       = current_state.env_state.max_v
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = _vmap_step(
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
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )
    return rollout_history, new_state, new_obs