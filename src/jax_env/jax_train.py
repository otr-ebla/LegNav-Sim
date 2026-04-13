"""
jax_train.py 
"""

import os
import functools
import jax
import jax.numpy as jnp
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_network import (
    EndToEndActorCritic,
    scale_actions_batched,
    squash_corrected_log_prob,
)


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
NUM_ENVS      = 512
ROLLOUT_STEPS = 128
OBS_SIZE      = 3 * 3 + 5 + 216 * 3    # 662

_VMAP_STEP_CACHE: dict = {}


def init_env_state(rng_key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0, scenario_idx: int = -1):
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3
    )

    cache_key = "default"
    if cache_key not in _VMAP_STEP_CACHE:
        _VMAP_STEP_CACHE.clear()
        step_auto = make_autoreset_env(reset_stacked, step_stacked)
        _VMAP_STEP_CACHE[cache_key] = jax.jit(
            jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None))
        )
    vmap_step = _VMAP_STEP_CACHE[cache_key]

    def _reset_with_dist(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)

    return env_obs, env_state, vmap_step


def rebuild_vmap_step(ghost_prob: float):
    ghost_robot = (ghost_prob >= 0.5)
    cache_key   = ghost_robot

    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob
    )

    _VMAP_STEP_CACHE.clear()
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    _VMAP_STEP_CACHE[cache_key] = jax.jit(
        # FIX Bug#3: in_axes aveva 5 voci, step_autoreset ne vuole 6
        # (key, state, action, max_goal_dist, scenario_idx, ghost_prob)
        jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None))
    )
    return _VMAP_STEP_CACHE[cache_key]


def batched_sample_action(rng_key, mean, logstd, max_v):
    std         = jnp.exp(logstd)
    noise       = jax.random.normal(rng_key, shape=mean.shape)
    raw_actions = mean + noise * std
    log_probs   = squash_corrected_log_prob(raw_actions, mean, logstd, max_v)
    return raw_actions, log_probs


@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts(
    rng_key,
    params,
    apply_fn,
    vmap_step,
    env_state,
    env_obs,
    max_goal_dist,
    scenario_idx,
    ghost_prob,
):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.
    STATELESS: no hidden carry. Network is pure feedforward (+ frame-stack attention).

    Returns:
      rollout_history  — dict of (ROLLOUT_STEPS, NUM_ENVS, ...) tensors
      new_state        — environment state dopo l'ultimo step
      new_obs          — osservazione dopo l'ultimo step
      last_val         — bootstrap value al termine del chunk
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        # Forward pass stateless: solo obs
        mean, logstd, values = apply_fn({"params": params}, current_obs)

        max_v = current_state.env_state.max_v
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd, max_v)
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions, max_goal_dist, scenario_idx, ghost_prob
        )

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "max_v":        max_v,
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
            "passive_col":  infos["passive_col"],
            "active_col":   infos["active_col"],
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )

    # Bootstrap: forward pass stateless sull'ultima osservazione
    _, _, last_val = apply_fn({"params": params}, new_obs)

    return rollout_history, new_state, new_obs, last_val