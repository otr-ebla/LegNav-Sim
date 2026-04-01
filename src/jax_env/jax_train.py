"""
jax_train.py — Rollout Collection
===================================
CHANGES vs previous version:

  GRU MEMORY SUPPORT:
    - collect_rollouts now maintains a `hidden` carry in the lax.scan loop.
    - At each step, `hidden` is updated by the network and then masked to zero
      for any environment that just terminated (done=True). This prevents a
      completed episode's memory from contaminating the next episode in the
      same environment slot.
    - The rollout buffer stores `hiddens` — the hidden state BEFORE each step
      (i.e. the input hidden to the network that produced actions/values). This
      is the "initial hidden" TBPTT needs to re-run the GRU during the update.
    - `last_hidden` is returned alongside `last_val` so ppo.py can bootstrap
      the GAE correctly.
    - The initial hidden passed to collect_rollouts is the live carry from the
      previous rollout chunk (TRUE TBPTT across rollout boundaries).

  UNCHANGED:
    - Ghost-prob curriculum and vmap_step caching logic.
    - Dynamic curriculum variables (max_goal_dist, scenario_idx).
    - ROLLOUT_STEPS = 64, NUM_ENVS = 8192.
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
    GRU_HIDDEN_SIZE,
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
OBS_SIZE      = 3 * 3 + 9 + 108 * 3    # 342

_VMAP_STEP_CACHE: dict = {}   # {ghost_robot_bool: vmap_step_fn}


def init_env_state(rng_key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0, scenario_idx: int = -1):
    """
    Initialise all NUM_ENVS environments and return the vmap_step closure.
    Only called once at startup, and when ghost_prob changes (see rebuild_vmap_step).
    """
    ghost_robot = (ghost_prob >= 0.5)

    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob
    )

    cache_key = ghost_robot
    if cache_key not in _VMAP_STEP_CACHE:
        _VMAP_STEP_CACHE.clear()
        step_auto = make_autoreset_env(reset_stacked, step_stacked)
        _VMAP_STEP_CACHE[cache_key] = jax.jit(
            jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))
        )
    vmap_step = _VMAP_STEP_CACHE[cache_key]

    def _reset_with_dist(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)

    return env_obs, env_state, vmap_step


def rebuild_vmap_step(ghost_prob: float):
    """
    Rebuild ONLY the vmap_step closure when ghost_prob changes at a curriculum
    transition — without touching env_state or env_obs (see original comments).
    """
    ghost_robot = (ghost_prob >= 0.5)
    cache_key   = ghost_robot

    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob
    )

    _VMAP_STEP_CACHE.clear()
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    _VMAP_STEP_CACHE[cache_key] = jax.jit(
        jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))
    )
    return _VMAP_STEP_CACHE[cache_key]


def batched_sample_action(rng_key, mean, logstd, max_v):
    """
    Sample raw actions and compute log_prob corrected for sigmoid/tanh squashing.
    max_v is the per-env max velocity array (shape: NUM_ENVS,).
    """
    std     = jnp.exp(logstd)
    noise   = jax.random.normal(rng_key, shape=mean.shape)
    raw_actions = mean + noise * std

    # Unified math logic. Stop duplicating equations.
    log_probs = squash_corrected_log_prob(raw_actions, mean, logstd, max_v)
    return raw_actions, log_probs


@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts(
    rng_key,
    params,
    apply_fn,
    vmap_step,
    env_state,
    env_obs,
    hidden,          # (NUM_ENVS, GRU_HIDDEN_SIZE) — live carry from previous chunk
    max_goal_dist,
    scenario_idx,
):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.

    Returns:
      rollout_history  — dict of (ROLLOUT_STEPS, NUM_ENVS, ...) tensors
                         including 'hiddens' = GRU input hidden at each step
      new_state        — environment state after the final step
      new_obs          — observation after the final step
      new_hidden       — GRU carry after the final step (for the next chunk)
      last_val         — critic bootstrap value at new_obs / new_hidden
    """
    def _env_step(carry, _):
        current_state, current_obs, current_hidden, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        # Forward pass — network now takes hidden and returns new_hidden
        mean, logstd, values, next_hidden = apply_fn(
            {"params": params}, current_obs, current_hidden
        )

        max_v = current_state.env_state.max_v
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd, max_v)
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions, max_goal_dist, scenario_idx
        )

        # ── Done-masking: reset hidden to zero for terminated environments ────
        # Shape: (NUM_ENVS, 1) broadcast over hidden dim
        mask = (1.0 - dones.astype(jnp.float32))[:, None]
        masked_hidden = next_hidden * mask

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "max_v":        max_v,
            "hiddens":      current_hidden,   # GRU input at this step — needed for TBPTT
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
            "passive_col":  infos["passive_col"],
            "active_col":   infos["active_col"],
        }
        return (next_state, next_obs, masked_hidden, current_rng), transition

    (new_state, new_obs, new_hidden, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, hidden, rng_key), None, length=ROLLOUT_STEPS
    )

    # Bootstrap critic value at the end of the chunk
    _, _, last_val, _ = apply_fn({"params": params}, new_obs, new_hidden)

    return rollout_history, new_state, new_obs, new_hidden, last_val