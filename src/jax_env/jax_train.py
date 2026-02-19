"""
jax_train.py — Massive Vectorised Rollout Collection
=====================================================
Fixes vs original:
  - Rollout loop now uses the ACTUAL network (params + apply_fn) instead of dummy random actions
  - log_probs and values are stored in the transition dict (required by PPO)
  - @jax.jit not nested inside vmap (removed inner @jax.jit from wrappers when used here)
  - step_auto uses split key for step/reset to prevent correlated randomness (fixed in wrappers)
  - Correct obs_size constant derived from stacked env

Usage:
  from jax_train import collect_rollouts, NUM_ENVS, ROLLOUT_STEPS
"""

import jax
import jax.numpy as jnp
from jax_env import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_network import sample_action

# ── Config ───────────────────────────────────────────────────────────────────
NUM_ENVS      = 4096    # parallel simulators on one GPU
ROLLOUT_STEPS = 128     # PPO horizon

# Obs size: pose_stack(9) + state_vec(5) + lidar_stack(324) = 338
OBS_SIZE = 3 * 3 + 5 + 108 * 3   # = 338

# ── Environment setup ────────────────────────────────────────────────────────
# Build stacked env then auto-reset wrapper
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto = make_autoreset_env(reset_stacked, step_stacked)

# Vectorise across NUM_ENVS (no nested jit — wrappers are jitted internally)
vmap_reset = jax.vmap(reset_stacked)
vmap_step  = jax.vmap(step_auto, in_axes=(0, 0, 0))


# ── Rollout Collection ───────────────────────────────────────────────────────

@jax.jit
def collect_rollouts(rng_key: jnp.ndarray, params: dict, apply_fn):
    """
    Collects NUM_ENVS * ROLLOUT_STEPS transitions entirely on the GPU via lax.scan.

    Returns:
      rollout_history : dict with keys obs, actions, log_probs, values, rewards, dones, next_obs
                        each of shape (ROLLOUT_STEPS, NUM_ENVS, ...)
      final_carry     : (final_stacked_state, final_obs, rng_key)
    """
    # ── Initialise all environments ──────────────────────────────────────────
    rng_key, reset_key = jax.random.split(rng_key)
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    initial_obs, initial_state = vmap_reset(reset_keys)

    # ── Single-step function for lax.scan ────────────────────────────────────
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry

        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        # --- Network inference on GPU ---
        mean, logstd, values = apply_fn({"params": params}, current_obs)

        # Split rng across envs for independent action sampling
        action_rngs = jax.random.split(action_rng, NUM_ENVS)
        actions, log_probs = jax.vmap(sample_action)(action_rngs, mean, logstd)

        # Clip actions to valid ranges for the robot kinematics
        # action[:,0] = linear velocity  ∈ [0, max_v]  (max_v varies per env; clip at 1.5)
        # action[:,1] = angular velocity ∈ [-1.5, 1.5]
        actions = jnp.stack([
            jnp.clip(actions[:, 0], 0.0,  1.5),
            jnp.clip(actions[:, 1], -1.5, 1.5)
        ], axis=-1)

        # --- Step all envs ---
        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(step_keys, current_state, actions)

        transition = {
            "obs":       current_obs,   # (NUM_ENVS, OBS_SIZE)
            "actions":   actions,        # (NUM_ENVS, 2)
            "log_probs": log_probs,      # (NUM_ENVS,)   ← FIX: was missing
            "values":    values,         # (NUM_ENVS,)   ← FIX: was missing
            "rewards":   rewards,        # (NUM_ENVS,)
            "dones":     dones,          # (NUM_ENVS,)
            "next_obs":  next_obs,       # (NUM_ENVS, OBS_SIZE)
        }

        next_carry = (next_state, next_obs, current_rng)
        return next_carry, transition

    # ── XLA-compiled scan loop ───────────────────────────────────────────────
    initial_carry = (initial_state, initial_obs, rng_key)

    final_carry, rollout_history = jax.lax.scan(
        _env_step,
        initial_carry,
        None,
        length=ROLLOUT_STEPS
    )

    # rollout_history values: (ROLLOUT_STEPS, NUM_ENVS, ...)
    return rollout_history, final_carry