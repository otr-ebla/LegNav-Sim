"""
jax_train_asym.py — rollout collection for the asymmetric-critic PPO.

Identical to jax_train.collect_rollouts except that a privileged frame-stack
(ground-truth shoe centres + body velocities of the humans within 5 m) is
carried alongside the observation and stored in every transition. The stack
lives in the scan carry rather than in the env state, so jax_env / jax_wrappers
— shared with SAC, TQC and Dreamer — are left untouched.

The actor never sees the privileged tensor; only the critic does.
"""

import functools
import jax
import jax.numpy as jnp

from legnav.core.jax_network import scale_actions_batched
from legnav.core.jax_privileged import (
    init_privileged_stack, push_privileged_stack, PRIV_OBS_SIZE,
)
from legnav.algorithms.jax_train import (
    init_env_state, batched_sample_action, NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)

__all__ = [
    "init_env_state_asym", "collect_rollouts_asym",
    "NUM_ENVS", "ROLLOUT_STEPS", "OBS_SIZE", "PRIV_OBS_SIZE",
]


def init_env_state_asym(rng_key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0,
                        scenario_idx: int = -1):
    """init_env_state + the reset-time privileged stack (current frame tiled)."""
    env_obs, env_state, vmap_step = init_env_state(
        rng_key, max_goal_dist=max_goal_dist, ghost_prob=ghost_prob, scenario_idx=scenario_idx
    )
    priv_stack = init_privileged_stack(env_state.env_state)   # (N, PRIV_STACK, PRIV_SIZE)
    return env_obs, env_state, priv_stack, vmap_step


@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts_asym(
    rng_key,
    params,
    apply_fn,          # AsymmetricActorCritic.apply → (mean, logstd, value)
    vmap_step,
    env_state,
    env_obs,
    priv_stack,        # (NUM_ENVS, PRIV_STACK, PRIV_SIZE)
    max_goal_dist,
    scenario_idx,
    ghost_prob,
    max_scenario,
):
    """
    Returns:
      rollout_history — dict of (ROLLOUT_STEPS, NUM_ENVS, ...) tensors, now
                        including "priv" of shape (T, N, PRIV_OBS_SIZE)
      new_state, new_obs, new_priv_stack — carried into the next update
      last_val        — bootstrap value, computed with the privileged critic
    """
    def _env_step(carry, _):
        current_state, current_obs, current_priv, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        priv_flat = current_priv.reshape(NUM_ENVS, PRIV_OBS_SIZE)
        mean, logstd, values = apply_fn({"params": params}, current_obs, priv_flat)

        max_v = current_state.env_state.max_v
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd, max_v)
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions, max_goal_dist, scenario_idx,
            ghost_prob, max_scenario
        )

        # next_state is post-autoreset: on a done env the stack is re-seeded.
        next_priv = push_privileged_stack(current_priv, next_state.env_state, dones)

        transition = {
            "obs":          current_obs,
            "priv":         priv_flat,
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
        return (next_state, next_obs, next_priv, current_rng), transition

    (new_state, new_obs, new_priv, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, priv_stack, rng_key), None, length=ROLLOUT_STEPS
    )

    _, _, last_val = apply_fn(
        {"params": params}, new_obs, new_priv.reshape(NUM_ENVS, PRIV_OBS_SIZE)
    )

    return rollout_history, new_state, new_obs, new_priv, last_val
