"""
dreamer_env.py — Environment Initialization for DreamerV3

"""

import jax
from jax_env_multi import reset_env, step_env
from jax_wrappers  import make_stacked_env, make_autoreset_env


def init_dreamer_envs(
    rng_key,
    num_envs:     int,
    max_goal_dist: float = 3.0,
    ghost_prob:   float  = 1.0,
    scenario_idx: int    = -1,
):
    """
    Builds and JIT-compiles the vectorized step/reset functions, then resets
    `num_envs` environments and returns their initial observations and states.

    Returns:
        env_obs    : initial observations  [num_envs, obs_dim]
        env_state  : initial env states    (pytree)
        vmap_step  : compiled vmap'd step  (step_keys, state, actions, max_goal_dist, scenario_idx)
    """
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob,
    )

    step_auto = make_autoreset_env(reset_stacked, step_stacked)

    # vmap over per-env keys, states, and actions; scalars are broadcast
    vmap_step = jax.jit(
        jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))
    )

    def _reset_with_dist(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx)

    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys         = jax.random.split(rng_key, num_envs)
    env_obs, env_state = vmap_reset(reset_keys)

    return env_obs, env_state, vmap_step