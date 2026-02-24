"""
jax_wrappers.py — Observation Stacking & Auto-Reset
====================================================
FIXES vs previous version:

  FIX — @jax.jit removed from reset_stacked, step_stacked, step_autoreset
    (Issue #8, coordinated with jax_env.py and jax_train.py):
    All three functions were @jax.jit decorated but are always called from
    within an outer jax.jit (the vmap in jax_train.init_env_state).
    Nested JIT prevents XLA from fusing across boundaries and adds tracing
    overhead. JIT now lives only at the outermost vmap level in jax_train.py.

  All previous changes (obs layout 342, slice-assign stacking, curriculum
  min_goal_dist forwarding) are carried forward unchanged.

Obs layout: [pose_stack(3*stack_dim=9) | state_vec(9) | lidar_stack(num_rays*stack_dim=324)]
Total: 9 + 9 + 324 = 342
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_env import EnvState, NUM_RAYS, SINGLE_OBS_SIZE, STATE_VEC_SIZE

POSE_SIZE = 3


@struct.dataclass
class StackedEnvState:
    env_state:   EnvState
    lidar_stack: jnp.ndarray   # (stack_dim, NUM_RAYS)
    pose_stack:  jnp.ndarray   # (stack_dim, POSE_SIZE)


def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3, num_rays: int = NUM_RAYS):
    """
    Temporal stacking wrapper.
    Obs layout: [pose_stack(3*stack_dim) | state_vec(STATE_VEC_SIZE) | lidar_stack(num_rays*stack_dim)]
    Total: 9 + 9 + 324 = 342 elements.

    FIX: @jax.jit removed from reset_stacked and step_stacked.
    JIT lives only at the outermost vmap level in jax_train.py.
    """

    def reset_stacked(key, min_goal_dist: float = 3.0):
        base_obs, base_state = base_reset_fn(key, min_goal_dist)
        pose      = base_obs[0:POSE_SIZE]
        state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        lidar     = base_obs[POSE_SIZE + STATE_VEC_SIZE:]

        lidar_stack = jnp.tile(lidar[None, :], (stack_dim, 1))
        pose_stack  = jnp.tile(pose[None,  :], (stack_dim, 1))

        stacked_state = StackedEnvState(
            env_state=base_state,
            lidar_stack=lidar_stack,
            pose_stack=pose_stack
        )
        flat_obs = jnp.concatenate([
            pose_stack.flatten(), state_vec, lidar_stack.flatten()
        ])
        return flat_obs, stacked_state

    def step_stacked(key, state: StackedEnvState, action):
        base_obs, new_base_state, reward, done, info = base_step_fn(key, state.env_state, action)

        new_pose      = base_obs[0:POSE_SIZE]
        new_state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[POSE_SIZE + STATE_VEC_SIZE:]

        new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
        new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)

        new_stacked_state = StackedEnvState(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack
        )
        flat_obs = jnp.concatenate([
            new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
        ])
        return flat_obs, new_stacked_state, reward, done, info

    return reset_stacked, step_stacked


def make_autoreset_env(reset_fn, step_fn, min_goal_dist: float = 3.0):
    """
    FIX: @jax.jit removed from step_autoreset.
    JIT lives only at the outermost vmap level in jax_train.py.
    """
    def step_autoreset(key, state, action):
        step_key, reset_key = jax.random.split(key)
        obs, next_state, reward, done, info = step_fn(step_key, state, action)
        reset_obs, reset_state = reset_fn(reset_key, min_goal_dist)

        def _select(reset_leaf, next_leaf):
            d = jnp.asarray(done)
            d = d.reshape((1,) * next_leaf.ndim) if next_leaf.ndim > 0 else d
            return jnp.where(d, reset_leaf, next_leaf)

        final_state = jax.tree_util.tree_map(_select, reset_state, next_state)
        final_obs   = jnp.where(done, reset_obs, obs)

        return final_obs, final_state, reward, done, info

    return step_autoreset