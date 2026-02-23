"""
jax_wrappers.py — Observation Stacking & Auto-Reset
====================================================
FIXES vs previous version:
  1. state_vec slice updated from [3:8] to [3:9] — state_vec is now size 6
     (added rear_prox scalar in jax_env.py).
  2. Autoreset broadcast fix retained (was already correct).

Obs layout: [pose_stack(3*stack_dim) | state_vec(6) | lidar_stack(num_rays*stack_dim)]
Total: 9 + 6 + 324 = 339
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_env import EnvState, NUM_RAYS, SINGLE_OBS_SIZE

STATE_VEC_SIZE = 6   # v, w, max_v_norm, goal_dist, goal_align, rear_prox
POSE_SIZE      = 3


@struct.dataclass
class StackedEnvState:
    env_state:   EnvState
    lidar_stack: jnp.ndarray   # (stack_dim, NUM_RAYS)
    pose_stack:  jnp.ndarray   # (stack_dim, 3)


def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3, num_rays: int = NUM_RAYS):
    """
    Temporal stacking wrapper.
    Obs layout: [pose_stack(3*stack_dim) | state_vec(6) | lidar_stack(num_rays*stack_dim)]
    Total: 9 + 6 + 324 = 339 elements.
    """

    @jax.jit
    def reset_stacked(key):
        base_obs, base_state = base_reset_fn(key)
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

    @jax.jit
    def step_stacked(key, state: StackedEnvState, action):
        base_obs, new_base_state, reward, done, info = base_step_fn(key, state.env_state, action)

        new_pose      = base_obs[0:POSE_SIZE]
        new_state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[POSE_SIZE + STATE_VEC_SIZE:]

        new_lidar_stack = jnp.roll(state.lidar_stack, shift=-1, axis=0).at[-1].set(new_lidar)
        new_pose_stack  = jnp.roll(state.pose_stack,  shift=-1, axis=0).at[-1].set(new_pose)

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


def make_autoreset_env(reset_fn, step_fn):
    """
    Auto-reset wrapper.
    done is scalar bool. For each pytree leaf of shape (d1,d2,...),
    reshape done to (1,...,1) with the same ndim so jnp.where broadcasts correctly.
    """

    @jax.jit
    def step_autoreset(key, state, action):
        step_key, reset_key = jax.random.split(key)

        obs, next_state, reward, done, info = step_fn(step_key, state, action)
        reset_obs, reset_state = reset_fn(reset_key)

        def _select(reset_leaf, next_leaf):
            d = done.reshape((1,) * reset_leaf.ndim) if reset_leaf.ndim > 0 else done
            return jnp.where(d, reset_leaf, next_leaf)

        final_state = jax.tree_util.tree_map(_select, reset_state, next_state)
        final_obs   = jnp.where(done, reset_obs, obs)

        return final_obs, final_state, reward, done, info

    return step_autoreset