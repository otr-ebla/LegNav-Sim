"""
jax_wrappers.py — Observation Stacking & Auto-Reset
====================================================
Fixes vs original:
  - make_autoreset_env: reset uses a freshly split key (not the same key as step)
  - jnp.where broadcasting fixed: done is scalar, state leaves can be any shape
  - StackedEnvState documented clearly
  - Obs size consistent with jax_env.py (3 + 5 + NUM_RAYS = 116 per frame)
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_env import EnvState, NUM_RAYS


@struct.dataclass
class StackedEnvState:
    """
    Wraps EnvState with circular buffers for temporal stacking.
      lidar_stack : (stack_dim, NUM_RAYS)
      pose_stack  : (stack_dim, 3)
    """
    env_state:   EnvState
    lidar_stack: jnp.ndarray   # (stack_dim, NUM_RAYS)
    pose_stack:  jnp.ndarray   # (stack_dim, 3)


def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3, num_rays: int = NUM_RAYS):
    """
    Returns (reset_stacked, step_stacked) wrapping the base env with temporal stacking.
    Stacked obs layout: [pose_stack(3*stack_dim) | state_vec(5) | lidar_stack(num_rays*stack_dim)]
    Total: 3*3 + 5 + 108*3 = 338 elements.
    """

    @jax.jit
    def reset_stacked(key: jnp.ndarray):
        base_obs, base_state = base_reset_fn(key)

        # base_obs layout: pose(3) | state_vec(5) | lidar(num_rays)
        pose      = base_obs[0:3]
        state_vec = base_obs[3:8]
        lidar     = base_obs[8:]

        lidar_stack = jnp.tile(lidar[None, :], (stack_dim, 1))  # (stack_dim, num_rays)
        pose_stack  = jnp.tile(pose[None,  :], (stack_dim, 1))  # (stack_dim, 3)

        stacked_state = StackedEnvState(
            env_state=base_state,
            lidar_stack=lidar_stack,
            pose_stack=pose_stack
        )

        flat_obs = jnp.concatenate([
            pose_stack.flatten(),    # 3 * stack_dim
            state_vec,               # 5
            lidar_stack.flatten()    # num_rays * stack_dim
        ])
        return flat_obs, stacked_state

    @jax.jit
    def step_stacked(key: jnp.ndarray, state: StackedEnvState, action: jnp.ndarray):
        base_obs, new_base_state, reward, done, info = base_step_fn(key, state.env_state, action)

        new_pose      = base_obs[0:3]
        new_state_vec = base_obs[3:8]
        new_lidar     = base_obs[8:]

        # Shift circular buffers (oldest frame out, newest in at index -1)
        new_lidar_stack = jnp.roll(state.lidar_stack, shift=-1, axis=0).at[-1].set(new_lidar)
        new_pose_stack  = jnp.roll(state.pose_stack,  shift=-1, axis=0).at[-1].set(new_pose)

        new_stacked_state = StackedEnvState(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack
        )

        flat_obs = jnp.concatenate([
            new_pose_stack.flatten(),
            new_state_vec,
            new_lidar_stack.flatten()
        ])
        return flat_obs, new_stacked_state, reward, done, info

    return reset_stacked, step_stacked


def make_autoreset_env(reset_fn, step_fn):
    """
    Auto-reset wrapper: if an episode ends (done=True) the env is immediately reset.
    The returned obs/state is always valid for the NEXT episode.

    FIX 1: reset_key is derived from a fresh split, not reusing the step key.
    FIX 2: jnp.where broadcasts `done` (scalar) against arbitrary-shaped leaves.
    """

    @jax.jit
    def step_autoreset(key: jnp.ndarray, state, action: jnp.ndarray):
        step_key, reset_key = jax.random.split(key)   # FIX 1: independent keys

        obs, next_state, reward, done, info = step_fn(step_key, state, action)

        reset_obs, reset_state = reset_fn(reset_key)

        # FIX 2: reshape `done` to broadcast against each leaf's shape
        final_state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(
                done.reshape((1,) * (r.ndim - 0) if r.ndim == 0 else (1,) * r.ndim).squeeze()
                if r.ndim > 0 else done,
                r, n
            ),
            reset_state,
            next_state
        )

        final_obs = jnp.where(done, reset_obs, obs)

        return final_obs, final_state, reward, done, info

    return step_autoreset