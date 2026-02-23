"""
jax_wrappers.py — Observation Stacking & Auto-Reset
====================================================
CHANGES vs previous version:

  Updated for new obs layout from jax_env.py:
    STATE_VEC_SIZE: 6 → 9  (rear_prox expanded from 1 scalar to 4 scalars)
    SINGLE_OBS_SIZE: 117 → 120
    Stacked OBS_SIZE: 339 → 342  (9 + 9 + 324)

  IMPROVEMENT — jnp.roll replaced with slice-assign (NEW):
    jnp.roll(axis=0) on a (3, N) array allocates a full copy and then
    applies a gather permutation. For temporal stacking with shift=-1 and
    at[-1].set(...), we can do this more efficiently with a direct slice:
      new_stack = jnp.concatenate([old_stack[1:], new_frame[None]], axis=0)
    This is equivalent, avoids the roll+gather, and XLA can fuse it more
    aggressively since the output shape is statically known.

  UNCHANGED — autoreset broadcast logic was already correct.

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

        # IMPROVEMENT: concatenate slice instead of roll+set — avoids gather permutation
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


def make_autoreset_env(reset_fn, step_fn):
    """
    Auto-reset wrapper.
    done is scalar bool. For each pytree leaf, reshape done to broadcast correctly.
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