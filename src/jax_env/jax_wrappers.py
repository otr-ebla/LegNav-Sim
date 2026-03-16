"""
jax_wrappers.py — Observation Stacking & Auto-Reset
====================================================
FIXES vs previous version:

  FIX — @jax.jit removed from reset_stacked, step_stacked, step_autoreset
    (Issue #8, coordinated with jax_env.py and jax_train.py).

  CHANGE — ghost_robot as closure variable (NOT a runtime argument):
    ghost_robot must be a Python bool baked into the function at construction
    time. Passing it as a runtime argument causes TracerBoolConversionError.

  CHANGE — ghost_prob curriculum support (CHANGE E from jax_env_multi.py):
    make_stacked_env now accepts ghost_prob (float in [0,1], default 1.0).
    ghost_prob=1.0  → always ghost (original training behaviour).
    ghost_prob=0.0  → never ghost (full eval behaviour).
    ghost_prob=p    → ghost_robot=True with probability p, resolved ONCE
                      at construction time by sampling a Python bool.
    The sampled bool is baked into the closure — JAX never sees a traced bool.

    For curriculum decay: rebuild make_stacked_env at each stage change in
    jax_ppo.py with the new ghost_prob, same as rebuilding for min_goal_dist.
    make_autoreset_env is ghost_agnostic (behaviour is locked in step_fn).

Obs layout: [pose_stack(3*stack_dim=9) | state_vec(9) | lidar_stack(num_rays*stack_dim=324)]
Total: 9 + 9 + 324 = 342
"""

import jax
import jax.numpy as jnp
import random as _random
from flax import struct
from jax_env import EnvState, NUM_RAYS, SINGLE_OBS_SIZE, STATE_VEC_SIZE

POSE_SIZE = 3


@struct.dataclass
class StackedEnvState:
    env_state:   EnvState
    lidar_stack: jnp.ndarray   # (stack_dim, NUM_RAYS)
    pose_stack:  jnp.ndarray   # (stack_dim, POSE_SIZE)


def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3,
                     num_rays: int = NUM_RAYS, ghost_robot: bool = True,
                     ghost_prob: float = 1.0):
    """
    Temporal stacking wrapper.

    ghost_robot : Python bool — kept for backward compatibility.
        If ghost_prob is not 1.0, ghost_robot is IGNORED and overridden by
        a Bernoulli sample of ghost_prob resolved here at construction time.

    ghost_prob  : float in [0, 1] — probability that ghost_robot=True for
        this particular stacked env instance.
        ghost_prob=1.0 → always ghost (default, original training behaviour).
        ghost_prob=0.0 → never ghost (eval mode).
        ghost_prob=p   → sampled once here; baked as a Python bool into the
                         closure so JAX never sees a traced conditional.

        CURRICULUM USAGE: call make_stacked_env again with the new ghost_prob
        at each curriculum stage change (same pattern as min_goal_dist).
    """
    # Resolve ghost_robot once at construction time — never a traced value.
    if ghost_prob < 1.0:
        ghost_robot = (_random.random() < ghost_prob)

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
        # ghost_robot is captured from the enclosing scope as a Python bool —
        # never a JAX-traced value, so the `if` inside step_env is safe.
        base_obs, new_base_state, reward, done, info = base_step_fn(
            key, state.env_state, action, ghost_robot=ghost_robot
        )

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
    Auto-reset wrapper.

    ghost_robot behaviour is already baked into step_fn (= step_stacked) via
    make_stacked_env's ghost_robot parameter — do NOT pass it again here.
    This function is ghost_robot-agnostic: it works for both training and eval
    because the correct behaviour is locked into the step_fn closure.
    """
    def step_autoreset(key, state, action):
        # ghost_robot is NOT forwarded here: step_fn (= step_stacked) already
        # closes over it from its own make_stacked_env call. step_stacked's
        # signature is (key, state, action) — passing ghost_robot again would
        # raise "unexpected keyword argument".
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