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

    For curriculum updates: rebuild make_stacked_env whenever ghost_prob changes
    (driven by get_continuous_curriculum in jax_ppo.py), same as rebuilding
    for max_goal_dist. make_autoreset_env is ghost-agnostic (behaviour is locked in step_fn).

  CHANGE — CALF ego-motion compensation obs layout:
    goal_stack (ego goal vector g_x, g_y over the last stack_dim frames) is
    kept in the flat obs, flattened next to kin_vec: together they form the
    global state g_T. kin_vec is NOT stacked (current frame only).
    A pose_stack (world [x, y, θ] at each frame time) is tracked as well and
    converted each step into per-frame ego-motion deltas (Δx, Δy, Δθ): the
    SE(2) pose of frame t-k expressed in the current robot frame. The network
    uses them to reproject past scans into the current frame (align t-k → t).

  CHANGE — strided frame stacking (RobotConfig.LIDAR_STACK_STRIDE):
    With a small DT consecutive scans are nearly identical, so stacking
    o_t, o_{t-1}, o_{t-2} carries little temporal signal. Instead we keep an
    internal full-resolution ring buffer and expose every STRIDE-th frame:
    o_t, o_{t-STRIDE}, o_{t-2*STRIDE}, .... The exposed obs still has stack_dim
    frames, so the layout/size and the network are unchanged. STRIDE=1 recovers
    the classic consecutive stack.

Obs layout: [kin_vec(5) | goal_stack(2*stack_dim=6) | ego_deltas(3*stack_dim=9) | lidar_stack(num_rays*stack_dim=648)]
Total: 5 + 6 + 9 + 648 = 668
"""

import jax
import jax.numpy as jnp
import random as _random
from flax import struct
from legnav.core.jax_env import EnvState, NUM_RAYS, SINGLE_OBS_SIZE, KIN_VEC_SIZE, GOAL_VEC_SIZE
from legnav.config import RobotConfig

GOAL_SIZE      = GOAL_VEC_SIZE
EGO_DELTA_SIZE = 3   # (Δx, Δy, Δθ) per frame, in the current robot frame


@struct.dataclass
class StackedEnvState:
    env_state:   EnvState
    lidar_stack: jnp.ndarray        # (buffer_len, NUM_RAYS) full-resolution ring buffer, oldest first
    goal_stack:  jnp.ndarray        # (buffer_len, GOAL_SIZE) ego goal vec per frame, oldest first
    pose_stack:  jnp.ndarray = None  # (buffer_len, 3) world pose [x, y, θ] at each frame time


def _ego_deltas(pose_stack: jnp.ndarray) -> jnp.ndarray:
    """
    SE(2) pose of each stored frame expressed in the newest frame.

    pose_stack: (stack_dim, 3) world poses [x, y, θ], oldest first.
    Returns     (stack_dim, 3) [Δx, Δy, Δθ] per frame — newest row is zeros.
    """
    cur = pose_stack[-1]
    dxw = pose_stack[:, 0] - cur[0]
    dyw = pose_stack[:, 1] - cur[1]
    c, s = jnp.cos(cur[2]), jnp.sin(cur[2])
    dx  =  c * dxw + s * dyw
    dy  = -s * dxw + c * dyw
    dth = (pose_stack[:, 2] - cur[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return jnp.stack([dx, dy, dth], axis=-1)


def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3,
                     num_rays: int = NUM_RAYS,
                     stride: int = RobotConfig.LIDAR_STACK_STRIDE):
    """Temporal frame stacking with a configurable stride.

    A full-resolution ring buffer of length ``buffer_len = (stack_dim-1)*stride+1``
    is kept internally; the flat observation exposes only ``stack_dim`` frames
    spaced ``stride`` apart — i.e. o_t, o_{t-stride}, o_{t-2*stride}, ....
    With ``stride=1`` this reduces to the classic o_t, o_{t-1}, o_{t-2} stack.
    The exposed observation layout (and its size) is independent of ``stride``,
    so the network is unchanged. The newest frame is always the buffer's last
    row, so downstream reads of ``state.lidar_stack[-1]`` stay valid.
    """
    buffer_len = (stack_dim - 1) * stride + 1
    # Oldest-first indices selecting the strided subset (last = newest frame).
    sel = jnp.arange(stack_dim) * stride

    def reset_stacked(key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0, scenario_idx: int = -1, min_goal_dist: float = 0.8, **kwargs):
        # Passes any extra dynamic args (like scenario_idx) gracefully down to the environment
        base_obs, base_state = base_reset_fn(key, max_goal_dist=max_goal_dist, min_goal_dist=min_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob, **kwargs)
        goal      = base_obs[0:GOAL_SIZE]
        kin_vec   = base_obs[GOAL_SIZE : GOAL_SIZE + KIN_VEC_SIZE]
        lidar     = base_obs[GOAL_SIZE + KIN_VEC_SIZE:]

        lidar_stack = jnp.tile(lidar[None, :], (buffer_len, 1))
        goal_stack  = jnp.tile(goal[None,  :], (buffer_len, 1))

        pose       = jnp.array([base_state.x, base_state.y, base_state.theta])
        pose_stack = jnp.tile(pose[None, :], (buffer_len, 1))

        stacked_state = StackedEnvState(
            env_state=base_state,
            lidar_stack=lidar_stack,
            goal_stack=goal_stack,
            pose_stack=pose_stack
        )
        flat_obs = jnp.concatenate([
            kin_vec, goal_stack[sel].flatten(),
            _ego_deltas(pose_stack[sel]).flatten(), lidar_stack[sel].flatten()
        ])
        return flat_obs, stacked_state

    def step_stacked(key, state: StackedEnvState, action):
        base_obs, new_base_state, reward, done, info = base_step_fn(
            key, state.env_state, action
        )

        new_goal      = base_obs[0:GOAL_SIZE]
        new_kin_vec   = base_obs[GOAL_SIZE : GOAL_SIZE + KIN_VEC_SIZE]
        new_lidar     = base_obs[GOAL_SIZE + KIN_VEC_SIZE:]

        new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
        new_goal_stack  = jnp.concatenate([state.goal_stack[1:],  new_goal[None]],  axis=0)

        new_pose       = jnp.array([new_base_state.x, new_base_state.y, new_base_state.theta])
        new_pose_stack = jnp.concatenate([state.pose_stack[1:], new_pose[None]], axis=0)

        new_stacked_state = StackedEnvState(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            goal_stack=new_goal_stack,
            pose_stack=new_pose_stack
        )
        flat_obs = jnp.concatenate([
            new_kin_vec, new_goal_stack[sel].flatten(),
            _ego_deltas(new_pose_stack[sel]).flatten(), new_lidar_stack[sel].flatten()
        ])
        return flat_obs, new_stacked_state, reward, done, info

    return reset_stacked, step_stacked


def make_autoreset_env(reset_fn, step_fn):
    def step_autoreset(key, state, action, max_goal_dist, scenario_idx, ghost_prob, max_scenario, min_goal_dist=0.8):
        step_key, reset_key = jax.random.split(key)
        obs, next_state, reward, done, info = step_fn(step_key, state, action)

        # max_scenario flows via **kwargs through reset_stacked → reset_env → generate_scenario
        # so the random draw on reset is capped to [0, max_scenario] (curriculum range).
        reset_obs, reset_state = reset_fn(reset_key, max_goal_dist=max_goal_dist,
                          scenario_idx=scenario_idx, ghost_prob=ghost_prob,
                          max_scenario=max_scenario, min_goal_dist=min_goal_dist)

        def _select(reset_leaf, next_leaf):
            d = jnp.asarray(done)
            d = d.reshape((1,) * next_leaf.ndim) if next_leaf.ndim > 0 else d
            return jnp.where(d, reset_leaf, next_leaf)

        final_state = jax.tree_util.tree_map(_select, reset_state, next_state)
        final_obs   = jnp.where(done, reset_obs, obs)

        # True successor obs of THIS transition: when done, final_obs is the NEW
        # episode's first obs — off-policy learners must bootstrap timeout
        # transitions (done, terminal=0) from the pre-reset obs, or the TD
        # target is computed on the wrong state.
        info = {**info, "final_obs": obs}

        return final_obs, final_state, reward, done, info

    return step_autoreset