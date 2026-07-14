"""
jax_privileged.py — Ground-truth human state for the asymmetric critic.

This vector is SIMULATION-ONLY: it is never given to the actor, because a real
TurtleBot cannot measure shoe centres or body velocities. It exists so that the
critic can estimate V(s) from the true state instead of inferring it from the
LiDAR stack (asymmetric actor-critic, Pinto et al. 2017).

Per-human features (PRIV_FEAT = 7), all in the CURRENT ROBOT FRAME:
  [0:2] left shoe centre  (x, y)   — foot_state[:, 0:2]
  [2:4] right shoe centre (x, y)   — foot_state[:, 2:4]
  [4:6] body velocity     (vx, vy) — people[:, 2:4]
  [6]   valid flag                 — 1.0 if the human is within PRIV_RANGE

The PRIV_HUMANS nearest humans within PRIV_RANGE metres are emitted, sorted by
body distance (nearest first). Empty slots — out-of-range humans and the
(-999, -999) dummy slots that pad `people` — are all-zero with valid = 0.
Positions are normalised by PRIV_RANGE and velocities by MAX_SPEED so the
critic MLP sees O(1) inputs.
"""

import jax
import jax.numpy as jnp

from legnav.core.jax_humans import MAX_SPEED

PRIV_RANGE  = 5.0                          # m — radius around the robot
PRIV_HUMANS = 6                            # nearest humans emitted
PRIV_FEAT   = 7                            # per human: 2+2+2 floats + valid flag
PRIV_STACK  = 3                            # frames, matches the LiDAR stack
PRIV_SIZE   = PRIV_HUMANS * PRIV_FEAT      # 42 — one frame
PRIV_OBS_SIZE = PRIV_SIZE * PRIV_STACK     # 126 — what the critic actually sees


def privileged_frame(env_state) -> jnp.ndarray:
    """Single-env ground-truth human frame. env_state: unstacked EnvState.

    Returns (PRIV_SIZE,) = (PRIV_HUMANS * PRIV_FEAT,).
    """
    rx, ry, rtheta = env_state.x, env_state.y, env_state.theta
    c, s = jnp.cos(rtheta), jnp.sin(rtheta)

    def to_ego(px, py):
        dx, dy = px - rx, py - ry
        return c * dx + s * dy, -s * dx + c * dy

    def rot_ego(vx, vy):
        return c * vx + s * vy, -s * vx + c * vy

    people     = env_state.people        # (N, 8)
    foot_state = env_state.foot_state    # (N, 10)

    bx, by = people[:, 0], people[:, 1]
    dist   = jnp.sqrt((bx - rx) ** 2 + (by - ry) ** 2 + 1e-8)
    valid  = dist < PRIV_RANGE           # dummies live at (-999, -999) → False

    lx, ly = to_ego(foot_state[:, 0], foot_state[:, 1])
    rx_, ry_ = to_ego(foot_state[:, 2], foot_state[:, 3])
    vx, vy = rot_ego(people[:, 2], people[:, 3])

    feats = jnp.stack([
        lx / PRIV_RANGE, ly / PRIV_RANGE,
        rx_ / PRIV_RANGE, ry_ / PRIV_RANGE,
        vx / MAX_SPEED,  vy / MAX_SPEED,
        jnp.ones_like(lx),
    ], axis=-1)                                        # (N, PRIV_FEAT)
    feats = jnp.where(valid[:, None], feats, 0.0)

    # Nearest-first ordering; invalid humans are pushed to the back of the sort
    # so that the leading slots are always the real, in-range ones.
    rank_key = jnp.where(valid, dist, jnp.inf)
    idx = jnp.argsort(rank_key)[:PRIV_HUMANS]
    return feats[idx].reshape(-1)                      # (PRIV_SIZE,)


# Batched over the NUM_ENVS axis of a vmapped StackedEnvState.
privileged_frame_batched = jax.vmap(privileged_frame)


def init_privileged_stack(env_state) -> jnp.ndarray:
    """Tile the current frame PRIV_STACK times — the reset-time history."""
    frame = privileged_frame_batched(env_state)                  # (B, PRIV_SIZE)
    return jnp.tile(frame[:, None, :], (1, PRIV_STACK, 1))       # (B, S, PRIV_SIZE)


def push_privileged_stack(stack: jnp.ndarray, env_state, done: jnp.ndarray) -> jnp.ndarray:
    """Roll a new frame into the stack; re-tile it on the envs that auto-reset.

    `env_state` is the POST-step state, so on a done env it is already the reset
    state and its history must be re-seeded rather than shifted in.
    """
    frame  = privileged_frame_batched(env_state)                 # (B, PRIV_SIZE)
    rolled = jnp.concatenate([stack[:, 1:], frame[:, None, :]], axis=1)
    tiled  = jnp.tile(frame[:, None, :], (1, PRIV_STACK, 1))
    return jnp.where(done[:, None, None], tiled, rolled)         # (B, S, PRIV_SIZE)
