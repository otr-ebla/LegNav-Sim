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

import os
import jax
import jax.numpy as jnp

from legnav.core.jax_humans import MAX_SPEED

# LEGNAV_PRIV_V2=1 (social fine-tuning, scenario 18): richer ground truth for the critic so it can
# price the social reward / terminal penalty exactly. Resolved at import — the critic's input size
# depends on it, so V1 checkpoints (default) stay loadable. Extra, all in the robot frame:
#   per human (+5): current waypoint (x, y), standing flag, group-member flag, predicted
#                   space-time conflict weight with the robot (same function as the reward);
#   per frame (+12): robot-to-group-link gaps (full / inner), robot position in the room,
#                    hidden actuation of the real-robot env (tau, a_max, alpha_max, delay2, cmd_buf).
PRIV_V2     = os.environ.get("LEGNAV_PRIV_V2", "0") == "1"

PRIV_RANGE  = 5.0                          # m — radius around the robot
PRIV_HUMANS = 8 if PRIV_V2 else 6          # nearest humans emitted (scenario 18 has up to 7 people)
PRIV_FEAT   = 12 if PRIV_V2 else 7         # per human: 2+2+2 floats + valid flag (+5 in V2)
PRIV_GLOBAL = 12 if PRIV_V2 else 0         # per frame, not tied to a human
PRIV_STACK  = 3                            # frames, matches the LiDAR stack
PRIV_SIZE   = PRIV_HUMANS * PRIV_FEAT + PRIV_GLOBAL     # one frame (V1: 42)
PRIV_OBS_SIZE = PRIV_SIZE * PRIV_STACK     # what the critic actually sees (V1: 126)


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

    cols = [
        lx / PRIV_RANGE, ly / PRIV_RANGE,
        rx_ / PRIV_RANGE, ry_ / PRIV_RANGE,
        vx / MAX_SPEED,  vy / MAX_SPEED,
        jnp.ones_like(lx),
    ]
    if PRIV_V2:
        from legnav.core import jax_env_multi as em
        wpx, wpy = em.human_waypoints(people)
        wx, wy = to_ego(wpx, wpy)
        standing = (people[:, 10] >= 0.0) & (people[:, 6] == people[:, 8]) & (people[:, 7] == people[:, 9])
        r_rob = em._ROBOT_CIRCUM_R if em._RECT_ROBOT else em.ROBOT_RADIUS
        conf = em.predicted_conflict(rx, ry, rtheta, env_state.v, env_state.w, env_state.v, env_state.w,
                                     1.0, bx, by, people[:, 2], people[:, 3], wpx, wpy, valid, r_rob)
        cols += [jnp.clip(wx / PRIV_RANGE, -2.0, 2.0), jnp.clip(wy / PRIV_RANGE, -2.0, 2.0),
                 standing.astype(jnp.float32), em.group_members(people).astype(jnp.float32), conf]
    feats = jnp.stack(cols, axis=-1)                   # (N, PRIV_FEAT)
    feats = jnp.where(valid[:, None], feats, 0.0)

    # Nearest-first ordering; invalid humans are pushed to the back of the sort
    # so that the leading slots are always the real, in-range ones.
    rank_key = jnp.where(valid, dist, jnp.inf)
    idx = jnp.argsort(rank_key)[:PRIV_HUMANS]
    out = feats[idx].reshape(-1)
    if not PRIV_V2:
        return out                                     # (PRIV_SIZE,)

    gap_full, gap_inner = em.group_space_gap(people, rx, ry, rtheta)
    zero = jnp.float32(0.0)
    cmd_buf = getattr(env_state, "cmd_buf", None)       # real-robot env only; zeros in the stock env
    cmd_buf = jnp.zeros(4) if cmd_buf is None else jnp.asarray(cmd_buf).reshape(-1)
    glob = jnp.concatenate([
        jnp.stack([jnp.minimum(gap_full / 3.0, 1.0), jnp.minimum(gap_inner / 3.0, 1.0),
                   rx / env_state.room_w, ry / env_state.room_h,
                   jnp.asarray(getattr(env_state, "tau", zero), jnp.float32),
                   jnp.asarray(getattr(env_state, "a_max", zero), jnp.float32),
                   jnp.asarray(getattr(env_state, "alpha_max", zero), jnp.float32),
                   jnp.asarray(getattr(env_state, "delay2", zero), jnp.float32)]),
        cmd_buf,
    ])                                                  # (12,)
    return jnp.concatenate([out, glob])


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
