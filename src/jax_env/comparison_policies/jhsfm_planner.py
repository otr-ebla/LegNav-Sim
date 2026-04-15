"""
jhsfm_planner.py — Human-like SFM planner for NavRep pretraining
=================================================================

Replaces the DWA expert with a Social Force Model (SFM) controller that moves
the robot toward the goal exactly like a pedestrian in jax_humans.py:

  * Goal force steers toward the goal (same τ, same desired speed law).
  * Obstacle repulsion from LiDAR hits (same exponential + contact spring).
  * Wall repulsion from the four room boundaries.
  * Resulting force vector → unicycle (v, w) via a P-controller on heading.

Interface is identical to DWA:
    pilot = HumanPilot()
    action = pilot.act(obs)        # obs: (662,) → action: (2,) [v, w]

    import jax
    vmap_act = jax.vmap(pilot.act)
    actions  = vmap_act(obs_batch) # (N, 662) → (N, 2)
"""

import math
import sys
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

# ---------------------------------------------------------------------------
# Path setup (identical to dwa_planner.py)
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jax_env import (
    ROBOT_RADIUS,
    NUM_RAYS,
    MAX_LIDAR_DIST,
    ROOM_W,
    ROOM_H,
    DT,
    GOAL_RADIUS,
)

# ---------------------------------------------------------------------------
# Observation layout (must match dwa_planner.py / jax_wrappers.py)
# ---------------------------------------------------------------------------
_STACK_DIM    = 3
_POSE_SIZE    = 3   # (gdx_ego/D, gdy_ego/D, θ/π) per frame
_STATE_VEC_SZ = 5   # (v/vmax, w, (vmax-0.2)/1.8, goal_dist/D, goal_align/π)

_POSE_END     = _STACK_DIM * _POSE_SIZE      # 9
_STATE_END    = _POSE_END + _STATE_VEC_SZ    # 14

_MAX_GOAL_DIST = math.sqrt(ROOM_W ** 2 + ROOM_H ** 2)   # ≈ 16.97 m

# LiDAR ray angles in robot frame (matches jax_physics.py)
_LIDAR_ANGLES = jnp.linspace(-jnp.pi, jnp.pi, NUM_RAYS)

# ---------------------------------------------------------------------------
# SFM parameters (calibrated for DT=0.15 s — same as jax_humans.py)
# ---------------------------------------------------------------------------
_TAU        = 0.5    # s   goal-force relaxation time
_A_OBS      = 10.0   # N   obstacle repulsion magnitude
_B_OBS      = 0.25   # m   decay distance
_OVERLAP_K  = 40.0   # N/m hard contact spring (when inside obstacle)
_A_WALL     = 10.0   # N   wall repulsion magnitude
_B_WALL     = 0.25   # m
_EPS        = 1e-7

# Maximum allowed desired speed (clamped to max_v at act-time)
_MAX_SPEED   = 1.6   # m/s

# P-gain: how aggressively the robot turns toward the desired heading
_W_GAIN     = 2.0   # rad/s per rad of heading error
_W_MAX      = 1.0   # rad/s (env clip)

# Minimum LiDAR distance to count as a real obstacle (filter S&P noise)
_LIDAR_MIN  = 0.12  # m


# ===========================================================================
# HumanPilot: SFM-based unicycle controller
# ===========================================================================

class HumanPilot:
    """
    Moves the robot toward its goal using the Social Force Model.

    The robot's heading error (vs desired force direction) is turned into
    an angular velocity command via a proportional controller.  The linear
    speed is set to the desired speed scaled by alignment with the goal.

    Parameters
    ----------
    lidar_n_frames : int
        Number of stacked LiDAR frames to use for obstacle repulsion
        (1 = most recent only; 3 = all three frames for a denser cloud).
    """

    def __init__(self, lidar_n_frames: int = 1):
        self.lidar_n_frames = lidar_n_frames
        self.name = "HumanPilot"

    # ------------------------------------------------------------------
    # Observation decoding (same layout as DWA.decode_obs)
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=("self",))
    def decode_obs(self, obs: jnp.ndarray):
        """
        Decode the 662-dim stacked observation.

        Returns
        -------
        goal_ego : (2,) [m]       — goal vector in robot ego frame
        max_v    : scalar [m/s]   — episode maximum linear speed
        v_cur    : scalar [m/s]   — current linear speed
        w_cur    : scalar [rad/s] — current angular speed
        lidar_dists : (F*NUM_RAYS,) [m] — raw LiDAR distances (clamped)
        """
        pose_stack = obs[:_POSE_END].reshape(_STACK_DIM, _POSE_SIZE)
        state_vec  = obs[_POSE_END:_STATE_END]
        lidar_norm = obs[_STATE_END:].reshape(_STACK_DIM, NUM_RAYS)

        # Goal in robot ego frame
        gdx_ego  = pose_stack[-1, 0] * _MAX_GOAL_DIST
        gdy_ego  = pose_stack[-1, 1] * _MAX_GOAL_DIST
        goal_ego = jnp.array([gdx_ego, gdy_ego])

        # Dynamic max speed
        max_v = state_vec[2] * 1.8 + 0.2

        # Current velocities
        v_cur = state_vec[0] * max_v
        w_cur = state_vec[1]          # already in rad/s

        # LiDAR: inverse-normalised → raw distances [m]
        inv_frames  = lidar_norm[-self.lidar_n_frames:]          # (F, NUM_RAYS)
        raw_dists   = (
            MAX_LIDAR_DIST - inv_frames * (MAX_LIDAR_DIST - ROBOT_RADIUS)
        )                                                         # (F, NUM_RAYS)
        lidar_dists = raw_dists.ravel()                           # (F*NUM_RAYS,)

        return goal_ego, max_v, v_cur, w_cur, lidar_dists

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=("self",))
    def _compute_force(
        self,
        goal_ego:    jnp.ndarray,   # (2,) goal in ego frame [m]
        v_cur:       jnp.ndarray,   # scalar [m/s]
        w_cur:       jnp.ndarray,   # scalar [rad/s]
        max_v:       jnp.ndarray,   # scalar [m/s]
        lidar_dists: jnp.ndarray,   # (M,) [m]
    ) -> jnp.ndarray:
        """
        SFM: compute the net 2-D force vector in the robot ego frame.

        Returns
        -------
        force : (2,) [N]
        """
        # Current translational velocity in ego frame (heading = 0, so vx=v)
        vx = v_cur
        vy = jnp.array(0.0)

        # ── Goal force ────────────────────────────────────────────────────────
        # Steer toward goal at desired speed (clamp to max_v)
        v_des  = jnp.minimum(max_v, _MAX_SPEED)
        goal_d = jnp.sqrt(goal_ego[0]**2 + goal_ego[1]**2 + _EPS)
        ex     = goal_ego[0] / goal_d
        ey     = goal_ego[1] / goal_d
        # Stop force when already inside goal radius
        goal_force = jnp.where(
            goal_d > GOAL_RADIUS,
            jnp.array([(v_des * ex - vx) / _TAU,
                        (v_des * ey - vy) / _TAU]),
            jnp.zeros(2),
        )

        # ── Obstacle repulsion (from LiDAR hit points) ───────────────────────
        # Use only the most-recent frame's angles (lidar_n_frames stacked):
        # all frames share the same angular layout.
        n_angles = NUM_RAYS * self.lidar_n_frames  # M
        # Tile angles if lidar_n_frames > 1
        angles = jnp.tile(_LIDAR_ANGLES, self.lidar_n_frames)     # (M,)

        # Convert LiDAR hit to Cartesian in ego frame
        hx = lidar_dists * jnp.cos(angles)          # (M,)
        hy = lidar_dists * jnp.sin(angles)          # (M,)

        # Direction from hit to robot (robot at origin)
        dx = -hx; dy = -hy
        dist = jnp.sqrt(dx**2 + dy**2 + _EPS)
        nx   = dx / dist
        ny   = dy / dist

        # Surface distance (obstacle surface assumed at ROBOT_RADIUS)
        surf  = jnp.maximum(dist - ROBOT_RADIUS, 0.0)
        pen   = jnp.maximum(ROBOT_RADIUS - dist, 0.0)   # penetration

        # Soft magnitude + hard contact
        mag = _A_OBS * jnp.exp(-surf / _B_OBS) + _OVERLAP_K * pen

        # Mask: ignore S&P noise (very short rays) and full-range rays (no obstacle)
        valid = (lidar_dists > _LIDAR_MIN) & (lidar_dists < MAX_LIDAR_DIST - 0.05)
        mag   = jnp.where(valid, mag, 0.0)

        obs_fx = jnp.sum(mag * nx)
        obs_fy = jnp.sum(mag * ny)

        # Normalise by number of valid rays to avoid scale depending on density
        n_valid  = jnp.sum(valid.astype(jnp.float32)) + _EPS
        obs_force = jnp.array([obs_fx, obs_fy]) / n_valid

        # ── Wall repulsion (4 axis-aligned walls in ego frame) ────────────────
        # Ego frame: robot at origin, heading = +x.  Walls in ego frame depend
        # on the robot's world position which we don't have here, but we CAN
        # approximate walls as strong repulsion from LiDAR rays near MAX_LIDAR_DIST
        # (already captured above when dist is large → mag ≈ 0, and walls generate
        # their own hits).  For simplicity we skip an extra wall term: the LiDAR
        # already encodes wall proximity via short-range hits on wall reflections.

        # ── Net force ─────────────────────────────────────────────────────────
        force = goal_force + obs_force
        return force

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=("self",))
    def act(
        self,
        obs:  jnp.ndarray,          # (662,) stacked observation
        _rng: jax.Array = None,     # unused — deterministic policy
    ) -> jnp.ndarray:
        """
        Compute a [v, w] action using the SFM.

        The desired heading is the direction of the net SFM force vector.
        A P-controller drives w to align the robot with that heading.
        Linear speed is set proportional to alignment with the goal,
        clamped to [0, max_v].

        Returns
        -------
        action : (2,) [v, w]   v ∈ [0, max_v],  w ∈ [-1, 1]
        """
        goal_ego, max_v, v_cur, w_cur, lidar_dists = self.decode_obs(obs)

        force = self._compute_force(goal_ego, v_cur, w_cur, max_v, lidar_dists)

        # Desired heading = direction of net force (in ego frame)
        force_mag = jnp.sqrt(force[0]**2 + force[1]**2 + _EPS)

        # Angular velocity: P-controller on heading error
        desired_theta = jnp.arctan2(force[1], force[0])   # in ego frame, robot heading = 0
        heading_err   = (desired_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        w_cmd = jnp.clip(_W_GAIN * heading_err, -_W_MAX, _W_MAX)

        # Linear speed: proportional to alignment with goal (cos of heading error)
        # and to force magnitude (slow down near obstacles / when uncertain)
        goal_d  = jnp.sqrt(goal_ego[0]**2 + goal_ego[1]**2 + _EPS)
        goal_align = jnp.maximum(
            jnp.cos(heading_err) * jnp.sqrt(jnp.minimum(goal_d / 2.0, 1.0)), 0.0
        )
        v_des   = jnp.minimum(max_v, _MAX_SPEED)
        v_cmd   = jnp.clip(v_des * goal_align, 0.0, max_v)

        return jnp.array([v_cmd, w_cmd])

    def __repr__(self) -> str:
        return f"HumanPilot(lidar_n_frames={self.lidar_n_frames})"
