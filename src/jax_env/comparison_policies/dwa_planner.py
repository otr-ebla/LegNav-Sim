"""
dwa_planner.py — Dynamic Window Approach (DWA) for indoor-rl-nav
=================================================================

Zero-shot navigation policy. No training required.
Adapted from the original implementation by Alberto's colleague (socialjym/LaserNav).

Key adaptations
---------------
* Standalone module — no BasePolicy inheritance, no socialjym dependencies.
* Decodes the **662-dim stacked observation** produced by ``make_stacked_env``
  (layout: pose_stack 9 + state_vec 5 + lidar_stack 648 = 662 dims).
* Works entirely in the robot's **ego frame** (robot at origin, facing +x).
  No absolute world coordinates needed.
* **Action space**: v ∈ [0, max_v], w ∈ [−1, +1] — rectangular grid,
  matching the clip used in jax_env_multi.py.
* ``max_v`` is decoded dynamically from the observation (varies per episode).
* **LiDAR angles**: ``compute_lidar`` in jax_physics.py generates NUM_RAYS=216
  rays at absolute angles ``theta + linspace(−π, π, 216)``, so in the robot
  frame they are at ``linspace(−π, π, 216)`` (ray 0 = directly behind,
  ray 108 ≈ directly ahead).
* The observation **frame stack is ordered oldest-to-newest**: index 0 is the
  oldest frame, index 2 is the most recent. DWA uses only the most recent
  frame (``lidar_norm[2]``).
* Both ``act()`` and ``cost()`` are ``@jax.jit``-compiled and ``jax.vmap``-
  compatible — they can be used inside vmapped evaluation loops directly.

Observation layout (from make_stacked_env, stack_dim=3)
--------------------------------------------------------
::

    obs[0:9]    = pose_stack.flatten()     # 3 frames × (gdx_ego/D, gdy_ego/D, θ/π)
    obs[9:14]   = state_vec               # (v/vmax, w, (vmax−0.2)/1.8, d/D, align/π)
    obs[14:662] = lidar_stack.flatten()   # 3 frames × 216 inverse-normalised rays

    Frame ordering within each block: index 0 = oldest, index 2 = most recent.
    D = sqrt(ROOM_W² + ROOM_H²) ≈ 16.97 m

Usage
-----
::

    from comparison_policies.dwa_planner import DWA

    dwa = DWA()
    action = dwa.act(obs)           # obs: (662,) → action: (2,) [v, w]

    # With cost diagnostics:
    action, costs = dwa.act_with_costs(obs)

    # Vectorised over N environments (JIT + vmap):
    import jax
    vmap_act = jax.jit(jax.vmap(dwa.act))
    actions = vmap_act(obs_batch)   # obs_batch: (N, 662) → actions: (N, 2)
"""

import math
import sys
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

# ---------------------------------------------------------------------------
# Resolve project paths so this module works when run directly or imported
# from any working directory.
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)          # src/jax_env/
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)       # src/
_ROOT_DIR    = os.path.dirname(_SRC_DIR)            # project root

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import physical constants from the parent environment
# ---------------------------------------------------------------------------
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
# Observation layout constants (must match jax_env.py + jax_wrappers.py)
# ---------------------------------------------------------------------------
_STACK_DIM    = 3               # number of stacked frames
_POSE_SIZE    = 3               # (gdx_ego/D, gdy_ego/D, θ/π) per frame
_STATE_VEC_SZ = 5               # (v/vmax, w, (vmax−0.2)/1.8, goal_dist/D, goal_align/π)

_POSE_END     = _STACK_DIM * _POSE_SIZE     # 9
_STATE_END    = _POSE_END + _STATE_VEC_SZ   # 14
# lidar block: obs[14:662] → reshape to (_STACK_DIM, NUM_RAYS)

_MAX_GOAL_DIST = math.sqrt(ROOM_W ** 2 + ROOM_H ** 2)   # ≈ 16.97 m


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle to [−π, π] without importing socialjym."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# ===========================================================================
# DWA Policy
# ===========================================================================

class DWA:
    """
    Dynamic Window Approach planner, adapted for indoor-rl-nav.

    Parameters
    ----------
    n_v : int
        Number of linear-speed samples in [0, max_v].  Default 9.
    n_w : int
        Number of angular-speed samples in [−w_max, +w_max].  Default 21.
    n_steps : int
        Number of kinematic prediction steps per candidate action.
        Prediction horizon = ``n_steps × dt``.  Default 5 (0.75 s at dt=0.15).
    heading_cost_coeff : float
        Weight for the goal-heading critic.
    clearance_cost_coeff : float
        Weight for the obstacle-clearance critic.
    velocity_cost_coeff : float
        Weight for the velocity (speed) critic.
    robot_radius : float
        Robot radius [m] used for collision checking.
    w_max : float
        Maximum angular speed [rad/s].  Matches our env's ±1 rad/s clip.
    dt : float
        Simulation time-step [s].  Must match the env's DT.
    num_rays : int
        Number of LiDAR rays (must match NUM_RAYS in jax_env.py).
    lidar_max_dist : float
        Maximum LiDAR range [m] (must match MAX_LIDAR_DIST in jax_env.py).
    lidar_n_frames : int
        How many stacked LiDAR frames to use for the point cloud.
        1 = only most recent; 3 = all frames (richer cloud, slower).
    clearance_min_dist : float
        Points closer than this distance [m] are ignored as likely spurious
        (Salt-&-Pepper noise artefacts at 0 m).  Default 0.05 m.
    """

    def __init__(
        self,
        n_v:                   int   = 9,
        n_w:                   int   = 21,
        n_steps:               int   = 5,
        heading_cost_coeff:    float = 0.5,
        clearance_cost_coeff:  float = 0.35,
        velocity_cost_coeff:   float = 0.15,
        robot_radius:          float = ROBOT_RADIUS,
        w_max:                 float = 1.0,
        dt:                    float = DT,
        num_rays:              int   = NUM_RAYS,
        lidar_max_dist:        float = MAX_LIDAR_DIST,
        lidar_n_frames:        int   = 1,
        clearance_min_dist:    float = 0.05,
        clearance_max_dist:    float = 4.0,    # ignore LiDAR points > this distance [m]
    ):
        # Store hyperparameters
        self.n_v                  = n_v
        self.n_w                  = n_w
        self.n_steps              = n_steps
        self.heading_cost_coeff   = heading_cost_coeff
        self.clearance_cost_coeff = clearance_cost_coeff
        self.velocity_cost_coeff  = velocity_cost_coeff
        self.robot_radius         = robot_radius
        self.w_max                = w_max
        self.dt                   = dt
        self.num_rays             = num_rays
        self.lidar_max_dist       = lidar_max_dist
        self.lidar_n_frames       = lidar_n_frames
        self.clearance_min_dist   = clearance_min_dist

        self.clearance_max_dist   = clearance_max_dist

        self.name = "DWA"

        # ------------------------------------------------------------------
        # LiDAR ray angles in the robot frame
        # jax_physics.py computes rays at: theta + linspace(−π, π, NUM_RAYS)
        # → in robot frame: linspace(−π, π, NUM_RAYS)
        # Ray 0 faces directly BEHIND, ray 108 faces directly AHEAD.
        # ------------------------------------------------------------------
        self.lidar_angles = jnp.linspace(-jnp.pi, jnp.pi, num_rays)

        # ------------------------------------------------------------------
        # Normalised action grid: v_norm ∈ [0,1], w ∈ [−w_max, +w_max]
        # At act-time we scale v_norm by the dynamic max_v from the obs.
        # Storing v_norm (not v) makes the grid static → JIT/vmap-friendly.
        # ------------------------------------------------------------------
        v_norms = jnp.linspace(0.0, 1.0, n_v)
        ws      = jnp.linspace(-w_max, w_max, n_w)
        V, W    = jnp.meshgrid(v_norms, ws, indexing="ij")
        self._action_norms = jnp.stack(             # (n_v * n_w, 2)
            [V.ravel(), W.ravel()], axis=-1
        )
        self._n_actions = n_v * n_w

    # =======================================================================
    # Kinematic model
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def motion(
        self,
        pose:   jnp.ndarray,   # (3,) [x, y, θ]
        action: jnp.ndarray,   # (2,) [v, w]
    ) -> jnp.ndarray:
        """
        Unicycle kinematic step.

        Uses the exact arc-length formula when |w| > ε, falling back to
        straight-line integration otherwise.  Identical to the colleague's
        original formulation.

        Returns
        -------
        jnp.ndarray
            Next pose (3,) [x, y, θ].
        """
        v, w = action[0], action[1]
        return lax.cond(
            jnp.abs(w) > 1e-5,
            lambda _: jnp.array([
                pose[0] + (v / w) * (jnp.sin(pose[2] + w * self.dt) - jnp.sin(pose[2])),
                pose[1] + (v / w) * (jnp.cos(pose[2]) - jnp.cos(pose[2] + w * self.dt)),
                _wrap_angle(pose[2] + w * self.dt),
            ]),
            lambda _: jnp.array([
                pose[0] + v * self.dt * jnp.cos(pose[2]),
                pose[1] + v * self.dt * jnp.sin(pose[2]),
                pose[2],
            ]),
            None,
        )

    # =======================================================================
    # Trajectory prediction
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _rollout(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Roll out ``n_steps`` unicycle steps from the origin [0, 0, 0]
        (ego frame: robot is always at the origin at the current time-step).

        Parameters
        ----------
        action : (2,) [v, w]

        Returns
        -------
        jnp.ndarray
            Trajectory (n_steps + 1, 3) including the start pose at row 0.
        """
        def _step(pose, _):
            next_pose = self.motion(pose, action)
            return next_pose, next_pose

        start = jnp.zeros(3)
        _, traj = lax.scan(_step, start, None, length=self.n_steps)
        # traj: (n_steps, 3) — prepend [0,0,0] for the full trajectory
        return jnp.concatenate([start[None, :], traj], axis=0)   # (n_steps+1, 3)

    # =======================================================================
    # Individual critic functions
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _velocity_critic(self, v_norm: jnp.ndarray) -> jnp.ndarray:
        """
        Prefer higher linear speeds.

        Parameters
        ----------
        v_norm : scalar in [0, 1] — normalised linear speed (v / max_v)

        Returns
        -------
        scalar cost in [0, 1]: 0 = maximum speed (best), 1 = stopped (worst).
        """
        return 1.0 - v_norm

    @partial(jit, static_argnames=("self",))
    def _goal_heading_critic(
        self,
        action:    jnp.ndarray,   # (2,) [v, w]  actual speeds
        goal_ego:  jnp.ndarray,   # (2,) [gdx, gdy] goal in robot frame [m]
    ) -> jnp.ndarray:
        """
        After rolling out ``action`` for ``n_steps``, compute how aligned
        the robot heading is with the direction to the goal.

        Everything is computed in the robot ego frame (robot starts at origin).

        Returns
        -------
        scalar cost in [0, 1]: 0 = perfectly aligned, 1 = facing away.
        """
        final_pose = self._rollout(action)[-1]          # [x_f, y_f, θ_f]
        # Direction from final robot pose to the goal
        goal_direction = jnp.arctan2(
            goal_ego[1] - final_pose[1],
            goal_ego[0] - final_pose[0],
        )
        heading_error = _wrap_angle(goal_direction - final_pose[2])
        return jnp.abs(heading_error) / jnp.pi          # [0, 1]

    @partial(jit, static_argnames=("self",))
    def _clearance_critic(
        self,
        action:      jnp.ndarray,   # (2,) [v, w]  actual speeds
        point_cloud: jnp.ndarray,   # (M, 2) obstacle points in robot frame [m]
    ) -> jnp.ndarray:
        """
        Minimum distance from the predicted trajectory to any obstacle point.

        Only considers points within ``clearance_max_dist`` of the robot origin
        to avoid spurious collisions from distant wall LiDAR hits.

        Returns
        -------
        scalar cost:
          * ``jnp.inf``  — collision (min_dist ≤ robot_radius)
          * ``1 / min_dist`` — otherwise (prefer larger clearance)
        """
        traj_xy = self._rollout(action)[1:, :2]          # (n_steps, 2) skip t=0

        # Distance of each LiDAR point from the robot origin (current position)
        point_norms = jnp.linalg.norm(point_cloud, axis=1)   # (M,)
        # Mask out points that are farther than clearance_max_dist — they are
        # walls or distant obstacles that the short prediction horizon cannot reach.
        _FAR_SENTINEL = self.clearance_max_dist * 10.0
        nearby_cloud = jnp.where(
            (point_norms <= self.clearance_max_dist)[:, None],
            point_cloud,
            _FAR_SENTINEL,
        )                                                     # (M, 2)

        def _min_dist_to_traj(point: jnp.ndarray) -> jnp.ndarray:
            dists = jnp.linalg.norm(traj_xy - point[None, :], axis=1)   # (n_steps,)
            return jnp.min(dists)

        distances = vmap(_min_dist_to_traj)(nearby_cloud)    # (M,)
        min_dist  = jnp.min(distances)

        return lax.cond(
            min_dist - self.robot_radius <= 0.0,
            lambda: jnp.array(jnp.inf),              # collision → ∞ cost
            lambda: 1.0 / (min_dist + 1e-6),         # prefer larger gap
        )

    # =======================================================================
    # Combined cost function
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def cost(
        self,
        v_norm:      jnp.ndarray,   # scalar ∈ [0,1]
        action:      jnp.ndarray,   # (2,) [v, w] — actual (v = v_norm * max_v)
        goal_ego:    jnp.ndarray,   # (2,) goal in robot frame [m]
        point_cloud: jnp.ndarray,   # (M, 2) obstacle points in robot frame [m]
    ) -> jnp.ndarray:
        """
        Weighted sum of all critic sub-costs for a single candidate action.

        Parameters
        ----------
        v_norm : scalar
            Normalised speed in [0, 1] (for the velocity critic).
        action : (2,)
            Actual [v, w] action used for kinematic rollout.
        goal_ego : (2,)
            Goal vector in the robot frame [m].
        point_cloud : (M, 2)
            Cartesian obstacle points in the robot frame [m].

        Returns
        -------
        scalar total cost (may be ``jnp.inf`` if a collision is forecasted).
        """
        vel_cost       = self._velocity_critic(v_norm)
        heading_cost   = self._goal_heading_critic(action, goal_ego)
        clearance_cost = self._clearance_critic(action, point_cloud)

        return (
            self.velocity_cost_coeff   * vel_cost
            + self.heading_cost_coeff  * heading_cost
            + self.clearance_cost_coeff * clearance_cost
        )

    # =======================================================================
    # Observation decoding
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def decode_obs(
        self,
        obs: jnp.ndarray,   # (662,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extract all DWA inputs from the 662-dim stacked observation.

        Observation layout (matches make_stacked_env with stack_dim=3)::

            obs[0:9]    pose_stack  — 3 × (gdx_ego/D, gdy_ego/D, θ/π)
            obs[9:14]   state_vec   — (v/vmax, w, (vmax-0.2)/1.8, d/D, align/π)
            obs[14:662] lidar_stack — 3 × 216 inverse-normalised readings

        Frame ordering: index 0 = oldest, index 2 = most recent.

        Returns
        -------
        goal_ego : (2,) [m]
            Goal vector in the robot ego frame: [Δx_ego, Δy_ego].
        max_v : scalar [m/s]
            Maximum linear speed for this episode.
        point_cloud : (lidar_n_frames * num_rays, 2) [m]
            LiDAR hits as Cartesian points in the robot frame.
            Points closer than ``clearance_min_dist`` are filtered out
            (they are likely Salt-&-Pepper noise artefacts at d ≈ 0).
        """
        # ── Split observation ─────────────────────────────────────────────
        pose_stack = obs[:_POSE_END].reshape(_STACK_DIM, _POSE_SIZE)
        state_vec  = obs[_POSE_END:_STATE_END]
        lidar_norm = obs[_STATE_END:].reshape(_STACK_DIM, self.num_rays)

        # ── Goal in ego frame (most recent frame = index 2) ───────────────
        # pose_stack[2, 0] = gdx_ego / _MAX_GOAL_DIST
        # pose_stack[2, 1] = gdy_ego / _MAX_GOAL_DIST
        gdx_ego  = pose_stack[-1, 0] * _MAX_GOAL_DIST
        gdy_ego  = pose_stack[-1, 1] * _MAX_GOAL_DIST
        goal_ego = jnp.array([gdx_ego, gdy_ego])

        # ── Max linear speed ──────────────────────────────────────────────
        # state_vec[2] = (max_v − 0.2) / 1.8
        max_v = state_vec[2] * 1.8 + 0.2

        # ── LiDAR → Cartesian point cloud in robot frame ──────────────────
        # lidar_norm[k] = inv_lidar = (MAX_LIDAR_DIST − raw_dist)/(MAX_LIDAR_DIST − ROBOT_RADIUS)
        # → raw_dist = MAX_LIDAR_DIST − inv_lidar × (MAX_LIDAR_DIST − ROBOT_RADIUS)
        # Use the ``lidar_n_frames`` most recent frames (indices −n:).
        inv_frames = lidar_norm[-self.lidar_n_frames:]         # (F, NUM_RAYS)
        raw_dists  = (
            self.lidar_max_dist
            - inv_frames * (self.lidar_max_dist - self.robot_radius)
        )                                                       # (F, NUM_RAYS) [m]

        xs = raw_dists * jnp.cos(self.lidar_angles[None, :])   # (F, NUM_RAYS)
        ys = raw_dists * jnp.sin(self.lidar_angles[None, :])
        pc_raw = jnp.stack([xs, ys], axis=-1).reshape(-1, 2)   # (F*NUM_RAYS, 2)

        # Filter out near-zero ghost rays (salt-&-pepper noise at d≈0)
        point_norms = jnp.linalg.norm(pc_raw, axis=1)          # (M,)
        valid       = point_norms > self.clearance_min_dist
        # Replace invalid points with a far-away sentinel (won't affect min-dist)
        _FAR = self.lidar_max_dist * 10.0
        point_cloud = jnp.where(valid[:, None], pc_raw, _FAR)  # (M, 2)

        return goal_ego, max_v, point_cloud

    # =======================================================================
    # Main entry points
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def act(
        self,
        obs:  jnp.ndarray,          # (662,) stacked observation
        _rng: jax.Array = None,     # unused — DWA is deterministic
    ) -> jnp.ndarray:
        """
        Select the best action for the current observation.

        The action grid is a rectangular lattice of
        ``(n_v × n_w)`` = ``(9 × 21) = 189`` candidate [v, w] pairs.
        All candidates are scored in parallel via ``jax.vmap`` and the
        lowest-cost non-``inf`` action is returned.

        Parameters
        ----------
        obs : (662,)
            Stacked observation from ``make_stacked_env``.
        _rng : jax.Array, optional
            Ignored (DWA is deterministic).

        Returns
        -------
        action : (2,) [v, w]
            Best action in ``[0, max_v] × [−1, 1]``.
        """
        goal_ego, max_v, point_cloud = self.decode_obs(obs)

        # Scale normalised v-grid by the dynamic max_v
        actual_vs = self._action_norms[:, 0] * max_v    # (n_actions,)
        actual_ws = self._action_norms[:, 1]             # (n_actions,)
        actions   = jnp.stack([actual_vs, actual_ws], axis=-1)  # (n_actions, 2)

        # Vectorised cost evaluation over all candidates
        costs = vmap(self.cost, in_axes=(0, 0, None, None))(
            self._action_norms[:, 0], actions, goal_ego, point_cloud
        )                                                # (n_actions,)

        best_idx = jnp.nanargmin(costs)
        best_action = actions[best_idx]                  # (2,)

        # ── Fallback: if ALL candidates forecast a collision (all costs = inf),
        # rotate in place toward the goal instead of committing to w=-1 (which is
        # just the lowest-index nan in a list of infs).
        all_inf   = jnp.all(jnp.isinf(costs))
        goal_align = jnp.arctan2(goal_ego[1], goal_ego[0])   # angle to goal in robot frame
        # Turn toward goal: sign of goal_align determines rotation direction
        fallback   = jnp.array([0.0, jnp.clip(goal_align, -self.w_max, self.w_max)])
        return jnp.where(all_inf, fallback, best_action)

    @partial(jit, static_argnames=("self",))
    def act_with_costs(
        self,
        obs:  jnp.ndarray,
        _rng: jax.Array = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Like :meth:`act` but also returns the full cost vector for diagnostics
        and visualisation.

        Returns
        -------
        action : (2,)
            Best [v, w] action.
        costs : (n_v * n_w,)
            Per-candidate cost values (``jnp.inf`` = forecasted collision).
        """
        goal_ego, max_v, point_cloud = self.decode_obs(obs)

        actual_vs = self._action_norms[:, 0] * max_v
        actual_ws = self._action_norms[:, 1]
        actions   = jnp.stack([actual_vs, actual_ws], axis=-1)

        costs = vmap(self.cost, in_axes=(0, 0, None, None))(
            self._action_norms[:, 0], actions, goal_ego, point_cloud
        )

        best_idx    = jnp.nanargmin(costs)
        best_action = actions[best_idx]

        all_inf    = jnp.all(jnp.isinf(costs))
        goal_align = jnp.arctan2(goal_ego[1], goal_ego[0])
        fallback   = jnp.array([0.0, jnp.clip(goal_align, -self.w_max, self.w_max)])
        return jnp.where(all_inf, fallback, best_action), costs

    # =======================================================================
    # Convenience: action-space metadata for visualisation
    # =======================================================================

    def get_action_grid(self, max_v: float) -> jnp.ndarray:
        """
        Return the full (n_v × n_w, 2) action grid scaled by ``max_v``.

        Useful for plotting the DWA action space distribution.
        """
        actual_vs = self._action_norms[:, 0] * max_v
        actual_ws = self._action_norms[:, 1]
        return jnp.stack([actual_vs, actual_ws], axis=-1)

    def __repr__(self) -> str:
        return (
            f"DWA(n_v={self.n_v}, n_w={self.n_w}, n_steps={self.n_steps}, "
            f"dt={self.dt}, w_max={self.w_max}, "
            f"h={self.heading_cost_coeff}, "
            f"c={self.clearance_cost_coeff}, "
            f"v={self.velocity_cost_coeff})"
        )
