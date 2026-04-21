"""
mppi_planner.py — Model Predictive Path Integral (MPPI) for indoor-rl-nav
==========================================================================
"""

import sys
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

# ---------------------------------------------------------------------------
# Resolve paths — same logic as dwa_planner.py
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparison_policies.dwa_planner import DWA, _MAX_GOAL_DIST   # reuse motion() and decode_obs()

# Large-but-finite collision cost in the obstacle critic. Picked so that
# (i)  exp(-collision_cost / temperature) ≈ 0 → colliding samples vanish
# (ii) we can cheaply detect "all rollouts collided" by thresholding on
#      total_cost, without having to thread a collision flag through lax.scan.
_COLLISION_COST = 1.0e6


# ===========================================================================
# MPPI Policy
# ===========================================================================

class MPPI(DWA):
    """
    Model Predictive Path Integral (MPPI) planner, adapted for indoor-rl-nav.

    **Stateful policy** — call :meth:`init_u_mean` at the start of each
    episode and pass the returned ``u_mean`` to every :meth:`act` call for
    warm-starting.

    Parameters
    ----------
    num_samples : int
        Number of sampled control sequences per step (K).  Default 512.
    horizon : int
        Prediction horizon in steps (H).  Horizon time = H × dt.  Default 20.
    temperature : float
        MPPI temperature λ.  Lower → greedier (best-cost wins all weight).
        Default 0.1.
    noise_sigma : (2,) array
        Standard deviation of control noise in normalised space:
        ``[σ_v_norm, σ_w]``.  Default [0.25, 0.5].
    velocity_cost_weight : float
        Weight for the speed critic (prefer higher speed).  Default 0.3.
    goal_distance_cost_weight : float
        Weight for the goal-distance step cost.  Default 1.5.
    obstacle_cost_weight : float
        Weight for the obstacle-clearance step cost.  Default 3.0.
    control_cost_weight : float
        Weight for the control-effort (smoothness) penalty.  Default 0.05.
    terminal_cost_weight : float
        Extra weight applied to goal-distance at the **final** rollout step.
        Default 5.0.
    obstacle_max_dist : float
        Only LiDAR points within this distance [m] are considered for the
        obstacle critic during rollout.  Default 5.0 m.
    **dwa_kwargs
        Any parameter of :class:`dwa_planner.DWA` (``n_v``, ``n_w``,
        ``robot_radius``, ``dt``, ``num_rays``, ``lidar_max_dist``,
        ``clearance_min_dist``, ``clearance_max_dist``, …).

    Notes
    -----
    * ``num_samples`` and ``horizon`` are **static** (determine array shapes).
      Changing them between calls triggers re-tracing.
    * The warm-start ``u_mean`` is normalised (v_norm ∈ [0,1], w ∈ [−1,1]).
      It is automatically shifted by one step at the end of :meth:`act`,
      so the carry across steps is ready to use immediately.
    """

    def __init__(
        self,
        # MPPI hyperparameters
        num_samples:              int   = 512,
        horizon:                  int   = 10,
        temperature:              float = 1.0,
        noise_sigma:              jnp.ndarray = jnp.array([0.25, 0.5]),
        # Cost weights (goal/terminal critics now normalised by _MAX_GOAL_DIST)
        velocity_cost_weight:     float = 0.3,
        goal_distance_cost_weight: float = 3.0,
        obstacle_cost_weight:     float = 3.0,
        control_cost_weight:      float = 0.05,
        terminal_cost_weight:     float = 10.0,
        # Warm-start bias (normalised v) to avoid the clip-at-0 asymmetry
        # that would otherwise dominate the first few steps after reset.
        warm_start_v_norm:        float = 0.3,
        # Obstacle filtering for rollout critic
        obstacle_max_dist:        float = 5.0,
        # Pass remaining kwargs to DWA
        **dwa_kwargs,
    ):
        # Initialise DWA parent (motion model + obs decoder)
        # DWA's action grid (n_v, n_w) is not used by MPPI — pass small defaults
        dwa_kwargs.setdefault("n_v", 3)
        dwa_kwargs.setdefault("n_w", 3)
        dwa_kwargs.setdefault("n_steps", 5)
        super().__init__(**dwa_kwargs)

        # MPPI-specific attributes
        self.num_samples               = num_samples
        self.horizon                   = horizon
        self.temperature               = temperature
        self.noise_sigma               = jnp.asarray(noise_sigma, dtype=jnp.float32)
        self.velocity_cost_weight      = velocity_cost_weight
        self.goal_distance_cost_weight = goal_distance_cost_weight
        self.obstacle_cost_weight      = obstacle_cost_weight
        self.control_cost_weight       = control_cost_weight
        self.terminal_cost_weight      = terminal_cost_weight
        self.warm_start_v_norm         = warm_start_v_norm
        self.obstacle_max_dist         = obstacle_max_dist

        self.name = "MPPI"

    # =======================================================================
    # Warm-start initialisation
    # =======================================================================

    def init_u_mean(self) -> jnp.ndarray:
        """
        Return the warm-start control sequence for a new episode.

        The v_norm channel is initialised at ``warm_start_v_norm`` (default 0.3)
        rather than zero: with u_mean=0 the sampling distribution is centred at
        the lower edge of the v action space, so ~50% of samples clip to v=0
        and the robot starts stalled. A small positive bias keeps the clip
        roughly symmetric in (0, 1).
        """
        u0 = jnp.zeros((self.horizon, 2), dtype=jnp.float32)
        return u0.at[:, 0].set(self.warm_start_v_norm)

    # =======================================================================
    # Action-space helpers
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _clamp_action_norm(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        """
        Clamp a **normalised** action to the valid rectangular action space:
        ``v_norm ∈ [0, 1]``, ``w ∈ [−1, +1]``.

        This replaces the triangle-constraint clamping used in the original
        socialjym version (which assumed a differential-drive with fixed wheel
        base).  Our env clips v and w independently.

        Parameters
        ----------
        action_norm : (2,) [v_norm, w]

        Returns
        -------
        (2,) clamped [v_norm, w]
        """
        return jnp.array([
            jnp.clip(action_norm[0], 0.0, 1.0),
            jnp.clip(action_norm[1], -1.0, 1.0),
        ])

    # =======================================================================
    # Step-cost critics (evaluated at each rollout pose)
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _velocity_critic(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        """Prefer higher linear speeds.  Cost = 1 − v_norm ∈ [0, 1]."""
        return 1.0 - action_norm[0]

    @partial(jit, static_argnames=("self",))
    def _goal_distance_critic(
        self,
        pose:     jnp.ndarray,   # (3,) [x, y, θ] in ego frame
        goal_ego: jnp.ndarray,   # (2,) [gdx, gdy] in ego frame [m]
    ) -> jnp.ndarray:
        """
        Distance from rollout pose to the goal, normalised by the room diagonal
        so the value lives in [0, ~1] regardless of map size. Keeps the per-step
        goal term on the same scale as the obstacle/velocity critics (≤ 1)
        instead of growing linearly with absolute distance in metres.
        """
        return jnp.linalg.norm(pose[:2] - goal_ego) / _MAX_GOAL_DIST

    @partial(jit, static_argnames=("self",))
    def _obstacle_critic(
        self,
        pose:        jnp.ndarray,   # (3,) [x, y, θ] in ego frame
        point_cloud: jnp.ndarray,   # (M, 2) obstacle points in ego frame [m]
    ) -> jnp.ndarray:
        """
        Clearance cost from the current rollout pose to the nearest obstacle.

        Only points within ``obstacle_max_dist`` of the **rollout pose**
        are considered, so the cost is not dominated by distant walls.

        Returns
        -------
        scalar cost:
          * ``1e6`` — forecasted collision (gap ≤ robot_radius)
          * ``1 / min_dist`` — otherwise (prefer larger clearance)
        """
        pos = pose[:2]
        # Distance from the current pose to each point in the cloud
        dists_to_cloud = jnp.linalg.norm(pos[None, :] - point_cloud, axis=1)  # (M,)

        # Mask points far from the current rollout position (irrelevant)
        dists_masked = jnp.where(
            dists_to_cloud <= self.obstacle_max_dist,
            dists_to_cloud,
            self.obstacle_max_dist * 10.0,
        )
        min_dist = jnp.min(dists_masked)

        return lax.cond(
            min_dist - self.robot_radius <= 0.0,
            lambda: jnp.array(_COLLISION_COST),   # collision → dominant cost
            lambda: 1.0 / (min_dist + 1e-6),
        )

    @partial(jit, static_argnames=("self",))
    def _control_critic(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        """Prefer smaller actions (smooth trajectories).  Cost = ||action_norm||."""
        return jnp.linalg.norm(action_norm)

    @partial(jit, static_argnames=("self",))
    def _step_cost(
        self,
        pose:        jnp.ndarray,   # (3,) current rollout pose in ego frame
        action_norm: jnp.ndarray,   # (2,) normalised action [v_norm, w]
        goal_ego:    jnp.ndarray,   # (2,) goal in ego frame [m]
        point_cloud: jnp.ndarray,   # (M, 2) obstacles in ego frame [m]
    ) -> jnp.ndarray:
        """Weighted sum of all per-step critics."""
        vel  = self.velocity_cost_weight      * self._velocity_critic(action_norm)
        goal = self.goal_distance_cost_weight * self._goal_distance_critic(pose, goal_ego)
        obs  = self.obstacle_cost_weight      * self._obstacle_critic(pose, point_cloud)
        ctrl = self.control_cost_weight       * self._control_critic(action_norm)
        return vel + goal + obs + ctrl

    # =======================================================================
    # Full trajectory rollout
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _rollout_and_cost(
        self,
        start_pose:        jnp.ndarray,   # (3,) ego-frame start = [0, 0, 0]
        controls_seq_norm: jnp.ndarray,   # (H, 2) normalised control sequence
        goal_ego:          jnp.ndarray,   # (2,) goal in ego frame [m]
        point_cloud:       jnp.ndarray,   # (M, 2) obstacle points [m]
        max_v:             jnp.ndarray,   # scalar max linear speed [m/s]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate one control sequence and return cumulative cost + trajectory.

        The rollout operates in the ego frame (robot starts at [0, 0, 0]).
        Per-step cost is evaluated at the POST-STEP pose (after applying the
        action). The start pose is always (0, 0, 0) — the ego origin — where
        the LiDAR scan was produced. Because salt-&-pepper noise and scan
        saturation create spurious "ghost" points at roughly robot_radius
        from the origin, evaluating the obstacle critic at the start pose
        would trigger a spurious collision for EVERY rollout and silently
        force the all-collision fallback every step. Skipping the pre-step
        evaluation (equivalently: aligning with DWA, which scores from
        ``traj[1:]``) avoids this failure mode.

        Parameters
        ----------
        start_pose : (3,)
        controls_seq_norm : (H, 2) normalised actions
        goal_ego : (2,) goal in ego frame [m]
        point_cloud : (M, 2) obstacle Cartesian points [m]
        max_v : scalar [m/s]

        Returns
        -------
        total_cost : scalar
        trajectory : (H+1, 3) — includes start_pose at row 0
        """
        def _step(carry, action_norm):
            pose, total_cost = carry
            # Scale to actual action for kinematic step
            actual_action = jnp.array([action_norm[0] * max_v, action_norm[1]])
            next_pose = self.motion(pose, actual_action)
            # Step cost evaluated at POST-step pose (see docstring)
            cost = self._step_cost(next_pose, action_norm, goal_ego, point_cloud)
            return (next_pose, total_cost + cost), next_pose   # save next_pose for trajectory

        (final_pose, total_cost), next_poses = lax.scan(
            _step, (start_pose, jnp.array(0.0)), controls_seq_norm
        )
        # next_poses: (H, 3) — trajectories after each action
        # Prepend start_pose to get full (H+1, 3) trajectory
        trajectory = jnp.concatenate([start_pose[None, :], next_poses], axis=0)

        # Terminal cost: heavily penalise distance to goal at end of horizon
        terminal_cost = (
            self.terminal_cost_weight
            * self._goal_distance_critic(final_pose, goal_ego)
        )
        return total_cost + terminal_cost, trajectory

    # =======================================================================
    # Main entry points
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def act(
        self,
        obs:    jnp.ndarray,   # (662,) stacked observation
        u_mean: jnp.ndarray,   # (H, 2) normalised warm-start control sequence
        rng:    jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the best action using MPPI and return an updated warm-start.

        This is the **lean version** that returns only the action and the
        updated ``u_mean``.  For diagnostics (trajectories, costs), use
        :meth:`act_with_info`.

        Parameters
        ----------
        obs : (662,)
            Stacked observation from ``make_stacked_env``.
        u_mean : (H, 2)
            Normalised warm-start; use :meth:`init_u_mean` at episode start.
        rng : jax.Array
            JAX PRNGKey (consumed; a fresh key is returned implicitly via
            the vmap design — no key returned here to keep the signature
            lean).

        Returns
        -------
        action : (2,) [v, w]
            Best action in ``[0, max_v] × [−1, 1]``.
        u_mean_new : (H, 2)
            Updated, shifted normalised control sequence for the next step.
        """
        action, u_mean_new, _, _ = self._mppi_step(obs, u_mean, rng)
        return action, u_mean_new

    @partial(jit, static_argnames=("self",))
    def act_with_info(
        self,
        obs:    jnp.ndarray,
        u_mean: jnp.ndarray,
        rng:    jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Like :meth:`act` but also returns sampled trajectories and costs for
        diagnostics and visualisation.

        Returns
        -------
        action : (2,)
        u_mean_new : (H, 2)
        trajectories : (K, H+1, 3)
            All K sampled trajectories in the ego frame.  Transform to world
            frame for visualisation by adding the robot's world-frame position.
        costs : (K,)
            Total cost of each sampled trajectory.
        """
        return self._mppi_step(obs, u_mean, rng)

    # =======================================================================
    # Core MPPI computation (shared by act and act_with_info)
    # =======================================================================

    @partial(jit, static_argnames=("self",))
    def _mppi_step(
        self,
        obs:    jnp.ndarray,
        u_mean: jnp.ndarray,
        rng:    jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full MPPI update: sample → rollout → reweight → update u_mean.

        Returns
        -------
        action       : (2,)
        u_mean_new   : (H, 2)
        trajectories : (K, H+1, 3)  ego frame
        costs        : (K,)
        """
        # ── 1. Decode observation ─────────────────────────────────────────
        goal_ego, max_v, point_cloud = self.decode_obs(obs)
        start_pose = jnp.zeros(3)   # ego frame: robot at origin

        # ── 2. Sample perturbations ε ~ N(0, Σ) ──────────────────────────
        # noise: (K, H, 2) where K = num_samples, H = horizon
        noise = (
            jax.random.normal(rng, (self.num_samples, self.horizon, 2))
            * self.noise_sigma[None, None, :]
        )

        # ── 3. Perturbed & clamped control sequences ─────────────────────
        # V: (K, H, 2)  — each row is a candidate normalised control sequence
        V          = u_mean[None, :, :] + noise          # (K, H, 2)
        V_clamped  = vmap(vmap(self._clamp_action_norm))(V)   # (K, H, 2)

        # ── 4. Parallel rollout: K trajectories → costs (K,) ─────────────
        costs, trajectories = vmap(
            self._rollout_and_cost,
            in_axes=(None, 0, None, None, None),
        )(start_pose, V_clamped, goal_ego, point_cloud, max_v)
        # costs: (K,),  trajectories: (K, H+1, 3)

        # ── 5. MPPI information-theoretic reweighting ─────────────────────
        beta    = jnp.min(costs)                                   # numerical stabiliser
        weights = jnp.exp(-(costs - beta) / self.temperature)      # (K,)
        weights = weights / (jnp.sum(weights) + 1e-8)              # normalise

        # ── 6. Weighted update of u_mean ─────────────────────────────────
        # Use the *effective* perturbations (V_clamped − u_mean) rather than
        # raw ε: when u_mean is near the action-space boundary and samples
        # get clipped, raw ε no longer reflects the control that was
        # actually simulated, which biases the update.
        deltas = V_clamped - u_mean[None, :, :]                    # (K, H, 2)
        perturbations = jnp.sum(
            weights[:, None, None] * deltas,
            axis=0,
        )                                                           # (H, 2)
        u_mean_updated = vmap(self._clamp_action_norm)(
            u_mean + perturbations
        )

        # ── 7. All-collision fallback ────────────────────────────────────
        # If every sampled rollout collided, the weighted blend above is
        # meaningless (all rollouts are equally bad) and MPPI would output
        # arbitrary noise. Detect by cost threshold (collisions add at least
        # obstacle_cost_weight · _COLLISION_COST per colliding step) and fall
        # back to a DWA-style "stop and rotate toward goal" action.
        all_collide = jnp.all(costs > 0.5 * _COLLISION_COST)
        goal_align  = jnp.arctan2(goal_ego[1], goal_ego[0])
        fallback_norm = jnp.array([0.0, jnp.clip(goal_align, -1.0, 1.0)])

        # ── 8. Extract first action (scale to actual speed) ───────────────
        first_norm = jnp.where(all_collide, fallback_norm, u_mean_updated[0])
        action = jnp.array([
            first_norm[0] * max_v,          # v ∈ [0, max_v]
            first_norm[1],                  # w ∈ [−1, 1]
        ])

        # ── 9. Shift warm-start: slide window left by 1 step ─────────────
        # Old u_mean[1:] become u_mean_new[:-1]; the last step is re-seeded
        # with the warm_start bias so the robot doesn't drift to v=0 over
        # the horizon when the policy is coasting.
        tail = jnp.array([[self.warm_start_v_norm, 0.0]], dtype=jnp.float32)
        u_mean_shifted = jnp.concatenate([u_mean_updated[1:], tail], axis=0)

        # On all-collision, reset the warm-start: keeping a window full of
        # colliding controls poisons the next few steps.
        u_mean_shifted = jnp.where(
            all_collide, self.init_u_mean(), u_mean_shifted
        )

        return action, u_mean_shifted, trajectories, costs

    # =======================================================================
    # Extra: human-readable repr
    # =======================================================================

    def __repr__(self) -> str:
        return (
            f"MPPI(K={self.num_samples}, H={self.horizon}, "
            f"λ={self.temperature}, σ={self.noise_sigma.tolist()}, "
            f"dt={self.dt}, "
            f"w_goal={self.goal_distance_cost_weight}, "
            f"w_obs={self.obstacle_cost_weight}, "
            f"w_vel={self.velocity_cost_weight}, "
            f"w_ctrl={self.control_cost_weight}, "
            f"w_term={self.terminal_cost_weight})"
        )
