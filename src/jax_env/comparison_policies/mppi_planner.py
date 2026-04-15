"""
mppi_planner.py — Model Predictive Path Integral (MPPI) for indoor-rl-nav
==========================================================================

Zero-shot stochastic control policy. No training required.
Adapted from the original implementation by Alberto's colleague (socialjym/LaserNav).

Reference
---------
Williams et al., "Model Predictive Path Integral Control: From Theory to
Parallel Computation and Practice", ICRA 2017.

Key adaptations
---------------
* Inherits from :class:`~comparison_policies.dwa_planner.DWA` to reuse
  ``motion()``, ``decode_obs()``, and ``_wrap_angle()``.
* Decodes the **662-dim stacked observation** (same as DWA).
* Works entirely in the **robot's ego frame** (robot at origin at call time).
* Action space: **rectangular** v_norm ∈ [0, 1], w ∈ [−1, +1].
  The original triangle constraint (wheels_distance) is dropped — our env
  clips v and w independently.
* ``u_mean`` (warm-start control sequence) is stored **normalised**:
  ``u_mean[t, 0]`` = v_norm ∈ [0, 1], ``u_mean[t, 1]`` = w ∈ [−1, 1].
  This makes the carry valid even when max_v changes between episodes.
* ``max_v`` is decoded dynamically from the observation and used only at
  rollout time (``actual_v = v_norm × max_v``).
* Both ``act()`` and helper methods are ``@jax.jit``-compiled and
  ``jax.vmap``-compatible.

MPPI algorithm (one call to ``act()``)
--------------------------------------
::

    1. Decode obs  →  goal_ego, max_v, point_cloud
    2. Sample ε ~ N(0, Σ) of shape (K, H, 2)
    3. Perturbed controls: V = clamp(u_mean + ε)      (K, H, 2)
    4. Rollout K trajectories → cumulative costs (K,)
    5. Weights: w_k = exp(−(cost_k − β) / λ)  (normalised)
    6. Update: u_mean += Σ w_k · ε_k
    7. Clamp and shift u_mean (slide window by 1 step)
    8. action = u_mean[0] × [max_v, 1]

Observation layout (same as dwa_planner.py)
-------------------------------------------
::

    obs[0:9]    pose_stack   — 3 × (gdx_ego/D, gdy_ego/D, θ/π),  oldest→newest
    obs[9:14]   state_vec    — (v/vmax, w, (vmax-0.2)/1.8, d/D, align/π)
    obs[14:662] lidar_stack  — 3 × 216 inverse-normalised rays,  oldest→newest

Usage
-----
::

    from comparison_policies.mppi_planner import MPPI

    mppi   = MPPI()
    u_mean = mppi.init_u_mean()   # (H, 2) zeros — call once per episode

    # Inside episode loop:
    rng, subkey = jax.random.split(rng)
    action, u_mean = mppi.act(obs, u_mean, subkey)

    # With full diagnostics (trajectories, costs):
    action, u_mean, trajectories, costs = mppi.act_with_info(obs, u_mean, subkey)
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

from comparison_policies.dwa_planner import DWA    # reuse motion() and decode_obs()


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
        horizon:                  int   = 20,
        temperature:              float = 0.1,
        noise_sigma:              jnp.ndarray = jnp.array([0.25, 0.5]),
        # Cost weights
        velocity_cost_weight:     float = 0.3,
        goal_distance_cost_weight: float = 1.5,
        obstacle_cost_weight:     float = 3.0,
        control_cost_weight:      float = 0.05,
        terminal_cost_weight:     float = 5.0,
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
        self.obstacle_max_dist         = obstacle_max_dist

        self.name = "MPPI"

    # =======================================================================
    # Warm-start initialisation
    # =======================================================================

    def init_u_mean(self) -> jnp.ndarray:
        """
        Return a zeroed control sequence for warm-starting a new episode.

        Returns
        -------
        jnp.ndarray
            Shape (horizon, 2) — normalised [v_norm=0, w=0] for all steps.
        """
        return jnp.zeros((self.horizon, 2), dtype=jnp.float32)

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
        """Euclidean distance from current rollout pose to the goal [m]."""
        return jnp.linalg.norm(pose[:2] - goal_ego)

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
            lambda: jnp.array(100.0),         # collision → large-but-finite cost
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
        Per-step cost is evaluated at the PRE-STEP pose (before applying the
        action), consistent with the original socialjym implementation.

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
            # Step cost evaluated at CURRENT (pre-step) pose
            cost = self._step_cost(pose, action_norm, goal_ego, point_cloud)
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

        # ── 6. Weighted update of u_mean (use ε, not V, per MPPI theory) ──
        # perturbations: (H, 2)  weighted sum over samples
        perturbations = jnp.sum(
            weights[:, None, None] * noise,
            axis=0,
        )
        u_mean_updated = vmap(self._clamp_action_norm)(
            u_mean + perturbations
        )   # (H, 2)

        # ── 7. Extract first action (scale to actual speed) ───────────────
        first_norm = u_mean_updated[0]      # (2,) normalised
        action = jnp.array([
            first_norm[0] * max_v,          # v ∈ [0, max_v]
            first_norm[1],                  # w ∈ [−1, 1]
        ])

        # ── 8. Shift warm-start: slide window left by 1 step ─────────────
        # Old u_mean[1:] become u_mean_new[:-1]; the last step is zeroed.
        u_mean_shifted = jnp.concatenate([
            u_mean_updated[1:],
            jnp.zeros((1, 2), dtype=jnp.float32),
        ], axis=0)   # (H, 2)

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
