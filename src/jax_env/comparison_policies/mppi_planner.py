import sys
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparison_policies.dwa_planner import DWA, _MAX_GOAL_DIST
from jax_env import GOAL_RADIUS

_COLLISION_COST = 1.0e6


class MPPI(DWA):
    """
    Model Predictive Path Integral (MPPI) planner for indoor-rl-nav.

    Mirrors the reference socialjym/LaserNav MPPI: critics are the raw
    goal distance, an unbounded clearance term, a velocity term that prefers
    the fastest speed allowed by the kinematic triangle, and a control term.
    The goal-distance critic is kept in metres (not normalised) so it
    dominates the cost — normalising it by the room diagonal makes the
    obstacle term overwhelm progress and the robot spins in place.

    Operates in the robot ego frame (robot at the origin) and in normalised
    action space ``[v_norm in [0,1], w in [-w_max, w_max]]``; the linear speed
    is scaled by the per-episode ``max_v`` decoded from the observation.
    """

    def __init__(
        self,
        num_samples:              int   = 1000,
        horizon:                  int   = 20,
        temperature:              float = 0.1,
        noise_sigma:              jnp.ndarray = jnp.array([0.4, 1.2]),
        velocity_cost_weight:     float = 0.5,
        goal_distance_cost_weight: float = 1.0,
        obstacle_cost_weight:     float = 3.0,
        control_cost_weight:      float = 0.1,
        terminal_cost_weight:     float = 5.0,
        obstacle_max_dist:        float = 2.5,
        **dwa_kwargs,
    ):
        dwa_kwargs.setdefault("n_v", 3)
        dwa_kwargs.setdefault("n_w", 3)
        dwa_kwargs.setdefault("n_steps", 5)
        super().__init__(**dwa_kwargs)

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

    def init_u_mean(self) -> jnp.ndarray:
        return jnp.zeros((self.horizon, 2), dtype=jnp.float32)

    @partial(jit, static_argnames=("self",))
    def _clamp_action_norm(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        # Kinematic triangle clamping in normalised space: v_norm >= 0,
        # w in [-w_max, w_max], and v_norm + |w| / w_max <= 1. Actions outside
        # the triangle are projected back onto it along the ray from the origin.
        v = jnp.maximum(action_norm[0], 0.0)
        w = jnp.clip(action_norm[1], -self.w_max, self.w_max)
        constraint_val = v + jnp.abs(w) / self.w_max
        final_scale = jnp.minimum(1.0, 1.0 / (constraint_val + 1e-5))
        return jnp.array([v * final_scale, w * final_scale])

    @partial(jit, static_argnames=("self",))
    def _velocity_critic(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        # Prefer the highest speed allowed by the current angular velocity
        # (the kinematic triangle). Pure rotation in place is not penalised.
        vmax_norm = 1.0 - jnp.abs(action_norm[1]) / self.w_max
        return lax.cond(
            vmax_norm > 0.0,
            lambda: (vmax_norm - action_norm[0]) / vmax_norm,
            lambda: 0.0,
        )

    @partial(jit, static_argnames=("self",))
    def _goal_distance_critic(self, pose: jnp.ndarray, goal_ego: jnp.ndarray) -> jnp.ndarray:
        # Raw Euclidean distance in metres — kept un-normalised so progress
        # toward the goal dominates the cost.
        return jnp.linalg.norm(pose[:2] - goal_ego)

    @partial(jit, static_argnames=("self",))
    def _obstacle_critic(self, pose: jnp.ndarray, point_cloud: jnp.ndarray) -> jnp.ndarray:
        distances = jnp.linalg.norm(pose[None, :2] - point_cloud, axis=1)
        min_dist  = jnp.min(distances)
        # 1/d clearance pressure, shifted so it vanishes at ``obstacle_max_dist``.
        # Beyond that range there is no incentive to flee, so the robot stops
        # running from people that are already comfortably far away.
        return lax.cond(
            min_dist - self.robot_radius <= 0.0,
            lambda: jnp.array(_COLLISION_COST),
            lambda: jnp.maximum(0.0, 1.0 / min_dist - 1.0 / self.obstacle_max_dist),
        )

    @partial(jit, static_argnames=("self",))
    def _control_critic(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(action_norm)

    @partial(jit, static_argnames=("self",))
    def _step_cost(
        self, pose: jnp.ndarray, action_norm: jnp.ndarray, goal_ego: jnp.ndarray, point_cloud: jnp.ndarray
    ) -> jnp.ndarray:
        vel  = self.velocity_cost_weight      * self._velocity_critic(action_norm)
        goal = self.goal_distance_cost_weight * self._goal_distance_critic(pose, goal_ego)
        obs  = self.obstacle_cost_weight      * self._obstacle_critic(pose, point_cloud)
        ctrl = self.control_cost_weight       * self._control_critic(action_norm)
        return vel + goal + obs + ctrl

    @partial(jit, static_argnames=("self",))
    def _rollout_and_cost(
        self, start_pose: jnp.ndarray, controls_seq_norm: jnp.ndarray, goal_ego: jnp.ndarray,
        point_cloud: jnp.ndarray, max_v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        def _step(carry, action_norm):
            pose, total_cost, reached = carry
            actual_action = jnp.array([action_norm[0] * max_v, action_norm[1]])
            next_pose = self.motion(pose, actual_action)
            cost = self._step_cost(next_pose, action_norm, goal_ego, point_cloud)
            # Absorbing goal: the real episode ends on arrival, so once a predicted
            # pose enters GOAL_RADIUS we freeze it and stop charging cost. Reaching
            # the goal quickly then beats hovering, and driving "past" a near goal
            # within the horizon is no longer penalised (kills the creep at the goal).
            cost = jnp.where(reached, 0.0, cost)
            out_pose = jnp.where(reached, pose, next_pose)
            new_reached = jnp.logical_or(
                reached, self._goal_distance_critic(next_pose, goal_ego) < GOAL_RADIUS
            )
            return (out_pose, total_cost + cost, new_reached), out_pose

        (final_pose, total_cost, reached), next_poses = lax.scan(
            _step, (start_pose, jnp.array(0.0), jnp.array(False)), controls_seq_norm
        )
        trajectory = jnp.concatenate([start_pose[None, :], next_poses], axis=0)

        terminal_cost = jnp.where(
            reached, 0.0,
            self.terminal_cost_weight * self._goal_distance_critic(final_pose, goal_ego),
        )
        return total_cost + terminal_cost, trajectory

    @partial(jit, static_argnames=("self",))
    def act(
        self, obs: jnp.ndarray, u_mean: jnp.ndarray, rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, u_mean_new, _, _ = self._mppi_step(obs, u_mean, rng)
        return action, u_mean_new

    @partial(jit, static_argnames=("self",))
    def act_with_info(
        self, obs: jnp.ndarray, u_mean: jnp.ndarray, rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._mppi_step(obs, u_mean, rng)

    @partial(jit, static_argnames=("self",))
    def _mppi_step(
        self, obs: jnp.ndarray, u_mean: jnp.ndarray, rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        goal_ego, max_v, point_cloud = self.decode_obs(obs)
        start_pose = jnp.zeros(3)

        noise = jax.random.normal(rng, (self.num_samples, self.horizon, 2)) * self.noise_sigma[None, None, :]
        V = u_mean[None, :, :] + noise
        V_clamped = vmap(vmap(self._clamp_action_norm))(V)

        costs, trajectories = vmap(
            self._rollout_and_cost, in_axes=(None, 0, None, None, None)
        )(start_pose, V_clamped, goal_ego, point_cloud, max_v)

        beta = jnp.min(costs)
        weights = jnp.exp(-(costs - beta) / self.temperature)
        weights = weights / (jnp.sum(weights) + 1e-5)

        perturbations = jnp.sum(weights[:, None, None] * noise, axis=0)
        u_mean_updated = vmap(self._clamp_action_norm)(u_mean + perturbations)

        first_norm = u_mean_updated[0]
        action = jnp.array([first_norm[0] * max_v, first_norm[1]])

        # Shift the plan forward one step; seed the freed tail with zeros.
        u_mean_shifted = jnp.roll(u_mean_updated, -1, axis=0).at[-1].set(jnp.zeros(2, jnp.float32))

        return action, u_mean_shifted, trajectories, costs

    def __repr__(self) -> str:
        return (f"MPPI(K={self.num_samples}, H={self.horizon}, λ={self.temperature})")
