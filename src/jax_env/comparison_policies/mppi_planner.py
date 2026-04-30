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

_COLLISION_COST = 1.0e6


class MPPI(DWA):
    """
    Model Predictive Path Integral (MPPI) planner for indoor-rl-nav.
    Operates strictly end-to-end from LiDAR observations. Assumes a static 
    world model for obstacles due to lack of an explicit tracking layer.
    Uses continuous 1/x obstacle gradients and kinematic triangle clamping.
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
        warm_start_v_norm:        float = 0.0,
        obstacle_max_dist:        float = 10.0,
        safe_distance:            float = 1.0,
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
        self.warm_start_v_norm         = warm_start_v_norm
        self.obstacle_max_dist         = obstacle_max_dist
        self.safe_distance             = safe_distance
        self.name = "MPPI"

    def init_u_mean(self) -> jnp.ndarray:
        u0 = jnp.zeros((self.horizon, 2), dtype=jnp.float32)
        return u0.at[:, 0].set(self.warm_start_v_norm)

    @partial(jit, static_argnames=("self",))
    def _clamp_action_norm(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        # Kinematic triangle clamping: v_norm >= 0, w_norm in [-1, 1], v_norm + |w_norm| <= 1.0
        v = jnp.maximum(action_norm[0], 0.0)
        w = jnp.clip(action_norm[1], -1.0, 1.0)
        constraint_val = v + jnp.abs(w)
        scale_factor = 1.0 / (constraint_val + 1e-5)
        final_scale = jnp.minimum(1.0, scale_factor)
        return jnp.array([v * final_scale, w * final_scale])

    @partial(jit, static_argnames=("self",))
    def _velocity_critic(self, action_norm: jnp.ndarray) -> jnp.ndarray:
        # Penalize low speed relative to the maximum allowed by the current angular velocity
        vmax_norm = 1.0 - jnp.abs(action_norm[1])
        return lax.cond(
            vmax_norm > 0.0,
            lambda: (vmax_norm - action_norm[0]) / vmax_norm,
            lambda: 0.0  # Complete turning in place is not penalized
        )

    @partial(jit, static_argnames=("self",))
    def _goal_distance_critic(self, pose: jnp.ndarray, goal_ego: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(pose[:2] - goal_ego) / _MAX_GOAL_DIST

    @partial(jit, static_argnames=("self",))
    def _obstacle_critic(self, pose: jnp.ndarray, point_cloud: jnp.ndarray) -> jnp.ndarray:
        pos = pose[:2]
        sq_dists = jnp.sum(jnp.square(pos[None, :] - point_cloud), axis=1)
        min_dist = jnp.sqrt(jnp.min(sq_dists))
        
        # Continuous 1/x gradient to apply soft pressure from obstacles
        return lax.cond(
            min_dist - self.robot_radius <= 0.0,
            lambda: jnp.array(_COLLISION_COST),
            lambda: 1.0 / (min_dist + 1e-5)
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
            pose, total_cost = carry
            actual_action = jnp.array([action_norm[0] * max_v, action_norm[1]])
            next_pose = self.motion(pose, actual_action)
            cost = self._step_cost(next_pose, action_norm, goal_ego, point_cloud)
            return (next_pose, total_cost + cost), next_pose

        (final_pose, total_cost), next_poses = lax.scan(
            _step, (start_pose, jnp.array(0.0)), controls_seq_norm
        )
        trajectory = jnp.concatenate([start_pose[None, :], next_poses], axis=0)

        terminal_cost = self.terminal_cost_weight * self._goal_distance_critic(final_pose, goal_ego)
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

        noise = (jax.random.normal(rng, (self.num_samples, self.horizon, 2)) * self.noise_sigma[None, None, :])
        V = u_mean[None, :, :] + noise
        V_clamped = vmap(vmap(self._clamp_action_norm))(V)

        costs, trajectories = vmap(
            self._rollout_and_cost, in_axes=(None, 0, None, None, None)
        )(start_pose, V_clamped, goal_ego, point_cloud, max_v)

        beta = jnp.min(costs)
        weights = jnp.exp(-(costs - beta) / self.temperature)
        weights = weights / (jnp.sum(weights) + 1e-8)

        # Pure update using raw noise, as in reference code
        perturbations = jnp.sum(weights[:, None, None] * noise, axis=0)
        u_mean_updated = vmap(self._clamp_action_norm)(u_mean + perturbations)

        # Fallback if mostly colliding
        collision_frac = jnp.mean(costs > 0.5 * _COLLISION_COST)
        all_collide = collision_frac > 0.95
        goal_align  = jnp.arctan2(goal_ego[1], goal_ego[0])
        # Maintain a slight forward crawl while turning to break free from local minima
        fallback_norm = jnp.array([0.2, jnp.clip(goal_align, -1.0, 1.0)])

        first_norm = jnp.where(all_collide, fallback_norm, u_mean_updated[0])
        action = jnp.array([first_norm[0] * max_v, first_norm[1]])

        tail = jnp.array([[self.warm_start_v_norm, 0.0]], dtype=jnp.float32)
        u_mean_shifted = jnp.concatenate([u_mean_updated[1:], tail], axis=0)
        u_mean_shifted = jnp.where(all_collide, self.init_u_mean(), u_mean_shifted)

        return action, u_mean_shifted, trajectories, costs

    def __repr__(self) -> str:
        return (f"MPPI(K={self.num_samples}, H={self.horizon}, λ={self.temperature})")