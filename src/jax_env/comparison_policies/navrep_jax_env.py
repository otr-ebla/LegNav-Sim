"""
navrep_jax_env.py — Gymnasium wrapper for the JAX indoor navigation environment.

Exposes a NavRep-compatible observation structure:
  observation_space = Dict{
      "lidar":       Box(0, 1, shape=(216,))   — inverted lidar scan (1=free, 0=obstacle)
      "robot_state": Box(-inf, inf, shape=(8,)) — [goal_dx, goal_dy, theta_norm,
                                                    v_norm, w, max_v_norm, dist_norm, align_norm]
  }
  action_space = Box([0., -1.], [1., 1.]) for [v_raw, w_raw]

Single-env wrapper (no vectorisation) suitable for SB3's DummyVecEnv.
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Path setup (mirrors ppo_mlp_baseline.py) ──────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jax_env_multi import reset_env, step_env  # noqa: E402

_LIDAR_DIM  = 216
_ROBOT_DIM  = 8   # pose(3) + state_vec(5)


def _split_obs(obs_j) -> dict:
    """Split flat 224-D JAX obs into NavRep-compatible numpy dict."""
    obs = np.asarray(obs_j)
    return {
        "robot_state": obs[:_ROBOT_DIM].astype(np.float32),
        "lidar":       obs[_ROBOT_DIM:_ROBOT_DIM + _LIDAR_DIM].astype(np.float32),
    }


class NavRepJaxEnv(gym.Env):
    """
    Single-step gymnasium env backed by the vectorisable JAX simulation.

    Parameters
    ----------
    scenario_idx : int
        -1 = random, 0-5 = fixed scenario (see jax_scenarios.py).
    max_goal_dist : float
        Maximum spawn distance to goal during reset.
    ghost_prob : float
        Probability that the robot is invisible to humans (1.0 = always ghost,
        matching the standard training protocol).
    seed : int
        Initial PRNG seed.
    """
    metadata = {"render_modes": []}

    def __init__(self, scenario_idx: int = -1, max_goal_dist: float = 3.0,
                 ghost_prob: float = 1.0, seed: int = 0):
        super().__init__()
        self.scenario_idx = scenario_idx
        self.max_goal_dist = max_goal_dist
        self.ghost_prob = ghost_prob
        self._key = jax.random.PRNGKey(seed)
        self._state = None

        self.observation_space = spaces.Dict({
            "lidar":       spaces.Box(0.0, 1.0, (_LIDAR_DIM,), np.float32),
            "robot_state": spaces.Box(-np.inf, np.inf, (_ROBOT_DIM,), np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], np.float32),
            high=np.array([1.0,  1.0], np.float32),
        )

        # JIT-compile with scenario_idx / ghost_prob closed over (they are static
        # in JAX's sense — they affect trace-time control flow in generate_scenario).
        _reset_partial = functools.partial(
            reset_env,
            max_goal_dist=max_goal_dist,
            scenario_idx=scenario_idx,
            ghost_prob=ghost_prob,
        )
        self._jit_reset = jax.jit(_reset_partial)
        self._jit_step  = jax.jit(step_env)

    # ── gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, k = jax.random.split(self._key)
        obs_j, self._state = self._jit_reset(k)
        return _split_obs(obs_j), {}

    def step(self, action):
        self._key, k = jax.random.split(self._key)
        obs_j, self._state, reward, done, _ = self._jit_step(
            k, self._state, jnp.array(action, jnp.float32)
        )
        return _split_obs(obs_j), float(reward), bool(done), False, {}
