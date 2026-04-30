"""
jhsfm_planner.py — Exact JHSFM planner for NavRep pretraining
=================================================================

Replaces the DWA expert with a Social Force Model (SFM) controller that moves
the robot toward the goal exactly like a pedestrian in jax_humans.py, using
the exact same absolute global state of the environment and force functions.

Interface:
    pilot = HumanPilot()
    action = pilot.act(state)        # state: EnvState → action: (2,) [v, w]

    import jax
    vmap_act = jax.vmap(pilot.act)
    actions  = vmap_act(state_batch) # (N,) EnvState → (N, 2)
"""

import sys
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)

for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jax_env import ROBOT_RADIUS, PEOPLE_RADIUS, ROOM_W, ROOM_H, DT, GOAL_RADIUS
from jax_humans import (
    _goal_force,
    _human_force,
    _wall_force,
    _circle_force,
    _box_force,
    MAX_SPEED,
    _EPS,
)

_W_GAIN = 2.0
_W_MAX  = 1.0


class HumanPilot:
    def __init__(self):
        self.name = "HumanPilot"

    @partial(jit, static_argnames=("self",))
    def act(
        self,
        state,                      # EnvState (single or properly vmapped)
        _rng: jax.Array = None,     # unused
    ) -> jnp.ndarray:
        # Extract robot state
        px, py = state.x, state.y
        theta = state.theta
        vx = state.v * jnp.cos(theta)
        vy = state.v * jnp.sin(theta)
        
        wp_x, wp_y = state.goal_x, state.goal_y
        max_v = state.max_v
        v_des = jnp.minimum(max_v, MAX_SPEED)

        # ── 1) Goal force ─────────────────────────────────────────────────────
        gfx, gfy = _goal_force(px, py, vx, vy, wp_x, wp_y, v_des)
        
        # Stop force when already inside goal radius
        goal_d = jnp.sqrt((wp_x - px)**2 + (wp_y - py)**2 + _EPS)
        gfx = jnp.where(goal_d > GOAL_RADIUS, gfx, 0.0)
        gfy = jnp.where(goal_d > GOAL_RADIUS, gfy, 0.0)

        # ── 2) Human-human forces ─────────────────────────────────────────────
        # The robot computes forces exactly as if it were a human dodging other humans.
        all_humans = state.people
        N = all_humans.shape[0]
        
        # We approximate the contact interaction radius as the average or sum
        # In jax_humans, `_human_force(..., r)` uses `2*r` for contact distance.
        # Thus `r` should be `(ROBOT_RADIUS + PEOPLE_RADIUS) / 2`.
        effective_r = (ROBOT_RADIUS + PEOPLE_RADIUS) / 2.0
        
        def hf(i):
            opx, opy = all_humans[i,0], all_humans[i,1]
            return _human_force(px, py, vx, vy, opx, opy, effective_r)
        
        hfxs, hfys = jax.vmap(hf)(jnp.arange(N))
        hfx, hfy = jnp.sum(hfxs), jnp.sum(hfys)

        # ── 3) Obstacle + wall forces ──────────────────────────────────────────
        wfx, wfy = _wall_force(px, py, ROOM_W, ROOM_H)
        cfx, cfy = _circle_force(px, py, vx, vy, state.obs_circles)
        bfx, bfy = _box_force(px, py, state.obs_boxes)
        
        # Sum acceleration (no robot force since this IS the robot)
        ax = gfx + hfx + wfx + cfx + bfx
        ay = gfy + hfy + wfy + cfy + bfy

        # Convert to [v, w] control
        fx, fy = ax, ay
        
        desired_theta = jnp.arctan2(fy, fx)
        heading_err = (desired_theta - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        w_cmd = jnp.clip(_W_GAIN * heading_err, -_W_MAX, _W_MAX)
        
        goal_align = jnp.maximum(
            jnp.cos(heading_err) * jnp.sqrt(jnp.minimum(goal_d / 2.0, 1.0)), 0.0
        )
        v_cmd = jnp.clip(v_des * goal_align, 0.0, max_v)
        
        return jnp.array([v_cmd, w_cmd])
