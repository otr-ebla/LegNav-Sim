"""
jhsfm_planner.py — Exact JHSFM planner (robot driven as a pedestrian)
=====================================================================

Drives the robot with the **exact same** Headed Social Force Model that moves
the pedestrians inside ``step_env`` (``jhsfm.hsfm.step``) — not a re-implementation.
The robot is inserted as one more headed agent in the crowd: a human circle with
the **robot's radius**, sharing the humans' force/torque parameters, reacting to
the other people and the static obstacles (walls, boxes, circles) in identical
fashion. Its resulting human-like motion over one env timestep is then mapped to
the robot's unicycle command ``[v, w]``.

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

from legnav.core.jax_env import ROBOT_RADIUS, DT
from legnav.core.jax_env_multi import build_hsfm_obstacles, N_SUBSTEPS, HSFM_DT, NUM_PEOPLE

try:
    from legnav.jhsfm.JHSFM.jhsfm.hsfm import step as hsfm_step
    from legnav.jhsfm.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    from legnav.jhsfm.JHSFM.jhsfm.hsfm import step as hsfm_step
    from legnav.jhsfm.JHSFM.jhsfm.utils import get_standard_humans_parameters

# JHSFM per-agent parameter layout (see jhsfm/utils.py):
#   [radius, mass, v_max, tau, ...]
_P_RADIUS = 0

_W_MAX = 1.0


class HumanPilot:
    def __init__(self):
        self.name = "HumanPilot"

    @partial(jit, static_argnames=("self",))
    def act(
        self,
        state,                      # EnvState (single or properly vmapped)
        _rng: jax.Array = None,     # unused
    ) -> jnp.ndarray:
        # ── Build the (N+1)-agent HSFM state: humans + robot as the last agent ──
        # JHSFM agent state is [px, py, bvx, bvy, theta, omega] with body-frame
        # velocity. A unicycle robot moving along its heading has bvx=v, bvy=0.
        h_state   = state.people[:, :6]                                   # (N, 6)
        robot_row = jnp.array([[state.x, state.y, state.v, 0.0,
                                state.theta, state.w]])                   # (1, 6)
        ext_state = jnp.concatenate([h_state, robot_row], axis=0)         # (N+1, 6)

        # Goals: each human's active waypoint (idx 0/1 → g1/g2); robot's goal last.
        idx_h    = state.people[:, 10]
        g1x, g1y = state.people[:, 6], state.people[:, 7]
        g2x, g2y = state.people[:, 8], state.people[:, 9]
        h_goals  = jnp.stack([jnp.where(idx_h == 0, g1x, g2x),
                              jnp.where(idx_h == 0, g1y, g2y)], axis=-1)  # (N, 2)
        ext_goals = jnp.concatenate(
            [h_goals, jnp.array([[state.goal_x, state.goal_y]])], axis=0) # (N+1, 2)

        # Same parameters and obstacles the env feeds the pedestrians. The only
        # robot-specific override is the footprint: a human circle with the
        # robot's radius. Desired walking speed and all force/torque coefficients
        # stay identical to the humans (the robot's max_v is applied as a hard cap
        # on the unicycle command below, not as the social-force desired speed).
        params = get_standard_humans_parameters(NUM_PEOPLE + 1)
        params = params.at[-1, _P_RADIUS].set(ROBOT_RADIUS)
        obstacles = build_hsfm_obstacles(state.obs_boxes, state.obs_circles, state.room_h)

        # Evolve the crowd + robot together with the env's substep schedule, so the
        # robot covers the same ground a pedestrian would over one env timestep.
        def _sub(_, ext):
            return hsfm_step(ext, ext_goals, params, obstacles, HSFM_DT)
        final_ext = jax.lax.fori_loop(0, N_SUBSTEPS, _sub, ext_state)
        r = final_ext[-1]   # robot's evolved HSFM state

        # ── Map the human-like net pose change over DT to a unicycle [v, w] ──────
        dtheta = (r[4] - state.theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        w_cmd  = jnp.clip(dtheta / DT, -_W_MAX, _W_MAX)
        disp   = jnp.hypot(r[0] - state.x, r[1] - state.y)
        v_cmd  = jnp.clip(disp / DT, 0.0, state.max_v)
        return jnp.array([v_cmd, w_cmd])
