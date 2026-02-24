"""
jax_env_multi.py — Core 2D Navigation Environment with JHSFM
=============================================================
PATCH — Leg-pair simulation:
  reset_env: initialises leg_phases with random offsets.
  step_env:  advances leg_phases each step via jax_legs.advance_phase().
  get_obs (from jax_env) already handles USE_LEGS via get_leg_circles().

No other changes vs previous version.
"""

import jax
import jax.numpy as jnp
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_env import (EnvState, get_obs,
                     ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT,
                     MAX_STEPS, GOAL_RADIUS, COMFORT_DIST, COMFORT_COEF,
                     HEADING_BONUS, PROGRESS_COEF)
from jax_scenarios import generate_scenario
from jax_legs import advance_feet, init_foot_state

try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    from jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

__all__ = ["reset_env", "step_env", "EnvState", "get_obs"]

HSFM_DT    = 0.05
N_SUBSTEPS = int(DT / HSFM_DT)
NUM_PEOPLE = 12


def build_hsfm_obstacles(obs_boxes):
    room_edges = jnp.array([
        [[0.0, 0.0], [ROOM_W, 0.0]],
        [[ROOM_W, 0.0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0.0, ROOM_H]],
        [[0.0, ROOM_H], [0.0, 0.0]]
    ])

    def box_to_edges(box):
        cx, cy, hw, hh = box
        valid = jnp.where(hw > 0.0, 1.0, 0.0)
        p1 = jnp.array([cx - hw, cy - hh]) * valid
        p2 = jnp.array([cx + hw, cy - hh]) * valid
        p3 = jnp.array([cx + hw, cy + hh]) * valid
        p4 = jnp.array([cx - hw, cy + hh]) * valid
        return jnp.stack([
            jnp.stack([p1, p2]), jnp.stack([p2, p3]),
            jnp.stack([p3, p4]), jnp.stack([p4, p1])
        ])

    box_edges = jax.vmap(box_to_edges)(obs_boxes)
    all_edges = jnp.concatenate([room_edges[None, ...], box_edges], axis=0)
    return jnp.tile(all_edges[None, ...], (NUM_PEOPLE + 1, 1, 1, 1, 1))


def reset_env(key: jax.Array, min_goal_dist: float = 3.0, scenario_idx: int = -1):
    k_main, k_legs = jax.random.split(key)

    rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people = \
        generate_scenario(k_main, min_goal_dist, scenario_idx)

    # ── NEW: staggered initial gait phases ────────────────────────────────────
    foot_state = init_foot_state(people, k_legs)

    state = EnvState(
        x=rx, y=ry, theta=rtheta, v=0.0, w=0.0,
        goal_x=gx, goal_y=gy, max_v=max_v,
        people=people, obs_circles=obs_circles, obs_boxes=obs_boxes,
        time_step=0,
        foot_state=foot_state,
    )
    return get_obs(state), state


def step_env(key, state, action):
    # 1. Robot Kinematics
    target_v = jnp.clip(action[0], 0.0, state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    mid_theta = state.theta + 0.5 * target_w * DT
    new_theta = (state.theta + target_w * DT + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    raw_x     = state.x + target_v * DT * jnp.cos(mid_theta)
    raw_y     = state.y + target_v * DT * jnp.sin(mid_theta)

    wall_collision = (raw_x < ROBOT_RADIUS) | (raw_x > ROOM_W - ROBOT_RADIUS) | \
                     (raw_y < ROBOT_RADIUS) | (raw_y > ROOM_H - ROBOT_RADIUS)
    new_x = jnp.clip(raw_x, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y = jnp.clip(raw_y, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    # 2. JHSFM substeps (unchanged)
    hsfm_params      = get_standard_humans_parameters(NUM_PEOPLE + 1)
    static_obstacles = build_hsfm_obstacles(state.obs_boxes)

    def _hsfm_substep(carry, _):
        h_state, r_state = carry
        idx  = state.people[:, 10]
        g1x, g1y = state.people[:, 6], state.people[:, 7]
        g2x, g2y = state.people[:, 8], state.people[:, 9]
        gx   = jnp.where(idx == 0, g1x, g2x)
        gy   = jnp.where(idx == 0, g1y, g2y)
        h_goals    = jnp.stack([gx, gy], axis=-1)
        r_goal     = jnp.array([state.goal_x, state.goal_y])
        ext_state  = jnp.concatenate([h_state, r_state[None, :]], axis=0)
        ext_goals  = jnp.concatenate([h_goals, r_goal[None, :]], axis=0)
        next_ext   = hsfm_step(ext_state, ext_goals, hsfm_params, static_obstacles, HSFM_DT)
        next_h     = next_ext[:-1]
        clamped_x  = jnp.clip(next_h[:, 0], 0.1, ROOM_W - 0.1)
        clamped_y  = jnp.clip(next_h[:, 1], 0.1, ROOM_H - 0.1)
        next_h     = next_h.at[:, 0].set(clamped_x).at[:, 1].set(clamped_y)
        return (next_h, r_state), None

    h_state_init = state.people[:, :6]
    r_state_init = jnp.array([new_x, new_y,
                               target_v * jnp.cos(new_theta),
                               target_v * jnp.sin(new_theta),
                               new_theta, target_w])
    (new_h_state, _), _ = jax.lax.scan(
        _hsfm_substep, (h_state_init, r_state_init), None, length=N_SUBSTEPS
    )

    # 3. Waypoint toggle
    idx_cur = state.people[:, 10]
    g1x, g1y = state.people[:, 6], state.people[:, 7]
    g2x, g2y = state.people[:, 8], state.people[:, 9]
    gx_cur   = jnp.where(idx_cur == 0, g1x, g2x)
    gy_cur   = jnp.where(idx_cur == 0, g1y, g2y)
    dist_to_goal = jnp.sqrt((new_h_state[:, 0] - gx_cur)**2 +
                             (new_h_state[:, 1] - gy_cur)**2)
    new_idx  = jnp.where(dist_to_goal < 0.5, 1.0 - idx_cur, idx_cur)
    new_people = jnp.concatenate([new_h_state, state.people[:, 6:10], new_idx[:, None]], axis=-1)

    # ── NEW: advance leg gait phases ──────────────────────────────────────────
    new_foot_state = advance_feet(state.foot_state, new_people, DT)

    # 4. Distance and Collision Logic
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x  - state.goal_x)**2 + (new_y  - state.goal_y)**2)

    dx_p = new_people[:, 0] - new_x
    dy_p = new_people[:, 1] - new_y
    dists_p = jnp.sqrt(dx_p**2 + dy_p**2)
    closest_human = jnp.min(dists_p)

    human_col_mask  = dists_p < (ROBOT_RADIUS + PEOPLE_RADIUS)
    human_collision = jnp.any(human_col_mask)

    heading_dot = dx_p * jnp.cos(new_theta) + dy_p * jnp.sin(new_theta)
    in_fov_mask = heading_dot > 0.0
    any_active  = jnp.any(human_col_mask & in_fov_mask)
    active_col  = human_collision & (target_v > 0.1) & any_active
    passive_col = human_collision & ~active_col

    dx_c = state.obs_circles[:, 0] - new_x
    dy_c = state.obs_circles[:, 1] - new_y
    closest_cir = jnp.min(jnp.sqrt(dx_c**2 + dy_c**2) - state.obs_circles[:, 2])

    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(new_x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(new_y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)
    closest_box = jnp.min(jax.vmap(_box_dist)(state.obs_boxes))

    obs_collision = (closest_cir < ROBOT_RADIUS) | (closest_box < ROBOT_RADIUS)
    collision     = human_collision | obs_collision | wall_collision
    timeout       = (state.time_step + 1) >= MAX_STEPS
    goal_reached  = new_dist < GOAL_RADIUS
    done          = goal_reached | collision | timeout

    # 5. Reward
    progress  = PROGRESS_COEF * (prev_dist - new_dist)
    step_pen  = -0.006
    smooth    = -0.5 * jnp.abs(target_w - state.w)
    speed_bon = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)

    goal_angle_cur = jnp.arctan2(state.goal_y - new_y, state.goal_x - new_x)
    align_cur      = (goal_angle_cur - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    heading_bon    = HEADING_BONUS * jnp.maximum(0.0, jnp.cos(align_cur)) * \
                     (target_v / jnp.maximum(state.max_v, 1e-3))
    comfort_pen    = -COMFORT_COEF * jnp.sum(jnp.maximum(0.0, 1.0 - dists_p / COMFORT_DIST))

    reward = progress + step_pen + smooth + speed_bon + heading_bon + comfort_pen
    reward = jnp.where(goal_reached, 25.0, reward)
    reward = jnp.where(obs_collision  & ~goal_reached, -7.0, reward)
    reward = jnp.where(wall_collision & ~obs_collision & ~goal_reached, -7.0, reward)
    reward = jnp.where(active_col  & ~obs_collision & ~wall_collision & ~goal_reached, -7.0, reward)
    reward = jnp.where(passive_col & ~obs_collision & ~wall_collision & ~goal_reached, -2.0, reward)
    reward = jnp.where(timeout & ~goal_reached & ~collision, -0.5, reward)

    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1,
        foot_state=new_foot_state,
    )

    obs  = get_obs(new_state)
    info = {
        "discount":      jnp.where(done, 0.0, 1.0),
        "goal_reached":  goal_reached,
        "collision":     collision,
        "passive_col":   passive_col,
        "closest_human": closest_human,
    }
    return obs, new_state, reward, done, info