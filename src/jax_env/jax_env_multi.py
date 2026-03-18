"""
jax_env_multi.py — Core 2D Navigation Environment with JHSFM
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
                     MAX_STEPS, GOAL_RADIUS,
                     HEADING_BONUS, PROGRESS_COEF, USE_LEGS)
from jax_scenarios import generate_scenario
from jax_legs import advance_feet, init_foot_state, get_shoe_boxes, get_leg_positions, LEG_RADIUS as _LEG_R

try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    from jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

__all__ = ["reset_env", "step_env", "EnvState", "get_obs"]

HSFM_DT    = 0.01

# ── Reward constants — aligned with reference Python env ─────────────────────

# Terminal rewards (matches reference step() function)
_R_GOAL          =  200.0   # reference: +200
_R_OBS_COL       =  -70.0   # reference: -70 (static collision)
_R_WALL_COL      =  -70.0   # reference: -70
_R_ACTIVE_COL    =  -70.0   # reference: -70 (active human collision)
_R_PASSIVE_COL   =  -15.0   # reference: -40 (passive human collision)
_R_TIMEOUT       =   -5.0   # reference: -5

# Progress shaping (reference: 2.5 * delta_dist)
_PROGRESS_COEF   =   2.5

# Step penalty (reference: -0.005)
_STEP_PEN        =  -0.05

# Jerk penalty — quadratic (reference: -1.5 * (delta_w)^2)
_JERK_WEIGHT     =   1.5

# Social reward zones (reference: DIST_INTIMATE/PERSONAL/SOCIAL)
_DIST_INTIMATE   =   0.45   # m
_DIST_PERSONAL   =   1.2    # m
_DIST_SOCIAL     =   2.0    # m

# A. Proxemic penalty: -2.0 * exp(-2.0 * safety_margin)  [only when moving]
_PROXEMIC_COEF   =   0.5
_PROXEMIC_DECAY  =   2.0

# B. Overspeed penalty: -5.0 * (v - safe_speed)^2  [when dist < PERSONAL]
_OVERSPEED_COEF  =   5.0

# C. Aim penalty: -0.5 * (1 - d/D_social)  [|rel_angle| < 0.5 and v > 0.2]
_AIM_COEF        =   0.5
_AIM_FOV         =   0.5    # rad ≈ 28.6°

# Heading bonus gate (keep — prevents heading bonus fighting obstacle avoidance)
HEADING_CLEARANCE_DIST = 3.0
_FWD_RAY_HALF_WIDTH    = 8

# Yield logic (keep — adds stop-and-go behaviour absent in reference)
_YIELD_PENALTY   =  -3.0
_YIELD_BONUS     =   10.0
_RESUME_BONUS    =   4.5
_RESUME_MIN_DIST =   0.8
N_SUBSTEPS = int(DT / HSFM_DT)
NUM_PEOPLE = 12

# Position used to hide the robot from HSFM when ghost_robot=True.
# Far outside the room so social forces from the robot on humans are zero.
_GHOST_POS = -999.0


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
    k_main, k_legs, k_obs = jax.random.split(key, 3)

    rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people = \
        generate_scenario(k_main, min_goal_dist, scenario_idx)
    
    max_v = 0.8

    foot_state = init_foot_state(people, k_legs)

    state = EnvState(
        x=rx, y=ry, theta=rtheta, v=0.0, w=0.0,
        goal_x=gx, goal_y=gy, max_v=max_v,
        people=people, obs_circles=obs_circles, obs_boxes=obs_boxes,
        time_step=0,
        foot_state=foot_state,
        time_stopped=0,
        sp_mask=jnp.zeros(108, dtype=jnp.bool_)
    )

    obs, sp_mask = get_obs(state, k_obs)
    state = state.replace(sp_mask=sp_mask)
    return obs, state


def step_env(key, state, action, ghost_robot: bool = True):
    """
    Advance the environment by one timestep.

    Parameters
    ----------
    key         : JAX PRNGKey
    state       : EnvState
    action      : (2,) array — [v_raw, w_raw]
    ghost_robot : bool (static, resolved at trace time)
        True  → robot is invisible to humans (training mode).
                 HSFM receives a dummy robot position (-999, -999) so human
                 trajectories are unaffected by the robot's presence.
        False → robot is visible to humans (evaluation mode).
                 HSFM receives the true robot position; humans avoid the robot,
                 making navigation easier — as intended for eval.
    """
    k_step, k_obs = jax.random.split(key)

    # ── 1. Robot Kinematics ───────────────────────────────────────────────────
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

    # ── 2. JHSFM substeps ─────────────────────────────────────────────────────
    hsfm_params      = get_standard_humans_parameters(NUM_PEOPLE + 1)
    static_obstacles = build_hsfm_obstacles(state.obs_boxes)

    # Ghost robot: hide position from HSFM so humans ignore the robot.
    # ghost_robot is a Python bool → resolved at trace time, zero overhead.
    if ghost_robot:
        hsfm_rx = jnp.array(_GHOST_POS)
        hsfm_ry = jnp.array(_GHOST_POS)
    else:
        hsfm_rx = new_x
        hsfm_ry = new_y

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

        # Do not clamp dummy humans — they live at -999 intentionally
        is_dummy_sub = idx < 0.0
        clamped_x  = jnp.where(is_dummy_sub, next_h[:, 0],
                                jnp.clip(next_h[:, 0], 0.1, ROOM_W - 0.1))
        clamped_y  = jnp.where(is_dummy_sub, next_h[:, 1],
                                jnp.clip(next_h[:, 1], 0.1, ROOM_H - 0.1))
        next_h     = next_h.at[:, 0].set(clamped_x).at[:, 1].set(clamped_y)
        return (next_h, r_state), None

    h_state_init = state.people[:, :6]
    # r_state uses ghost position for HSFM but true kinematics elsewhere
    r_state_init = jnp.array([hsfm_rx, hsfm_ry,
                               target_v * jnp.cos(new_theta),
                               target_v * jnp.sin(new_theta),
                               new_theta, target_w])
    (new_h_state, _), _ = jax.lax.scan(
        _hsfm_substep, (h_state_init, r_state_init), None, length=N_SUBSTEPS
    )

    # ── 3. Waypoint toggle & Respawn Logic ────────────────────────────────────
    idx_cur = state.people[:, 10]
    g1x, g1y = state.people[:, 6], state.people[:, 7]
    g2x, g2y = state.people[:, 8], state.people[:, 9]
    gx_cur   = jnp.where(idx_cur == 0, g1x, g2x)
    gy_cur   = jnp.where(idx_cur == 0, g1y, g2y)

    dist_to_goal = jnp.sqrt((new_h_state[:, 0] - gx_cur)**2 +
                             (new_h_state[:, 1] - gy_cur)**2)

    # Only toggle active humans (idx >= 0) to stop dummies from reviving
    new_idx = jnp.where((dist_to_goal < 0.5) & (idx_cur >= 0.0),
                        1.0 - idx_cur, idx_cur)

    # --- ADVANCED RESPAWN LOGIC ---
    k_respawn1, k_respawn2 = jax.random.split(k_step)
    is_dummy = state.people[:, 10] < 0.0

    # SYNCED: g1y is firmly back to 1.0. This reactivates the teleportation!
    is_parallel     = (g1x == g2x) & (jnp.abs(g1y - 1.0) < 0.1)
    is_bottleneck   = jnp.abs(g2y) < 0.1
    is_teleport_scenario = is_parallel | is_bottleneck

    reached_bottom = new_h_state[:, 1] < 1.5
    needs_respawn  = (reached_bottom & is_teleport_scenario) & ~is_dummy

    # Generate random X coordinates (narrowed for the middle lanes)
    rand_x_corr = jax.random.uniform(k_respawn1, (NUM_PEOPLE,), minval=4.8, maxval=7.2)
    rand_x_full = jax.random.uniform(k_respawn1, (NUM_PEOPLE,), minval=1.0, maxval=ROOM_W - 1.0)
    
    # Identify wall-walkers in the parallel scenario (spawned exactly at 4.5 or 7.5)
    is_wall_walker = (jnp.abs(g1x - 4.5) < 0.1) | (jnp.abs(g1x - 7.5) < 0.1)
    
    # Wall-walkers keep their exact lane (g1x), others get randomized inside the corridor
    rand_x_prl = jnp.where(is_wall_walker, g1x, rand_x_corr)
    rand_x = jnp.where(is_parallel, rand_x_prl, rand_x_full)
    
    rand_y = jnp.full((NUM_PEOPLE,), ROOM_H - 0.2)

    new_idx = jnp.where(needs_respawn, 0.0, new_idx)

    dummy_x = jnp.full((NUM_PEOPLE,), _GHOST_POS)
    dummy_y = jnp.full((NUM_PEOPLE,), _GHOST_POS)

    final_px = jnp.where(is_dummy, dummy_x,
               jnp.where(needs_respawn, rand_x, new_h_state[:, 0]))
    final_py = jnp.where(is_dummy, dummy_y,
               jnp.where(needs_respawn, rand_y, new_h_state[:, 1]))
    final_vx = jnp.where(needs_respawn, 0.0, new_h_state[:, 2])
    final_vy = jnp.where(needs_respawn, 0.0, new_h_state[:, 3])

    respawned_h_state = jnp.stack([
        final_px, final_py,
        final_vx, final_vy,
        new_h_state[:, 4], new_h_state[:, 5]   # theta, omega unchanged
    ], axis=-1)

    new_people = jnp.concatenate(
        [respawned_h_state, state.people[:, 6:10], new_idx[:, None]], axis=-1
    )

    # 1. Advance the continuous gait phase for everyone
    advanced_foot_state = advance_feet(state.foot_state, new_people, DT)
    
    # 2. Generate a clean set of feet perfectly centered under the new body coordinates
    fresh_foot_state = init_foot_state(new_people, k_respawn2) 
    
    # 3. Overwrite the foot state ONLY for humans that just teleported
    new_foot_state = jnp.where(needs_respawn[:, None], fresh_foot_state, advanced_foot_state)

    # ── 4. Distance helpers ───────────────────────────────────────────────────
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x  - state.goal_x)**2 + (new_y  - state.goal_y)**2)

    left_xy, right_xy = get_leg_positions(new_foot_state)   # (N,2) each

    # --- MODIFIED: Use stable JHSFM body coordinates for the center ---
    center_x = new_people[:, 0]
    center_y = new_people[:, 1]

    dx_p = center_x - new_x
    dy_p = center_y - new_y
    dists_p = jnp.sqrt(dx_p**2 + dy_p**2)

    # Mask dummies from all distance/collision logic
    active_mask    = new_people[:, 10] >= 0.0
    dists_p_active = jnp.where(active_mask, dists_p, jnp.inf)
    closest_human  = jnp.min(dists_p_active)

    # ── 5. Collision Detection ───────────────────────────────────────────────────

    heading_dot  = dx_p * jnp.cos(new_theta) + dy_p * jnp.sin(new_theta)
    in_fwd_fov   = heading_dot > 0.0       # human is ahead of the robot
    in_prox      = dists_p_active < 1.5    # within 1.5 m
    robot_moving = target_v >= 0.1         # robot is moving

    # ── 5a. Human body collisions (active vs passive) ──────────────────────────
    # USE_LEGS=False: circle-vs-circle model.
    #   Threshold = ROBOT_RADIUS + PEOPLE_RADIUS (full body cylinders).
    #
    # USE_LEGS=True: shoe-box model is the primary contact surface (see 5c).
    #   Body threshold is tightened to ROBOT_RADIUS + LEG_RADIUS to catch only
    #   direct leg/torso contacts that slip past the shoe AABB.
    if USE_LEGS:
        body_thresh = ROBOT_RADIUS + _LEG_R
    else:
        body_thresh = ROBOT_RADIUS + PEOPLE_RADIUS

    human_col_mask  = (dists_p < body_thresh) & active_mask
    human_collision = jnp.any(human_col_mask)

    # Active body collision: robot moved into a human that was in front & close
    active_body     = human_col_mask & in_fwd_fov & in_prox
    any_active_body = jnp.any(active_body)
    active_col_body  = human_collision & robot_moving & any_active_body
    passive_col_body = human_collision & ~active_col_body

    # ── 5b. Static obstacle collisions ────────────────────────────────────────────
    dx_c = state.obs_circles[:, 0] - new_x
    dy_c = state.obs_circles[:, 1] - new_y
    closest_cir = jnp.min(jnp.sqrt(dx_c**2 + dy_c**2) - state.obs_circles[:, 2])

    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(new_x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(new_y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)

    closest_box = jnp.min(jax.vmap(_box_dist)(state.obs_boxes))

    static_obs_collision = (closest_cir < ROBOT_RADIUS) | (closest_box < ROBOT_RADIUS)

    # ── 5c. Shoe-box collisions (USE_LEGS=True only, resolved at trace time) ───
    # When USE_LEGS=True the shoe AABB is the primary human contact surface.
    # Each shoe contact is classified active/passive with the same logic as body
    # contacts: active if robot was moving toward the shoe's owner and v >= 0.1.
    # When USE_LEGS=False these flags are all False — collision uses the body
    # circle threshold from 5a instead.
    if USE_LEGS:
        shoe_boxes = get_shoe_boxes(new_people, new_foot_state)   # (2*N, 4)

        # Per-shoe distances (2N shoes)
        shoe_dists_raw = jax.vmap(_box_dist)(shoe_boxes)           # (2*N,)

        # Map each shoe back to its owner human index (left shoes: 0..N-1,
        # right shoes: N..2N-1) so we can apply the active_mask per owner.
        N = NUM_PEOPLE
        owner_idx      = jnp.concatenate([jnp.arange(N), jnp.arange(N)])   # (2N,)
        owner_active   = active_mask[owner_idx]                             # (2N,)
        shoe_dists     = jnp.where(owner_active, shoe_dists_raw, jnp.inf)  # (2N,)

        shoe_collision_mask = shoe_dists < ROBOT_RADIUS                     # (2N,)
        any_shoe_col        = jnp.any(shoe_collision_mask)

        owner_heading_dot = heading_dot[owner_idx]   # (2N,)
        owner_in_fwd      = owner_heading_dot > 0.0
        owner_in_prox     = dists_p_active[owner_idx] < 1.5

        active_shoe      = shoe_collision_mask & owner_in_fwd & owner_in_prox
        any_active_shoe  = jnp.any(active_shoe)
        active_col_shoe  = any_shoe_col & robot_moving & any_active_shoe
        passive_col_shoe = any_shoe_col & ~active_col_shoe
    else:
        any_shoe_col     = jnp.array(False)
        active_col_shoe  = jnp.array(False)
        passive_col_shoe = jnp.array(False)

    # ── 5d. Aggregate collision flags ─────────────────────────────────────────────
    obs_collision = static_obs_collision

    # active_col / passive_col cover ONLY human contacts.
    # Wall and static-obstacle contacts are always in obs_collision /
    # wall_collision and NEVER bleed into passive_col.
    active_col  = active_col_body  | active_col_shoe
    passive_col = (passive_col_body | passive_col_shoe) & ~obs_collision & ~wall_collision

    collision    = human_collision | any_shoe_col | obs_collision | wall_collision
    timeout      = (state.time_step + 1) >= MAX_STEPS
    goal_reached = new_dist < GOAL_RADIUS
    done         = goal_reached | collision | timeout

    # ── 6. Reward — aligned with reference Python env ─────────────────────────

    # — 6a. Progress + step penalty + jerk (quadratic, matches reference) ------
    progress = _PROGRESS_COEF * (prev_dist - new_dist)    # 2.5 * delta_dist
    step_pen = _STEP_PEN                                   # -0.005
    jerk_pen = -_JERK_WEIGHT * (target_w - state.w) ** 2  # -1.5 * (delta_w)^2

    # — 6b. Heading bonus gated by forward clearance (JAX addition) ------------
    fwd_clearance  = jnp.clip(closest_human / HEADING_CLEARANCE_DIST, 0.0, 1.0)
    goal_angle_cur = jnp.arctan2(state.goal_y - new_y, state.goal_x - new_x)
    align_cur      = (goal_angle_cur - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    heading_bon    = (HEADING_BONUS
                      * jnp.maximum(0.0, jnp.cos(align_cur))
                      * (target_v / jnp.maximum(state.max_v, 1e-3))
                      * fwd_clearance)

    # — 6c. Social reward (matches reference 4A + 4B + 4C) --------------------
    # closest_rel_angle: angle to the center of the nearest human.
    nearest_idx       = jnp.argmin(dists_p_active)
    closest_rel_angle = jnp.arctan2(dy_p[nearest_idx], dx_p[nearest_idx]) - new_theta
    closest_rel_angle = (closest_rel_angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    in_social_zone = closest_human < _DIST_SOCIAL

    # A. Proxemic penalty (exponential, only when moving)
    safety_margin  = jnp.maximum(0.0, closest_human - PEOPLE_RADIUS - ROBOT_RADIUS)
    proxemic_pen   = _PROXEMIC_COEF * jnp.exp(-_PROXEMIC_DECAY * safety_margin)
    social_a       = jnp.where(in_social_zone & (target_v > 0.1), -proxemic_pen, 0.0)

    # B. Overspeed penalty: -5 * (v - safe_speed)^2 when dist < PERSONAL
    safe_speed = 0.3 + jnp.clip(closest_human - _DIST_INTIMATE, 0.0, None) * 0.8
    overspeed  = jnp.maximum(0.0, target_v - safe_speed)
    social_b   = jnp.where(closest_human < _DIST_PERSONAL,
                            -_OVERSPEED_COEF * overspeed ** 2, 0.0)

    # C. Aim penalty: pointing at human while moving
    social_c = jnp.where(
        in_social_zone & (jnp.abs(closest_rel_angle) < _AIM_FOV) & (target_v > 0.2),
        -_AIM_COEF * (1.0 - closest_human / _DIST_SOCIAL),
        0.0
    )

    social_reward = social_a + social_b + social_c

    # — 6d. Yield reward (JAX addition — stop-and-go near crossing humans) -----
    rel_angles = jnp.arctan2(
        new_people[:, 1] - new_y,
        new_people[:, 0] - new_x
    ) - new_theta
    rel_angles = (rel_angles + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    YIELD_DIST = 1.5
    YIELD_FOV  = 1.57

    in_yield_zone      = (dists_p_active < YIELD_DIST) & (jnp.abs(rel_angles) < YIELD_FOV)
    is_yield_situation = jnp.any(in_yield_zone)

    closest_yield_dist = jnp.min(jnp.where(in_yield_zone, dists_p_active, 100.0))
    urgency = jnp.where(
        is_yield_situation,
        (YIELD_DIST - closest_yield_dist) / YIELD_DIST,
        0.0
    )

    is_stopped = target_v <= 0.1

    new_time_stopped = jnp.where(
        is_yield_situation,
        jnp.where(is_stopped, state.time_stopped + 1, state.time_stopped),
        0,
    )

    was_in_yield     = state.time_stopped > 0
    yield_cleared    = was_in_yield & ~is_yield_situation
    was_close_enough = closest_yield_dist < _RESUME_MIN_DIST
    resume_bonus     = jnp.where(
        yield_cleared & ~is_stopped & was_close_enough,
        _RESUME_BONUS, 0.0
    )

    decay         = jnp.maximum(0.0, 1.0 - (new_time_stopped / 20.0))
    moving_frac   = target_v / jnp.maximum(state.max_v, 1e-3)
    yield_penalty = _YIELD_PENALTY * urgency * moving_frac
    yield_bonus   = _YIELD_BONUS   * urgency * decay

    yield_reward = jnp.where(
        is_yield_situation,
        jnp.where(is_stopped, yield_bonus, yield_penalty),
        0.0
    )

    # — 6e. Total step reward --------------------------------------------------
    reward = (progress + step_pen + jerk_pen + heading_bon
              + social_reward + yield_reward + resume_bonus)

    # — 6f. Terminal overrides (priority: goal > obs > wall > active > passive > timeout)
    reward = jnp.where(goal_reached,
                       _R_GOAL, reward)           # +200
    reward = jnp.where(obs_collision & ~goal_reached,
                       _R_OBS_COL, reward)        # -70
    reward = jnp.where(wall_collision & ~obs_collision & ~goal_reached,
                       _R_WALL_COL, reward)       # -70
    reward = jnp.where(active_col  & ~obs_collision & ~wall_collision & ~goal_reached,
                       _R_ACTIVE_COL, reward)     # -70
    reward = jnp.where(passive_col & ~obs_collision & ~wall_collision & ~goal_reached,
                       _R_PASSIVE_COL, reward)    # -40
    reward = jnp.where(timeout & ~goal_reached & ~collision,
                       _R_TIMEOUT, reward)        # -5

    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1,
        foot_state=new_foot_state,
        time_stopped=new_time_stopped,
    )

    obs, sp_mask = get_obs(new_state, k_obs)
    new_state = new_state.replace(sp_mask=sp_mask)

    info = {
        "discount":      jnp.where(done, 0.0, 1.0),
        "goal_reached":  goal_reached,
        "collision":     collision,
        "passive_col":   passive_col,
        "active_col":    active_col,
        "closest_human": closest_human,
        "sp_mask":       sp_mask,
        "timeout":       timeout,
    }
    return obs, new_state, reward, done, info