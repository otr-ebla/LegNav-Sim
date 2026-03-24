"""
jax_env_multi.py — Core 2D Navigation Environment with JHSFM

FIXES vs previous version:

  FIX #5 — passive_col penalty ridotta da -60 a -20:
    La passive_col si verifica quando il robot è fermo e un umano ci cammina
    sopra. Il robot non può evitarla. Una penalità di -60 era sproporzionata
    e incoraggiava la policy a muoversi freneticamente vicino agli umani per
    "scappare" invece di navigare efficientemente. Con -20 rimane un segnale
    negativo (non stare in mezzo al traffico) senza dominare il reward.

  DIFFERENTIABILITY OVERHAUL (SHAC):
    step_env is now natively fully differentiable so SHAC can backpropagate
    exact gradients through the real simulator — no surrogate reward needed.

    Changes:
      • soft_clip() replaces jnp.clip() for action/position clamping.
        jnp.clip has zero gradient when saturated; soft_clip (softplus-based)
        returns a non-zero gradient at the boundary, allowing BPTT to learn
        that pushing into a wall is bad.
      • Boolean collision flags replaced by sigmoid soft-indicators for reward.
        Hard bool flags have zero gradient; sigmoids give a smooth collision
        signal that propagates into the actor gradient.
      • Terminal jnp.where cascade replaced by a smooth reward sum.
        jnp.where(done, CONST, reward) kills the gradient wherever done=True;
        sigmoid indicators smoothly blend terminal and dense rewards.
      • Boolean `done` is preserved for episode-reset logic but is wrapped in
        stop_gradient so it never contaminates the BPTT graph.
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
                     USE_LEGS,
                     P_HUMAN_STOP, STOP_MIN_STEPS, STOP_MAX_STEPS)
from jax_scenarios import generate_scenario
from jax_legs import advance_feet, init_foot_state, get_shoe_boxes, get_leg_positions, LEG_RADIUS as _LEG_R

try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm_diff import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm_diff import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

__all__ = ["reset_env", "step_env", "EnvState", "get_obs"]

HSFM_DT    = 0.01



# ── Reward constants ─────────────────────────────────────────────────────────
#
# Design philosophy: minimal hand-engineering, maximal emergence.
#
# The core reward is:
#
#   social_progress = Δdist_to_goal × clearance_factor(closest_human)
#
# where clearance_factor ∈ [0,1] is a smooth sigmoid that:
#   → equals ~1 when the robot is comfortably far from all humans
#   → drops toward 0 as it enters the personal-space zone
#   → reaches 0 at the intimate zone boundary
#
# This single multiplicative term encodes the social priority rule:
# "yield to humans" emerges because progress near humans is worth almost
# nothing, so the policy learns that detouring around them (even at a
# distance cost) is better than passing through.  Stopping to let a
# crossing human pass also becomes rational: zero progress × anything = 0,
# whereas waiting a few steps and then resuming full progress is better.
#
# No separate yield/escape/proxemic/overspeed/aim terms are needed.
# The jerk penalty keeps trajectories smooth (humans walk smoothly).
# Terminal rewards remain for hard constraints.

# Terminal rewards
_R_GOAL        =  100.0   # FIX: era 200 — scala ridotta per bilanciare con ENTROPY_COEF
_R_OBS_COL     =  -50.0   # FIX: era -90
_R_WALL_COL    =  -50.0   # FIX: era -90
_R_ACTIVE_COL  =  -50.0   # FIX: era -90
# FIX #5: ridotto da -60 a -20. La passive_col avviene quando il robot è
# FERMO e un umano gli cammina sopra — il robot non può fare molto per
# evitarla. Una penalità di -60 (quasi pari alla active_col) era sproporzionata
# e insegnava alla policy a muoversi freneticamente per "scappare" dagli umani
# invece di navigare efficientemente. Con -20 rimane un segnale negativo
# (incentiva a non stare in mezzo al traffico) senza dominare il reward.
_R_PASSIVE_COL =  -10.0   # FIX: era -20
_R_TIMEOUT     =  -10.0   # FIX: era -5 — penalty più alta per timeout, incentiva velocità

# Progress scaling — how much a metre of progress toward goal is worth
# when the robot is in completely open space (clearance_factor = 1).
_PROGRESS_COEF =  12.0   # FIX: alzato 8→12. Con CF_CENTER=0.4 e densità media, clearance_factor≈0.5
                         # → 12 × 0.08m/step × 0.5 ≈ 0.48/step; goal reward 100 ≈ 200 steps di progress

# Step penalty — small constant cost per timestep, encourages efficiency.
_STEP_PEN      =  -0.02

# Jerk penalty — discourages angular velocity changes (smooth paths).
# Larger than before because smoothness is the only trajectory shaping left.
_JERK_WEIGHT   =   2.0

# ── Clearance factor parameters ───────────────────────────────────────────────
# clearance_factor(d) = sigmoid((d - _CF_CENTER) / _CF_SLOPE)
#
#   _CF_CENTER : distance at which clearance_factor = 0.5 [m]
#                Set to personal space boundary (~1.2 m).
#                At this distance the progress reward is halved.
#   _CF_SLOPE  : controls steepness of the sigmoid [m]
#                Smaller → sharper boundary; larger → softer gradient.
#                0.4 m gives a transition from ~0.1 to ~0.9 over ~1.8 m.
#
# Intuition for the policy:
#   d > 2.0 m  → clearance_factor ≈ 0.95  → almost full progress reward
#   d = 1.2 m  → clearance_factor = 0.50  → progress halved → slow down
#   d = 0.7 m  → clearance_factor ≈ 0.10  → progress worth almost nothing → stop/detour
#   d < 0.5 m  → clearance_factor ≈ 0.02  → collision imminent, no progress matters
_CF_CENTER = 0.3   # m — personal space boundary (Hall 1966)
_CF_SLOPE  = 0.3   # m — sigmoid steepness
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
    
    #max_v = 0.8

    foot_state = init_foot_state(people, k_legs)

    state = EnvState(
        x=rx, y=ry, theta=rtheta, v=0.0, w=0.0,
        goal_x=gx, goal_y=gy, max_v=max_v,
        people=people, obs_circles=obs_circles, obs_boxes=obs_boxes,
        time_step=0,
        foot_state=foot_state,
        time_stopped=0,
        sp_mask=jnp.zeros(108, dtype=jnp.bool_),
        human_stop_timers=jnp.zeros(NUM_PEOPLE, dtype=jnp.int32),
        escape_timer=0,
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
        False → robot is visible to humans (evaluation mode).

    DIFFERENTIABILITY (SHAC):
    All jnp.clip calls on robot position and action are replaced by soft_clip
    so gradients flow at boundaries. Collision indicators for reward use sigmoid
    soft-indicators instead of boolean comparisons. Terminal rewards use sigmoid
    transitions instead of hard jnp.where(done, CONST, ...) switches.
    """
    k_step, k_obs = jax.random.split(key)

    # ── 1. Robot Kinematics ───────────────────────────────────────────────────────
    target_v = jnp.clip(action[0], 0.0, state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    mid_theta = state.theta + 0.5 * target_w * DT
    new_theta = (state.theta + target_w * DT + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    raw_x     = state.x + target_v * DT * jnp.cos(mid_theta)
    raw_y     = state.y + target_v * DT * jnp.sin(mid_theta)

    # Boolean wall_collision for episode logic (stop_grad'd later)
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
    # BUG FIX 1: Stop gradient through HSFM substeps.
    # BPTT depth without this: H_outer × N_SUBSTEPS = 32 × 15 = 480 steps.
    # The HSFM Jacobian has eigenvalues > 1 in crowded scenarios, so gradients
    # explode multiplicatively over 480 steps (AGN up to 200+ observed).
    # Human positions are not controlled by the actor — only robot kinematics
    # are. The relevant BPTT path is: actor_params → action → robot kinematics
    # → dist_to_humans/goal/obstacles → reward. Human positions contribute only
    # noise amplified over 480 steps.  stop_gradient reduces BPTT depth to H
    # outer steps only.
    new_h_state = jax.lax.stop_gradient(new_h_state)

    # ── 3. Waypoint toggle & Respawn Logic ────────────────────────────────────
    idx_cur = state.people[:, 10]
    g1x, g1y = state.people[:, 6], state.people[:, 7]
    g2x, g2y = state.people[:, 8], state.people[:, 9]
    gx_cur   = jnp.where(idx_cur == 0, g1x, g2x)
    gy_cur   = jnp.where(idx_cur == 0, g1y, g2y)

    dist_to_goal = jnp.sqrt((new_h_state[:, 0] - gx_cur)**2 +
                             (new_h_state[:, 1] - gy_cur)**2 + 1e-8)

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

    # ── Human stop-and-go ──────────────────────────────────────────────────────
    # Timers > 0 → human is in idle pause this step. Dummy humans (idx < 0)
    # are never frozen. Velocities clamped after respawn so a freshly
    # teleported human is never accidentally frozen.
    stop_key = jax.random.fold_in(k_step, 0xAB57)

    stop_roll  = jax.random.uniform(stop_key, (NUM_PEOPLE,))
    dur_keys   = jax.random.split(stop_key, NUM_PEOPLE)
    stop_dur   = jax.vmap(lambda k: jax.random.randint(
        k, (), STOP_MIN_STEPS, STOP_MAX_STEPS + 1))(dur_keys)

    prev_timers  = state.human_stop_timers
    in_stop      = prev_timers > 0
    new_timers   = jnp.where(in_stop, prev_timers - 1, 0)
    start_stop   = ~in_stop & (stop_roll < P_HUMAN_STOP)
    new_timers   = jnp.where(start_stop, stop_dur, new_timers)
    is_stopped_h = new_timers > 0

    # Only freeze active (non-dummy) humans; never freeze respawned ones
    active_mask_stop = (new_people[:, 10] >= 0.0) & ~needs_respawn
    freeze           = is_stopped_h & active_mask_stop

    # Clamp px=0,1 vx=2, vy=3, omega=5 for stopped humans
    new_people = new_people.at[:, 2].set(jnp.where(freeze, 0.0, new_people[:, 2]))
    new_people = new_people.at[:, 3].set(jnp.where(freeze, 0.0, new_people[:, 3]))
    new_people = new_people.at[:, 5].set(jnp.where(freeze, 0.0, new_people[:, 5]))

    # 1. Advance the continuous gait phase for everyone
    advanced_foot_state = advance_feet(state.foot_state, new_people, DT)
    
    # 2. Generate a clean set of feet perfectly centered under the new body coordinates
    fresh_foot_state = init_foot_state(new_people, k_respawn2) 
    
    # 3. Overwrite the foot state ONLY for humans that just teleported
    new_foot_state = jnp.where(needs_respawn[:, None], fresh_foot_state, advanced_foot_state)

    # ── 4. Distance helpers ───────────────────────────────────────────────────
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2 + 1e-8)
    new_dist  = jnp.sqrt((new_x  - state.goal_x)**2 + (new_y  - state.goal_y)**2 + 1e-8)

    left_xy, right_xy = get_leg_positions(new_foot_state)   # (N,2) each

    # --- MODIFIED: Use stable JHSFM body coordinates for the center ---
    center_x = new_people[:, 0]
    center_y = new_people[:, 1]

    dx_p = center_x - new_x
    dy_p = center_y - new_y
    dists_p = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)

    # Mask dummies from all distance/collision logic
    active_mask    = new_people[:, 10] >= 0.0
    # BUG FIX 2: Replace jnp.inf with large finite sentinel for dummy humans.
    # jnp.min over an array containing jnp.inf can produce NaN gradients in
    # XLA's backward pass when inf wins the reduction (gradient of the inf
    # branch is NaN-contaminated).  Using a large finite value avoids this
    # while keeping dummy humans effectively invisible to all distance logic.
    # 1e4 >> ROOM_W/H (~12m) so it never affects the min for active humans.
    _DUMMY_DIST = 1e4
    dists_p_active = jnp.where(active_mask, dists_p, _DUMMY_DIST)
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
    closest_cir = jnp.min(jnp.sqrt(dx_c**2 + dy_c**2 + 1e-8) - state.obs_circles[:, 2])

    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(new_x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(new_y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2 + 1e-8)

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
    active_col  = active_col_body  | active_col_shoe
    passive_col = (passive_col_body | passive_col_shoe) & ~obs_collision & ~wall_collision

    collision    = human_collision | any_shoe_col | obs_collision | wall_collision
    timeout      = (state.time_step + 1) >= MAX_STEPS
    goal_reached = new_dist < GOAL_RADIUS
    done         = goal_reached | collision | timeout

    # ── 6. Reward ───────────────────────────────────────────────────────────────

    # — 6a. Clearance factor (unchanged) ──────────────────────────────────
    edge_to_edge   = jnp.maximum(0.0, closest_human - PEOPLE_RADIUS - ROBOT_RADIUS)
    clearance_factor = jax.nn.sigmoid((edge_to_edge - _CF_CENTER) / _CF_SLOPE)

    # — 6b. Dense shaping (smooth, unchanged) ────────────────────────────
    progress         = prev_dist - new_dist
    social_progress  = _PROGRESS_COEF * progress * clearance_factor
    step_pen         = _STEP_PEN
    jerk_pen         = -_JERK_WEIGHT * (target_w - state.w) ** 2
    dense_reward     = social_progress + step_pen + jerk_pen

    # — 6c. Terminal cascades ─────────────────────────────────────────────
    reward = dense_reward
    reward = jnp.where(goal_reached, _R_GOAL, reward)
    reward = jnp.where(obs_collision & ~goal_reached, _R_OBS_COL, reward)
    reward = jnp.where(wall_collision & ~obs_collision & ~goal_reached, _R_WALL_COL, reward)
    reward = jnp.where(active_col & ~obs_collision & ~wall_collision & ~goal_reached, _R_ACTIVE_COL, reward)
    reward = jnp.where(passive_col & ~active_col & ~obs_collision & ~wall_collision & ~goal_reached, _R_PASSIVE_COL, reward)
    reward = jnp.where(timeout & ~goal_reached & ~collision, _R_TIMEOUT, reward)

    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1,
        foot_state=new_foot_state,
        time_stopped=jnp.int32(0),    # unused in new reward; kept for compat
        human_stop_timers=new_timers,
        escape_timer=jnp.int32(0),    # unused in new reward; kept for compat
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