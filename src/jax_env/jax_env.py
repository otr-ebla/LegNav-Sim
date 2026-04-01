"""
jax_env.py — Core 2D Navigation Environment
============================================
Previous fixes: A (LiDAR anchor), B (passive_col), C (resample cap),
                D (no nested JIT), E (person spawn clearance).
Obs layout (single frame): pose(3) + state_vec(9) + lidar(NUM_RAYS) = 120
Stacked × 3: 9 + 9 + 324 = 342  (UNCHANGED)
"""

import math
import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar
from jax_humans import update_all_humans
from jax_legs import get_leg_circles, get_shoe_boxes, advance_feet, init_foot_state, LEG_RADIUS
from config import SimConfig, RobotConfig, LidarConfig

# ── Feature flags ─────────────────────────────────────────────────────────────
# Flip USE_LEGS to False for cylinder-model baseline/ablation training.
# Eval scripts can override: import jax_env; jax_env.USE_LEGS = False
USE_LEGS = True

# DIFFERENTIABILITY: Set SENSOR_NOISE = False in SHAC training to get a
# deterministic gradient path through the observation.  Noise adds variance to
# BPTT gradients (the random gates are not differentiable) and can cause
# spurious zero/NaN grad contributions.
# Eval scripts should keep SENSOR_NOISE = True for realistic sensor simulation.
# Override: import jax_env; jax_env.SENSOR_NOISE = False
SENSOR_NOISE = True

# ── Constants ─────────────────────────────────────────────────────────────────
DT             = RobotConfig.DT
MAX_STEPS      = SimConfig.MAX_STEPS
NUM_RAYS       = LidarConfig.NUM_RAYS
REAR_RAYS      = 4
NUM_PEOPLE     = 18
NUM_OBS_CIR    = 7
NUM_OBS_BOX    = 7
ROOM_W         = 12.0
ROOM_H         = 12.0
ROBOT_RADIUS   = RobotConfig.RADIUS
PEOPLE_RADIUS  = SimConfig.HUMANS_RADIUS
MAX_LIDAR_DIST = 12.0
FOV            = math.pi         # 180° forward-facing LiDAR
GOAL_RADIUS    = 0.3
COMFORT_DIST   = 1.0
COMFORT_COEF   = 0.03
HEADING_BONUS  = 0.005
PROGRESS_COEF  = 1.0

MAX_RESAMPLE_ITERS = 200
DEFAULT_MIN_GOAL_DIST = 3.0

STATE_VEC_SIZE  = 9   # v, w, max_v_norm, goal_dist, goal_align, rear_prox×4
_MAX_GOAL_DIST  = math.sqrt(ROOM_W**2 + ROOM_H**2)
SINGLE_OBS_SIZE = 3 + STATE_VEC_SIZE + NUM_RAYS   # 120

# ── Human idle / stop-and-go behaviour ─────────────────────────────────────────────
# Each step an active human has P_HUMAN_STOP probability of starting a stop.
# Duration is uniform in [STOP_MIN_STEPS, STOP_MAX_STEPS]. During a stop
# vx=vy=0 (velocity clamped after update_all_humans).
P_HUMAN_STOP    = 0.003   # ~0.3 % per step → avg one stop every ~22 s at dt=0.15
STOP_MIN_STEPS  = 25       # 1.05 s minimum pause
STOP_MAX_STEPS  = 60      # 6.0  s maximum pause


@struct.dataclass
class EnvState:
    x:           jnp.float32
    y:           jnp.float32
    theta:       jnp.float32
    v:           jnp.float32
    w:           jnp.float32
    goal_x:      jnp.float32
    goal_y:      jnp.float32
    max_v:       jnp.float32
    people:      jnp.ndarray   # (NUM_PEOPLE, 8)
    obs_circles: jnp.ndarray   # (NUM_OBS_CIR, 3)  [cx, cy, r]
    obs_boxes:   jnp.ndarray   # (NUM_OBS_BOX, 4)  [cx, cy, hw, hh]
    time_step:   jnp.int32
    # ── NEW: per-human gait phase ────────────────────────────────────────────
    foot_state:         jnp.ndarray
    time_stopped:       jnp.int32
    sp_mask:            jnp.ndarray
    human_stop_timers:  jnp.ndarray
    escape_timer:       jnp.int32     # unused in new reward; kept for checkpoint compat
    is_ghost:           jnp.ndarray


# ── Helpers ───────────────────────────────────────────────────────────────────

def _min_dist_to_circles(x, y, circles):
    dx = circles[:, 0] - x
    dy = circles[:, 1] - y
    return jnp.min(jnp.sqrt(dx**2 + dy**2) - circles[:, 2])


def _min_dist_to_boxes(x, y, boxes):
    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)
    return jnp.min(jax.vmap(_box_dist)(boxes))


def _is_safe(x, y, clearance, obs_circles, obs_boxes):
    wall_ok = (x > clearance) & (x < ROOM_W - clearance) & \
              (y > clearance) & (y < ROOM_H - clearance)
    cir_ok  = _min_dist_to_circles(x, y, obs_circles) > clearance
    box_ok  = _min_dist_to_boxes(x, y, obs_boxes) > clearance
    return wall_ok & cir_ok & box_ok


# ── Observation ───────────────────────────────────────────────────────────────

def get_obs(state: EnvState, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the flat observation vector.

    PATCH: people_circles is now replaced by leg circles from jax_legs.
    The number of circles fed to compute_lidar changes:
      USE_LEGS=True  → 2*N_people + N_obs_circles  (leg-pair model)
      USE_LEGS=False → N_people + N_obs_circles     (cylinder model)
    compute_lidar handles variable circle counts via vmap, so this is fine.

    OBS_SIZE = 342 is UNCHANGED — the output vector layout is identical.
    """
    # ── Build human geometry for LiDAR ───────────────────────────────────────
    # USE_LEGS is a Python bool → resolved at trace time, no conditional overhead
    human_circles = get_leg_circles(state.people, state.foot_state, use_legs=USE_LEGS)
    all_circles = jnp.concatenate([human_circles, state.obs_circles], axis=0)

    # Front LiDAR sweep — 108 rays, FOV=π
    # NOTE: shoes are NOT included here — LiDAR only sees leg circles, not shoes.
    raw_lidar = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )

    # ── NEW: Apply Vectorized Sensor Noise (only when SENSOR_NOISE=True) ────────
    # SHAC training sets SENSOR_NOISE=False for deterministic gradient flow.
    # SENSOR_NOISE is a Python bool resolved at trace time — zero overhead.
    if SENSOR_NOISE:
        k_gauss, k_sp = jax.random.split(key)
        # 1. Gaussian Noise (e.g., 0.05m standard deviation)
        gaussian = jax.random.normal(k_gauss, raw_lidar.shape) * 0.05
        # 2. Salt & Pepper Noise (3% of rays)
        sp_rand = jax.random.uniform(k_sp, raw_lidar.shape)
        sp_mask = sp_rand < 0.03            # The 3% affected rays
        is_max  = sp_rand < 0.015           # 1.5% drop to maximum
        # Apply Gaussian first, then overwrite with Salt/Pepper extremes
        noisy_lidar = jnp.where(sp_mask, jnp.where(is_max, MAX_LIDAR_DIST, 0.0), raw_lidar + gaussian)
        noisy_lidar = jnp.clip(noisy_lidar, 0.0, MAX_LIDAR_DIST)
    else:
        # No noise: clean differentiable LiDAR for SHAC BPTT
        noisy_lidar = raw_lidar
        sp_mask = jnp.zeros(raw_lidar.shape, dtype=bool)
    # ──────────────────────────────────────────────────────────────────────────

    # Rear sweep — 4 rays, FOV=0.75π
    _REAR_FOV = float(jnp.pi * 0.75)
    rear_raw = compute_lidar(
        state.x, state.y, state.theta + jnp.pi,
        all_circles, state.obs_boxes,
        REAR_RAYS, _REAR_FOV, MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )

    inv_lidar = jnp.clip(
        (MAX_LIDAR_DIST - noisy_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )
    rear_prox_vec = jnp.clip(
        (MAX_LIDAR_DIST - rear_raw) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )

    dx    = state.goal_x - state.x
    dy    = state.goal_y - state.y
    cos_t = jnp.cos(-state.theta)
    sin_t = jnp.sin(-state.theta)
    gdx_ego = cos_t * dx - sin_t * dy
    gdy_ego = sin_t * dx + cos_t * dy

    goal_dist  = jnp.sqrt(dx**2 + dy**2)
    goal_angle = jnp.arctan2(dy, dx)
    goal_align = (goal_angle - state.theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    s_theta    = state.theta / jnp.pi

    pose_vec = jnp.array([
        gdx_ego / _MAX_GOAL_DIST,
        gdy_ego / _MAX_GOAL_DIST,
        s_theta,
    ])

    state_vec_scalars = jnp.array([
        state.v / jnp.maximum(state.max_v, 1e-3),
        state.w,
        (state.max_v - 0.2) / 1.8,
        goal_dist / _MAX_GOAL_DIST,
        goal_align / jnp.pi,
    ])
    state_vec = jnp.concatenate([state_vec_scalars, rear_prox_vec])  # (9,)

    return jnp.concatenate([pose_vec, state_vec, inv_lidar]), sp_mask  # 3+9+108 = 120


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_env(key: jnp.ndarray, max_goal_dist: float = 3.0, **kwargs):
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k_legs, k_obs = jax.random.split(key, 11)

    max_v  = jax.random.uniform(k1, minval=0.2, maxval=2.0)
    margin = ROBOT_RADIUS + 0.5

    # ── Circular obstacles ────────────────────────────────────────────────────
    cir_keys = jax.random.split(k8, NUM_OBS_CIR)

    def init_circle(ck):
        c1, c2, c3 = jax.random.split(ck, 3)
        cx = jax.random.uniform(c1, minval=1.5, maxval=ROOM_W - 1.5)
        cy = jax.random.uniform(c2, minval=1.5, maxval=ROOM_H - 1.5)
        r  = jax.random.uniform(c3, minval=0.15, maxval=0.45)
        return jnp.array([cx, cy, r])

    obs_circles = jax.vmap(init_circle)(cir_keys)

    # ── Rectangular obstacles ─────────────────────────────────────────────────
    box_keys = jax.random.split(k9, NUM_OBS_BOX)

    def init_box(bk):
        b1, b2, b3, b4 = jax.random.split(bk, 4)
        cx = jax.random.uniform(b1, minval=1.5, maxval=ROOM_W - 1.5)
        cy = jax.random.uniform(b2, minval=1.5, maxval=ROOM_H - 1.5)
        hw = jax.random.uniform(b3, minval=0.20, maxval=0.70)
        hh = jax.random.uniform(b4, minval=0.20, maxval=0.70)
        return jnp.array([cx, cy, hw, hh])

    obs_boxes = jax.vmap(init_box)(box_keys)

    # ── Robot spawn ───────────────────────────────────────────────────────────
    ROBOT_CLEARANCE = ROBOT_RADIUS + 0.35

    def _robot_cond(carry):
        rx, ry, k, i = carry
        return ~_is_safe(rx, ry, ROBOT_CLEARANCE, obs_circles, obs_boxes) & \
               (i < MAX_RESAMPLE_ITERS)

    def _robot_body(carry):
        _, _, k, i = carry
        k, ka, kb = jax.random.split(k, 3)
        rx = jax.random.uniform(ka, minval=margin, maxval=ROOM_W - margin)
        ry = jax.random.uniform(kb, minval=margin, maxval=ROOM_H - margin)
        return rx, ry, k, i + 1

    rx0 = jax.random.uniform(k2, minval=margin, maxval=ROOM_W - margin)
    ry0 = jax.random.uniform(k3, minval=margin, maxval=ROOM_H - margin)
    rx, ry, k2, _ = jax.lax.while_loop(_robot_cond, _robot_body, (rx0, ry0, k2, 0))
    rtheta = jax.random.uniform(k4, minval=-jnp.pi, maxval=jnp.pi)

    # ── Goal spawn ────────────────────────────────────────────────────────────
    GOAL_CLEARANCE = GOAL_RADIUS + 0.3

    def _goal_cond(carry):
        gx, gy, k, i = carry
        dist = jnp.sqrt((gx - rx)**2 + (gy - ry)**2)
        # Keeps goal outside the robot (0.8m) but within the curriculum max_goal_dist
        bad_dist = (dist < 0.8) | (dist > max_goal_dist) 
        return (bad_dist | ~_is_safe(gx, gy, GOAL_CLEARANCE, obs_circles, obs_boxes)) & \
               (i < MAX_RESAMPLE_ITERS)

    def _goal_body(carry):
        _, _, k, i = carry
        k, ka, kb = jax.random.split(k, 3)
        gx = jax.random.uniform(ka, minval=margin, maxval=ROOM_W - margin)
        gy = jax.random.uniform(kb, minval=margin, maxval=ROOM_H - margin)
        return gx, gy, k, i + 1

    k5a, k5b = jax.random.split(k5)
    gx0 = jax.random.uniform(k5a, minval=margin, maxval=ROOM_W - margin)
    gy0 = jax.random.uniform(k5b, minval=margin, maxval=ROOM_H - margin)
    gx, gy, _, _ = jax.lax.while_loop(_goal_cond, _goal_body, (gx0, gy0, k6, 0))

    # ── People ────────────────────────────────────────────────────────────────
    PERSON_CLEARANCE   = PEOPLE_RADIUS + 0.15
    PERSON_ROBOT_CLEAR = ROBOT_RADIUS + PEOPLE_RADIUS + 0.3
    PERSON_GOAL_CLEAR  = GOAL_RADIUS + PEOPLE_RADIUS + 0.1
    people_keys = jax.random.split(k7, NUM_PEOPLE)

    def init_person(pkey):
        pk1, pk2, pk3, pk4, pk5 = jax.random.split(pkey, 5)
        angle = jax.random.uniform(pk3, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(pk4, minval=0.4, maxval=1.4)

        def _p_cond(carry):
            px, py, k, i = carry
            obs_safe    = _is_safe(px, py, PERSON_CLEARANCE, obs_circles, obs_boxes)
            robot_clear = jnp.sqrt((px - rx)**2 + (py - ry)**2) > PERSON_ROBOT_CLEAR
            goal_clear  = jnp.sqrt((px - gx)**2 + (py - gy)**2) > PERSON_GOAL_CLEAR
            return (~obs_safe | ~robot_clear | ~goal_clear) & (i < MAX_RESAMPLE_ITERS)

        def _p_body(carry):
            _, _, k, i = carry
            k, ka, kb = jax.random.split(k, 3)
            px = jax.random.uniform(ka, minval=1.0, maxval=ROOM_W - 1.0)
            py = jax.random.uniform(kb, minval=1.0, maxval=ROOM_H - 1.0)
            return px, py, k, i + 1

        px0 = jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W - 1.0)
        py0 = jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H - 1.0)
        px, py, _, _ = jax.lax.while_loop(_p_cond, _p_body, (px0, py0, pk5, 0))

        vx = speed * jnp.cos(angle)
        vy = speed * jnp.sin(angle)
        return jnp.array([px, py, vx, vy, angle, 0.0, -1.0, speed])

    people = jax.vmap(init_person)(people_keys)

    # ── NEW: random initial gait phases ───────────────────────────────────────
    # Stagger phases so humans aren't all in sync at episode start.
    foot_state = init_foot_state(people, k_legs)

    state = EnvState(
        x=rx, y=ry, theta=rtheta,
        v=0.0, w=0.0,
        goal_x=gx, goal_y=gy,
        max_v=max_v,
        people=people,
        obs_circles=obs_circles,
        obs_boxes=obs_boxes,
        time_step=0,
        foot_state=foot_state,
        time_stopped=0,
        sp_mask=jnp.zeros(NUM_RAYS, dtype=jnp.bool_),
        human_stop_timers=jnp.zeros(NUM_PEOPLE, dtype=jnp.int32),
        escape_timer=0,
        is_ghost=jnp.array(False, dtype=jnp.bool_),
    )

    obs, sp_mask = get_obs(state, k_obs)
    state = state.replace(sp_mask=sp_mask)
    return obs, state


# ── Step ──────────────────────────────────────────────────────────────────────

def step_env(key: jnp.ndarray, state: EnvState, action: jnp.ndarray, **kwargs):
    dt = DT

    target_v = jnp.clip(action[0], 0.0,  state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    # Midpoint integration
    mid_theta = state.theta + 0.5 * target_w * dt
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    raw_x     = state.x + target_v * dt * jnp.cos(mid_theta)
    raw_y     = state.y + target_v * dt * jnp.sin(mid_theta)

    wall_collision = (
        (raw_x < ROBOT_RADIUS) | (raw_x > ROOM_W - ROBOT_RADIUS) |
        (raw_y < ROBOT_RADIUS) | (raw_y > ROOM_H - ROBOT_RADIUS)
    )
    new_x = jnp.clip(raw_x, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y = jnp.clip(raw_y, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    human_key, k_obs = jax.random.split(key)
    new_people = update_all_humans(
        state.people, human_key, dt,
        new_x, new_y, new_theta, target_v,
        ROOM_W, ROOM_H, PEOPLE_RADIUS,
        state.obs_circles, state.obs_boxes
    )

    # ── Human stop-and-go ───────────────────────────────────────────────────────
    # Timers in state.human_stop_timers > 0 → human is in idle pause.
    # Each step: decrement active timers; for humans whose timer just
    # hit 0, sample whether to start a new stop (P_HUMAN_STOP) and draw
    # a random duration.  All JIT-compatible — no Python control flow on
    # traced values.
    stop_key             = jax.random.fold_in(human_key, 0xAB57)
    stop_roll    = jax.random.uniform(stop_key, (NUM_PEOPLE,))          # U[0,1)
    dur_keys     = jax.random.split(stop_key, NUM_PEOPLE)
    stop_dur     = jax.vmap(lambda k: jax.random.randint(
        k, (), STOP_MIN_STEPS, STOP_MAX_STEPS + 1))(dur_keys)           # (N,) int

    prev_timers  = state.human_stop_timers                               # (N,)
    in_stop      = prev_timers > 0                                       # currently paused
    new_timers   = jnp.where(in_stop, prev_timers - 1, 0)               # decrement

    # Humans whose timer just expired AND roll < P_HUMAN_STOP start a new stop
    start_stop   = ~in_stop & (stop_roll < P_HUMAN_STOP)
    new_timers   = jnp.where(start_stop, stop_dur, new_timers)

    # Still paused after decrement = should be frozen this step
    is_stopped_h = new_timers > 0                                        # (N,)

    # Prevent JHSFM drift by carrying over the old state entirely, with zeroed velocities.
    frozen_people = state.people.at[:, 2:4].set(0.0)
    new_people = jnp.where(is_stopped_h[:, None], frozen_people, new_people)

    # ── Advance leg gait phases ───────────────────────────────────────────────
    new_foot_state = advance_feet(state.foot_state, new_people, dt)

    # ── Distances & Angles ────────────────────────────────────────────────────
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x  - state.goal_x)**2 + (new_y  - state.goal_y)**2)

    dx_p    = new_people[:, 0] - new_x
    dy_p    = new_people[:, 1] - new_y
    dists_p = jnp.sqrt(dx_p**2 + dy_p**2)
    closest_human = jnp.min(dists_p)

    human_angles = jnp.arctan2(dy_p, dx_p)
    rel_angles   = (human_angles - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # ── Human Collisions (Active vs Passive) ─────────────────────────────────
    # USE_LEGS=False: circle-vs-circle model.
    #   Collision = robot circle overlaps human body circle (radius PEOPLE_RADIUS).
    #
    # USE_LEGS=True: shoe-box model.
    #   Human body collision threshold is tightened to LEG_RADIUS (the actual
    #   simulated geometry) because the shoe AABBs already capture the main
    #   contact surface. Body-circle hits at this threshold represent only
    #   direct leg/torso contact that slipped past the shoe AABB check.
    #   Shoe-box contacts (active/passive) are computed separately below and
    #   merged into the final active_col / passive_col flags.
    is_in_front = jnp.abs(rel_angles) < (jnp.pi / 2.0)   # forward 180° FOV
    in_prox     = dists_p < 1.5                            # within 1.5 m
    is_moving   = target_v >= 0.1                          # robot moving

    if USE_LEGS:
        body_thresh = ROBOT_RADIUS + LEG_RADIUS
    else:
        body_thresh = ROBOT_RADIUS + PEOPLE_RADIUS

    col_mask     = dists_p < body_thresh
    active_mask  = col_mask & is_in_front & in_prox & is_moving
    passive_mask = col_mask & ~active_mask

    active_col_body  = jnp.any(active_mask)
    passive_col_body = jnp.any(passive_mask)

    # ── Obstacle Collisions ───────────────────────────────────────────────────
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

    # ── Shoe-box collisions — USE_LEGS=True only ──────────────────────────────
    # When USE_LEGS=True the shoe AABB is the primary human contact surface.
    # Each shoe contact is classified active/passive with the same logic as
    # body contacts (owner heading dot-product + proximity + robot speed).
    if USE_LEGS:
        shoe_boxes_step  = get_shoe_boxes(new_people, new_foot_state)   # (2N, 4)
        shoe_dists_raw   = jax.vmap(_box_dist)(shoe_boxes_step)         # (2N,)

        # Map each shoe to its owner: left shoes 0..N-1, right shoes N..2N-1
        N_p            = new_people.shape[0]
        owner_idx      = jnp.concatenate([jnp.arange(N_p), jnp.arange(N_p)])
        shoe_dists     = shoe_dists_raw                                 # no dummy mask in jax_env

        shoe_col_mask  = shoe_dists < ROBOT_RADIUS                     # (2N,)
        any_shoe_col   = jnp.any(shoe_col_mask)

        owner_in_front = is_in_front[owner_idx]
        owner_in_prox  = in_prox[owner_idx]

        active_shoe    = shoe_col_mask & owner_in_front & owner_in_prox & is_moving
        active_col_shoe  = jnp.any(active_shoe)
        passive_col_shoe = any_shoe_col & ~active_col_shoe
    else:
        any_shoe_col     = jnp.array(False)
        active_col_shoe  = jnp.array(False)
        passive_col_shoe = jnp.array(False)

    # Merge body + shoe collision flags
    # active_col / passive_col cover ONLY human contacts.
    # Wall and static-obstacle contacts are always in obs_collision /
    # wall_collision and NEVER bleed into passive_col.
    active_col  = active_col_body  | active_col_shoe
    passive_col = (passive_col_body | passive_col_shoe) & ~obs_collision & ~wall_collision

    # ── End of Episode Flags ──────────────────────────────────────────────────
    human_collision = active_col | passive_col
    collision       = human_collision | obs_collision | wall_collision
    timeout         = (state.time_step + 1) >= MAX_STEPS
    goal_reached    = new_dist < GOAL_RADIUS
    done            = goal_reached | collision | timeout

    # ── Dense Rewards ─────────────────────────────────────────────────────────

    # Progress: ~0.18/step at 1.2 m/s. Primary navigation signal, unchanged.
    progress = PROGRESS_COEF * (prev_dist - new_dist)

    # Step penalty: -0.006 → -0.02.
    # Old value accumulated to only -2.4 over a full 400-step timeout — nearly
    # free. At -0.02 a timeout costs -8, making efficiency matter.
    step_pen = -0.02

    # Smoothness: unchanged.
    smooth = -0.5 * jnp.abs(target_w - state.w)

    # Speed bonus: unchanged.
    speed_bon = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)

    # Heading bonus: unchanged.
    goal_angle_cur = jnp.arctan2(state.goal_y - new_y, state.goal_x - new_x)
    align_cur      = (goal_angle_cur - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    heading_bon    = HEADING_BONUS * jnp.maximum(0.0, jnp.cos(align_cur)) * \
                     (target_v / jnp.maximum(state.max_v, 1e-3))

    # Comfort penalty scaled with speed: more dangerous to invade comfort at high v
    comfort_pen = -0.15 * jnp.sum(
        jnp.maximum(0.0, 1.0 - dists_p / COMFORT_DIST)
    ) * (1.0 + target_v / jnp.maximum(state.max_v, 1e-3))

    YIELD_DIST = 1.5   # aligned with benchmark_eval.py
    YIELD_FOV  = 1.5708

    in_yield_zone      = (dists_p < YIELD_DIST) & (jnp.abs(rel_angles) < YIELD_FOV)
    is_yield_situation = jnp.any(in_yield_zone)

    closest_yield_dist = jnp.min(jnp.where(in_yield_zone, dists_p, 100.0))
    urgency = jnp.where(is_yield_situation,
                        (YIELD_DIST - closest_yield_dist) / YIELD_DIST, 0.0)

    is_stopped = target_v <= 0.1

    new_time_stopped = jnp.where(
        is_yield_situation,
        jnp.where(is_stopped, state.time_stopped + 1, state.time_stopped),
        0,
    )

    # Speed bonus suppressed in yield zone: stopping should never cost speed points
    speed_bon_yield = jnp.where(is_yield_situation, 0.0, speed_bon)

    # Yield penalty is absolute (not normalised by max_v):
    # moving at 2 m/s costs ~5x more than at 0.4 m/s — strong deterrent at any speed
    yield_penalty = -7.5 * urgency * target_v
    # Yield bonus: full value while urgency is present, no time-decay
    yield_bonus   =  0.5 * urgency

    yield_reward = jnp.where(
        is_yield_situation,
        jnp.where(is_stopped, yield_bonus, yield_penalty),
        0.0
    )

    # ── Terminal Reward Overrides ─────────────────────────────────────────────
    reward = progress + step_pen + smooth + speed_bon_yield + heading_bon + comfort_pen + yield_reward

    reward = jnp.where(goal_reached, 200.0, reward)
    reward = jnp.where(obs_collision  & ~goal_reached, -70.0, reward)
    reward = jnp.where(wall_collision & ~obs_collision & ~goal_reached, -70.0, reward)
    reward = jnp.where(active_col  & ~obs_collision & ~wall_collision & ~goal_reached, -72.0, reward)

    reward = jnp.where(passive_col & ~active_col & ~obs_collision & ~wall_collision & ~goal_reached, -22.0, reward)
    reward = jnp.where(timeout & ~goal_reached & ~collision, -0.5, reward)

    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1,
        foot_state=new_foot_state,
        time_stopped=new_time_stopped,
        human_stop_timers=new_timers,
    )

    obs, sp_mask = get_obs(new_state, k_obs)
    new_state = new_state.replace(sp_mask=sp_mask)

    info = {
        "discount":      jnp.where(done, 0.0, 1.0),
        "goal_reached":  goal_reached,
        "collision":     collision,
        "passive_col":   passive_col,
        "active_col":    active_col,    # <-- Added this line
        "closest_human": closest_human,
        "sp_mask":       sp_mask,
    }
    return obs, new_state, reward, done, info