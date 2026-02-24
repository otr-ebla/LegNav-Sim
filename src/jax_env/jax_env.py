"""
jax_env.py — Core 2D Navigation Environment
============================================
FIXES vs previous version:

  FIX A — LiDAR anchor was wrong (Bug #1):
    The old code made a single 360° sweep anchored at theta, so ray 0 pointed
    straight BEHIND the robot (theta - π). The first 108 rays covered nearly
    the full circle starting from the rear — not the intended forward 180°.
    FIX: two separate compute_lidar calls:
      • Front: NUM_RAYS=108 rays, FOV=π, anchored at theta
                → ray 0 at theta-π/2 (right), ray 107 at theta+π/2 (left) ✓
      • Rear:  REAR_RAYS=4  rays, FOV=π*0.75, anchored at theta+π
                → 4 probes spread across the rear hemisphere ✓

  FIX B — passive_col logic corrected (Bug #2):
    Old code: active = human_collision & robot moving, passive = robot stopped.
    Wrong: ignores whether the human was actually visible in the LiDAR FOV.
    CORRECT definition:
      active  = human_collision & target_v > 0.1 & human in forward 180° FOV
                (robot was moving AND the human was visible — robot's fault)
      passive = human_collision & ~active
                (robot stopped, OR human came from outside the forward FOV)
    FOV test: dot(robot_heading, vector_to_human) > 0, matching the 180° LiDAR.

  FIX C — while_loop rejection sampler had no iteration cap (Bug #5):
    Dense obstacle configs could hang indefinitely on GPU.
    FIX: carry includes an iteration counter; after MAX_RESAMPLE_ITERS the
    loop exits and the last sampled position is used (guaranteed non-infinite).

  FIX D — @jax.jit removed from reset_env and step_env (Issue #8):
    Both functions were @jax.jit decorated but are always called from within
    an outer jax.jit (via vmap in jax_train.py). Nested JIT prevents kernel
    fusion and adds tracing overhead.
    FIX: removed @jax.jit from reset_env and step_env. JIT lives only at the
    outermost vmap level in jax_train.py.

  FIX E — Person spawn ignored robot and goal positions (Bug #9):
    The old _p_cond only checked static obstacle clearance. People could
    spawn directly on top of the robot or goal, causing an immediate collision
    penalty on step 0 before the agent could act. The agent was unfairly
    punished for a reset-time placement it had no control over.
    FIX: _p_cond now also rejects positions within (ROBOT_RADIUS +
    PEOPLE_RADIUS + 0.3) of the robot spawn and within GOAL_RADIUS of the
    goal. The extra 0.3 m margin prevents a person from immediately walking
    into the robot on the very first step.

  All previous fixes (FIX 1-5) and improvements (A-D) are carried forward.

Obs layout (single frame): pose(3) + state_vec(9) + lidar(NUM_RAYS) = 120
Stacked × 3: 9 + 9 + 324 = 342
"""

import math
import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar
from jax_humans import update_all_humans

# ── Constants ─────────────────────────────────────────────────────────────────
DT             = 0.15
MAX_STEPS      = 400
NUM_RAYS       = 108
REAR_RAYS      = 4
NUM_PEOPLE     = 10
NUM_OBS_CIR    = 6
NUM_OBS_BOX    = 6
ROOM_W         = 12.0
ROOM_H         = 12.0
ROBOT_RADIUS   = 0.2
PEOPLE_RADIUS  = 0.2
MAX_LIDAR_DIST = 12.0
FOV            = math.pi         # 180° forward-facing LiDAR
GOAL_RADIUS    = 0.3
COMFORT_DIST   = 1.0
COMFORT_COEF   = 0.03
HEADING_BONUS  = 0.015
PROGRESS_COEF  = 5.0

# Rejection-sampler iteration cap — prevents infinite while_loop on GPU
MAX_RESAMPLE_ITERS = 200

DEFAULT_MIN_GOAL_DIST = 3.0

STATE_VEC_SIZE  = 9   # v, w, max_v_norm, goal_dist, goal_align, rear_prox×4
_MAX_GOAL_DIST  = math.sqrt(ROOM_W**2 + ROOM_H**2)
SINGLE_OBS_SIZE = 3 + STATE_VEC_SIZE + NUM_RAYS   # 120


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

def get_obs(state: EnvState) -> jnp.ndarray:
    people_circles = jnp.stack([
        state.people[:, 0],
        state.people[:, 1],
        jnp.full(NUM_PEOPLE, PEOPLE_RADIUS)
    ], axis=-1)
    all_circles = jnp.concatenate([people_circles, state.obs_circles], axis=0)

    # FIX A: Two separate LiDAR calls instead of one broken 360° sweep.
    #
    # Front sweep — 108 rays, FOV=π, anchored at theta:
    #   compute_lidar spans [theta - FOV/2 .. theta + FOV/2]
    #                     = [theta - π/2  .. theta + π/2]
    #   ray 0  → theta - π/2  (right side of robot) ✓
    #   ray 107 → theta + π/2 (left  side of robot) ✓
    raw_lidar = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )

    # Rear sweep — 4 rays, FOV=0.75π, anchored at theta+π (pointing backward):
    #   spans [theta + π - 0.375π .. theta + π + 0.375π]
    #        = [theta + 0.625π    .. theta + 1.375π]
    _REAR_FOV = float(jnp.pi * 0.75)
    rear_raw = compute_lidar(
        state.x, state.y, state.theta + jnp.pi,
        all_circles, state.obs_boxes,
        REAR_RAYS, _REAR_FOV, MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )

    inv_lidar = jnp.clip(
        (MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )
    rear_prox_vec = jnp.clip(
        (MAX_LIDAR_DIST - rear_raw) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )  # shape (4,)

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

    return jnp.concatenate([pose_vec, state_vec, inv_lidar])  # 3+9+108 = 120


# ── Reset ─────────────────────────────────────────────────────────────────────

# FIX D: @jax.jit removed — JIT lives at the outermost vmap level in jax_train.py.
def reset_env(key: jnp.ndarray, min_goal_dist: float = DEFAULT_MIN_GOAL_DIST):
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)

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

    # FIX C: carry = (x, y, key, iter_count); exit after MAX_RESAMPLE_ITERS.
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
        too_close = jnp.sqrt((gx - rx)**2 + (gy - ry)**2) < min_goal_dist
        return (too_close | ~_is_safe(gx, gy, GOAL_CLEARANCE, obs_circles, obs_boxes)) & \
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

    # ── People — safe spawn with obstacle, robot, and goal clearance ─────────
    PERSON_CLEARANCE  = PEOPLE_RADIUS + 0.15
    # FIX E: minimum separation from robot at spawn to avoid instant collision
    # on step 0. 0.3 m margin gives the human ~1 step of travel before it
    # could physically reach the robot even at max human speed (1.4 m/s * 0.15 s = 0.21 m).
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
            # FIX E: also reject overlap with robot spawn and goal positions.
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

    state = EnvState(
        x=rx, y=ry, theta=rtheta,
        v=0.0, w=0.0,
        goal_x=gx, goal_y=gy,
        max_v=max_v,
        people=people,
        obs_circles=obs_circles,
        obs_boxes=obs_boxes,
        time_step=0
    )
    return get_obs(state), state


# ── Step ──────────────────────────────────────────────────────────────────────

# FIX D: @jax.jit removed — JIT lives at the outermost vmap level in jax_train.py.
def step_env(key: jnp.ndarray, state: EnvState, action: jnp.ndarray):
    dt = DT

    target_v = jnp.clip(action[0], 0.0,  state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    # Midpoint integration
    mid_theta = state.theta + 0.5 * target_w * dt
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    raw_x     = state.x + target_v * dt * jnp.cos(mid_theta)
    raw_y     = state.y + target_v * dt * jnp.sin(mid_theta)

    # Wall collision detected on raw position before clamp
    wall_collision = (
        (raw_x < ROBOT_RADIUS) | (raw_x > ROOM_W - ROBOT_RADIUS) |
        (raw_y < ROBOT_RADIUS) | (raw_y > ROOM_H - ROBOT_RADIUS)
    )
    new_x = jnp.clip(raw_x, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y = jnp.clip(raw_y, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    human_key, _ = jax.random.split(key)
    new_people = update_all_humans(
        state.people, human_key, dt,
        new_x, new_y, new_theta, target_v,
        ROOM_W, ROOM_H, PEOPLE_RADIUS,
        state.obs_circles, state.obs_boxes
    )

    # ── Distances ─────────────────────────────────────────────────────────────
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x  - state.goal_x)**2 + (new_y  - state.goal_y)**2)

    dx_p    = new_people[:, 0] - new_x
    dy_p    = new_people[:, 1] - new_y
    dists_p = jnp.sqrt(dx_p**2 + dy_p**2)
    closest_human = jnp.min(dists_p)

    # ── Collisions ────────────────────────────────────────────────────────────
    human_col_mask  = dists_p < (ROBOT_RADIUS + PEOPLE_RADIUS)
    human_collision = jnp.any(human_col_mask)

    # Active collision: human in forward 180° FOV AND robot is moving.
    # The human was visible in LiDAR — the robot's fault for not avoiding.
    # FOV test: dot(robot_heading, vector_to_human) > 0
    #   heading = (cos(new_theta), sin(new_theta))
    #   vector to human = (dx_p, dy_p) — unnormalised, sign is sufficient.
    # Passive: robot was stopped OR all colliding humans came from behind the FOV.
    heading_dot = dx_p * jnp.cos(new_theta) + dy_p * jnp.sin(new_theta)  # (NUM_PEOPLE,)
    in_fov_mask = heading_dot > 0.0                                        # forward 180°
    any_active  = jnp.any(human_col_mask & in_fov_mask)

    active_col  = human_collision & (target_v > 0.1) & any_active
    passive_col = human_collision & ~active_col

    # Obstacle collisions
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

    # ── Reward ────────────────────────────────────────────────────────────────
    progress  = PROGRESS_COEF * (prev_dist - new_dist)
    step_pen  = -0.006
    smooth    = -0.5 * jnp.abs(target_w - state.w)
    speed_bon = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)

    goal_angle_cur = jnp.arctan2(state.goal_y - new_y, state.goal_x - new_x)
    align_cur      = (goal_angle_cur - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    heading_bon    = HEADING_BONUS * jnp.maximum(0.0, jnp.cos(align_cur)) * \
                     (target_v / jnp.maximum(state.max_v, 1e-3))

    comfort_pen = -COMFORT_COEF * jnp.sum(
        jnp.maximum(0.0, 1.0 - dists_p / COMFORT_DIST)
    )

    reward = progress + step_pen + smooth + speed_bon + heading_bon + comfort_pen

    # Explicit priority: goal > obs > wall > active_human > passive > timeout
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
        time_step=state.time_step + 1
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