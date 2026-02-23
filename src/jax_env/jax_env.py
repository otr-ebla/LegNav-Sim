"""
jax_env.py — Core 2D Navigation Environment
============================================
FIXES & IMPROVEMENTS vs previous version:

  FIX 1 (carried) — wall_collision computed on raw position before clamp.

  FIX 2 (carried) — rear_angles dead code removed.

  FIX 3 (carried) — Reward priority order made explicit and unambiguous.

  FIX 4 (carried) — passive_col correct active/passive semantics.

  FIX 5 (carried) — People spawn with obstacle safety check.

  IMPROVEMENT A (carried) — Social comfort-zone penalty.

  IMPROVEMENT B (carried) — Goal direction in robot frame (ego-centric).

  IMPROVEMENT C (carried) — rear_prox uses all 4 rays.

  IMPROVEMENT D (carried) — Fused single LiDAR call.

  CURRICULUM (NEW) — Goal distance curriculum via min_goal_dist parameter:
    reset_env now accepts an optional min_goal_dist argument (default 3.0 m,
    matching the original hardcoded value). jax_ppo.py passes a scalar that
    starts at CURRICULUM_START_DIST and grows as success rate improves.
    - Stage 0 (suc <  30%): min_goal_dist = 1.0 m  — very easy warm-up
    - Stage 1 (suc <  55%): min_goal_dist = 2.5 m  — medium distances
    - Stage 2 (suc <  70%): min_goal_dist = 4.5 m  — close to full range
    - Stage 3 (suc >= 70%): min_goal_dist = 6.0 m  — full difficulty
    The while_loop resampling guarantees the constraint is always satisfied.

  REWARD SHAPING (NEW) — Denser progress signal:
    - progress coefficient raised 3.0 → 5.0 (stronger gradient signal)
    - heading_bonus: small reward (+0.015 max) when well-aligned with goal
      and moving forward — encourages pointing toward goal before advancing
    - step_pen tightened -0.004 → -0.006 for stronger time pressure
    - goal reward raised 20.0 → 25.0 for clearer terminal signal

Obs layout (single frame): pose(3) + state_vec(9) + lidar(NUM_RAYS) = 120
Stacked × 3: 9 + 9 + 324 = 342
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar
from jax_humans import update_all_humans

# ── Constants ─────────────────────────────────────────────────────────────────
DT             = 0.15
MAX_STEPS      = 400
NUM_RAYS       = 108
REAR_RAYS      = 4        # kept as separate scalars in state_vec (not collapsed)
NUM_PEOPLE     = 6
NUM_OBS_CIR    = 6
NUM_OBS_BOX    = 6
ROOM_W         = 12.0
ROOM_H         = 12.0
ROBOT_RADIUS   = 0.2
PEOPLE_RADIUS  = 0.2
MAX_LIDAR_DIST = 12.0
FOV            = jnp.pi          # 180° forward-facing LiDAR
GOAL_RADIUS    = 0.3             # success threshold (metres)
COMFORT_DIST   = 1.0             # social comfort zone radius (m)
COMFORT_COEF   = 0.03            # reward penalty per-human per-step at distance 0
HEADING_BONUS  = 0.015           # max reward for well-aligned + moving forward
PROGRESS_COEF  = 5.0             # progress reward coefficient (was 3.0)

# Curriculum: default min goal distance — overridden by jax_ppo.py at runtime
DEFAULT_MIN_GOAL_DIST = 3.0

# IMPROVEMENT C: state_vec now has 9 entries (was 6)
# v, w, max_v_norm, goal_dist_norm, goal_align_norm, rear_prox×4
STATE_VEC_SIZE = 9

_MAX_GOAL_DIST = float(jnp.sqrt(ROOM_W**2 + ROOM_H**2))

# Single-frame obs: pose(3) + state_vec(9) + lidar(NUM_RAYS)
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

@jax.jit
def get_obs(state: EnvState) -> jnp.ndarray:
    people_circles = jnp.stack([
        state.people[:, 0],
        state.people[:, 1],
        jnp.full(NUM_PEOPLE, PEOPLE_RADIUS)
    ], axis=-1)
    all_circles = jnp.concatenate([people_circles, state.obs_circles], axis=0)

    # IMPROVEMENT D: single fused 360° LiDAR call, then split front/rear by index.
    # Front 108 rays centred at theta (FOV=π), rear 4 rays centred at theta+π (FOV=π).
    # We run one sweep of (NUM_RAYS + REAR_RAYS) rays over the full 2π circle
    # anchored at theta - π/2 so that:
    #   rays[0 : NUM_RAYS]           = forward hemisphere
    #   rays[NUM_RAYS : NUM_RAYS+4]  = rear 4 samples
    _TOTAL_RAYS = NUM_RAYS + REAR_RAYS      # 112
    _FULL_FOV   = 2.0 * float(jnp.pi)
    all_raw = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        _TOTAL_RAYS, _FULL_FOV, MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )
    raw_lidar = all_raw[:NUM_RAYS]
    rear_raw  = all_raw[NUM_RAYS:]          # shape (REAR_RAYS,) = (4,)

    inv_lidar = jnp.clip(
        (MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )
    # IMPROVEMENT C: keep all 4 rear scalars (was collapsed to min → 3 wasted)
    rear_prox_vec = jnp.clip(
        (MAX_LIDAR_DIST - rear_raw) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )  # shape (4,)

    # IMPROVEMENT B: ego-centric goal vector instead of global (x/ROOM_W, y/ROOM_H).
    # Rotate goal offset into robot frame — rotationally invariant representation.
    dx    = state.goal_x - state.x
    dy    = state.goal_y - state.y
    cos_t = jnp.cos(-state.theta)
    sin_t = jnp.sin(-state.theta)
    gdx_ego = cos_t * dx - sin_t * dy   # forward component
    gdy_ego = sin_t * dx + cos_t * dy   # lateral component

    goal_dist  = jnp.sqrt(dx**2 + dy**2)
    goal_angle = jnp.arctan2(dy, dx)
    goal_align = (goal_angle - state.theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    s_theta    = state.theta / jnp.pi

    # pose_vec: (ego_goal_dx_norm, ego_goal_dy_norm, theta_norm)
    # Normalise ego goal by max possible distance so it's ~[-1, 1]
    pose_vec = jnp.array([
        gdx_ego / _MAX_GOAL_DIST,
        gdy_ego / _MAX_GOAL_DIST,
        s_theta,
    ])

    # state_vec: 5 scalars + 4 rear rays = 9 total
    state_vec_scalars = jnp.array([
        state.v / jnp.maximum(state.max_v, 1e-3),   # [0, 1]
        state.w,                                      # [-1, 1]
        (state.max_v - 0.2) / 1.8,                   # [0, 1]  properly normalised
        goal_dist / _MAX_GOAL_DIST,                   # [0, 1]
        goal_align / jnp.pi,                          # [-1, 1]
    ])
    state_vec = jnp.concatenate([state_vec_scalars, rear_prox_vec])  # (9,)

    return jnp.concatenate([pose_vec, state_vec, inv_lidar])  # 3+9+108 = 120


# ── Reset ─────────────────────────────────────────────────────────────────────

@jax.jit
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

    def _robot_cond(carry):
        rx, ry, k = carry
        return ~_is_safe(rx, ry, ROBOT_CLEARANCE, obs_circles, obs_boxes)

    def _robot_body(carry):
        _, _, k = carry
        k, ka, kb = jax.random.split(k, 3)
        rx = jax.random.uniform(ka, minval=margin, maxval=ROOM_W - margin)
        ry = jax.random.uniform(kb, minval=margin, maxval=ROOM_H - margin)
        return rx, ry, k

    rx0 = jax.random.uniform(k2, minval=margin, maxval=ROOM_W - margin)
    ry0 = jax.random.uniform(k3, minval=margin, maxval=ROOM_H - margin)
    rx, ry, k2 = jax.lax.while_loop(_robot_cond, _robot_body, (rx0, ry0, k2))
    rtheta = jax.random.uniform(k4, minval=-jnp.pi, maxval=jnp.pi)

    # ── Goal spawn ────────────────────────────────────────────────────────────
    GOAL_CLEARANCE = GOAL_RADIUS + 0.3

    def _goal_cond(carry):
        gx, gy, k = carry
        too_close = jnp.sqrt((gx - rx)**2 + (gy - ry)**2) < min_goal_dist
        return too_close | ~_is_safe(gx, gy, GOAL_CLEARANCE, obs_circles, obs_boxes)

    def _goal_body(carry):
        _, _, k = carry
        k, ka, kb = jax.random.split(k, 3)
        gx = jax.random.uniform(ka, minval=margin, maxval=ROOM_W - margin)
        gy = jax.random.uniform(kb, minval=margin, maxval=ROOM_H - margin)
        return gx, gy, k

    k5a, k5b = jax.random.split(k5)
    gx0 = jax.random.uniform(k5a, minval=margin, maxval=ROOM_W - margin)
    gy0 = jax.random.uniform(k5b, minval=margin, maxval=ROOM_H - margin)
    gx, gy, _ = jax.lax.while_loop(_goal_cond, _goal_body, (gx0, gy0, k6))

    # ── People — FIX 5: safe spawn with obstacle clearance ────────────────────
    # Each person resamples until clear of obstacles and walls.
    PERSON_CLEARANCE = PEOPLE_RADIUS + 0.15
    people_keys = jax.random.split(k7, NUM_PEOPLE)

    def init_person(pkey):
        pk1, pk2, pk3, pk4, pk5 = jax.random.split(pkey, 5)
        angle = jax.random.uniform(pk3, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(pk4, minval=0.4, maxval=1.4)

        # Safe position sampling
        def _p_cond(carry):
            px, py, k = carry
            return ~_is_safe(px, py, PERSON_CLEARANCE, obs_circles, obs_boxes)

        def _p_body(carry):
            _, _, k = carry
            k, ka, kb = jax.random.split(k, 3)
            px = jax.random.uniform(ka, minval=1.0, maxval=ROOM_W - 1.0)
            py = jax.random.uniform(kb, minval=1.0, maxval=ROOM_H - 1.0)
            return px, py, k

        px0 = jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W - 1.0)
        py0 = jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H - 1.0)
        px, py, _ = jax.lax.while_loop(_p_cond, _p_body, (px0, py0, pk5))

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

@jax.jit
def step_env(key: jnp.ndarray, state: EnvState, action: jnp.ndarray):
    dt = DT

    target_v = jnp.clip(action[0], 0.0,  state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    # Midpoint integration
    mid_theta = state.theta + 0.5 * target_w * dt
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    raw_x     = state.x + target_v * dt * jnp.cos(mid_theta)
    raw_y     = state.y + target_v * dt * jnp.sin(mid_theta)

    # FIX 1 (carried): detect wall collision on RAW position before clamp
    wall_collision = (
        (raw_x < ROBOT_RADIUS) | (raw_x > ROOM_W - ROBOT_RADIUS) |
        (raw_y < ROBOT_RADIUS) | (raw_y > ROOM_H - ROBOT_RADIUS)
    )
    new_x = jnp.clip(raw_x, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y = jnp.clip(raw_y, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    # Update humans (uses two subkeys to avoid key reuse)
    human_key, step_key2 = jax.random.split(key)
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

    # FIX 4: correct active/passive semantics
    # active  = robot drove into someone (human in front FOV, robot moving)
    # passive = human walked into a stationary/slow robot (human NOT in front, or robot slow)
    angles_p   = jnp.arctan2(dy_p, dx_p)
    rel_angles = (angles_p - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    in_front   = jnp.abs(rel_angles) <= (jnp.pi / 2.0)

    active_col  = jnp.any(human_col_mask & in_front)  & (target_v > 0.1)
    passive_col = jnp.any(human_col_mask & ~in_front) & (target_v < 0.1)

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
    progress  = PROGRESS_COEF * (prev_dist - new_dist)   # was 3.0, now 5.0
    step_pen  = -0.006                                    # was -0.004, tighter time pressure
    smooth    = -0.5 * jnp.abs(target_w - state.w)
    speed_bon = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)

    # Heading bonus: reward being aligned with goal AND moving forward
    goal_angle_cur = jnp.arctan2(state.goal_y - new_y, state.goal_x - new_x)
    align_cur      = (goal_angle_cur - new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    heading_bon    = HEADING_BONUS * jnp.maximum(0.0, jnp.cos(align_cur)) * (target_v / jnp.maximum(state.max_v, 1e-3))

    # IMPROVEMENT A: social comfort-zone soft penalty
    comfort_pen = -COMFORT_COEF * jnp.sum(
        jnp.maximum(0.0, 1.0 - dists_p / COMFORT_DIST)
    )

    reward = progress + step_pen + smooth + speed_bon + heading_bon + comfort_pen

    # FIX 3 (carried): explicit priority — goal > obs > wall > active_human > passive > timeout
    reward = jnp.where(goal_reached, 25.0, reward)                                              # was 20.0
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