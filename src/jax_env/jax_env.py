"""
jax_env.py — Core 2D Navigation Environment
============================================
FIXES vs previous version:
  1. Circle collision threshold: was `closest_cir < 0.0` (ignores robot body),
     now `closest_cir < ROBOT_RADIUS` — consistent with box collision.
  2. Goal RNG: was using same key k5/k6 for both jnp.where branches (both
     uniform calls evaluated with same key → correlated draws). Now uses
     k5a/k5b split per axis.
  3. Spawn safety: robot start and goal positions are now validated to be at
     least (ROBOT_RADIUS + max_obs_r + 0.1) away from all obstacles and from
     each other, using jax.lax.while_loop resampling.
  4. Reward rebalancing: terminal rewards scaled down 10× (±20/−7) so the
     value function can bootstrap in a stable range. Step rewards kept as-is.
  5. FOV kept at π (180°) but a 4-scalar "rear proximity" feature added to
     state_vec so the agent can sense what's directly behind it.

Obs layout (single frame): pose(3) + state_vec(6) + lidar(NUM_RAYS) = 117
Stacked × 3 = 9 + 6 + 324 = 339
NOTE: OBS_SIZE in jax_train.py updated accordingly.
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar
from jax_humans import update_all_humans

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_STEPS    = 400
NUM_RAYS     = 108
NUM_PEOPLE   = 6
NUM_OBS_CIR  = 6
NUM_OBS_BOX  = 6
ROOM_W       = 12.0
ROOM_H       = 12.0
ROBOT_RADIUS = 0.2
PEOPLE_RADIUS= 0.2
MAX_LIDAR_DIST = 12.0
FOV          = jnp.pi          # 180° forward-facing LiDAR
REAR_RAYS    = 4               # extra rear-proximity scalars added to state_vec
GOAL_RADIUS  = 0.3             # success threshold (metres)

# Single-frame obs size: pose(3) + state_vec(6) + lidar(NUM_RAYS)
# state_vec: v, w, max_v_norm, goal_dist_norm, goal_align_norm, rear_prox_norm
SINGLE_OBS_SIZE = 3 + 6 + NUM_RAYS   # 117


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
    """Minimum surface distance from point (x,y) to any circle (centre_dist - r)."""
    dx = circles[:, 0] - x
    dy = circles[:, 1] - y
    return jnp.min(jnp.sqrt(dx**2 + dy**2) - circles[:, 2])


def _min_dist_to_boxes(x, y, boxes):
    """Minimum distance from point (x,y) to any AABB surface."""
    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)
    return jnp.min(jax.vmap(_box_dist)(boxes))


# ── Observation ───────────────────────────────────────────────────────────────

@jax.jit
def get_obs(state: EnvState) -> jnp.ndarray:
    people_circles = jnp.stack([
        state.people[:, 0],
        state.people[:, 1],
        jnp.full(NUM_PEOPLE, PEOPLE_RADIUS)
    ], axis=-1)

    all_circles = jnp.concatenate([people_circles, state.obs_circles], axis=0)

    # Forward-facing LiDAR (180°)
    raw_lidar = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )
    inv_lidar = jnp.clip(
        (MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )

    # FIX: rear proximity — 4 rays spanning 180° behind the robot
    # Gives the agent awareness of what is directly behind it.
    rear_fov    = jnp.pi
    rear_theta  = state.theta + jnp.pi   # pointing backwards
    rear_angles = rear_theta - rear_fov * 0.5 + jnp.arange(REAR_RAYS) * (rear_fov / (REAR_RAYS - 1))
    rear_raw    = compute_lidar(
        state.x, state.y, rear_theta,
        all_circles, state.obs_boxes,
        REAR_RAYS, float(rear_fov), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )
    # Single rear proximity scalar: normalised min rear distance
    rear_prox = jnp.clip(
        (MAX_LIDAR_DIST - jnp.min(rear_raw)) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0
    )

    # Pose (normalised to [~-1, 1])
    s_x     = state.x     / ROOM_W
    s_y     = state.y     / ROOM_H
    s_theta = state.theta / jnp.pi

    # Goal
    dx         = state.goal_x - state.x
    dy         = state.goal_y - state.y
    goal_dist  = jnp.sqrt(dx**2 + dy**2)
    goal_angle = jnp.arctan2(dy, dx)
    goal_align = (goal_angle - state.theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    MAX_GOAL_DIST = jnp.sqrt(ROOM_W**2 + ROOM_H**2)

    pose_vec  = jnp.array([s_x, s_y, s_theta])
    state_vec = jnp.array([
        state.v / jnp.maximum(state.max_v, 1e-3),  # [0, 1]
        state.w,                                     # [-1, 1]
        state.max_v / 2.0,                           # [0.1, 1]
        goal_dist  / MAX_GOAL_DIST,                  # [0, 1]
        goal_align / jnp.pi,                         # [-1, 1]
        rear_prox,                                   # [0, 1]  FIX: rear awareness
    ])

    return jnp.concatenate([pose_vec, state_vec, inv_lidar])


# ── Reset ─────────────────────────────────────────────────────────────────────

def _is_safe(x, y, clearance, obs_circles, obs_boxes):
    """Returns True if (x,y) is at least `clearance` from all obstacles and walls."""
    wall_ok = (x > clearance) & (x < ROOM_W - clearance) & \
              (y > clearance) & (y < ROOM_H - clearance)
    cir_ok  = _min_dist_to_circles(x, y, obs_circles) > clearance
    box_ok  = _min_dist_to_boxes(x, y, obs_boxes)    > clearance
    return wall_ok & cir_ok & box_ok


@jax.jit
def reset_env(key: jnp.ndarray):
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)

    max_v  = jax.random.uniform(k1, minval=0.2, maxval=2.0)
    margin = ROBOT_RADIUS + 0.5   # tighter wall margin after spawn safety

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

    # ── Robot spawn — FIX: resample until safe ────────────────────────────────
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

    # ── Goal spawn — FIX: split keys + resample until safe + far from robot ──
    GOAL_CLEARANCE  = GOAL_RADIUS + 0.3
    MIN_GOAL_DIST   = 3.0   # goal must be at least this far from robot

    def _goal_cond(carry):
        gx, gy, k = carry
        too_close = jnp.sqrt((gx - rx)**2 + (gy - ry)**2) < MIN_GOAL_DIST
        return too_close | ~_is_safe(gx, gy, GOAL_CLEARANCE, obs_circles, obs_boxes)

    def _goal_body(carry):
        _, _, k = carry
        k, ka, kb = jax.random.split(k, 3)
        gx = jax.random.uniform(ka, minval=margin, maxval=ROOM_W - margin)
        gy = jax.random.uniform(kb, minval=margin, maxval=ROOM_H - margin)
        return gx, gy, k

    # FIX: use separate keys k5a, k5b for the two branches — not the same key
    k5a, k5b = jax.random.split(k5)
    gx0 = jax.random.uniform(k5a, minval=margin, maxval=ROOM_W - margin)
    gy0 = jax.random.uniform(k5b, minval=margin, maxval=ROOM_H - margin)
    gx, gy, _ = jax.lax.while_loop(_goal_cond, _goal_body, (gx0, gy0, k6))

    # ── People (billiard balls) ───────────────────────────────────────────────
    people_keys = jax.random.split(k7, NUM_PEOPLE)

    def init_person(pkey):
        pk1, pk2, pk3, pk4 = jax.random.split(pkey, 4)
        px    = jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W - 1.0)
        py    = jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H - 1.0)
        angle = jax.random.uniform(pk3, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(pk4, minval=0.4, maxval=1.4)
        vx    = speed * jnp.cos(angle)
        vy    = speed * jnp.sin(angle)
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
    dt = 0.2

    target_v = jnp.clip(action[0], 0.0,  state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0)

    # Midpoint integration
    mid_theta = state.theta + 0.5 * target_w * dt
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    new_x     = state.x + target_v * dt * jnp.cos(mid_theta)
    new_y     = state.y + target_v * dt * jnp.sin(mid_theta)

    # Wall clamp
    new_x = jnp.clip(new_x, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y = jnp.clip(new_y, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    # Update humans
    human_key, _ = jax.random.split(key)
    new_people = update_all_humans(
        state.people, human_key, dt,
        new_x, new_y, new_theta, target_v,
        ROOM_W, ROOM_H, PEOPLE_RADIUS,
        state.obs_circles, state.obs_boxes
    )

    # ── Distances ──────────────────────────────────────────────────────────────
    prev_dist = jnp.sqrt((state.x  - state.goal_x)**2 + (state.y  - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x    - state.goal_x)**2 + (new_y    - state.goal_y)**2)

    # Nearest human (center-to-center)
    dx_p = new_people[:, 0] - new_x
    dy_p = new_people[:, 1] - new_y
    closest_human = jnp.min(jnp.sqrt(dx_p**2 + dy_p**2))

    # Nearest circular obstacle — surface distance from robot centre
    dx_c = state.obs_circles[:, 0] - new_x
    dy_c = state.obs_circles[:, 1] - new_y
    closest_cir = jnp.min(jnp.sqrt(dx_c**2 + dy_c**2) - state.obs_circles[:, 2])

    # Nearest box obstacle — centre-to-surface distance
    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(new_x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(new_y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)
    closest_box = jnp.min(jax.vmap(_box_dist)(state.obs_boxes))

    # Wall clearance
    wall_clearance = jnp.minimum(
        jnp.minimum(new_x, ROOM_W - new_x),
        jnp.minimum(new_y, ROOM_H - new_y)
    ) - ROBOT_RADIUS

    # ── Rewards ────────────────────────────────────────────────────────────────
    progress  = 3.0 * (prev_dist - new_dist)
    step_pen  = -0.004
    smooth    = -0.5 * jnp.abs(target_w - state.w)
    speed_bon = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)
    reward    = progress + step_pen + smooth + speed_bon

    # ── Terminations ───────────────────────────────────────────────────────────
    goal_reached = new_dist < GOAL_RADIUS
    human_collision = closest_human < (ROBOT_RADIUS + PEOPLE_RADIUS)
    # FIX: circle threshold now uses ROBOT_RADIUS (was 0.0, ignored robot body)
    obs_collision   = (closest_cir < ROBOT_RADIUS) | (closest_box < ROBOT_RADIUS)
    wall_collision  = wall_clearance < 0.0
    collision       = human_collision | obs_collision | wall_collision
    timeout         = (state.time_step + 1) >= MAX_STEPS
    done            = goal_reached | collision | timeout

    # FIX: terminal rewards scaled down 10× → value function can bootstrap
    # stably; curriculum of positive/negative is preserved in ratio.
    reward = jnp.where(goal_reached,              20.0, reward)
    reward = jnp.where(collision & ~goal_reached, -7.0, reward)
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
        "closest_human": closest_human,
    }
    return obs, new_state, reward, done, info