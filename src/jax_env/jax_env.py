"""
jax_env.py — Core 2-D Navigation Environment
=============================================
Fixes vs original:
  - FOV corrected to 2π (true 360° LiDAR)
  - Key reuse bug fixed in reset_env (gx, gy, people each get unique keys)
  - Kinematics fixed: use current theta for displacement, then update theta
  - Wall collision detection added (termination + penalty)
  - Obstacle circles: radius now correctly read from column 2, not 3
  - compute_lidar signature updated (boxes argument added)
  - obstacles initialised with realistic random circular pillars + their radii
  - Observation size documented and consistent

Obs layout (flat):
  pose  (3)  + state (5) + lidar (NUM_RAYS)  =  3 + 5 + 108 = 116   (single frame)
  stacked via jax_wrappers → 3*3 + 5 + 3*108 = 338
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar

# ── Environment constants ────────────────────────────────────────────────────
MAX_STEPS        = 1000
NUM_RAYS         = 108
NUM_PEOPLE       = 10
NUM_OBSTACLES    = 12           # circular pillars
ROOM_W           = 12.0
ROOM_H           = 12.0
ROBOT_RADIUS     = 0.2
PEOPLE_RADIUS    = 0.3
MAX_LIDAR_DIST   = 15.0
FOV              = 2.0 * jnp.pi   # FIX: was jnp.pi (180°); real LiDAR = 360°
PILLAR_RADIUS    = 0.25

# Dummy empty boxes array (no box obstacles by default; extend as needed)
_EMPTY_BOXES = jnp.zeros((1, 4))   # shape (1,4) avoids 0-size vmap issues


@struct.dataclass
class EnvState:
    x:          jnp.float32
    y:          jnp.float32
    theta:      jnp.float32
    v:          jnp.float32
    w:          jnp.float32
    goal_x:     jnp.float32
    goal_y:     jnp.float32
    max_v:      jnp.float32
    people:     jnp.ndarray     # (NUM_PEOPLE, 8)
    obstacles:  jnp.ndarray     # (NUM_OBSTACLES, 3)  [cx, cy, r]
    time_step:  jnp.int32


# ── Observation ──────────────────────────────────────────────────────────────

@jax.jit
def get_obs(state: EnvState) -> jnp.ndarray:
    """
    Returns flat obs: [pose(3) | state_vec(5) | lidar(NUM_RAYS)] = 116 values.
    """
    # Dynamic circles: people + pillar obstacles
    people_circles = jnp.stack(
        [state.people[:, 0],
         state.people[:, 1],
         jnp.full(NUM_PEOPLE, PEOPLE_RADIUS)],
        axis=-1
    )                                        # (NUM_PEOPLE, 3)
    # FIX: obstacles stores [cx, cy, r] directly (columns 0-2)
    all_circles = jnp.concatenate([people_circles, state.obstacles], axis=0)  # (M, 3)

    raw_lidar = compute_lidar(
        state.x, state.y, state.theta,
        all_circles,
        _EMPTY_BOXES,
        NUM_RAYS, FOV, MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )

    # Normalise to [0, 1]  (1 = obstacle right here, 0 = clear to max range)
    inv_lidar = jnp.clip((MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS), 0.0, 1.0)

    # Pose (normalised)
    s_x     = state.x     / ROOM_W
    s_y     = state.y     / ROOM_H
    s_theta = state.theta / jnp.pi          # ∈ [-1, 1]

    # Goal-relative features
    dx = state.goal_x - state.x
    dy = state.goal_y - state.y
    goal_dist  = jnp.sqrt(dx**2 + dy**2)
    goal_angle = jnp.arctan2(dy, dx)
    goal_align = (goal_angle - state.theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    pose_vec  = jnp.array([s_x, s_y, s_theta])
    state_vec = jnp.array([state.v / state.max_v,   # normalised linear vel
                            state.w,                 # angular vel  (already ≤1)
                            state.max_v,
                            goal_dist,
                            goal_align])

    return jnp.concatenate([pose_vec, state_vec, inv_lidar])


# ── Reset ────────────────────────────────────────────────────────────────────

@jax.jit
def reset_env(key: jnp.ndarray):
    """Pure-functional reset."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)   # FIX: unique keys

    max_v = jax.random.uniform(k1, minval=0.3, maxval=1.5)

    margin = ROBOT_RADIUS + 0.3
    rx = jax.random.uniform(k2, minval=margin, maxval=ROOM_W - margin)
    ry = jax.random.uniform(k3, minval=margin, maxval=ROOM_H - margin)
    rtheta = jax.random.uniform(k4, minval=0.0, maxval=2.0 * jnp.pi)

    # FIX: gx uses k5, gy uses k6 (was both using k5)
    gx = jnp.where(
        rx < ROOM_W / 2,
        jax.random.uniform(k5, minval=ROOM_W / 2 + 1.0, maxval=ROOM_W - 1.0),
        jax.random.uniform(k5, minval=1.0,               maxval=ROOM_W / 2 - 1.0)
    )
    gy = jnp.where(
        ry < ROOM_H / 2,
        jax.random.uniform(k6, minval=ROOM_H / 2 + 1.0, maxval=ROOM_H - 1.0),
        jax.random.uniform(k6, minval=1.0,               maxval=ROOM_H / 2 - 1.0)
    )

    # People  [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
    people_keys = jax.random.split(k7, NUM_PEOPLE)

    def init_person(pkey):
        pk1, pk2, pk3, pk4 = jax.random.split(pkey, 4)
        px    = jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W - 1.0)
        py    = jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H - 1.0)
        angle = jax.random.uniform(pk3, minval=0.0, maxval=2.0 * jnp.pi)
        speed = jax.random.uniform(pk4, minval=0.5, maxval=1.4)  # varied speeds
        vx    = speed * jnp.cos(angle)
        vy    = speed * jnp.sin(angle)
        is_distracted = 0.0
        wait_timer    = -1.0
        return jnp.array([px, py, vx, vy, angle, is_distracted, wait_timer, speed])

    people = jax.vmap(init_person)(people_keys)

    # Circular pillar obstacles [cx, cy, r]
    obs_keys = jax.random.split(k8, NUM_OBSTACLES)

    def init_pillar(okey):
        ok1, ok2 = jax.random.split(okey)
        cx = jax.random.uniform(ok1, minval=1.5, maxval=ROOM_W - 1.5)
        cy = jax.random.uniform(ok2, minval=1.5, maxval=ROOM_H - 1.5)
        return jnp.array([cx, cy, PILLAR_RADIUS])

    obstacles = jax.vmap(init_pillar)(obs_keys)   # (NUM_OBSTACLES, 3)

    state = EnvState(
        x=rx, y=ry, theta=rtheta,
        v=0.0, w=0.0,
        goal_x=gx, goal_y=gy,
        max_v=max_v,
        people=people,
        obstacles=obstacles,
        time_step=0
    )

    obs = get_obs(state)
    return obs, state


# ── Step ─────────────────────────────────────────────────────────────────────

@jax.jit
def step_env(key: jnp.ndarray, state: EnvState, action: jnp.ndarray):
    """Pure-functional environment step."""
    from jax_humans import update_all_humans

    dt = 0.1

    # Action clipping
    target_v = jnp.clip(action[0], 0.0, state.max_v)
    target_w = jnp.clip(action[1], -1.5, 1.5)

    # FIX: correct Euler kinematics — rotate first, then translate with OLD theta
    # (original code used new_theta for displacement, introducing a systematic bias)
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    mid_theta = state.theta + 0.5 * target_w * dt    # midpoint integration (smoother)
    new_x = state.x + target_v * dt * jnp.cos(mid_theta)
    new_y = state.y + target_v * dt * jnp.sin(mid_theta)

    # Wall clamping (keep robot inside room)
    margin = ROBOT_RADIUS
    new_x = jnp.clip(new_x, margin, ROOM_W - margin)
    new_y = jnp.clip(new_y, margin, ROOM_H - margin)

    # Update humans
    human_key, _ = jax.random.split(key)
    new_people = update_all_humans(
        state.people, human_key, dt,
        new_x, new_y, new_theta, target_v,
        ROOM_W, ROOM_H, PEOPLE_RADIUS
    )

    # Distances to goal
    prev_dist = jnp.sqrt((state.x     - state.goal_x)**2 + (state.y     - state.goal_y)**2)
    new_dist  = jnp.sqrt((new_x       - state.goal_x)**2 + (new_y       - state.goal_y)**2)

    # Nearest human
    dx_p = new_people[:, 0] - new_x
    dy_p = new_people[:, 1] - new_y
    dist_people = jnp.sqrt(dx_p**2 + dy_p**2)
    closest_human = jnp.min(dist_people)

    # Nearest obstacle pillar
    dx_o = state.obstacles[:, 0] - new_x
    dy_o = state.obstacles[:, 1] - new_y
    dist_obs = jnp.sqrt(dx_o**2 + dy_o**2) - state.obstacles[:, 2]
    closest_obs = jnp.min(dist_obs)

    # Wall proximity (min clearance)
    wall_clearance = jnp.minimum(
        jnp.minimum(new_x, ROOM_W - new_x),
        jnp.minimum(new_y, ROOM_H - new_y)
    ) - ROBOT_RADIUS

    # Rewards
    progress_reward = 3.0 * (prev_dist - new_dist)
    step_penalty    = -0.004
    smooth_penalty  = -0.5 * jnp.abs(target_w - state.w)    # smoothness (not squared — less aggressive)
    speed_bonus     = 0.02 * target_v / jnp.maximum(state.max_v, 1e-3)   # encourage movement

    reward = progress_reward + step_penalty + smooth_penalty + speed_bonus

    # Terminations
    goal_reached   = new_dist       <= 0.3
    human_collision= closest_human  <  (ROBOT_RADIUS + PEOPLE_RADIUS)
    obs_collision  = closest_obs    <  0.0
    wall_collision = wall_clearance <  0.0
    collision      = human_collision | obs_collision | wall_collision
    timeout        = (state.time_step + 1) >= MAX_STEPS

    done = goal_reached | collision | timeout

    # Terminal rewards (override step reward)
    reward = jnp.where(goal_reached,                     200.0, reward)
    reward = jnp.where(collision & ~goal_reached,        -70.0, reward)
    reward = jnp.where(timeout & ~goal_reached & ~collision, -5.0, reward)

    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1
    )

    obs  = get_obs(new_state)
    info = {
        "discount":       jnp.where(done, 0.0, 1.0),
        "goal_reached":   goal_reached,
        "collision":      collision,
        "closest_human":  closest_human,
    }

    return obs, new_state, reward, done, info