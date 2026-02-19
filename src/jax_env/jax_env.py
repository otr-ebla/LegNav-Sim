import jax
import jax.numpy as jnp
from flax import struct
from jax_physics import compute_lidar, get_ray_wall_intersections, get_ray_circles_intersections
from jax_humans import update_all_humans

# Environment Constants
MAX_STEPS = 1000
NUM_RAYS = 108
NUM_PEOPLE = 10
NUM_OBSTACLES = 12
ROOM_W = 12.0
ROOM_H = 12.0
ROBOT_RADIUS = 0.2
PEOPLE_RADIUS = 0.2
MAX_LIDAR_DIST = 15.0
FOV = jnp.pi  # Assuming 360 degrees for 108 rays

@struct.dataclass
class EnvState:
    x: jnp.float32
    y: jnp.float32
    theta: jnp.float32
    v: jnp.float32
    w: jnp.float32
    goal_x: jnp.float32
    goal_y: jnp.float32
    max_v: jnp.float32
    people: jnp.ndarray 
    obstacles: jnp.ndarray 
    time_step: jnp.int32

@jax.jit
def get_obs(state: EnvState) -> jnp.ndarray:
    """
    Computes the observation vector. 
    Matches the old Dict space but flattened for maximum speed.
    """
    # 1. Lidar
    # Extract circles from people (px, py, radius) and circular obstacles
    people_circles = jnp.stack([state.people[:, 0], state.people[:, 1], jnp.full(NUM_PEOPLE, PEOPLE_RADIUS)], axis=-1)
    
    # Assuming for simplicity all obstacles here are circles [cx, cy, r]
    obs_circles = state.obstacles[:, 1:4] 
    
    all_circles = jnp.concatenate([people_circles, obs_circles], axis=0)
    
    raw_lidar = compute_lidar(state.x, state.y, state.theta, all_circles, NUM_RAYS, FOV, MAX_LIDAR_DIST, ROOM_W, ROOM_H)
    
    # Normalize lidar
    inv_lidar = jnp.clip((MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - 0.12), 0.0, 1.0)
    
    # 2. Pose
    s_x = state.x / ROOM_W
    s_y = state.y / ROOM_H
    s_theta = state.theta / jnp.pi
    
    # 3. State Vector
    dx = state.goal_x - state.x
    dy = state.goal_y - state.y
    goal_distance = jnp.sqrt(dx**2 + dy**2)
    angle_to_goal = jnp.arctan2(dy, dx)
    goal_error_alignment = (angle_to_goal - state.theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    state_vec = jnp.array([state.v, state.w, state.max_v, goal_distance, goal_error_alignment])
    pose_vec = jnp.array([s_x, s_y, s_theta])
    
    # Return a flat array: [pose(3) + state(5) + lidar(108)] = 116 elements
    return jnp.concatenate([pose_vec, state_vec, inv_lidar])

@jax.jit
def reset_env(key: jnp.ndarray) -> tuple[jnp.ndarray, EnvState]:
    """
    Resets the environment state using pure functional randomness.
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # Randomize max velocity
    max_v = jax.random.uniform(k1, minval=0.2, maxval=2.0)
    
    # Randomize robot pose safely within margins
    margin = ROBOT_RADIUS + 0.2
    rx = jax.random.uniform(k2, minval=margin, maxval=ROOM_W - margin)
    ry = jax.random.uniform(k3, minval=margin, maxval=ROOM_H - margin)
    rtheta = jax.random.uniform(k4, minval=0.0, maxval=2 * jnp.pi)
    
    # Place goal on the opposite side of the room to ensure distance
    gx = jnp.where(rx < ROOM_W/2, jax.random.uniform(k5, minval=ROOM_W/2 + 1.0, maxval=ROOM_W - 1.0), jax.random.uniform(k5, minval=1.0, maxval=ROOM_W/2 - 1.0))
    gy = jnp.where(ry < ROOM_H/2, jax.random.uniform(k5, minval=ROOM_H/2 + 1.0, maxval=ROOM_H - 1.0), jax.random.uniform(k5, minval=1.0, maxval=ROOM_H/2 - 1.0))

    # Initialize people (simplified grid/random distribution)
    # [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
    people_keys = jax.random.split(k5, NUM_PEOPLE)
    
    def init_person(pkey):
        pk1, pk2, pk3 = jax.random.split(pkey, 3)
        px = jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W - 1.0)
        py = jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H - 1.0)
        angle = jax.random.uniform(pk3, minval=0.0, maxval=2 * jnp.pi)
        speed = 1.0
        vx = speed * jnp.cos(angle)
        vy = speed * jnp.sin(angle)
        return jnp.array([px, py, vx, vy, angle, 0.0, -1.0, speed])

    people = jax.vmap(init_person)(people_keys)

    # Initialize static dummy obstacles (zeros for now, can be randomized similarly)
    obstacles = jnp.zeros((NUM_OBSTACLES, 5)) 

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

@jax.jit
def step_env(key: jnp.ndarray, state: EnvState, action: jnp.ndarray) -> tuple[jnp.ndarray, EnvState, jnp.float32, jnp.bool_, dict]:
    """
    Steps the environment forward by dt.
    """
    dt = 0.1
    target_v = jnp.clip(action[0], 0.0, state.max_v)
    target_w = jnp.clip(action[1], -1.0, 1.0) # RobotConfig.MAX_W
    
    # Kinematics
    new_theta = (state.theta + target_w * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_x = state.x + target_v * dt * jnp.cos(new_theta)
    new_y = state.y + target_v * dt * jnp.sin(new_theta)
    
    # Update Humans
    human_key, _ = jax.random.split(key)
    new_people = update_all_humans(state.people, human_key, dt, new_x, new_y, new_theta, target_v, ROOM_W, ROOM_H, PEOPLE_RADIUS)
    
    # Distances
    prev_dist = jnp.sqrt((state.x - state.goal_x)**2 + (state.y - state.goal_y)**2)
    new_dist = jnp.sqrt((new_x - state.goal_x)**2 + (new_y - state.goal_y)**2)
    
    # Nearest human calculation
    dx_people = new_people[:, 0] - new_x
    dy_people = new_people[:, 1] - new_y
    dist_people = jnp.sqrt(dx_people**2 + dy_people**2)
    closest_human_dist = jnp.min(dist_people)
    
    # Rewards
    progress_reward = 2.5 * (prev_dist - new_dist)
    step_penalty = -0.005
    jerk_penalty = -1.5 * ((target_w - state.w)**2)
    
    reward = progress_reward + step_penalty + jerk_penalty
    
    # Terminations
    goal_reached = new_dist <= 0.3
    collision = closest_human_dist < (ROBOT_RADIUS + PEOPLE_RADIUS)
    timeout = state.time_step + 1 >= MAX_STEPS
    
    done = goal_reached | collision | timeout
    
    reward = jnp.where(goal_reached, 200.0, reward)
    reward = jnp.where(collision, -70.0, reward)
    reward = jnp.where(timeout & ~goal_reached & ~collision, -5.0, reward)
    
    new_state = state.replace(
        x=new_x, y=new_y, theta=new_theta,
        v=target_v, w=target_w,
        people=new_people,
        time_step=state.time_step + 1
    )
    
    obs = get_obs(new_state)
    info = {"discount": jnp.where(done, 0.0, 1.0)} # Required for JAX RL frameworks
    
    return obs, new_state, reward, done, info