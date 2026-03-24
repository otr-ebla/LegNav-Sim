import jax.numpy as jnp
from jax import jit, vmap

exp_clip = 50
eps_divisioni = 1e-3

@jit
def wrap_angle(theta: float) -> float:
    """Wraps the angle to the interval [-pi, pi]."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jit
def get_linear_velocity(theta: float, body_velocity: jnp.ndarray) -> jnp.ndarray:
    """
    Computes linear velocity in the world frame.
    Unrolled math is significantly faster than instantiating a 2x2 matrix and using matmul.
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        c * body_velocity[0] - s * body_velocity[1],
        s * body_velocity[0] + c * body_velocity[1]
    ])

@jit
def compute_edge_closest_point(reference_point: jnp.ndarray, edge: jnp.ndarray):
    """
    Computes the closest point of the edge to the reference point.
    Branchless XLA implementation using jnp.where for NaN padding.
    """
    a = edge[0]
    b = edge[1]
    ap = reference_point - a
    ab = b - a
    
    den = jnp.dot(ab, ab) + 1e-8
    t = jnp.clip(jnp.dot(ap, ab) / den, 0.0, 1.0)
    h = a + t * ab
    
    # jnp.hypot is highly optimized on GPUs for 2D distances
    dist = jnp.hypot(h[0] - reference_point[0], h[1] - reference_point[1])
    
    is_nan = jnp.any(jnp.isnan(edge))
    closest_point = jnp.where(is_nan, jnp.array([1e6, 1e6]), h)
    min_distance = jnp.where(is_nan, 1e6, dist)
    
    return closest_point, min_distance

vectorized_compute_edge_closest_point = vmap(compute_edge_closest_point, in_axes=(None, 0))

@jit
def compute_obstacle_closest_point(reference_point: jnp.ndarray, obstacle: jnp.ndarray) -> jnp.ndarray:
    """Returns the closest point of the obstacle to the reference point."""
    closest_points, min_distances = vectorized_compute_edge_closest_point(reference_point, obstacle)
    best_pt = closest_points[jnp.argmin(min_distances)]
    
    is_nan = jnp.all(jnp.isnan(obstacle))
    return jnp.where(is_nan, jnp.array([1e6, 1e6]), best_pt)

vectorized_compute_obstacle_closest_point = vmap(compute_obstacle_closest_point, in_axes=(None, 0))

@jit
def pairwise_social_force(human_state: jnp.ndarray, other_human_state: jnp.ndarray, parameters: jnp.ndarray, other_human_parameters: jnp.ndarray):
    """
    Computes social force. 
    Branchless implementation: masks out self-interaction mathematically instead of using exact float equality.
    """
    rij = parameters[0] + other_human_parameters[0] + parameters[18] + other_human_parameters[18]
    diff = human_state[:2] - other_human_state[:2]
    dist = jnp.hypot(diff[0], diff[1])
    
    nij = diff / (dist + eps_divisioni)
    real_dist = rij - dist
    tij = jnp.array([-nij[1], nij[0]])
    
    human_linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
    other_human_linear_velocity = get_linear_velocity(other_human_state[4], other_human_state[2:4])
    delta_vij = jnp.dot(other_human_linear_velocity - human_linear_velocity, tij)
    
    # Smooth contact boundary
    contact = jnp.maximum(0.0, real_dist)
    
    force = (
        (parameters[4] * jnp.exp(real_dist / parameters[6]) + parameters[12] * contact) * nij +
        (parameters[8] * jnp.exp(real_dist / parameters[10]) + parameters[13] * contact * delta_vij) * tij
    )
    
    # Hardware multiplexer: If the distance is near zero, it is the same agent. Mask force to 0.0.
    return jnp.where(dist < 1e-4, 0.0, force)

vectorized_pairwise_social_force = vmap(pairwise_social_force, in_axes=(None, 0, None, 0))

@jit
def compute_obstacle_force(human_state: jnp.ndarray, obstacle: jnp.ndarray, parameters: jnp.ndarray):
    """Computes the obstacle force. Branchless execution."""
    diff = human_state[:2] - obstacle
    dist = jnp.hypot(diff[0], diff[1])
    
    niw = diff / (dist + eps_divisioni)
    tiw = jnp.array([-niw[1], niw[0]])
    
    linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
    delta_viw = -jnp.dot(linear_velocity, tiw)
    real_dist = parameters[0] - dist + parameters[18]
    
    contact = jnp.maximum(0.0, real_dist)
    force = (
        (parameters[5] * jnp.exp(real_dist / parameters[7]) + parameters[12] * contact) * niw +
        (-parameters[9] * jnp.exp(real_dist / parameters[11]) - parameters[13] * contact) * delta_viw * tiw
    )
    
    is_nan = jnp.any(jnp.isnan(obstacle))
    return jnp.where(is_nan, 0.0, force)

vectorized_compute_obstacle_force = vmap(compute_obstacle_force, in_axes=(None, 0, None))

@jit
def single_update(idx: int, humans_state: jnp.ndarray, human_goal: jnp.ndarray, parameters: jnp.ndarray, obstacles: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Euler step for a single human, entirely branchless."""
    self_state = humans_state[idx]
    self_parameters = parameters[idx]
    
    linear_velocity = get_linear_velocity(self_state[4], self_state[2:4])
    diff = human_goal - self_state[:2]
    dist = jnp.hypot(diff[0], diff[1])
    
    # Desired force
    raw_desired = self_parameters[1] * (((diff / (dist + 1e-8)) * self_parameters[2]) - linear_velocity) / self_parameters[3]
    desired_force = jnp.where(dist > self_parameters[0], raw_desired, 0.0)
    
    # Social force
    social_force = jnp.sum(vectorized_pairwise_social_force(self_state, humans_state, self_parameters, parameters), axis=0)
    
    # Obstacle force
    closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
    num_real_obstacles = jnp.sum(~jnp.isnan(closest_points[:, 0]))
    raw_obs_force = jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0)
    obstacle_force = raw_obs_force / jnp.maximum(num_real_obstacles, 1.0)
    
    # Torque
    input_force = desired_force + social_force + obstacle_force
    input_force_norm = jnp.hypot(input_force[0], input_force[1])
    input_force_angle = jnp.arctan2(input_force[1], input_force[0])
    
    inertia = (self_parameters[1] * self_parameters[0] * self_parameters[0]) / 2.0
    k_theta = inertia * self_parameters[17] * input_force_norm
    k_omega = inertia * (1.0 + self_parameters[16]) * jnp.sqrt(jnp.maximum(1e-8, (self_parameters[17] * input_force_norm) / self_parameters[16]))
    
    torque = -k_theta * wrap_angle(self_state[4] - input_force_angle) - k_omega * self_state[5]
    torque = jnp.clip(torque, -100.0, 100.0)

    # Global force
    c, s = jnp.cos(self_state[4]), jnp.sin(self_state[4])
    global_force = jnp.array([
        jnp.dot(input_force, jnp.array([c, s])),
        self_parameters[14] * jnp.dot(social_force + obstacle_force, jnp.array([-s, c])) - self_parameters[15] * self_state[3]
    ])
    
    # Integrate position and angles
    new_px = self_state[0] + dt * linear_velocity[0]
    new_py = self_state[1] + dt * linear_velocity[1]
    new_th = wrap_angle(self_state[4] + dt * self_state[5])
    new_om = jnp.clip(self_state[5] + dt * (torque / inertia), -10.0, 10.0)
    
    # Integrate velocities
    new_bvx = self_state[2] + dt * (global_force[0] / self_parameters[1])
    new_bvy = self_state[3] + dt * (global_force[1] / self_parameters[1])
    
    # Smooth velocity projection (replaces lax.cond)
    speed = jnp.hypot(new_bvx, new_bvy)
    scale = self_parameters[2] / jnp.maximum(speed, self_parameters[2])
    new_bvx *= scale
    new_bvy *= scale

    # Compile the final state
    updated_human_state = jnp.array([new_px, new_py, new_bvx, new_bvy, new_th, new_om])
    
    return updated_human_state

# obstacles in_axes=None: all agents share the same (num_obs_groups, 4, 2, 2) array.
# Previously in_axes=0 forced the caller to tile obstacles (NUM_PEOPLE+1) times —
# a 13x memory waste with zero behavioural difference since all tiles were identical.
vectorized_single_update = vmap(single_update, in_axes=(0, None, 0, None, None, None))

@jit
def step(humans_state: jnp.ndarray, humans_goal: jnp.ndarray, parameters: jnp.ndarray, obstacles: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Executes one time step (dt) for the humans' state using the Headed Social Force Model (HSFM).
    obstacles: (num_obs_groups, 4, 2, 2) — shared by all agents (not tiled per-agent).
    """
    # Pre-build index array: static shape → XLA treats as constant, not dynamic alloc.
    indices = jnp.arange(humans_state.shape[0])
    return vectorized_single_update(indices, humans_state, humans_goal, parameters, obstacles, dt)