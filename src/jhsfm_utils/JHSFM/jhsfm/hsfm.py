import jax.numpy as jnp
from jax import jit, vmap, lax, debug

# TODO: Add human grid cell filtering for faster step computation

exp_clip = 50
eps_divisioni = 1e-3

@jit
def wrap_angle(theta:float) -> float:
    """
    This function wraps the angle to the interval [-pi, pi]
    
    args:
    - theta: angle to be wrapped
    
    output:
    - wrapped_theta: angle wrapped to the interval [-pi, pi]
    """
    wrapped_theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return wrapped_theta

@jit
def get_linear_velocity(theta:float, body_velocity: jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the linear velocity of the agent in the world frame
    
    args:
    - theta: angle of the agent in the world frame
    - body_velocity: velocity of the agent in its body frame
    
    output: 
    - linear_velocity: velocity of the agent in the world frame
    """
    rotational_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    linear_velocity = jnp.matmul(rotational_matrix, body_velocity)
    return linear_velocity

@jit
def compute_edge_closest_point(reference_point:jnp.ndarray, edge:jnp.ndarray):
    """
    This function computes the closest point of the edge to the reference point and confronts it with the current closest point and min dist to the obstacle.
    
    args:
    - reference_point: shape is (2,) in the form (px, py)
    - edge: shape is (2, 2) where each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)

    output:
    - closest_point: shape is (2,) in the form (cx, cy)
    - min_distance: min distance to the closest point
    """
    @jit
    def _not_nan(reference_point:jnp.ndarray, edge:jnp.ndarray):
        # a = edge[0]
        # b = edge[1]
        # t = (jnp.dot(reference_point - a, b - a)) / (jnp.linalg.norm(b - a) ** 2)
        # t_lb = lax.cond(t>0,lambda x: x,lambda x: 0.,t)
        # t_star = lax.cond(t_lb<1,lambda x: x,lambda x: 1.,t_lb)
        # h = a + t_star * (b - a)
        # dist = jnp.linalg.norm(h - reference_point)
        a = edge[0]
        b = edge[1]
        ap = reference_point - a
        ab = b - a
        den = jnp.dot(ab, ab) + 1e-8
        t = jnp.dot(ap, ab) / den       # scalar
        t = jnp.clip(t, 0.0, 1.0)
        h = a + t * ab
        dist = jnp.linalg.norm(h - reference_point)

        return h, dist
    closest_point, min_distance = lax.cond(
        jnp.any(jnp.isnan(edge)), # In case the edge is a dummy edge (NaN), closest point and min distance should remain the current ones
        lambda _: (jnp.array([jnp.nan, jnp.nan]), jnp.float32(1_000_000.)),
        lambda _: _not_nan(reference_point, edge),
        None)
    return closest_point, min_distance
vectorized_compute_edge_closest_point = vmap(compute_edge_closest_point, in_axes=(None, 0))

@jit
def compute_obstacle_closest_point(reference_point:jnp.ndarray, obstacle:jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the closest point of the obstacle to the reference point
    
    args:
    - reference_point: shape is (2,) in the form (px, py)
    - obstacle: shape is (n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)

    output:
    - closest_point: shape is (2,) in the form (cx, cy)
    """
    @jit
    def _not_nan(reference_point:jnp.ndarray, obstacle:jnp.ndarray):
        closest_points, min_distances = vectorized_compute_edge_closest_point(reference_point, obstacle)
        return closest_points[jnp.argmin(min_distances)]
    closest_point = lax.cond(
        jnp.all(jnp.isnan(obstacle)), # In case the obstacle is a dummy obstacle (NaN), closest point should be nan
        lambda _: jnp.full((2,), jnp.nan),
        lambda _: _not_nan(reference_point, obstacle),
        None)
    return closest_point 
vectorized_compute_obstacle_closest_point = vmap(compute_obstacle_closest_point, in_axes=(None, 0))

@jit
def pairwise_social_force(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray):
    """
    This function computes the social force between a pair of humans

    args:
    - human_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    - other_humans_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    - parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)
    - other_humans_parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)

    output:
    - social_force: shape is (2,) in the form (fx, fy)
    """
    @jit
    def compute_social_force(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray):
        rij = parameters[0] + other_human_parameters[0] + parameters[18] + other_human_parameters[18]
        diff = human_state[:2] - other_human_state[:2]
        dist = jnp.linalg.norm(diff)
        nij = diff / (dist+eps_divisioni)
        real_dist = rij - dist
        tij = jnp.array([-nij[1], nij[0]])
        human_linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
        other_human_linear_velocity = get_linear_velocity(other_human_state[4], other_human_state[2:4])
        delta_vij = jnp.dot(other_human_linear_velocity - human_linear_velocity, tij)
        pairwise_social_force = (parameters[4] * jnp. exp(real_dist / parameters[6]) + parameters[12] * jnp.max(jnp.array([0, real_dist]))) * nij + (parameters[8] * jnp. exp(real_dist / parameters[10]) + parameters[13] * jnp.max(jnp.array([0, real_dist])) * delta_vij) * tij
        #pairwise_social_force = (parameters[4] * jnp.exp(jnp.clip(real_dist / parameters[6], -exp_clip, exp_clip)) + parameters[12] * real_dist) * nij + (parameters[8] * jnp.exp(jnp.clip(real_dist / parameters[10],-exp_clip,exp_clip)) + parameters[13] * real_dist * delta_vij) * tij
        return pairwise_social_force
    pairwise_social_force = lax.cond(
        jnp.all(human_state == other_human_state), # In case the human is the same as the other human, social force should not be computed
        lambda _: jnp.zeros((2,)),
        lambda _: compute_social_force(human_state, other_human_state, parameters, other_human_parameters),
        None)
    return pairwise_social_force
vectorized_pairwise_social_force = vmap(pairwise_social_force, in_axes=(None, 0, None, 0))

@jit
def compute_obstacle_force(human_state:jnp.ndarray, obstacle:jnp.ndarray, parameters:jnp.ndarray):
    """
    This function computes the obstacle force between a human and an obstacle.
    
    args:
    - human_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    - obstacle: shape is (2,) in the form (ox, oy)
    - parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)

    output:
    - obstacle_force: shape is (2,) in the form (fx, fy
    """
    @jit
    def _not_nan(human_state:jnp.ndarray, obstacle:jnp.ndarray, parameters:jnp.ndarray):
        diff = human_state[:2] - obstacle
        dist = jnp.linalg.norm(diff)
        niw = diff / (dist+eps_divisioni)
        tiw = jnp.array([-niw[1], niw[0]])
        linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
        delta_viw = - jnp.dot(linear_velocity, tiw)
        real_dist = parameters[0] - dist + parameters[18]
        obstacle_force = lax.cond(real_dist > 0, lambda x: x * (parameters[5] * jnp. exp(real_dist / parameters[7]) + parameters[12] * real_dist) * niw + (-parameters[9] * jnp. exp(real_dist / parameters[11]) - parameters[13] * real_dist) * delta_viw * tiw, lambda x: x * (parameters[5] * jnp. exp(real_dist / parameters[7])) * niw + (-parameters[9] * jnp. exp(real_dist / parameters[11])) * delta_viw * tiw, jnp.ones((2,)))
        #obstacle_force = lax.cond(real_dist > 0, lambda x: x * (parameters[5] * jnp.exp(jnp.clip(real_dist / parameters[7], -exp_clip, exp_clip)) + parameters[12] * real_dist) * niw + (-parameters[9] * jnp.exp(jnp.clip(real_dist / parameters[11], -exp_clip, exp_clip)) - parameters[13] * real_dist) * delta_viw * tiw, lambda x: x * (parameters[5] * jnp.exp(jnp.clip(real_dist / parameters[7], -exp_clip, exp_clip))) * niw + (-parameters[9] * jnp.exp(jnp.clip(real_dist / parameters[11], -exp_clip, exp_clip))) * delta_viw * tiw, jnp.ones((2,)))
        return obstacle_force
    obstacle_force = lax.cond(
        jnp.any(jnp.isnan(obstacle)), # In case the obstacle is a dummy obstacle (NaN), obstacle force should be zero (no real obstacle, just padding)
        lambda _: jnp.zeros((2,)),
        lambda _: _not_nan(human_state, obstacle, parameters),
        None)
    return obstacle_force
vectorized_compute_obstacle_force = vmap(compute_obstacle_force, in_axes=(None, 0, None))

@jit
def single_update(idx:int, humans_state:jnp.ndarray, human_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for a single human using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - idx: human index in the state, goal and parameter vectors
    - humans_state: shape is (n_humans, 6) in the form is (px, py, bvx, bvy, theta, omega)
    - humans_goal: shape is (2,) in the form (gx, gy)
    - parameters: shape is (n_humans, 19) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_human_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    """
    self_state = humans_state[idx]
    self_parameters = parameters[idx]
    # Desired force computation
    linear_velocity = get_linear_velocity(self_state[4], self_state[2:4])
    diff = human_goal - self_state[:2]
    dist = jnp.linalg.norm(diff)
    desired_force =  lax.cond(
        dist > self_parameters[0],
        lambda _: (self_parameters[1] * (((diff / dist) * self_parameters[2]) - linear_velocity) / self_parameters[3]),
        lambda _: jnp.zeros((2,)),
        None)
    # Social force computation
    # social_force = lax.fori_loop(
    #     0, 
    #     len(humans_state), 
    #     lambda j, acc: lax.cond(
    #         j != idx, 
    #         lambda acc: acc + pairwise_social_force(self_state, humans_state[j], self_parameters, parameters[j]), 
    #         lambda acc: acc, 
    #         acc), 
    #     jnp.zeros((2,)))
    social_force = jnp.sum(vectorized_pairwise_social_force(self_state, humans_state, self_parameters, parameters), axis=0)
    # Obstacle force computation
    closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
    num_real_obstacles = jnp.sum(~jnp.isnan(closest_points[:,0]))
    obstacle_force = lax.cond(
        num_real_obstacles > 0,
        lambda _: jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0) / num_real_obstacles,
        lambda _: jnp.zeros((2,)),
        None
    )
    # Torque computation
    input_force = desired_force + social_force + obstacle_force
    input_force_norm = jnp.linalg.norm(input_force)
    #input_force_norm = jnp.min(jnp.array([input_force_norm, 100 * self_parameters[1]])) # Limit force to avoid numerical issues
    input_force_angle = jnp.arctan2(input_force[1], input_force[0])
    inertia = (self_parameters[1] * self_parameters[0] * self_parameters[0]) / 2
    k_theta = inertia * self_parameters[17] * input_force_norm
    k_omega = inertia * (1 + self_parameters[16]) * jnp.sqrt((self_parameters[17] * input_force_norm) / self_parameters[16])
    torque = - k_theta * wrap_angle(self_state[4] - input_force_angle) - k_omega * self_state[5]
    torque = jnp.clip(torque, -100.0, 100.0)

    # Global force computation
    global_force = jnp.array([
        jnp.dot(input_force, jnp.array([jnp.cos(self_state[4]), jnp.sin(self_state[4])])),
        self_parameters[14] * jnp.dot(social_force + obstacle_force, jnp.array([-jnp.sin(self_state[4]), jnp.cos(self_state[4])])) - self_parameters[15] * self_state[3]])
    # Update
    updated_human_state = jnp.zeros((6,))
    updated_human_state = updated_human_state.at[0].set(self_state[0] + dt * linear_velocity[0])
    updated_human_state = updated_human_state.at[1].set(self_state[1] + dt * linear_velocity[1])
    updated_human_state = updated_human_state.at[4].set(wrap_angle(self_state[4] + dt * self_state[5]))
    updated_human_state = updated_human_state.at[2].set(self_state[2] + dt * (global_force[0] / self_parameters[1]))
    updated_human_state = updated_human_state.at[3].set(self_state[3] + dt * (global_force[1] / self_parameters[1]))
    updated_human_state = updated_human_state.at[5].set(jnp.clip(self_state[5] + dt * (torque / inertia), -10.0, 10.0))

    # Bound body velocity
    updated_human_state = updated_human_state.at[2:4].set(
        lax.cond(
            jnp.linalg.norm(updated_human_state[2:4]) > self_parameters[2], 
            lambda x: (x / jnp.linalg.norm(x)) * self_parameters[2], 
            lambda x: x, 
            updated_human_state[2:4]
        )
    )
    # DEBUGGING
    # debug.print("\n")
    # debug.print("jax.debug.print(closest_points) -> {x}", x=closest_points)
    # debug.print("jax.debug.print(min_distances) -> {x}", x=min_distances)
    # debug.print("jax.debug.print(torque) -> {x}", x=torque)
    # debug.print("jax.debug.print(input_force) -> {x}", x=input_force)
    # debug.print("jax.debug.print(desired_force) -> {x}", x=desired_force)
    # debug.print("jax.debug.print(social_force) -> {x}", x=social_force)
    # debug.print("jax.debug.print(obstacle_force) -> {x}", x=obstacle_force)
    # debug.print("jax.debug.print(global_force) -> {x}", x=global_force)
    # debug.print("jax.debug.print(updated_human_state) -> {x}", x=updated_human_state)
    updated_human_state = jnp.nan_to_num(updated_human_state, nan=0.0, posinf=1e3, neginf=-1e3)

    return updated_human_state
vectorized_single_update = vmap(single_update, in_axes=(0, None, 0, None, 0, None))

@jit
def step(humans_state:jnp.ndarray, humans_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for the humans' state using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
    - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
    - parameters: shape is (n_humans, 19) where each row is (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_humans, n_obstacles, n_edges, 2, 2) where each human can be assigned a different set of obstacles. Each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
    """
    updated_humans_state = vectorized_single_update(jnp.arange(len(humans_state)), humans_state, humans_goal, parameters, obstacles, dt)
    return updated_humans_state
