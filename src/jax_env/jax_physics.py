import jax
import jax.numpy as jnp

# =============================================================================
# VECTORIZED PHYSICS ENGINE
# =============================================================================

@jax.jit
def get_ray_wall_intersections(x0: jnp.float32, y0: jnp.float32, angles: jnp.ndarray, room_w: jnp.float32, room_h: jnp.float32) -> jnp.ndarray:
    """
    Calculates intersections between N rays and the 4 room walls.
    angles: array of shape (N,) containing ray angles.
    Returns: array of shape (N,) with the minimum distance to a wall for each ray.
    """
    dx = jnp.cos(angles)
    dy = jnp.sin(angles)
    
    # Avoid division by zero using a tiny epsilon
    eps = 1e-7
    dx = jnp.where(jnp.abs(dx) < eps, eps, dx)
    dy = jnp.where(jnp.abs(dy) < eps, eps, dy)

    # Time to intersection (t) for all 4 walls for all N rays
    t_left   = (0.0 - x0) / dx
    t_right  = (room_w - x0) / dx
    t_bottom = (0.0 - y0) / dy
    t_top    = (room_h - y0) / dy

    # A valid intersection must be strictly in front of the ray (t > 0)
    # If invalid, we set the distance to infinity
    t_left   = jnp.where(t_left > 0, t_left, jnp.inf)
    t_right  = jnp.where(t_right > 0, t_right, jnp.inf)
    t_bottom = jnp.where(t_bottom > 0, t_bottom, jnp.inf)
    t_top    = jnp.where(t_top > 0, t_top, jnp.inf)

    # The actual distance is the minimum of the valid wall intersections
    min_dist = jnp.minimum(
        jnp.minimum(t_left, t_right),
        jnp.minimum(t_bottom, t_top)
    )
    
    return min_dist

@jax.vmap
def _intersect_ray_circle(ray_x: float, ray_y: float, dx: float, dy: float, cx: float, cy: float, r: float):
    """
    Core math for a single ray-circle intersection.
    Decorated with @jax.vmap to automatically broadcast across arrays.
    """
    fx = ray_x - cx
    fy = ray_y - cy
    
    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - (r * r)
    disc = b * b - 4.0 * c
    
    # If discriminant is negative, no intersection (return infinity)
    # jnp.where(condition, true_value, false_value)
    sqrt_disc = jnp.sqrt(jnp.maximum(0.0, disc))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    
    # We want the smallest positive t
    valid_t1 = jnp.where(t1 > 0, t1, jnp.inf)
    valid_t2 = jnp.where(t2 > 0, t2, jnp.inf)
    
    min_t = jnp.minimum(valid_t1, valid_t2)
    
    # If disc < 0, override with infinity
    return jnp.where(disc < 0, jnp.inf, min_t)

@jax.jit
def get_ray_circles_intersections(x0: jnp.float32, y0: jnp.float32, angles: jnp.ndarray, circles: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates intersections between N rays and M circles (people or round obstacles).
    angles: shape (N,)
    circles: shape (M, 3) where columns are [cx, cy, radius]
    Returns: array of shape (N,) with the closest circle intersection per ray.
    """
    dx = jnp.cos(angles)
    dy = jnp.sin(angles)
    
    # We need to compute an N x M matrix of intersections.
    # jax.vmap needs to know which axes to map over.
    # First vmap maps over the N rays.
    # Second vmap maps over the M circles.
    
    # Define a function that intersects 1 ray against M circles
    intersect_1_ray_M_circles = jax.vmap(
        _intersect_ray_circle,
        in_axes=(None, None, None, None, 0, 0, 0) # Map over circles (cx, cy, r)
    )
    
    # Define a function that maps the previous function over N rays
    intersect_N_rays_M_circles = jax.vmap(
        intersect_1_ray_M_circles,
        in_axes=(None, None, 0, 0, None, None, None) # Map over rays (dx, dy)
    )
    
    cx = circles[:, 0]
    cy = circles[:, 1]
    r  = circles[:, 2]
    
    # distances shape: (N, M) -> N rays, M distances
    distances = intersect_N_rays_M_circles(x0, y0, dx, dy, cx, cy, r)
    
    # For each ray (axis 1), find the closest circle
    min_distances = jnp.min(distances, axis=1)
    
    return min_distances

@jax.jit
def compute_lidar(x: jnp.float32, y: jnp.float32, theta: jnp.float32, circles: jnp.ndarray, num_rays: int, fov: float, max_dist: float, room_w: float, room_h: float) -> jnp.ndarray:
    """
    Full Lidar sweep computation.
    """
    start_angle = theta - (fov / 2.0)
    
    # Generate array of angles: shape (N,)
    angles = start_angle + jnp.arange(num_rays) * (fov / (num_rays - 1))
    
    # 1. Intersect with walls
    wall_dists = get_ray_wall_intersections(x, y, angles, room_w, room_h)
    
    # 2. Intersect with all circles (people + circular obstacles)
    circle_dists = get_ray_circles_intersections(x, y, angles, circles)
    
    # 3. The final Lidar reading is the minimum among walls, circles, and max_dist
    final_dists = jnp.minimum(wall_dists, circle_dists)
    final_dists = jnp.clip(final_dists, 0.0, max_dist)
    
    return final_dists