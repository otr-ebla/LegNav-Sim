"""
jax_physics.py — Vectorized 2D Physics & LiDAR Engine
======================================================
Fixes vs original:
  - Removed erroneous @jax.vmap decorator from _intersect_ray_circle (caused crash)
  - Corrected vmap axis logic in get_ray_circles_intersections
  - Fixed min reduction axis after double-vmap (N,M) → (N,)
  - Added obstacle segment (box wall) intersections for rectangular obstacles
  - All functions are pure, jit-compatible, and vmap-safe
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Wall Intersections
# ---------------------------------------------------------------------------

@jax.jit
def get_ray_wall_intersections(
    x0: float, y0: float,
    angles: jnp.ndarray,
    room_w: float, room_h: float
) -> jnp.ndarray:
    """
    N rays vs 4 axis-aligned room walls.
    angles : (N,)
    returns: (N,) minimum positive distance to a wall.
    """
    dx = jnp.cos(angles)
    dy = jnp.sin(angles)

    eps = 1e-7
    dx = jnp.where(jnp.abs(dx) < eps, jnp.sign(dx) * eps + eps, dx)
    dy = jnp.where(jnp.abs(dy) < eps, jnp.sign(dy) * eps + eps, dy)

    t_left   = (0.0    - x0) / dx
    t_right  = (room_w - x0) / dx
    t_bottom = (0.0    - y0) / dy
    t_top    = (room_h - y0) / dy

    t_left   = jnp.where(t_left   > 1e-5, t_left,   jnp.inf)
    t_right  = jnp.where(t_right  > 1e-5, t_right,  jnp.inf)
    t_bottom = jnp.where(t_bottom > 1e-5, t_bottom, jnp.inf)
    t_top    = jnp.where(t_top    > 1e-5, t_top,    jnp.inf)

    return jnp.minimum(
        jnp.minimum(t_left, t_right),
        jnp.minimum(t_bottom, t_top)
    )


# ---------------------------------------------------------------------------
# Circle Intersections  (FIX: removed top-level @jax.vmap — was double-vmapped)
# ---------------------------------------------------------------------------

def _intersect_ray_circle_scalar(
    ray_x: float, ray_y: float,
    dx: float, dy: float,
    cx: float, cy: float, r: float
) -> float:
    """
    Analytic ray-circle intersection for a SINGLE ray and SINGLE circle.
    Ray: P(t) = (ray_x + t*dx, ray_y + t*dy),  direction must be unit-length.
    Returns smallest positive t, or jnp.inf if no intersection.
    """
    fx = ray_x - cx
    fy = ray_y - cy

    # a = dx²+dy² = 1  (unit direction)
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c          # a=1

    sqrt_disc = jnp.sqrt(jnp.maximum(0.0, disc))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    valid_t1 = jnp.where(t1 > 1e-5, t1, jnp.inf)
    valid_t2 = jnp.where(t2 > 1e-5, t2, jnp.inf)
    min_t    = jnp.minimum(valid_t1, valid_t2)

    return jnp.where(disc < 0.0, jnp.inf, min_t)


@jax.jit
def get_ray_circles_intersections(
    x0: float, y0: float,
    angles: jnp.ndarray,
    circles: jnp.ndarray
) -> jnp.ndarray:
    """
    N rays vs M circles.
    angles  : (N,)
    circles : (M, 3)  columns = [cx, cy, radius]
    returns : (N,)  closest circle hit per ray.

    Strategy: vmap over rays (outer), then vmap over circles (inner).
    Both vmaps are applied HERE — never on the scalar kernel above.
    """
    dx = jnp.cos(angles)   # (N,)
    dy = jnp.sin(angles)   # (N,)

    cx = circles[:, 0]     # (M,)
    cy = circles[:, 1]     # (M,)
    r  = circles[:, 2]     # (M,)

    # Inner vmap: 1 ray vs M circles  →  (M,)
    def one_ray_vs_all_circles(dxi, dyi):
        return jax.vmap(
            lambda cxi, cyi, ri: _intersect_ray_circle_scalar(x0, y0, dxi, dyi, cxi, cyi, ri)
        )(cx, cy, r)

    # Outer vmap: N rays  →  (N, M)
    distances = jax.vmap(one_ray_vs_all_circles)(dx, dy)

    # Closest circle per ray  (N,)
    return jnp.min(distances, axis=1)


# ---------------------------------------------------------------------------
# Rectangular (box) obstacle intersections — new, more realistic
# ---------------------------------------------------------------------------

def _intersect_ray_aabb_scalar(
    x0: float, y0: float,
    dx: float, dy: float,
    bx: float, by: float,   # box centre
    bw: float, bh: float,   # half-widths
) -> float:
    """Slab-method AABB intersection for 1 ray, 1 box."""
    eps = 1e-7
    idx = 1.0 / jnp.where(jnp.abs(dx) < eps, jnp.sign(dx) * eps + eps, dx)
    idy = 1.0 / jnp.where(jnp.abs(dy) < eps, jnp.sign(dy) * eps + eps, dy)

    tx1 = (bx - bw - x0) * idx
    tx2 = (bx + bw - x0) * idx
    ty1 = (by - bh - y0) * idy
    ty2 = (by + bh - y0) * idy

    tmin = jnp.maximum(jnp.minimum(tx1, tx2), jnp.minimum(ty1, ty2))
    tmax = jnp.minimum(jnp.maximum(tx1, tx2), jnp.maximum(ty1, ty2))

    hit = (tmax >= jnp.maximum(tmin, 1e-5))
    t   = jnp.where(tmin > 1e-5, tmin, tmax)
    return jnp.where(hit & (t > 1e-5), t, jnp.inf)


@jax.jit
def get_ray_boxes_intersections(
    x0: float, y0: float,
    angles: jnp.ndarray,
    boxes: jnp.ndarray,         # (K, 4)  [cx, cy, half_w, half_h]
) -> jnp.ndarray:
    """N rays vs K boxes → (N,)"""
    dx = jnp.cos(angles)
    dy = jnp.sin(angles)
    bx, by, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    def one_ray(dxi, dyi):
        return jax.vmap(
            lambda bxi, byi, bwi, bhi: _intersect_ray_aabb_scalar(x0, y0, dxi, dyi, bxi, byi, bwi, bhi)
        )(bx, by, bw, bh)

    distances = jax.vmap(one_ray)(dx, dy)   # (N, K)
    return jnp.min(distances, axis=1)


# ---------------------------------------------------------------------------
# Full LiDAR sweep
# ---------------------------------------------------------------------------

@jax.jit
def compute_lidar(
    x: float, y: float, theta: float,
    circles: jnp.ndarray,       # (M, 3) [cx, cy, r]
    boxes: jnp.ndarray,         # (K, 4) [cx, cy, hw, hh]
    num_rays: int,
    fov: float,
    max_dist: float,
    room_w: float,
    room_h: float,
) -> jnp.ndarray:
    """
    Full 360° (or arbitrary FOV) LiDAR sweep.
    Returns (num_rays,) clipped distances.
    """
    # Evenly spaced angles across full FOV (endpoint-inclusive)
    angles = theta - fov * 0.5 + jnp.arange(num_rays) * (fov / (num_rays - 1))

    wall_dists   = get_ray_wall_intersections(x, y, angles, room_w, room_h)
    circle_dists = get_ray_circles_intersections(x, y, angles, circles)
    box_dists    = get_ray_boxes_intersections(x, y, angles, boxes)

    final = jnp.minimum(jnp.minimum(wall_dists, circle_dists), box_dists)
    return jnp.clip(final, 0.0, max_dist)