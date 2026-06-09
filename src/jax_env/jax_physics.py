"""
jax_physics.py — Vectorized 2D Physics & LiDAR Engine
======================================================
obs_boxes format: (K, 8) — four ordered 2D vertices stored flat as
    [x0,y0, x1,y1, x2,y2, x3,y3] defining a convex quadrilateral.
"""

import functools
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Wall Intersections — accepts pre-computed dx, dy
# ---------------------------------------------------------------------------

def _get_ray_wall_intersections(x0, y0, dx, dy, room_w, room_h):
    """N rays vs 4 axis-aligned walls. dx,dy:(N,) → returns (N,)"""
    eps = 1e-7
    dx = jnp.where(jnp.abs(dx) < eps, jnp.sign(dx) * eps + eps, dx)
    dy = jnp.where(jnp.abs(dy) < eps, jnp.sign(dy) * eps + eps, dy)

    t_left   = (0.0    - x0) / dx
    t_right  = (room_w - x0) / dx
    t_bottom = (0.0    - y0) / dy
    t_top    = (room_h - y0) / dy

    # DIFF FIX D: replace jnp.inf with finite no_hit to avoid NaN gradients
    t_left   = jnp.where(t_left   > 1e-5, t_left,   jnp.inf)
    t_right  = jnp.where(t_right  > 1e-5, t_right,  jnp.inf)
    t_bottom = jnp.where(t_bottom > 1e-5, t_bottom, jnp.inf)
    t_top    = jnp.where(t_top    > 1e-5, t_top,    jnp.inf)

    return jnp.minimum(
        jnp.minimum(t_left, t_right),
        jnp.minimum(t_bottom, t_top)
    )


# ---------------------------------------------------------------------------
# Circle Intersections — accepts pre-computed dx, dy
# ---------------------------------------------------------------------------

def _intersect_ray_circle_scalar(ray_x, ray_y, dx, dy, cx, cy, r, no_hit):
    """Single ray vs single circle. Returns smallest positive t or no_hit."""
    fx = ray_x - cx
    fy = ray_y - cy
    b  = 2.0 * (fx * dx + fy * dy)
    c  = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c

    sqrt_disc = jnp.sqrt(jnp.maximum(1e-8, disc))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    # Use jnp.inf directly
    valid_t1 = jnp.where(t1 > 1e-5, t1, jnp.inf)
    valid_t2 = jnp.where(t2 > 1e-5, t2, jnp.inf)
    min_t    = jnp.minimum(valid_t1, valid_t2)
    return jnp.where(disc < 0.0, jnp.inf, min_t)


def _intersect_ray_circle_scalar(ray_x, ray_y, dx, dy, cx, cy, r):
    """Single ray vs single circle. Returns smallest positive t or jnp.inf."""
    fx = ray_x - cx
    fy = ray_y - cy
    b  = 2.0 * (fx * dx + fy * dy)
    c  = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c

    sqrt_disc = jnp.sqrt(jnp.maximum(1e-8, disc))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    valid_t1 = jnp.where(t1 > 1e-5, t1, jnp.inf)
    valid_t2 = jnp.where(t2 > 1e-5, t2, jnp.inf)
    min_t    = jnp.minimum(valid_t1, valid_t2)
    return jnp.where(disc < 0.0, jnp.inf, min_t)


def _get_ray_circles_intersections(x0, y0, dx, dy, circles):
    """N rays vs M circles."""
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]

    def one_ray(dxi, dyi):
        return jax.vmap(
            lambda cxi, cyi, ri: _intersect_ray_circle_scalar(x0, y0, dxi, dyi, cxi, cyi, ri)
        )(cx, cy, r)

    distances = jax.vmap(one_ray)(dx, dy)
    return jnp.min(distances, axis=1)


# ---------------------------------------------------------------------------
# Convex-quad (4-vertex) Ray Intersections
# ---------------------------------------------------------------------------
# Each box is stored as 8 floats: [x0,y0, x1,y1, x2,y2, x3,y3]  (ordered CCW).
# We intersect the ray against each of the 4 edges (segments) and take the
# minimum positive hit distance.

def _intersect_ray_segment_scalar(rx, ry, dx, dy, ax, ay, bx, by):
    """Ray-segment intersection: ray origin (rx,ry) + t*(dx,dy), segment A→B.
    Returns smallest positive t or jnp.inf if no hit."""
    # Parametric: ray: P = R + t*D,  segment: Q = A + s*(B-A), 0≤s≤1
    # Solve: R + t*D = A + s*(B-A)
    ex, ey = bx - ax, by - ay          # segment direction
    # denom = D × E  (2D cross product)
    denom = dx * ey - dy * ex
    safe_denom = jnp.where(jnp.abs(denom) < 1e-10, 1e-10, denom)
    # t and s from Cramer's rule
    fx, fy = ax - rx, ay - ry
    t = (fx * ey - fy * ex) / safe_denom
    s = (fx * dy - fy * dx) / safe_denom
    hit = (jnp.abs(denom) >= 1e-10) & (t > 1e-5) & (s >= 0.0) & (s <= 1.0)
    return jnp.where(hit, t, jnp.inf)


def _intersect_ray_quad_scalar(rx, ry, dxi, dyi, box):
    """Single ray vs single convex quad (box: 8-float vertex array)."""
    x0v, y0v = box[0], box[1]
    x1v, y1v = box[2], box[3]
    x2v, y2v = box[4], box[5]
    x3v, y3v = box[6], box[7]
    t0 = _intersect_ray_segment_scalar(rx, ry, dxi, dyi, x0v, y0v, x1v, y1v)
    t1 = _intersect_ray_segment_scalar(rx, ry, dxi, dyi, x1v, y1v, x2v, y2v)
    t2 = _intersect_ray_segment_scalar(rx, ry, dxi, dyi, x2v, y2v, x3v, y3v)
    t3 = _intersect_ray_segment_scalar(rx, ry, dxi, dyi, x3v, y3v, x0v, y0v)
    return jnp.minimum(jnp.minimum(t0, t1), jnp.minimum(t2, t3))


def _get_ray_boxes_intersections(x0, y0, dx, dy, boxes):
    """N rays vs K convex quads.  boxes: (K, 8)."""
    def one_ray(dxi, dyi):
        return jax.vmap(
            lambda b: _intersect_ray_quad_scalar(x0, y0, dxi, dyi, b)
        )(boxes)

    distances = jax.vmap(one_ray)(dx, dy)   # (N, K)
    return jnp.min(distances, axis=1)        # (N,)


# ---------------------------------------------------------------------------
# Full LiDAR sweep — single JIT entry point, fused trig computation
# static_argnums: num_rays (5), fov (6), max_dist (7), room_w (8), room_h (9)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def compute_lidar(x, y, theta, circles, boxes, num_rays, fov, max_dist, room_w, room_h):
    angles = theta - fov * 0.5 + jnp.arange(num_rays) * (fov / (num_rays - 1))

    dx = jnp.cos(angles)
    dy = jnp.sin(angles)

    # We no longer need the finite sentinel. Pass jnp.inf implicitly or update sub-functions.
    wall_dists   = _get_ray_wall_intersections(x, y, dx, dy, room_w, room_h)
    circle_dists = _get_ray_circles_intersections(x, y, dx, dy, circles)
    box_dists    = _get_ray_boxes_intersections(x, y, dx, dy, boxes)

    final = jnp.minimum(jnp.minimum(wall_dists, circle_dists), box_dists)
    return jnp.clip(final, 0.0, max_dist)