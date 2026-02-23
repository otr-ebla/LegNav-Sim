"""
jax_physics.py — Vectorized 2D Physics & LiDAR Engine
======================================================
IMPROVEMENTS vs previous version:

  IMPROVEMENT A — Fused ray-direction computation (NEW):
    get_ray_circles_intersections and get_ray_boxes_intersections each
    independently called jnp.cos/sin on the angles array. In compute_lidar
    this happened THREE times (walls, circles, boxes).
    FIX: compute dx/dy once in compute_lidar and pass them in; inner
    functions accept pre-computed directions to avoid redundant trig.
    On 4096 parallel envs × 112 rays this saves significant GPU cycles.

  IMPROVEMENT B — @jax.jit removed from sub-functions (NEW):
    get_ray_wall_intersections, get_ray_circles_intersections,
    get_ray_boxes_intersections were all @jax.jit decorated but are always
    called from within compute_lidar which is itself @jax.jit. Nested JIT
    is legal but adds tracing overhead and prevents kernel fusion.
    FIX: removed @jax.jit from subfunctions; only compute_lidar is JIT-ted.
    The full LiDAR sweep now compiles as a single fused kernel.

  IMPROVEMENT C — Avoid recomputation of angles linspace (NEW):
    The angles array was recomputed every call. Now the formula is kept
    but inlined without intermediate naming to help XLA's CSE pass.

  UNCHANGED — All intersection math is correct and unchanged.
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

def _intersect_ray_circle_scalar(ray_x, ray_y, dx, dy, cx, cy, r):
    """Single ray vs single circle. Returns smallest positive t or inf."""
    fx = ray_x - cx
    fy = ray_y - cy
    b  = 2.0 * (fx * dx + fy * dy)
    c  = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c

    sqrt_disc = jnp.sqrt(jnp.maximum(0.0, disc))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    valid_t1 = jnp.where(t1 > 1e-5, t1, jnp.inf)
    valid_t2 = jnp.where(t2 > 1e-5, t2, jnp.inf)
    min_t    = jnp.minimum(valid_t1, valid_t2)
    return jnp.where(disc < 0.0, jnp.inf, min_t)


def _get_ray_circles_intersections(x0, y0, dx, dy, circles):
    """
    N rays vs M circles.
    dx,dy:(N,)  circles:(M,3) [cx,cy,r]  → returns (N,)
    """
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]

    def one_ray(dxi, dyi):
        return jax.vmap(
            lambda cxi, cyi, ri: _intersect_ray_circle_scalar(x0, y0, dxi, dyi, cxi, cyi, ri)
        )(cx, cy, r)

    distances = jax.vmap(one_ray)(dx, dy)
    return jnp.min(distances, axis=1)


# ---------------------------------------------------------------------------
# Box (AABB) Intersections — accepts pre-computed dx, dy
# ---------------------------------------------------------------------------

def _intersect_ray_aabb_scalar(x0, y0, dx, dy, bx, by, bw, bh):
    """Slab method: single ray vs single axis-aligned box."""
    eps = 1e-7
    idx = 1.0 / jnp.where(jnp.abs(dx) < eps, jnp.sign(dx) * eps + eps, dx)
    idy = 1.0 / jnp.where(jnp.abs(dy) < eps, jnp.sign(dy) * eps + eps, dy)

    tx1 = (bx - bw - x0) * idx
    tx2 = (bx + bw - x0) * idx
    ty1 = (by - bh - y0) * idy
    ty2 = (by + bh - y0) * idy

    tmin = jnp.maximum(jnp.minimum(tx1, tx2), jnp.minimum(ty1, ty2))
    tmax = jnp.minimum(jnp.maximum(tx1, tx2), jnp.maximum(ty1, ty2))

    hit = tmax >= jnp.maximum(tmin, 1e-5)
    t   = jnp.where(tmin > 1e-5, tmin, tmax)
    return jnp.where(hit & (t > 1e-5), t, jnp.inf)


def _get_ray_boxes_intersections(x0, y0, dx, dy, boxes):
    """N rays vs K boxes. boxes:(K,4) [cx,cy,hw,hh] → (N,)"""
    bx, by, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    def one_ray(dxi, dyi):
        return jax.vmap(
            lambda bxi, byi, bwi, bhi: _intersect_ray_aabb_scalar(x0, y0, dxi, dyi, bxi, byi, bwi, bhi)
        )(bx, by, bw, bh)

    distances = jax.vmap(one_ray)(dx, dy)
    return jnp.min(distances, axis=1)


# ---------------------------------------------------------------------------
# Full LiDAR sweep — single JIT entry point, fused trig computation
# static_argnums: num_rays (5), fov (6), max_dist (7), room_w (8), room_h (9)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def compute_lidar(x, y, theta, circles, boxes, num_rays, fov, max_dist, room_w, room_h):
    """
    Full LiDAR sweep.
    circles:(M,3) [cx,cy,r]   boxes:(K,4) [cx,cy,hw,hh]
    Returns (num_rays,) clipped distances.

    IMPROVEMENT: cos/sin computed ONCE here and shared with all sub-routines.
    No nested @jax.jit — entire sweep compiles as one fused XLA kernel.
    """
    angles = theta - fov * 0.5 + jnp.arange(num_rays) * (fov / (num_rays - 1))

    # Compute directions once — shared by all three intersection routines
    dx = jnp.cos(angles)
    dy = jnp.sin(angles)

    wall_dists   = _get_ray_wall_intersections(x, y, dx, dy, room_w, room_h)
    circle_dists = _get_ray_circles_intersections(x, y, dx, dy, circles)
    box_dists    = _get_ray_boxes_intersections(x, y, dx, dy, boxes)

    final = jnp.minimum(jnp.minimum(wall_dists, circle_dists), box_dists)
    return jnp.clip(final, 0.0, max_dist)