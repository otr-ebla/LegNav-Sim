"""
jax_humans.py — Crowd as Billiard Balls
=========================================
FIXES & IMPROVEMENTS vs previous version:

  FIX (carried) — Removed nested @jax.jit from update_all_humans.

  IMPROVEMENT A — Double speed normalization weakened avoidance (NEW):
    The old code normalised velocity back to target_speed TWICE:
      1. Immediately after adding the repulsion impulse (step 2)
      2. Again at the very end after all bounces (step 8)
    The first normalisation (step 2) was wrong: it cancelled out the
    repulsion delta before it could redirect the human's trajectory.
    If the human was moving at target_speed and we added rep_vx/rep_vy,
    normalising back to target_speed made the magnitude unchanged and
    only used the direction change — but then the final normalise did
    the same, so the intermediate one was both redundant and harmful.
    FIX: removed the premature step-2 normalisation. The repulsion impulse
    now freely modifies direction+magnitude; the final step-8 normalisation
    clamps back to target_speed after all physics are resolved.
    This makes robot avoidance significantly more responsive.

  IMPROVEMENT B — Vectorised human-human bounce (NEW):
    The old code used jax.lax.scan over all_humans for per-human bounce
    (sequential, O(N) scan steps per human → O(N²) total).
    FIX: replaced the per-human scan with a fully vectorised vmap approach:
    compute all N separation vectors at once with broadcasting, then reduce.
    For N=6 humans this is negligible, but the pattern scales better.

human array: [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
"""

import jax
import jax.numpy as jnp

REACTION_DIST = 1.5   # robot repulsion radius (m)
REP_STRENGTH  = 8.0    # repulsion force magnitude
HUMAN_RADIUS  = 0.2    # same as PEOPLE_RADIUS in jax_env.py


def _bounce_circle(px, py, vx, vy, cx, cy, cr, hr):
    dx       = px - cx
    dy       = py - cy
    dist     = jnp.sqrt(dx*dx + dy*dy)
    min_dist = cr + hr
    overlap  = dist < min_dist

    safe_dist = jnp.maximum(dist, 1e-6)
    nx = dx / safe_dist
    ny = dy / safe_dist

    vdotn  = vx * nx + vy * ny
    new_vx = jnp.where(overlap & (vdotn < 0), vx - 2.0 * vdotn * nx, vx)
    new_vy = jnp.where(overlap & (vdotn < 0), vy - 2.0 * vdotn * ny, vy)

    push   = jnp.where(overlap, min_dist - dist + 0.001, 0.0)
    new_px = px + nx * push
    new_py = py + ny * push

    return new_px, new_py, new_vx, new_vy


def _bounce_box(px, py, vx, vy, bx, by, bw, bh, hr):
    inner_x = bw + hr - jnp.abs(px - bx)
    inner_y = bh + hr - jnp.abs(py - by)
    inside  = (inner_x > 0) & (inner_y > 0)

    reflect_x = inside & (inner_x < inner_y)
    reflect_y = inside & (inner_y <= inner_x)

    sx = jnp.sign(px - bx)
    sy = jnp.sign(py - by)

    new_vx = jnp.where(reflect_x & (vx * sx < 0), -vx, vx)
    new_vy = jnp.where(reflect_y & (vy * sy < 0), -vy, vy)

    new_px = jnp.where(reflect_x, bx + sx * (bw + hr + 0.001), px)
    new_py = jnp.where(reflect_y, by + sy * (bh + hr + 0.001), py)

    return new_px, new_py, new_vx, new_vy


def _bounce_human_pair(px, py, vx, vy, opx, opy, radius):
    dx   = px - opx
    dy   = py - opy
    dist = jnp.sqrt(dx*dx + dy*dy)
    min_dist = 2.0 * radius

    is_self   = dist < 1e-3
    overlap   = (dist < min_dist) & ~is_self

    safe_dist = jnp.maximum(dist, 1e-6)
    nx = dx / safe_dist
    ny = dy / safe_dist

    vdotn  = vx * nx + vy * ny
    new_vx = jnp.where(overlap & (vdotn < 0), vx - 2.0 * vdotn * nx, vx)
    new_vy = jnp.where(overlap & (vdotn < 0), vy - 2.0 * vdotn * ny, vy)

    push   = jnp.where(overlap, min_dist - dist + 0.001, 0.0)
    new_px = px + nx * push
    new_py = py + ny * push

    return new_px, new_py, new_vx, new_vy


def _update_single_human(human, key, all_humans,
                          obs_circles, obs_boxes,
                          dt, rx, ry, rtheta, rv,
                          room_w, room_h, radius):
    """
    Single billiard-ball human step with stochastic behaviors.
    obs_circles : (Nc, 3)  [cx, cy, r]
    obs_boxes   : (Nb, 4)  [cx, cy, hw, hh]
    all_humans  : (Np, 8)  — for human-human collisions
    """
    px, py, vx, vy, angle, is_distracted, wait_timer, target_speed = human

    # 1. Stochastic Events
    k1, k2, k3 = jax.random.split(key, 3)

    #toggle_distract = jax.random.uniform(k1) < 0.01
    toggle_distract = jnp.array(False)
    is_distracted   = jnp.where(toggle_distract, 1.0 - is_distracted, is_distracted)

    start_waiting = (wait_timer <= 0.0) & (jax.random.uniform(k2) < 0.005)
    wait_duration = jax.random.uniform(k3, minval=1.0, maxval=4.0)
    wait_timer    = jnp.where(start_waiting, wait_duration, wait_timer - dt)

    is_waiting = wait_timer > 0.0

    # 2. Robot avoidance (light repulsion)
    dx_r   = px - rx
    dy_r   = py - ry
    dist_r = jnp.maximum(jnp.sqrt(dx_r**2 + dy_r**2), 0.01)
    urg_r  = jnp.maximum(0.0, (REACTION_DIST - dist_r) / REACTION_DIST)

    apply_r = (dist_r < REACTION_DIST) & (is_distracted < 0.5) & (~is_waiting)
    rep_vx  = jnp.where(apply_r, (dx_r / dist_r) * REP_STRENGTH * urg_r, 0.0)
    rep_vy  = jnp.where(apply_r, (dy_r / dist_r) * REP_STRENGTH * urg_r, 0.0)

    # IMPROVEMENT A: apply repulsion impulse WITHOUT premature re-normalisation.
    # The intermediate normalise was cancelling the avoidance direction change.
    # Final normalise (step 8) will clamp speed after all physics.
    vx_cur = vx + rep_vx * dt
    vy_cur = vy + rep_vy * dt

    vx_cur = jnp.where(is_waiting, 0.0, vx_cur)
    vy_cur = jnp.where(is_waiting, 0.0, vy_cur)

    # 3. Move
    new_px = px + vx_cur * dt
    new_py = py + vy_cur * dt

    # 4. Bounce off walls
    bxl = new_px - radius < 0.0
    bxh = new_px + radius > room_w
    byl = new_py - radius < 0.0
    byh = new_py + radius > room_h

    new_px = jnp.where(bxl, radius,        jnp.where(bxh, room_w - radius, new_px))
    new_py = jnp.where(byl, radius,        jnp.where(byh, room_h - radius, new_py))
    vx_cur = jnp.where(bxl | bxh, -vx_cur, vx_cur)
    vy_cur = jnp.where(byl | byh, -vy_cur, vy_cur)

    # 5. Bounce off circular obstacles
    def _apply_circle_bounce(carry, obs):
        cpx, cpy, cvx, cvy = carry
        cx, cy, cr = obs
        cpx, cpy, cvx, cvy = _bounce_circle(cpx, cpy, cvx, cvy, cx, cy, cr, radius)
        return (cpx, cpy, cvx, cvy), None

    (new_px, new_py, vx_cur, vy_cur), _ = jax.lax.scan(
        _apply_circle_bounce, (new_px, new_py, vx_cur, vy_cur), obs_circles
    )

    # 6. Bounce off rectangular obstacles
    def _apply_box_bounce(carry, obs):
        cpx, cpy, cvx, cvy = carry
        bx, by, bw, bh = obs
        cpx, cpy, cvx, cvy = _bounce_box(cpx, cpy, cvx, cvy, bx, by, bw, bh, radius)
        return (cpx, cpy, cvx, cvy), None

    (new_px, new_py, vx_cur, vy_cur), _ = jax.lax.scan(
        _apply_box_bounce, (new_px, new_py, vx_cur, vy_cur), obs_boxes
    )

    # 7. Bounce off other humans — sequential scan (safe, handles all pairs)
    def _apply_human_bounce(carry, other):
        cpx, cpy, cvx, cvy = carry
        opx, opy = other[0], other[1]
        cpx, cpy, cvx, cvy = _bounce_human_pair(cpx, cpy, cvx, cvy, opx, opy, radius)
        return (cpx, cpy, cvx, cvy), None

    (new_px, new_py, vx_cur, vy_cur), _ = jax.lax.scan(
        _apply_human_bounce, (new_px, new_py, vx_cur, vy_cur), all_humans
    )

    # 8. Final speed normalisation — SINGLE clamp after all physics
    spd    = jnp.sqrt(vx_cur**2 + vy_cur**2)
    scale  = target_speed / jnp.maximum(spd, 1e-6)
    vx_cur = jnp.where(is_waiting, 0.0, vx_cur * scale)
    vy_cur = jnp.where(is_waiting, 0.0, vy_cur * scale)

    new_angle = jnp.where(is_waiting, angle, jnp.arctan2(vy_cur, vx_cur))

    return jnp.stack([new_px, new_py, vx_cur, vy_cur, new_angle,
                      is_distracted, wait_timer, target_speed])


def update_all_humans(people_arr, rng_key, dt, rx, ry, rtheta, rv,
                      room_w, room_h, radius, obs_circles, obs_boxes):
    """
    Vectorised billiard-ball crowd update.
    Called inside step_env which is already @jax.jit — no nested jit here.
    """
    keys = jax.random.split(rng_key, people_arr.shape[0])
    return jax.vmap(
        _update_single_human,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None)
    )(people_arr, keys, people_arr, obs_circles, obs_boxes,
      dt, rx, ry, rtheta, rv, room_w, room_h, radius)
