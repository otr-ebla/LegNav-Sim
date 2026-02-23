"""
jax_humans.py — Crowd as Billiard Balls
=========================================
FIX in questa versione:

  BUG 4 — Doppio @jax.jit su update_all_humans (PERFORMANCE):
    update_all_humans era decorata @jax.jit ma viene chiamata DENTRO step_env
    che è già @jax.jit. Il JIT innestato è legale in JAX ma causa overhead di
    tracing aggiuntivo ad ogni compilazione e può interferire con ottimizzazioni
    di fusione del grafo computazionale del JIT esterno.
    FIX: rimosso @jax.jit da update_all_humans. Il JIT di step_env coprirà
    l'intera funzione inclusa la vmap crowd.

  INVARIATO — Tutto il resto (elastic bounce, human-human collision, wall
  bounce, obstacle bounce, stochastic behaviors) era già corretto.

human array: [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
"""

import jax
import jax.numpy as jnp

REACTION_DIST  = 1.0    # robot repulsion radius (m)
REP_STRENGTH   = 6.0    # repulsion force magnitude
HUMAN_RADIUS   = 0.2    # same as PEOPLE_RADIUS in jax_env.py


def _bounce_circle(px, py, vx, vy, cx, cy, cr, hr):
    """
    Elastic bounce of a human (radius hr) off a circle obstacle (cx,cy,cr).
    """
    dx   = px - cx
    dy   = py - cy
    dist = jnp.sqrt(dx*dx + dy*dy)
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
    """
    Elastic bounce off an axis-aligned box (centre bx,by half-widths bw,bh).
    """
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
    """
    Elastic bounce of this human off another human at (opx, opy).
    Self-collision (dist ≈ 0) is masked out safely.
    """
    dx   = px - opx
    dy   = py - opy
    dist = jnp.sqrt(dx*dx + dy*dy)
    min_dist = 2.0 * radius

    # Mask: skip self (dist < 1e-3) and non-overlapping pairs
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

    toggle_distract = jax.random.uniform(k1) < 0.01
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

    apply_r= (dist_r < REACTION_DIST) & (is_distracted < 0.5) & (~is_waiting)
    rep_vx = jnp.where(apply_r, (dx_r / dist_r) * REP_STRENGTH * urg_r, 0.0)
    rep_vy = jnp.where(apply_r, (dy_r / dist_r) * REP_STRENGTH * urg_r, 0.0)

    vx_rep = vx + rep_vx * dt
    vy_rep = vy + rep_vy * dt

    spd    = jnp.sqrt(vx_rep**2 + vy_rep**2)
    scale  = target_speed / jnp.maximum(spd, 1e-6)
    vx_cur = vx_rep * scale
    vy_cur = vy_rep * scale

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

    # 7. Bounce off other humans
    def _apply_human_bounce(carry, other):
        cpx, cpy, cvx, cvy = carry
        opx, opy = other[0], other[1]
        cpx, cpy, cvx, cvy = _bounce_human_pair(cpx, cpy, cvx, cvy, opx, opy, radius)
        return (cpx, cpy, cvx, cvy), None

    (new_px, new_py, vx_cur, vy_cur), _ = jax.lax.scan(
        _apply_human_bounce, (new_px, new_py, vx_cur, vy_cur), all_humans
    )

    # 8. Final speed re-normalisation
    spd   = jnp.sqrt(vx_cur**2 + vy_cur**2)
    scale = target_speed / jnp.maximum(spd, 1e-6)
    vx_cur = jnp.where(is_waiting, 0.0, vx_cur * scale)
    vy_cur = jnp.where(is_waiting, 0.0, vy_cur * scale)

    new_angle = jnp.where(is_waiting, angle, jnp.arctan2(vy_cur, vx_cur))

    return jnp.stack([new_px, new_py, vx_cur, vy_cur, new_angle,
                      is_distracted, wait_timer, target_speed])


# FIX BUG 4: rimosso @jax.jit — questa funzione è chiamata DENTRO step_env che
# è già @jax.jit. Il doppio JIT innestato causa overhead di tracing senza benefici.
def update_all_humans(people_arr, rng_key, dt, rx, ry, rtheta, rv,
                      room_w, room_h, radius, obs_circles, obs_boxes):
    """
    Vectorised billiard-ball crowd update with human-human collisions.
    obs_circles : (Nc, 3)   obs_boxes : (Nb, 4)
    """
    keys = jax.random.split(rng_key, people_arr.shape[0])
    return jax.vmap(
        _update_single_human,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None)
    )(people_arr, keys, people_arr, obs_circles, obs_boxes,
      dt, rx, ry, rtheta, rv, room_w, room_h, radius)