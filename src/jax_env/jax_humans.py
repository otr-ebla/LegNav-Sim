"""
jax_humans.py — Stable Social Force Model for dt=0.15s
=======================================================
The previous HSFM implementation oscillated because the original force
magnitudes (Ai=2000 N, k1=120000 N/m) are tuned for dt≈0.01s. At dt=0.15s
Euler integration is unconditionally unstable with those values.

This implementation uses the SFM structure from Helbing & Molnár (1995) with:
  - Forces rescaled for dt=0.15s stability
  - Hard position correction (push-out) after integration so people
    CANNOT penetrate walls, boxes, or circles regardless of force magnitudes
  - Velocity-based steering (no torque/inertia — those need tiny dt)
  - Smooth heading derived from velocity direction with low-pass filter

PEOPLE ARRAY (N, 8) — unchanged:
  [0] px  [1] py  [2] vx  [3] vy  [4] theta
  [5] distracted  [6] waypoint_x  [7] desired_spd
"""

import jax
import jax.numpy as jnp

# ── Force tuning for dt=0.15s ─────────────────────────────────────────────────
# Rule of thumb: F/m * dt < v_max  →  F < m*v_max/dt = 75*1.3/0.15 ≈ 650 N
# We use much smaller values so a single step never overshoots.

A_HUMAN    = 8.0     # N   repulsion magnitude (human-human)
B_HUMAN    = 0.4     # m   decay distance
OVERLAP_K  = 40.0    # N/m hard contact spring (only when overlapping)

A_OBS      = 10.0    # N   repulsion from obstacles / walls
B_OBS      = 0.25    # m

A_ROBOT    = 8.0
B_ROBOT    = 0.45    # m

TAU        = 0.5     # s   goal-force relaxation time
MAX_SPEED  = 1.6     # m/s
DIST_MULT  = 0.55    # distracted speed fraction

# Anisotropy: downweight forces from behind
LAMBDA_A   = 0.3

_WAYPOINT_R = 0.5
_WP_MARGIN  = 0.8
_EPS        = 1e-7
_ROBOT_R    = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n2(dx, dy):
    """(distance, nx, ny) with eps guard."""
    d = jnp.sqrt(dx*dx + dy*dy + _EPS)
    return d, dx/d, dy/d

def _aniso(nx, ny, vx, vy):
    spd = jnp.sqrt(vx*vx + vy*vy + _EPS)
    cos_phi = (vx*nx + vy*ny) / spd
    return LAMBDA_A + (1.0 - LAMBDA_A) * (1.0 + cos_phi) * 0.5

def _wpx_to_wpy(wpx, room_h):
    frac = (wpx * 2.6180339887) % 1.0
    return _WP_MARGIN + frac * (room_h - 2.0 * _WP_MARGIN)

def _sample_wp(key, room_w, room_h):
    k1, k2 = jax.random.split(key)
    wpx = jax.random.uniform(k1, minval=_WP_MARGIN, maxval=room_w - _WP_MARGIN)
    wpy = jax.random.uniform(k2, minval=_WP_MARGIN, maxval=room_h - _WP_MARGIN)
    return wpx, wpy


# ── Force components ──────────────────────────────────────────────────────────

def _goal_force(px, py, vx, vy, wpx, wpy, v_des):
    """Steer toward waypoint: f = (v_des*e_goal - v) / tau."""
    dx, dy = wpx - px, wpy - py
    _, ex, ey = _n2(dx, dy)
    return (v_des*ex - vx) / TAU, (v_des*ey - vy) / TAU

def _human_force(px, py, vx, vy, opx, opy, r):
    """Repulsion + hard contact from one other person."""
    dx, dy = px-opx, py-opy
    dist, nx, ny = _n2(dx, dy)
    # Soft exponential
    mag  = A_HUMAN * jnp.exp(-(dist - 2*r) / B_HUMAN)
    w    = _aniso(nx, ny, vx, vy)
    # Hard contact spring (only when overlapping)
    pen  = jnp.maximum(2*r - dist, 0.0)
    hard = OVERLAP_K * pen
    return (mag*w + hard)*nx, (mag*w + hard)*ny

def _wall_force(px, py, room_w, room_h):
    """Smooth repulsion from 4 walls."""
    fx  = A_OBS * jnp.exp(-px / B_OBS)
    fx -= A_OBS * jnp.exp(-(room_w - px) / B_OBS)
    fy  = A_OBS * jnp.exp(-py / B_OBS)
    fy -= A_OBS * jnp.exp(-(room_h - py) / B_OBS)
    return fx, fy

def _circle_force(px, py, vx, vy, circles):
    """Repulsion from circular obstacles."""
    def one(c):
        cx, cy, r = c
        dx, dy = px-cx, py-cy
        dist, nx, ny = _n2(dx, dy)
        surf = jnp.maximum(dist - r, 0.0)
        mag  = A_OBS * jnp.exp(-surf / B_OBS)
        pen  = jnp.maximum(r - dist, 0.0)   # inside circle
        return (mag + OVERLAP_K*pen)*nx, (mag + OVERLAP_K*pen)*ny
    fxs, fys = jax.vmap(one)(circles)
    return jnp.sum(fxs), jnp.sum(fys)

def _box_force(px, py, boxes):
    """Repulsion from AABB boxes."""
    def one(b):
        cx, cy, hw, hh = b
        cpx = jnp.clip(px, cx-hw, cx+hw)
        cpy = jnp.clip(py, cy-hh, cy+hh)
        dx, dy = px-cpx, py-cpy
        dist, nx, ny = _n2(dx, dy)
        # inside box: dist≈0, need strong push
        inside = (jnp.abs(px-cx) < hw) & (jnp.abs(py-cy) < hh)
        # direction when inside: push to nearest face
        face_dx = hw - jnp.abs(px-cx)
        face_dy = hh - jnp.abs(py-cy)
        sign_x  = jnp.sign(px-cx)
        sign_y  = jnp.sign(py-cy)
        nx_in   = jnp.where(face_dx < face_dy, sign_x, 0.0)
        ny_in   = jnp.where(face_dx < face_dy, 0.0, sign_y)
        pen     = jnp.where(face_dx < face_dy, face_dx, face_dy)
        nx_use  = jnp.where(inside, nx_in, nx)
        ny_use  = jnp.where(inside, ny_in, ny)
        mag     = A_OBS * jnp.exp(-dist / B_OBS) + OVERLAP_K * jnp.where(inside, pen, 0.0)
        return mag*nx_use, mag*ny_use
    fxs, fys = jax.vmap(one)(boxes)
    return jnp.sum(fxs), jnp.sum(fys)

def _robot_force(px, py, rx, ry, distract):
    dx, dy = px-rx, py-ry
    dist, nx, ny = _n2(dx, dy)
    surf = jnp.maximum(dist - _ROBOT_R, 0.0)
    mag  = A_ROBOT * jnp.exp(-surf / B_ROBOT)
    sens = jnp.where(distract > 0.5, 0.3, 1.0)
    return mag*nx*sens, mag*ny*sens


# ── Hard push-out (position correction) ──────────────────────────────────────
# Forces alone can't prevent penetration at dt=0.15. We explicitly resolve
# overlaps AFTER integration so people can never be inside obstacles.

def _pushout_walls(px, py, r, room_w, room_h):
    px = jnp.clip(px, r, room_w - r)
    py = jnp.clip(py, r, room_h - r)
    return px, py

def _pushout_circles(px, py, r, circles):
    def one(carry, c):
        cpx, cpy = carry
        cx, cy, cr = c
        dx, dy = cpx-cx, cpy-cy
        dist, nx, ny = _n2(dx, dy)
        min_dist = cr + r
        overlap  = dist < min_dist
        push     = jnp.where(overlap, min_dist - dist + 1e-3, 0.0)
        return (cpx + push*nx, cpy + push*ny), None
    (px, py), _ = jax.lax.scan(one, (px, py), circles)
    return px, py

def _pushout_boxes(px, py, r, boxes):
    def one(carry, b):
        cpx, cpy = carry
        cx, cy, hw, hh = b
        inside = (jnp.abs(cpx-cx) < hw+r) & (jnp.abs(cpy-cy) < hh+r)
        # expanded box: [cx-hw-r, cx+hw+r] x [cy-hh-r, cy+hh+r]
        # push to nearest face of expanded box
        face_dx = (hw+r) - jnp.abs(cpx-cx)
        face_dy = (hh+r) - jnp.abs(cpy-cy)
        sign_x  = jnp.sign(cpx-cx)
        sign_y  = jnp.sign(cpy-cy)
        # push along shortest axis
        push_x  = face_dx * sign_x
        push_y  = face_dy * sign_y
        use_x   = face_dx < face_dy
        dpx     = jnp.where(inside, jnp.where(use_x, push_x, 0.0), 0.0)
        dpy     = jnp.where(inside, jnp.where(use_x, 0.0, push_y), 0.0)
        return (cpx + dpx, cpy + dpy), None
    (px, py), _ = jax.lax.scan(one, (px, py), boxes)
    return px, py


# ── Single human update ───────────────────────────────────────────────────────

def _update_one(human, key, all_humans, obs_circles, obs_boxes,
                dt, rx, ry, room_w, room_h, radius):
    px, py, vx, vy = human[0], human[1], human[2], human[3]
    theta, distract, wp_x, des_spd = human[4], human[5], human[6], human[7]

    k_wp, = jax.random.split(key, 1)

    v_des = jnp.where(distract > 0.5, des_spd * DIST_MULT, des_spd)

    # Waypoint management
    wp_y    = _wpx_to_wpy(wp_x, room_h)
    dist_wp = jnp.sqrt((px-wp_x)**2 + (py-wp_y)**2)
    need_wp = (dist_wp < _WAYPOINT_R) | (wp_x < 0.0)
    nwpx, nwpy = _sample_wp(k_wp, room_w, room_h)
    wp_x = jnp.where(need_wp, nwpx, wp_x)
    wp_y = jnp.where(need_wp, nwpy, wp_y)

    # Goal force
    gfx, gfy = _goal_force(px, py, vx, vy, wp_x, wp_y, v_des)

    # Human-human forces (vmap, zero out self)
    N = all_humans.shape[0]
    def hf(i):
        opx, opy = all_humans[i,0], all_humans[i,1]
        fx, fy = _human_force(px, py, vx, vy, opx, opy, radius)
        is_self = jnp.sqrt((px-opx)**2 + (py-opy)**2) < 1e-3
        return jnp.where(is_self, 0.0, fx), jnp.where(is_self, 0.0, fy)
    hfxs, hfys = jax.vmap(hf)(jnp.arange(N))
    hfx, hfy = jnp.sum(hfxs), jnp.sum(hfys)

    # Obstacle + wall forces
    wfx, wfy = _wall_force(px, py, room_w, room_h)
    cfx, cfy = _circle_force(px, py, vx, vy, obs_circles)
    bfx, bfy = _box_force(px, py, obs_boxes)
    rfx, rfy = _robot_force(px, py, rx, ry, distract)

    # Sum and clamp acceleration (prevents single-step blow-up)
    ax = gfx + hfx + wfx + cfx + bfx + rfx
    ay = gfy + hfy + wfy + cfy + bfy + rfy
    a_max = MAX_SPEED / dt      # never change velocity by more than MAX_SPEED in one step
    amag  = jnp.sqrt(ax*ax + ay*ay) + _EPS
    ax    = jnp.where(amag > a_max, ax * a_max / amag, ax)
    ay    = jnp.where(amag > a_max, ay * a_max / amag, ay)

    # Euler velocity update
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt

    # Clamp speed
    spd    = jnp.sqrt(new_vx**2 + new_vy**2) + _EPS
    new_vx = jnp.where(spd > MAX_SPEED, new_vx * MAX_SPEED / spd, new_vx)
    new_vy = jnp.where(spd > MAX_SPEED, new_vy * MAX_SPEED / spd, new_vy)

    # Position update
    new_px = px + new_vx * dt
    new_py = py + new_vy * dt

    # ── Hard push-out: obstacles cannot be penetrated ─────────────────────────
    new_px, new_py = _pushout_walls(new_px, new_py, radius, room_w, room_h)
    new_px, new_py = _pushout_circles(new_px, new_py, radius, obs_circles)
    new_px, new_py = _pushout_boxes(new_px, new_py, radius, obs_boxes)

    # Heading: smooth low-pass toward velocity direction
    moving    = spd > 0.1
    vel_theta = jnp.arctan2(new_vy, new_vx)
    # Blend: 80% old heading, 20% new — smooth rotation
    d_theta   = (vel_theta - theta + jnp.pi) % (2*jnp.pi) - jnp.pi
    new_theta = jnp.where(moving, theta + 0.25 * d_theta, theta)

    # NaN guard
    new_px    = jnp.nan_to_num(new_px, nan=px)
    new_py    = jnp.nan_to_num(new_py, nan=py)
    new_vx    = jnp.nan_to_num(new_vx, nan=0.0)
    new_vy    = jnp.nan_to_num(new_vy, nan=0.0)
    new_theta = jnp.nan_to_num(new_theta, nan=theta)

    return jnp.stack([new_px, new_py, new_vx, new_vy,
                      new_theta, distract, wp_x, des_spd])


# ── Public API ────────────────────────────────────────────────────────────────

def update_all_humans(people_arr, rng_key, dt, rx, ry, rtheta, rv,
                      room_w, room_h, radius, obs_circles, obs_boxes):
    """Drop-in replacement. people_arr: (N,8) → (N,8)."""
    N    = people_arr.shape[0]
    keys = jax.random.split(rng_key, N)
    return jax.vmap(
        _update_one,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
    )(people_arr, keys, people_arr, obs_circles, obs_boxes,
      dt, rx, ry, room_w, room_h, radius)