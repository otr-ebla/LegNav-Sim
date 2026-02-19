"""
jax_humans.py — Crowd Simulation
=================================
Fixes vs original:
  - Removed @jax.vmap decorator from _update_single_human (double-vmap bug)
  - Removed redundant explicit vmap in update_all_humans (was double-vmapping)
  - is_distracted now stochastically toggled so the feature is actually used
  - Human-human repulsion added for more realistic crowd behaviour
  - Speed normalisation bug fixed (max_reaction_speed clamping logic cleaned)
  - Varied walking speeds are preserved from init (target_speed respected)
"""

import jax
import jax.numpy as jnp

# Constants
REACTION_DIST    = 1.2
WANDER_STRENGTH  = 0.6
MIN_STOP_TIME    = 1.0
MAX_STOP_TIME    = 5.0
STOP_PROBABILITY = 0.10         # per second
DISTRACT_PROB    = 0.05         # probability per second to toggle distraction
HUMAN_REP_DIST   = 0.8         # humans repel each other within this distance


def _update_single_human(
    human: jnp.ndarray,
    key: jnp.ndarray,
    all_humans: jnp.ndarray,   # (N, 8) full crowd for H-H repulsion
    dt: float,
    rx: float, ry: float, rtheta: float, rv: float,
    room_w: float, room_h: float, radius: float
) -> jnp.ndarray:
    """
    Updates a SINGLE human (no @jax.vmap here — applied externally in update_all_humans).
    human format: [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
    """
    px, py, vx, vy, angle, is_distracted, wait_timer, target_speed = human

    k1, k2, k3, k4 = jax.random.split(key, 4)
    rand_stop    = jax.random.uniform(k1)
    rand_wait    = jax.random.uniform(k2, minval=MIN_STOP_TIME, maxval=MAX_STOP_TIME)
    rand_wander  = jax.random.uniform(k3, minval=-1.0, maxval=1.0)
    rand_distract= jax.random.uniform(k4)

    # --- State machine ---
    new_timer  = wait_timer - dt
    is_waiting = new_timer > 0.0

    trigger_stop  = rand_stop < (STOP_PROBABILITY * dt) & ~is_waiting
    updated_timer = jnp.where(trigger_stop, rand_wait, new_timer)

    # Toggle distraction stochastically
    new_distracted = jnp.where(
        rand_distract < (DISTRACT_PROB * dt),
        1.0 - is_distracted,   # flip
        is_distracted
    )

    # Wander: update heading only if walking
    new_angle = jnp.where(
        is_waiting | trigger_stop,
        angle,
        angle + rand_wander * WANDER_STRENGTH * dt
    )
    new_angle = (new_angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # Desired velocity
    currently_stopped = updated_timer > 0.0
    des_vx = jnp.where(currently_stopped, 0.0, jnp.cos(new_angle) * target_speed)
    des_vy = jnp.where(currently_stopped, 0.0, jnp.sin(new_angle) * target_speed)

    # --- Robot repulsion ---
    dx_r  = px - rx
    dy_r  = py - ry
    dist_r= jnp.sqrt(dx_r**2 + dy_r**2)
    dist_r= jnp.maximum(dist_r, 0.01)

    urgency_r   = jnp.maximum(0.0, (REACTION_DIST - dist_r) / REACTION_DIST)
    rep_mag_r   = 12.0 * urgency_r**2

    nx_r = dx_r / dist_r
    ny_r = dy_r / dist_r
    tx_r = -ny_r
    ty_r =  nx_r

    cross_r    = jnp.cos(rtheta) * dy_r - jnp.sin(rtheta) * dx_r
    side_sign  = jnp.where(cross_r > 0, 1.0, -1.0)
    dodge_r    = 2.0 * urgency_r

    apply_r = (dist_r < REACTION_DIST) & (new_distracted < 0.5)
    rep_vx  = jnp.where(apply_r, nx_r * rep_mag_r * 0.3 + tx_r * side_sign * dodge_r, 0.0)
    rep_vy  = jnp.where(apply_r, ny_r * rep_mag_r * 0.3 + ty_r * side_sign * dodge_r, 0.0)

    # --- Human-human repulsion (social force) ---
    dxh  = px - all_humans[:, 0]   # (N,)
    dyh  = py - all_humans[:, 1]
    disth= jnp.sqrt(dxh**2 + dyh**2)
    disth= jnp.maximum(disth, 0.01)

    urg_h = jnp.maximum(0.0, (HUMAN_REP_DIST - disth) / HUMAN_REP_DIST)   # (N,)
    # Exclude self (distance ~0): already handled by max(disth,0.01) but weight = 0 when urg=0
    hh_vx = jnp.sum((dxh / disth) * urg_h * 3.0)
    hh_vy = jnp.sum((dyh / disth) * urg_h * 3.0)

    final_vx = des_vx + rep_vx + hh_vx
    final_vy = des_vy + rep_vy + hh_vy

    # Speed cap
    spd = jnp.sqrt(final_vx**2 + final_vy**2)
    max_spd = jnp.maximum(target_speed * 1.6, 0.6)
    scale   = jnp.where(spd > max_spd, max_spd / jnp.maximum(spd, 1e-6), 1.0)
    final_vx *= scale
    final_vy *= scale

    # Move
    new_px = px + final_vx * dt
    new_py = py + final_vy * dt

    # Bounce off walls
    bxl = new_px - radius < 0.0
    bxh = new_px + radius > room_w
    byl = new_py - radius < 0.0
    byh = new_py + radius > room_h

    new_px   = jnp.where(bxl, radius,          jnp.where(bxh, room_w - radius, new_px))
    final_vx = jnp.where(bxl | bxh, -final_vx, final_vx)
    new_py   = jnp.where(byl, radius,          jnp.where(byh, room_h - radius, new_py))
    final_vy = jnp.where(byl | byh, -final_vy, final_vy)

    return jnp.stack([new_px, new_py, final_vx, final_vy, new_angle,
                      new_distracted, updated_timer, target_speed])


@jax.jit
def update_all_humans(
    people_arr: jnp.ndarray,
    rng_key: jnp.ndarray,
    dt: float,
    rx: float, ry: float, rtheta: float, rv: float,
    room_w: float, room_h: float,
    radius: float
) -> jnp.ndarray:
    """
    Vectorised crowd update.
    FIX: vmap is applied HERE only (not also on _update_single_human).
    """
    num_people = people_arr.shape[0]
    keys = jax.random.split(rng_key, num_people)

    # Pass full people_arr for H-H repulsion; vmap maps over (human, key) axis 0
    return jax.vmap(
        _update_single_human,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
    )(people_arr, keys, people_arr, dt, rx, ry, rtheta, rv, room_w, room_h, radius)