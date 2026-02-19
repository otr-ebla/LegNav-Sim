import jax
import jax.numpy as jnp

# Constants
REACTION_DIST = 1.0
WANDER_STRENGTH = 0.8
MIN_STOP_TIME = 1.0
MAX_STOP_TIME = 5.0
STOP_PROBABILITY = 0.15

@jax.vmap
def _update_single_human(human: jnp.ndarray, key: jnp.ndarray, dt: float, rx: float, ry: float, rtheta: float, rv: float, room_w: float, room_h: float, radius: float):
    """
    Updates a single human. 
    @jax.vmap will broadcast this across the entire human array.
    human array format: [px, py, vx, vy, angle, is_distracted, wait_timer, target_speed]
    """
    px, py, vx, vy, angle, is_distracted, wait_timer, target_speed = human

    # 1. PRNG Splitting (Generating 3 parallel random numbers for this specific human)
    k1, k2, k3 = jax.random.split(key, 3)
    rand_stop = jax.random.uniform(k1)
    rand_wait_time = jax.random.uniform(k2, minval=MIN_STOP_TIME, maxval=MAX_STOP_TIME)
    rand_wander = jax.random.uniform(k3, minval=-1.0, maxval=1.0)

    # 2. State Machine: Waiting vs Walking (Branchless)
    new_timer = wait_timer - dt
    is_waiting = new_timer > 0.0

    # Logic for Walking state
    trigger_stop = rand_stop < (STOP_PROBABILITY * dt)
    
    # If we trigger a stop, set the timer. Otherwise, keep the decremented timer.
    updated_timer = jnp.where(trigger_stop & ~is_waiting, rand_wait_time, new_timer)
    
    # Wandering logic: add noise to angle if walking
    new_angle = jnp.where(is_waiting | trigger_stop, angle, angle + rand_wander * WANDER_STRENGTH * dt)
    new_angle = (new_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Desired velocities
    des_vx = jnp.where(updated_timer > 0.0, 0.0, jnp.cos(new_angle) * target_speed)
    des_vy = jnp.where(updated_timer > 0.0, 0.0, jnp.sin(new_angle) * target_speed)

    # 3. Repulsion from Robot
    dx = px - rx
    dy = py - ry
    dist = jnp.sqrt(dx*dx + dy*dy)
    dist = jnp.maximum(dist, 0.01) # Prevent division by zero
    
    urgency = jnp.maximum(0.0, (REACTION_DIST - dist) / REACTION_DIST)
    rep_force_mag = 13.0 * (urgency ** 2)
    
    nx = dx / dist
    ny = dy / dist
    tx = -ny
    ty = nx
    
    # Cross product to determine dodge direction
    r_cos = jnp.cos(rtheta)
    r_sin = jnp.sin(rtheta)
    cross_prod = (r_cos * dy) - (r_sin * dx)
    side_sign = jnp.where(cross_prod > 0, 1.0, -1.0)
    
    # Apply repulsion only if close enough AND not distracted
    apply_repulsion = (dist < REACTION_DIST) & (is_distracted < 0.5)
    dodge_mag = 2.0 * urgency
    
    final_vx = jnp.where(apply_repulsion, des_vx + (nx * rep_force_mag * 0.3) + (tx * side_sign * dodge_mag), des_vx)
    final_vy = jnp.where(apply_repulsion, des_vy + (ny * rep_force_mag * 0.3) + (ty * side_sign * dodge_mag), des_vy)
    
    # Cap speed
    current_speed = jnp.sqrt(final_vx**2 + final_vy**2)
    max_reaction_speed = jnp.maximum(target_speed * 1.5, 0.5)
    scale = jnp.where(current_speed > max_reaction_speed, max_reaction_speed / jnp.maximum(current_speed, 1e-5), 1.0)
    
    final_vx = final_vx * scale
    final_vy = final_vy * scale

    # 4. Move and Bounce off walls
    new_px = px + final_vx * dt
    new_py = py + final_vy * dt
    
    bounce_x_low = new_px - radius < 0
    bounce_x_high = new_px + radius > room_w
    bounce_y_low = new_py - radius < 0
    bounce_y_high = new_py + radius > room_h
    
    new_px = jnp.where(bounce_x_low, radius, jnp.where(bounce_x_high, room_w - radius, new_px))
    final_vx = jnp.where(bounce_x_low | bounce_x_high, -final_vx, final_vx)
    
    new_py = jnp.where(bounce_y_low, radius, jnp.where(bounce_y_high, room_h - radius, new_py))
    final_vy = jnp.where(bounce_y_low | bounce_y_high, -final_vy, final_vy)

    # Return the updated row
    return jnp.stack([new_px, new_py, final_vx, final_vy, new_angle, is_distracted, updated_timer, target_speed])

@jax.jit
def update_all_humans(people_arr: jnp.ndarray, rng_key: jnp.ndarray, dt: float, rx: float, ry: float, rtheta: float, rv: float, room_w: float, room_h: float, radius: float) -> jnp.ndarray:
    """
    Main entry point to update the whole crowd.
    """
    num_people = people_arr.shape[0]
    
    # Split the main key into N subkeys, one for each human
    keys = jax.random.split(rng_key, num_people)
    
    # _update_single_human expects (human_1D, key_1D, scalars...)
    # We map over axis 0 of people_arr and axis 0 of keys. All other args are static (None).
    vmap_update = jax.vmap(_update_single_human, in_axes=(0, 0, None, None, None, None, None, None, None, None))
    
    updated_people = vmap_update(people_arr, keys, dt, rx, ry, rtheta, rv, room_w, room_h, radius)
    return updated_people