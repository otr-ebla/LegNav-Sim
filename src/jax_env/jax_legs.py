"""
jax_legs.py — Planted-Foot Gait Model (Body-Centric Reference Frame)
======================================================================
"""

import jax
import jax.numpy as jnp
from config import SimConfig

# ── Constants ─────────────────────────────────────────────────────────────────
LEG_RADIUS   = SimConfig.LEG_RADIUS     # m   — LiDAR cross-section
PEOPLE_RADIUS = SimConfig.PEOPLE_RADIUS    # m   — body cylinder radius (mirrors jax_env.PEOPLE_RADIUS)
HIP_WIDTH    = SimConfig.HIP_WIDTH     # m   — lateral separation between feet
STEP_SPEED   = 2.5      # m/s — reference speed for cadence scaling
STEP_FREQ    = 3.5        # half-steps/s — lower = longer, more visible strides
SPEED_THRESH = 0.1    # m/s — below this, feet freeze
LEG_LEASH_MAX = 0.6   # m   — max allowed foot-to-body distance (leash safety net)

# Shoe geometry
SHOE_LENGTH  = SimConfig.SHOE_LENGTH    # m   — toe-to-heel length
SHOE_WIDTH   = SimConfig.SHOE_WIDTH  # m   — matches leg circle diameter (0.17 m)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fwd_lat(theta):
    """Forward and lateral unit vectors from heading angle theta (scalar or (N,))."""
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    fwd = jnp.stack([cos_t, sin_t], axis=-1)   # (..., 2)
    lat = jnp.stack([-sin_t, cos_t], axis=-1)  # (..., 2) — 90° CCW = left
    return fwd, lat





# ── Initialisation ────────────────────────────────────────────────────────────

def init_foot_state(people: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Initialise foot state from the people array at reset time.

    people : (N, ≥5)  [px, py, vx, vy, theta, ...]
    key    : JAX random key for staggering initial phases

    Returns foot_state (N, 10):
      [0:2]  left_xy         — world positions (for rendering/collision)
      [2:4]  right_xy
      [4]    phase
      [5]    stance
      [6:8]  swing_target_local — local body-frame offset [fwd, lat]
      [8:10] swing_start_local  — local body-frame offset [fwd, lat]
    """
    N     = people.shape[0]
    px    = people[:, 0]
    py    = people[:, 1]
    theta = people[:, 4]

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    lat_x = -sin_t
    lat_y =  cos_t

    half = HIP_WIDTH * 0.5

    # World positions — used for rendering and LiDAR collision detection
    left_xy  = jnp.stack([px + lat_x * half, py + lat_y * half], axis=-1)   # (N,2)
    right_xy = jnp.stack([px - lat_x * half, py - lat_y * half], axis=-1)   # (N,2)

    # Stagger phases so humans aren't all in sync
    phase = jax.random.uniform(key, (N,), minval=0.0, maxval=1.0)

    left_theta  = theta
    right_theta = theta

    return jnp.concatenate([
        left_xy,                 # [0:2] current left pos
        right_xy,                # [2:4] current right pos
        phase[:, None],          # [4]   gait phase
        left_xy,                 # [5:7] left anchor (where foot was planted)
        right_xy,                # [7:9] right anchor (where foot was planted)
        jnp.zeros((N, 1)),       # [9]   padding to keep thetas stable at indices 10,11
        left_theta[:, None],     # [10]  left heading
        right_theta[:, None],    # [11]  right heading
    ], axis=-1)   # (N, 12)




def _update_single(fs_i, person_i, dt):
    """Update one human's foot state using real-time Inverse Kinematics."""
    left_curr    = fs_i[0:2]
    right_curr   = fs_i[2:4]
    phase        = fs_i[4]
    left_anchor  = fs_i[5:7]
    right_anchor = fs_i[7:9]
    left_theta   = fs_i[10]
    right_theta  = fs_i[11]

    px, py, vx, vy, theta = person_i[0], person_i[1], person_i[2], person_i[3], person_i[4]
    speed = jnp.hypot(vx, vy) + 1e-8
    body_xy = jnp.array([px, py])

    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    fwd = jnp.array([cos_t, sin_t])
    lat = jnp.array([-sin_t, cos_t])

    is_moving = speed > SPEED_THRESH

    # 1. Phase advance (Corrected cadence without hardcode)
    cadence   = STEP_FREQ * jnp.clip(speed / STEP_SPEED, 0.3, 1.0)
    new_phase = (phase + cadence * dt) % 1.0

    # 2. Dynamic Targets: Where feet SHOULD land based on CURRENT velocity and body pos
    stride_len   = speed * (1.0 / cadence) * 0.5 - SHOE_LENGTH * 0.5
    target_left  = body_xy + fwd * stride_len + lat * (HIP_WIDTH * 0.5)
    target_right = body_xy + fwd * stride_len - lat * (HIP_WIDTH * 0.5)

    # 3. Detect Phase Crossings to plant the feet and update anchors
    crossed_half = (phase < 0.5) & (new_phase >= 0.5)
    crossed_zero = (phase > 0.5) & (new_phase < 0.5)
    
    # When a foot plants, its anchor becomes the current target.
    new_left_anchor  = jnp.where(crossed_zero, target_left,  left_anchor)
    new_right_anchor = jnp.where(crossed_half, target_right, right_anchor)

    # 4. Leash Safety Net: Applied directly to the anchors
    hip_left  = body_xy + lat * (HIP_WIDTH * 0.5)
    hip_right = body_xy - lat * (HIP_WIDTH * 0.5)
    l_dist = jnp.hypot(new_left_anchor[0] - px, new_left_anchor[1] - py)
    r_dist = jnp.hypot(new_right_anchor[0] - px, new_right_anchor[1] - py)
    
    new_left_anchor  = jnp.where(l_dist > LEG_LEASH_MAX, hip_left,  new_left_anchor)
    new_right_anchor = jnp.where(r_dist > LEG_LEASH_MAX, hip_right, new_right_anchor)

    # 5. Swing Interpolation: lerp from anchor to real-time target
    is_left_stance = new_phase < 0.5
    t_right = new_phase * 2.0
    t_left  = (new_phase - 0.5) * 2.0
    
    swing_left  = new_left_anchor  * (1.0 - t_left)  + target_left  * t_left
    swing_right = new_right_anchor * (1.0 - t_right) + target_right * t_right

    new_left_curr  = jnp.where(is_left_stance, new_left_anchor, swing_left)
    new_right_curr = jnp.where(is_left_stance, swing_right, new_right_anchor)

    # 6. Yaw Orientations
    fast_enough = speed > 0.05
    new_left_theta  = jnp.where(fast_enough & ~is_left_stance, theta, left_theta)
    new_right_theta = jnp.where(fast_enough & is_left_stance, theta, right_theta)

    # 7. Freeze everything if stationary to prevent RL noise
    new_phase        = jnp.where(is_moving, new_phase,        phase)
    new_left_curr    = jnp.where(is_moving, new_left_curr,    left_curr)
    new_right_curr   = jnp.where(is_moving, new_right_curr,   right_curr)
    new_left_anchor  = jnp.where(is_moving, new_left_anchor,  left_anchor)
    new_right_anchor = jnp.where(is_moving, new_right_anchor, right_anchor)
    new_left_theta   = jnp.where(is_moving, new_left_theta,   left_theta)
    new_right_theta  = jnp.where(is_moving, new_right_theta,  right_theta)

    return jnp.concatenate([
        new_left_curr,          # [0:2]
        new_right_curr,         # [2:4]
        new_phase[None],        # [4]
        new_left_anchor,        # [5:7]
        new_right_anchor,       # [7:9]
        jnp.zeros((1,)),        # [9] Padding
        new_left_theta[None],   # [10]
        new_right_theta[None],  # [11]
    ])


def advance_feet(
    foot_state: jnp.ndarray,   # (N, 10)
    people:     jnp.ndarray,   # (N, ≥5)
    dt:         float,
) -> jnp.ndarray:
    """Advance all humans' foot states by one env timestep (fully vmapped)."""
    return jax.vmap(_update_single, in_axes=(0, 0, None))(foot_state, people, dt)


# ── Position extraction ───────────────────────────────────────────────────────

def get_leg_positions(foot_state: jnp.ndarray) -> tuple:
    """
    Extract current foot world positions.

    Returns:
      left_legs  : (N, 2)
      right_legs : (N, 2)
    """
    return foot_state[:, 0:2], foot_state[:, 2:4]


# ── LiDAR circle array ────────────────────────────────────────────────────────

def get_leg_circles(
    people:     jnp.ndarray,   # (N, ≥5)  — used only for cylinder fallback
    foot_state: jnp.ndarray,   # (N, 10)
    use_legs:   bool = True,
) -> jnp.ndarray:
    """
    Return [cx, cy, r] circle array for compute_lidar.

    use_legs=True  → (2N, 3)  individual leg circles
    use_legs=False → (N,  3)  body cylinders (fallback)
    """
    N = people.shape[0]

    if use_legs:
        left_xy, right_xy = get_leg_positions(foot_state)
        all_x = jnp.concatenate([left_xy[:, 0], right_xy[:, 0]], axis=0)
        all_y = jnp.concatenate([left_xy[:, 1], right_xy[:, 1]], axis=0)
        all_r = jnp.full((2 * N,), LEG_RADIUS)
        return jnp.stack([all_x, all_y, all_r], axis=-1)   # (2N, 3)
    else:
        return jnp.stack([
            people[:, 0],
            people[:, 1],
            jnp.full((N,), PEOPLE_RADIUS),
        ], axis=-1)   # (N, 3)


# ── Shoe AABB array ───────────────────────────────────────────────────────────

def get_shoe_boxes(
    people:     jnp.ndarray,   # (N, ≥5)  — provides heading theta
    foot_state: jnp.ndarray,   # (N, 10)
) -> jnp.ndarray:
    """
    Return axis-aligned bounding boxes (AABBs) for all shoes.

    Each shoe is a rectangle of SHOE_LENGTH × SHOE_WIDTH centred at:
        shoe_centre = foot_xy + fwd * (SHOE_LENGTH / 2)
    oriented along the person's walking heading (theta).

    Because jax_physics only supports AABBs, we compute the axis-aligned
    bounding box of the rotated rectangle:
        hw = |cos θ| * (L/2) + |sin θ| * (W/2)
        hh = |sin θ| * (L/2) + |cos θ| * (W/2)

    Returns (2N, 4)  [cx, cy, hw, hh]  — left shoes first, then right shoes.
    """
    left_xy,  right_xy  = get_leg_positions(foot_state)   # (N, 2) each

    # Stack into (2N,) arrays directly
    all_xy    = jnp.concatenate([left_xy, right_xy], axis=0)         # (2N, 2)
    
    # Left and right thetas are already correctly updated in the step logic
    all_theta = jnp.concatenate([foot_state[:, 10], foot_state[:, 11]], axis=0) # (2N,)

    # Compute forward vectors for every single shoe independently
    fwd, _ = _fwd_lat(all_theta)                                   # (2N, 2)

    half_L = SHOE_LENGTH * 0.5
    half_W = SHOE_WIDTH  * 0.5

    # Shoe centres: planted position + forward offset of half the shoe length
    all_cx = all_xy + fwd * half_L                                 # (2N, 2)

    # AABB half-extents using the independent foot angle
    abs_cos = jnp.abs(jnp.cos(all_theta))                          # (2N,)
    abs_sin = jnp.abs(jnp.sin(all_theta))                          # (2N,)
    all_hw = abs_cos * half_L + abs_sin * half_W                   # (2N,)
    all_hh = abs_sin * half_L + abs_cos * half_W                   # (2N,)

    return jnp.stack([all_cx[:, 0], all_cx[:, 1], all_hw, all_hh], axis=-1)   # (2N, 4)


# ── Backward compatibility shim ───────────────────────────────────────────────

def advance_phase(phase_offsets, people, dt):
    """DEPRECATED — use advance_feet(foot_state, people, dt) instead."""
    raise RuntimeError(
        "advance_phase is no longer used. Call advance_feet(state.foot_state, "
        "new_people, dt) and store foot_state (N,10) in EnvState."
    )