"""
jax_legs.py — Planted-Foot Gait Model (Body-Centric Reference Frame)
======================================================================
Fix 1 — Body-centric interpolation (eliminates rubber-band stretching):
  swing_target and swing_start are stored as LOCAL OFFSETS in the person's
  body frame at step-initiation time: [fwd_component, lat_component].
  Each frame, world positions are reconstructed as:
      body_xy + R(theta) @ local_offset
  Because the offset is relative to the pelvis, any JHSFM push-out that
  teleports the body automatically carries the feet with it. The absolute
  world anchors that caused stretching are gone.

Fix 2 — Kinematic leash (permanent safety net):
  After world-space reconstruction, if any foot is more than LEG_LEASH_MAX
  metres from the body centre it is snapped back to the natural hip baseline.
  This costs a single jnp.where per foot and prevents any residual anomaly
  from slow drift, extreme push-outs, or scenario teleportation.

Fix 3 (pre-existing) — Forward anticipation in _next_plant:
  Target is placed at body + fwd*(speed*swing_duration*0.8) ± hip/2 so the
  foot lands roughly under the future hip rather than behind it.

STATE stored per human (inside EnvState.foot_state):
  foot_state : (N, 10) float32
    [0:2]  left_foot_xy       — current WORLD position of left foot  (for rendering)
    [2:4]  right_foot_xy      — current WORLD position of right foot (for rendering)
    [4]    phase              — gait phase in [0, 1)
    [5]    stance_leg         — 0.0=left is stance / right swings,
                                1.0=right is stance / left swings
    [6:8]  swing_target_local — swing target as LOCAL body-frame offset [fwd, lat]
    [8:10] swing_start_local  — swing start  as LOCAL body-frame offset [fwd, lat]
                                (frozen at the moment a new step begins)

PUBLIC API:
  init_foot_state(people, key)                    → foot_state (N, 10)
  advance_feet(foot_state, people, dt)            → foot_state (N, 10)
  get_leg_positions(foot_state)                   → left_xy (N,2), right_xy (N,2)
  get_leg_circles(people, foot_state, use_legs)   → (2N,3) or (N,3)
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


def _next_plant(side, speed):
    """
    Compute where the swing foot should plant next, as a LOCAL body-frame offset.

    Returning a local offset [fwd_component, lat_component] instead of an
    absolute world coordinate means the target is immune to JHSFM push-outs:
    when the body teleports, the offset stays valid and the world position is
    reconstructed correctly from the new body position each frame.

      side  : float  +1.0 = left foot,  -1.0 = right foot
      speed : float  current walking speed (m/s)

    Returns local_offset (2,): [fwd_offset, lat_offset]
    """
    speed_factor   = jnp.clip(speed / STEP_SPEED, 0.3, 1.0)
    cadence        = STEP_FREQ * speed_factor
    swing_duration = 1.0 / cadence

    fwd_offset = speed * swing_duration * 0.5 - SHOE_LENGTH * 0.5   # forward offset places foot under future hip, not behind
    lat_offset = HIP_WIDTH * 0.5 * side
    return jnp.array([fwd_offset, lat_offset])   # (2,) local offset


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

    # Derive stance from phase: 0→0.5 = right swings (left stance), 0.5→1 = left swings
    stance = jnp.where(phase < 0.5, 0.0, 1.0)   # 0=left stance, 1=right stance

    # Local offsets at rest: [0 fwd, ±half lat] — foot is directly at hip baseline
    # These are the local-frame offsets; no forward anticipation at init (speed=0)
    left_local  = jnp.stack([jnp.zeros(N), jnp.full(N,  half)], axis=-1)   # (N,2)
    right_local = jnp.stack([jnp.zeros(N), jnp.full(N, -half)], axis=-1)   # (N,2)

    # swing_target_local = local offset of the swinging foot's *current* planted pos
    swing_target_local = jnp.where(
        stance[:, None] == 0.0,
        right_local,   # right is swinging
        left_local,    # left  is swinging
    )
    # swing_start_local matches target at init (no step in progress)
    swing_start_local = swing_target_local

    left_theta  = theta
    right_theta = theta

    return jnp.concatenate([
        left_xy,                 # [0:2]
        right_xy,                # [2:4]
        phase[:, None],          # [4]
        stance[:, None],         # [5]
        swing_target_local,      # [6:8]
        swing_start_local,       # [8:10]
        left_theta[:, None],     # [10]
        right_theta[:, None],    # [11]
    ], axis=-1)   # (N, 12)


# ── Per-step update ───────────────────────────────────────────────────────────

def _local_to_world(local_offset, body_xy, fwd, lat):
    """
    Transform a 2D local body-frame offset to world coordinates.

      local_offset : (2,)  [fwd_component, lat_component]
      body_xy      : (2,)  body centre in world space
      fwd          : (2,)  forward unit vector
      lat          : (2,)  lateral unit vector

    Returns world_pos (2,).
    """
    return body_xy + fwd * local_offset[0] + lat * local_offset[1]


def _world_to_local(world_pos, body_xy, fwd, lat):
    """
    Convert a world position to a local body-frame offset.

      world_pos : (2,)  position in world space
      body_xy   : (2,)  body centre in world space
      fwd       : (2,)  forward unit vector
      lat       : (2,)  lateral unit vector

    Returns local_offset (2,) = [fwd_component, lat_component].
    """
    delta = world_pos - body_xy
    return jnp.array([jnp.dot(delta, fwd), jnp.dot(delta, lat)])


def _update_single(fs_i, person_i, dt):
    """
    Update one human's foot state for one env timestep.

    fs_i     : (10,)  foot state
    person_i : (≥5,)  people row [px, py, vx, vy, theta, ...]
    dt       : float  env timestep (seconds)

    All interpolation happens in LOCAL body-frame coordinates.
    World positions [0:4] are reconstructed from local offsets each frame
    using the current body pose, so JHSFM push-outs carry the feet with the
    body automatically (Fix 1).  A leash clamp (Fix 2) then guarantees no
    foot can ever stray more than LEG_LEASH_MAX metres from the body.
    """
    left_xy           = fs_i[0:2]    # world — used as planted anchor when stance
    right_xy          = fs_i[2:4]    # world — used as planted anchor when stance
    phase             = fs_i[4]
    stance            = fs_i[5]      # 0 = left stance / right swings
                                     # 1 = right stance / left swings
    swing_target_local = fs_i[6:8]   # local offset [fwd, lat]
    swing_start_local  = fs_i[8:10]  # local offset [fwd, lat], frozen at step start
    left_theta        = fs_i[10]
    right_theta       = fs_i[11]

    px    = person_i[0]
    py    = person_i[1]
    vx    = person_i[2]
    vy    = person_i[3]
    theta = person_i[4]
    speed = jnp.sqrt(vx**2 + vy**2 + 1e-8)

    body_xy = jnp.array([px, py])
    fwd, lat = _fwd_lat(theta[None])   # (1,2) each
    fwd = fwd[0]; lat = lat[0]         # (2,)

    is_moving = speed > SPEED_THRESH

    # ── Phase advance ─────────────────────────────────────────────────────────
    speed_factor = jnp.clip(speed / STEP_SPEED, 0.3, 1.0)
    cadence      = 1#STEP_FREQ * speed_factor
    new_phase    = (phase + cadence * dt) % 1.0

    # crossed = True when phase wrapped around 0 → new step began
    crossed = new_phase < phase

    # ── Interpolate swing foot in LOCAL space, then lift to world ────────────
    # t advances smoothly within [0, 1) over one half-cycle.
    # Both swing_start_local and swing_target_local are frozen local offsets,
    # so their world positions track the body automatically on every frame.
    t = new_phase
    swing_local = swing_start_local * (1.0 - t) + swing_target_local * t
    swing_xy    = _local_to_world(swing_local, body_xy, fwd, lat)

    # ── On plant: swap stance, compute new step targets ───────────────────────
    new_stance = jnp.where(crossed, 1.0 - stance, stance)

    # New local targets for the foot that will next swing
    # (after crossing, the previously-stance foot becomes the new swing foot)
    next_left_local  = _next_plant(+1.0, speed)
    next_right_local = _next_plant(-1.0, speed)

    new_swing_target_local = jnp.where(
        crossed,
        jnp.where(stance == 0.0, next_left_local, next_right_local),
        swing_target_local,
    )

    # Freeze swing_start_local at the moment a new step begins.
    # The new start = where the newly-swinging foot is currently planted,
    # expressed as a local offset from the body at the crossing instant.
    left_local_now  = _world_to_local(left_xy,  body_xy, fwd, lat)
    right_local_now = _world_to_local(right_xy, body_xy, fwd, lat)
    new_swing_start_on_cross = jnp.where(
        stance == 0.0,
        left_local_now,   # left was stance → left now becomes the swing foot
        right_local_now,  # right was stance → right now becomes the swing foot
    )
    new_swing_start_local = jnp.where(crossed, new_swing_start_on_cross, swing_start_local)

    # ── Write swing position to the correct foot ──────────────────────────────
    # Stance foot stays at its planted world position (not modified by swing interp)
    new_right_xy = jnp.where(stance == 0.0, swing_xy,  right_xy)
    new_left_xy  = jnp.where(stance == 1.0, swing_xy,  left_xy)

    # On plant: snap the foot that finished swinging to its exact target (world)
    target_world = _local_to_world(swing_target_local, body_xy, fwd, lat)
    new_right_xy = jnp.where(crossed & (stance == 0.0), target_world, new_right_xy)
    new_left_xy  = jnp.where(crossed & (stance == 1.0), target_world, new_left_xy)

    # ── Fix 2: Kinematic leash ────────────────────────────────────────────────
    # If any foot drifts more than LEG_LEASH_MAX from the body (can happen after
    # extreme JHSFM push-outs or scenario teleportation), snap it back to the
    # natural hip baseline in world space.
    left_hip_world  = _local_to_world(jnp.array([0.0,  HIP_WIDTH * 0.5]), body_xy, fwd, lat)
    right_hip_world = _local_to_world(jnp.array([0.0, -HIP_WIDTH * 0.5]), body_xy, fwd, lat)

    left_dist  = jnp.sqrt(jnp.sum((new_left_xy  - body_xy)**2) + 1e-8)
    right_dist = jnp.sqrt(jnp.sum((new_right_xy - body_xy)**2) + 1e-8)

    new_left_xy  = jnp.where(left_dist  > LEG_LEASH_MAX, left_hip_world,  new_left_xy)
    new_right_xy = jnp.where(right_dist > LEG_LEASH_MAX, right_hip_world, new_right_xy)

    # foot_state[10] and [11] store the completely frozen planted anchors.
    # They update to the body's current heading ONLY on the exact frame the foot plants.
    new_left_theta  = jnp.where(crossed & (stance == 1.0), theta, left_theta)
    new_right_theta = jnp.where(crossed & (stance == 0.0), theta, right_theta)

    # ── Freeze everything if stationary ───────────────────────────────────────
    new_phase              = jnp.where(is_moving, new_phase,              phase)
    new_stance             = jnp.where(is_moving, new_stance,             stance)
    new_swing_target_local = jnp.where(is_moving, new_swing_target_local, swing_target_local)
    new_swing_start_local  = jnp.where(is_moving, new_swing_start_local,  swing_start_local)
    new_left_xy            = jnp.where(is_moving, new_left_xy,            left_xy)
    new_right_xy           = jnp.where(is_moving, new_right_xy,           right_xy)
    new_left_theta         = jnp.where(is_moving, new_left_theta,         left_theta)
    new_right_theta        = jnp.where(is_moving, new_right_theta,        right_theta)

    return jnp.concatenate([
        new_left_xy,                   # [0:2]
        new_right_xy,                  # [2:4]
        new_phase[None],               # [4]
        new_stance[None],              # [5]
        new_swing_target_local,        # [6:8]
        new_swing_start_local,         # [8:10]
        new_left_theta[None],          # [10]
        new_right_theta[None],         # [11]
    ])   # (12,)


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

    # Extract the frozen planted anchors
    left_anchor  = foot_state[:, 10]
    right_anchor = foot_state[:, 11]
    
    # Extract phase (t) and stance to interpolate the swinging foot dynamically
    t      = foot_state[:, 4]
    stance = foot_state[:, 5]
    theta  = people[:, 4]  # Body heading
    
    # Smoothly interpolate the swinging foot from its planted anchor to the body heading
    d_left  = (theta - left_anchor + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    d_right = (theta - right_anchor + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    
    actual_left  = left_anchor  + jnp.where(stance == 1.0, t * d_left, 0.0)
    actual_right = right_anchor + jnp.where(stance == 0.0, t * d_right, 0.0)

    # Stack into (2N,) arrays directly
    all_xy    = jnp.concatenate([left_xy, right_xy], axis=0)         # (2N, 2)
    all_theta = jnp.concatenate([actual_left, actual_right], axis=0) # (2N,)

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