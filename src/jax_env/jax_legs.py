"""
jax_legs.py — Planted-Foot Gait Model
======================================
Fixes feet falling behind the human body.

ROOT CAUSES FIXED:
  1. No forward offset in _next_plant — target landed under the *current* hip,
     not where the hip will be when the foot lands, so feet always trailed.
  2. swing_start was recomputed from the current planted position every frame
     instead of being frozen at the moment a new step begins — this caused the
     interpolation arc to re-anchor behind the body each frame.
  3. t = new_phase was used as the interpolation parameter, but new_phase resets
     to ~0 after every plant event, snapping the swing foot back to swing_start
     at the beginning of each half-cycle.

FIXES:
  • foot_state expanded to (N, 10): added swing_start_xy [8:10], frozen on
    each plant event and used as the stable interpolation origin.
  • t = new_phase / 1.0 is now valid because swing_start is frozen, so the
    foot smoothly travels from swing_start → swing_target over one half-cycle.
  • _next_plant adds a forward offset: body_xy + fwd * (speed * swing_duration * 0.5)
    so the target anticipates where the hip will be when the foot lands.

STATE stored per human (all in world space, inside EnvState.foot_state):
  foot_state : (N, 10) float32
    [0:2]  left_foot_xy    — current world position of left foot
    [2:4]  right_foot_xy   — current world position of right foot
    [4]    phase           — gait phase in [0, 1)
    [5]    stance_leg      — 0.0=left is stance / right swings,
                             1.0=right is stance / left swings
    [6:8]  swing_target_xy — where the swing foot is heading (world space)
    [8:10] swing_start_xy  — where the swing foot started (frozen at step begin)

PUBLIC API:
  init_foot_state(people, key)                    → foot_state (N, 10)
  advance_feet(foot_state, people, dt)            → foot_state (N, 10)
  get_leg_positions(foot_state)                   → left_xy (N,2), right_xy (N,2)
  get_leg_circles(people, foot_state, use_legs)   → (2N,3) or (N,3)
"""

import jax
import jax.numpy as jnp

# ── Constants ─────────────────────────────────────────────────────────────────
LEG_RADIUS   = 0.085    # m   — LiDAR cross-section
HIP_WIDTH    = 0.18    # m   — lateral separation between feet at plant
STEP_SPEED   = 2.5     # m/s — reference speed for cadence scaling
STEP_FREQ    = 10     # half-steps/s per leg at STEP_SPEED
SPEED_THRESH = 0.1    # m/s — below this, feet freeze


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fwd_lat(theta):
    """Forward and lateral unit vectors from heading angle theta (scalar or (N,))."""
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    fwd = jnp.stack([cos_t, sin_t], axis=-1)   # (..., 2)
    lat = jnp.stack([-sin_t, cos_t], axis=-1)  # (..., 2) — 90° CCW = left
    return fwd, lat


def _next_plant(body_xy, fwd, lat, side, speed):
    """
    Compute where the swing foot should plant next.

    Anticipates forward body motion during the swing phase so the foot lands
    roughly under the hip — not behind it.

      body_xy : (2,)   current body centre in world space
      fwd     : (2,)   forward unit vector
      lat     : (2,)   lateral unit vector (left = +1)
      side    : float  +1.0 = left foot,  -1.0 = right foot
      speed   : float  current walking speed (m/s)

    Returns target (2,) in world space.
    """
    speed_factor   = jnp.clip(speed / STEP_SPEED, 0.3, 1.0)
    cadence        = STEP_FREQ * speed_factor          # half-steps/s
    swing_duration = 1.0 / cadence                    # seconds for one half-step

    # Anticipate where the body centre will be at mid-swing
    # (0.5 × duration gives the midpoint; the foot lands after a full duration
    #  but placing target at body + full-duration keeps feet under the body)
    fwd_offset = speed * swing_duration * 0.5

    target = body_xy + fwd * fwd_offset + lat * (HIP_WIDTH * 0.5 * side)
    return target   # (2,)


# ── Initialisation ────────────────────────────────────────────────────────────

def init_foot_state(people: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Initialise foot state from the people array at reset time.

    people : (N, ≥5)  [px, py, vx, vy, theta, ...]
    key    : JAX random key for staggering initial phases

    Returns foot_state (N, 10):
      [0:2] left_xy  [2:4] right_xy  [4] phase  [5] stance
      [6:8] swing_target  [8:10] swing_start
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

    # Feet planted at natural hip positions
    left_xy  = jnp.stack([px + lat_x * half, py + lat_y * half], axis=-1)   # (N,2)
    right_xy = jnp.stack([px - lat_x * half, py - lat_y * half], axis=-1)   # (N,2)

    # Stagger phases so humans aren't all in sync
    phase = jax.random.uniform(key, (N,), minval=0.0, maxval=1.0)

    # Derive stance from phase: 0→0.5 = right swings (left stance), 0.5→1 = left swings
    stance = jnp.where(phase < 0.5, 0.0, 1.0)   # 0=left stance, 1=right stance

    # Swing target = current position of the swinging foot (no step pending yet)
    swing_target = jnp.where(
        stance[:, None] == 0.0,
        right_xy,   # right is swinging → its own planted pos
        left_xy,    # left  is swinging → its own planted pos
    )

    # Swing start matches swing target at init (feet not moving yet)
    swing_start = swing_target

    return jnp.concatenate([
        left_xy,          # [0:2]
        right_xy,         # [2:4]
        phase[:, None],   # [4]
        stance[:, None],  # [5]
        swing_target,     # [6:8]
        swing_start,      # [8:10]
    ], axis=-1)   # (N, 10)


# ── Per-step update ───────────────────────────────────────────────────────────

def _update_single(fs_i, person_i, dt):
    """
    Update one human's foot state for one env timestep.

    fs_i     : (10,)  foot state
    person_i : (≥5,)  people row [px, py, vx, vy, theta, ...]
    dt       : float  env timestep (seconds)
    """
    left_xy      = fs_i[0:2]
    right_xy     = fs_i[2:4]
    phase        = fs_i[4]
    stance       = fs_i[5]    # 0 = left stance / right swings
                               # 1 = right stance / left swings
    swing_target = fs_i[6:8]
    swing_start  = fs_i[8:10]

    px    = person_i[0]
    py    = person_i[1]
    vx    = person_i[2]
    vy    = person_i[3]
    theta = person_i[4]
    speed = jnp.sqrt(vx**2 + vy**2)

    body_xy = jnp.array([px, py])
    fwd, lat = _fwd_lat(theta[None])   # (1,2)
    fwd = fwd[0]; lat = lat[0]         # (2,)

    is_moving = speed > SPEED_THRESH

    # ── Phase advance ─────────────────────────────────────────────────────────
    speed_factor = jnp.clip(speed / STEP_SPEED, 0.3, 1.0)
    cadence      = STEP_FREQ * speed_factor
    new_phase    = (phase + cadence * dt) % 1.0

    # crossed = True when phase wrapped around 0 → new step began
    crossed = new_phase < phase

    # ── Interpolate swing foot from frozen start → target ─────────────────────
    # t ∈ [0, 1) advances smoothly within a half-cycle.
    # swing_start is frozen at the moment the step began, so the arc is stable.
    t        = new_phase   # 0 at step start, approaches 1 at plant
    swing_xy = swing_start * (1.0 - t) + swing_target * t

    # ── On plant: snap foot to target, swap stance, compute new step ──────────
    new_stance = jnp.where(crossed, 1.0 - stance, stance)

    # Next swing targets for each foot (with forward anticipation)
    next_left_target  = _next_plant(body_xy, fwd, lat, +1.0, speed)
    next_right_target = _next_plant(body_xy, fwd, lat, -1.0, speed)

    # After crossing: the *previously* stance foot becomes the new swing foot.
    #   stance==0 before cross → right just planted → left now swings
    #   stance==1 before cross → left  just planted → right now swings
    new_swing_target = jnp.where(
        crossed,
        jnp.where(stance == 0.0, next_left_target, next_right_target),
        swing_target,
    )

    # Freeze swing_start at the moment a new step begins (crossed).
    # new swing_start = where the new swing foot currently sits (its planted pos).
    new_swing_start_on_cross = jnp.where(
        stance == 0.0,
        left_xy,    # left was stance → left now becomes the swing foot
        right_xy,   # right was stance → right now becomes the swing foot
    )
    new_swing_start = jnp.where(crossed, new_swing_start_on_cross, swing_start)

    # ── Write swing position to the correct foot ──────────────────────────────
    # stance==0 → right is swinging
    new_right_xy = jnp.where(stance == 0.0, swing_xy,    right_xy)
    new_left_xy  = jnp.where(stance == 1.0, swing_xy,    left_xy)

    # On plant: snap the foot that just finished swinging to its exact target
    new_right_xy = jnp.where(crossed & (stance == 0.0), swing_target, new_right_xy)
    new_left_xy  = jnp.where(crossed & (stance == 1.0), swing_target, new_left_xy)

    # ── Freeze everything if stationary ───────────────────────────────────────
    new_phase        = jnp.where(is_moving, new_phase,        phase)
    new_stance       = jnp.where(is_moving, new_stance,       stance)
    new_swing_target = jnp.where(is_moving, new_swing_target, swing_target)
    new_swing_start  = jnp.where(is_moving, new_swing_start,  swing_start)
    new_left_xy      = jnp.where(is_moving, new_left_xy,      left_xy)
    new_right_xy     = jnp.where(is_moving, new_right_xy,     right_xy)

    return jnp.concatenate([
        new_left_xy,              # [0:2]
        new_right_xy,             # [2:4]
        new_phase[None],          # [4]
        new_stance[None],         # [5]
        new_swing_target,         # [6:8]
        new_swing_start,          # [8:10]
    ])   # (10,)


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
        from jax_env import PEOPLE_RADIUS
        return jnp.stack([
            people[:, 0],
            people[:, 1],
            jnp.full((N,), PEOPLE_RADIUS),
        ], axis=-1)   # (N, 3)


# ── Backward compatibility shim ───────────────────────────────────────────────

def advance_phase(phase_offsets, people, dt):
    """DEPRECATED — use advance_feet(foot_state, people, dt) instead."""
    raise RuntimeError(
        "advance_phase is no longer used. Call advance_feet(state.foot_state, "
        "new_people, dt) and store foot_state (N,10) in EnvState."
    )