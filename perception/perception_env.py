"""
perception/perception_env.py — Stationary-robot data environment
==================================================================
Generates supervised training data for the perception module by reusing the
LegNav simulator (legnav.core): scenarios, HSFM pedestrians and the
Planted-Foot Gait Model are identical to navigation training, but the robot
never moves — it sits at its spawn pose while people walk around it.

Each step yields:
  lidar_stack : (STACK_DIM, NUM_RAYS)  noisy inverted-normalized scans
                (same normalization the navigation policy uses)
  labels      : per-ray person mask, per-ray offset to the owning person's
                body centre (ego frame), per-person centres + visibility.

Ground truth is geometric (noise-free): a ray is a "person ray" when its
closest hit is one of the leg circles. Offsets point from the geometric ray
endpoint to the body centre of the person owning that leg, so the network
learns to fuse the two-leg signature into a single tracked position.
"""

import jax
import jax.numpy as jnp
from flax import struct

from legnav.core.jax_env import (
    reset_env, EnvState,
    ROBOT_RADIUS, PEOPLE_RADIUS,
    P_HUMAN_STOP, STOP_MIN_STEPS, STOP_MAX_STEPS,
)
from legnav.core.jax_physics import compute_lidar, _intersect_ray_circle_scalar
from legnav.core.jax_humans import update_all_humans
from legnav.core.jax_legs import get_leg_circles, advance_feet
from legnav.config import SimConfig

from perception.config import PerceptionConfig as PC

NUM_RAYS   = PC.NUM_RAYS
FOV        = PC.FOV
MAX_DIST   = PC.MAX_DIST
STACK_DIM  = PC.STACK_DIM
NUM_PEOPLE = SimConfig.JAX_NUM_PEOPLE
DT         = 0.15  # mirrors RobotConfig.DT


@struct.dataclass
class PerceptionState:
    env_state:   EnvState
    lidar_stack: jnp.ndarray   # (STACK_DIM, NUM_RAYS) inverted-normalized


# ── Static decoy clutter (hard negatives) ─────────────────────────────────────

def _inject_decoy_clutter(es: EnvState, key: jnp.ndarray) -> EnvState:
    """Replace part of the random obstacles with leg-like static decoys.

    Leg-pair circles, single poles and thin slats are exactly the static
    geometry the leg-pair decoder confuses with people. They enter the scan
    as ordinary obstacles and are labeled 0 by compute_ground_truth, so the
    network is explicitly trained to reject leg-shaped objects that carry no
    gait history across the 8-frame stack.
    """
    k_pair, k_pole, k_slat = jax.random.split(key, 3)

    def _keep_off_robot(x, y):
        """Push a decoy out of the robot's immediate vicinity (min 1.2 m)."""
        dx, dy = x - es.x, y - es.y
        d = jnp.sqrt(dx * dx + dy * dy) + 1e-6
        push = jnp.maximum(1.2 / d, 1.0)
        return es.x + dx * push, es.y + dy * push

    def _rand_xy(k):
        kx, ky = jax.random.split(k)
        x = jax.random.uniform(kx, minval=1.0, maxval=es.room_w - 1.0)
        y = jax.random.uniform(ky, minval=1.0, maxval=es.room_h - 1.0)
        return _keep_off_robot(x, y)

    # Leg-like circle pairs at anatomical separation → (2*DECOY_PAIRS, 3)
    def _make_pair(k):
        k_xy, k_sep, k_ang, k_r = jax.random.split(k, 4)
        cx, cy = _rand_xy(k_xy)
        sep = jax.random.uniform(k_sep, minval=PC.DECOY_SEP_MIN,
                                 maxval=PC.DECOY_SEP_MAX)
        ang = jax.random.uniform(k_ang, minval=0.0, maxval=jnp.pi)
        r = jax.random.uniform(k_r, (2,), minval=PC.DECOY_R_MIN,
                               maxval=PC.DECOY_R_MAX)
        ox, oy = 0.5 * sep * jnp.cos(ang), 0.5 * sep * jnp.sin(ang)
        return jnp.array([[cx - ox, cy - oy, r[0]],
                          [cx + ox, cy + oy, r[1]]])

    pairs = jax.vmap(_make_pair)(
        jax.random.split(k_pair, PC.DECOY_PAIRS)).reshape(-1, 3)

    # Single leg-sized poles → (DECOY_POLES, 3)
    def _make_pole(k):
        k_xy, k_r = jax.random.split(k)
        x, y = _rand_xy(k_xy)
        r = jax.random.uniform(k_r, minval=PC.DECOY_R_MIN,
                               maxval=PC.DECOY_R_MAX)
        return jnp.array([x, y, r])

    poles = jax.vmap(_make_pole)(jax.random.split(k_pole, PC.DECOY_POLES))

    # Thin slats (door jambs / wall stubs) → (DECOY_SLATS, 8) corner quads
    def _make_slat(k):
        k_xy, k_hw, k_hh, k_ang = jax.random.split(k, 4)
        cx, cy = _rand_xy(k_xy)
        hw = jax.random.uniform(k_hw, minval=0.02, maxval=0.07)
        hh = jax.random.uniform(k_hh, minval=0.08, maxval=0.60)
        ang = jax.random.uniform(k_ang, minval=0.0, maxval=jnp.pi)
        ca, sa = jnp.cos(ang), jnp.sin(ang)
        lx = jnp.array([-hw,  hw,  hw, -hw])
        ly = jnp.array([-hh, -hh,  hh,  hh])
        return jnp.stack([cx + ca * lx - sa * ly,
                          cy + sa * lx + ca * ly], axis=-1).reshape(8)

    slats = jax.vmap(_make_slat)(jax.random.split(k_slat, PC.DECOY_SLATS))

    decoy_circles = jnp.concatenate([pairs, poles], axis=0)
    n_cir = decoy_circles.shape[0]
    return es.replace(
        obs_circles=es.obs_circles.at[:n_cir].set(decoy_circles),
        obs_boxes=es.obs_boxes.at[:PC.DECOY_SLATS].set(slats),
    )


# ── Ground truth ──────────────────────────────────────────────────────────────

def compute_ground_truth(state: EnvState):
    """
    Per-ray and per-person labels from the geometric (noise-free) scene.

    Returns a dict:
      person_mask : (NUM_RAYS,)    bool  — ray's closest hit is a leg
      offsets     : (NUM_RAYS, 2)  float — (body_ego - endpoint_ego) / OFFSET_SCALE
      leg_offsets : (NUM_RAYS, 2)  float — (leg_ego  - endpoint_ego) / LEG_OFFSET_SCALE
      centers     : (NUM_PEOPLE,2) float — person body centres, ego frame
      visible     : (NUM_PEOPLE,)  bool  — at least one ray hits this person
    """
    # Ray directions — must match compute_lidar's angle convention exactly
    angles_world = state.theta - FOV * 0.5 + jnp.arange(NUM_RAYS) * (FOV / (NUM_RAYS - 1))
    dxr, dyr = jnp.cos(angles_world), jnp.sin(angles_world)

    # Distance of every ray to every leg circle (2N legs)
    leg_circles = get_leg_circles(state.people, state.foot_state, use_legs=True)  # (2N, 3)

    def ray_legs(dxi, dyi):
        return jax.vmap(
            lambda c: _intersect_ray_circle_scalar(
                state.x, state.y, dxi, dyi, c[0], c[1], c[2])
        )(leg_circles)

    t_legs    = jax.vmap(ray_legs)(dxr, dyr)        # (NUM_RAYS, 2N)
    d_leg     = jnp.min(t_legs, axis=1)             # (NUM_RAYS,)
    owner_leg = jnp.argmin(t_legs, axis=1)          # (NUM_RAYS,)
    owner     = owner_leg % NUM_PEOPLE              # left legs 0..N-1, right N..2N-1

    # Full geometric scan (legs + static obstacles + walls) for occlusion test
    all_circles = jnp.concatenate([leg_circles, state.obs_circles], axis=0)
    d_full = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_DIST, state.room_w, state.room_h,
    )

    person_mask = (d_leg < MAX_DIST) & (d_leg <= d_full + PC.GT_EPS)

    # Ego-frame geometry
    cos_t, sin_t = jnp.cos(-state.theta), jnp.sin(-state.theta)
    pdx = state.people[:, 0] - state.x
    pdy = state.people[:, 1] - state.y
    centers = jnp.stack([cos_t * pdx - sin_t * pdy,
                         sin_t * pdx + cos_t * pdy], axis=-1)   # (N, 2)

    angles_ego = -FOV * 0.5 + jnp.arange(NUM_RAYS) * (FOV / (NUM_RAYS - 1))
    endpoints = d_leg[:, None] * jnp.stack(
        [jnp.cos(angles_ego), jnp.sin(angles_ego)], axis=-1)    # (NUM_RAYS, 2)

    offsets = (centers[owner] - endpoints) / PC.OFFSET_SCALE
    offsets = jnp.where(person_mask[:, None], offsets, 0.0)

    # Offset to the centre of the individual leg circle the ray hit —
    # supervises the leg head used by the leg-pair decoder.
    ldx = leg_circles[:, 0] - state.x
    ldy = leg_circles[:, 1] - state.y
    legs_ego = jnp.stack([cos_t * ldx - sin_t * ldy,
                          sin_t * ldx + cos_t * ldy], axis=-1)   # (2N, 2)
    leg_offsets = (legs_ego[owner_leg] - endpoints) / PC.LEG_OFFSET_SCALE
    leg_offsets = jnp.where(person_mask[:, None], leg_offsets, 0.0)

    visible = jax.vmap(
        lambda i: jnp.any(person_mask & (owner == i))
    )(jnp.arange(NUM_PEOPLE))

    return {
        "person_mask": person_mask,
        "offsets":     offsets,
        "leg_offsets": leg_offsets,
        "centers":     centers,
        "visible":     visible,
    }


# ── Noisy observation frame ───────────────────────────────────────────────────

def _noisy_inv_frame(state: EnvState, key: jnp.ndarray) -> jnp.ndarray:
    """One noisy inverted-normalized scan — same model as jax_env.get_obs."""
    leg_circles = get_leg_circles(state.people, state.foot_state, use_legs=True)
    all_circles = jnp.concatenate([leg_circles, state.obs_circles], axis=0)
    raw = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_DIST, state.room_w, state.room_h,
    )
    k_gauss, k_sp = jax.random.split(key)
    gaussian = jax.random.normal(k_gauss, raw.shape) * 0.05
    sp_mask  = jax.random.uniform(k_sp, raw.shape) < 0.06
    noisy = jnp.where(sp_mask, MAX_DIST, raw + gaussian)
    noisy = jnp.clip(noisy, 0.0, MAX_DIST)
    return jnp.clip((MAX_DIST - noisy) / (MAX_DIST - ROBOT_RADIUS), 0.0, 1.0)


# ── Reset / Step ──────────────────────────────────────────────────────────────

def reset_perception(key: jnp.ndarray):
    """Sample a scenario via the standard env reset; the robot will stay put."""
    k_env, k_decoy, k_frame = jax.random.split(key, 3)
    _, env_state = reset_env(k_env)
    env_state = _inject_decoy_clutter(env_state, k_decoy)
    frame = _noisy_inv_frame(env_state, k_frame)
    state = PerceptionState(
        env_state=env_state,
        lidar_stack=jnp.tile(frame[None, :], (STACK_DIM, 1)),
    )
    return state, compute_ground_truth(env_state)


def step_perception(key: jnp.ndarray, state: PerceptionState):
    """Advance pedestrians one DT with the robot frozen in place."""
    es = state.env_state
    human_key, frame_key = jax.random.split(key)

    new_people = update_all_humans(
        es.people, human_key, DT,
        es.x, es.y, es.theta, 0.0,            # robot pose fixed, speed 0
        es.room_w, es.room_h, PEOPLE_RADIUS,
        es.obs_circles, es.obs_boxes,
    )

    # Human stop-and-go (same model as jax_env.step_env) — stationary people
    # are an important perception case: legs settle into the still pose.
    stop_key  = jax.random.fold_in(human_key, 0xAB57)
    stop_roll = jax.random.uniform(stop_key, (NUM_PEOPLE,))
    dur_keys  = jax.random.split(stop_key, NUM_PEOPLE)
    stop_dur  = jax.vmap(lambda k: jax.random.randint(
        k, (), STOP_MIN_STEPS, STOP_MAX_STEPS + 1))(dur_keys)

    prev_timers = es.human_stop_timers
    in_stop     = prev_timers > 0
    new_timers  = jnp.where(in_stop, prev_timers - 1, 0)
    start_stop  = ~in_stop & (stop_roll < P_HUMAN_STOP)
    new_timers  = jnp.where(start_stop, stop_dur, new_timers)
    is_stopped  = new_timers > 0

    frozen_people = es.people.at[:, 2:4].set(0.0)
    new_people = jnp.where(is_stopped[:, None], frozen_people, new_people)

    new_foot_state = advance_feet(es.foot_state, new_people, DT)

    new_es = es.replace(
        people=new_people,
        foot_state=new_foot_state,
        human_stop_timers=new_timers,
        time_step=es.time_step + 1,
    )

    frame = _noisy_inv_frame(new_es, frame_key)
    new_stack = jnp.concatenate([state.lidar_stack[1:], frame[None]], axis=0)
    new_state = PerceptionState(env_state=new_es, lidar_stack=new_stack)

    return new_state, compute_ground_truth(new_es)
