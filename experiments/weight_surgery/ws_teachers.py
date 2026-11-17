"""Behaviour-level 'teachers' built around the original network (obs-only, no privileged state).
They are NOT checkpoints: they are diagnostic probes and distillation targets."""
import numpy as np, jax, jax.numpy as jnp
import ws_lib as L
from legnav.core.jax_env import MAX_LIDAR_DIST, ROBOT_RADIUS, NUM_RAYS

_res = 2.0 * jnp.pi / (NUM_RAYS - 1)
PHI = -jnp.pi + jnp.arange(NUM_RAYS) * _res


def mirror_obs(o):
    """Left/right reflection of the 668-D observation (y -> -y)."""
    kin = o[..., :5] * jnp.array([1., -1., 1., 1., -1.])
    goal = (o[..., 5:11].reshape(o.shape[:-1] + (3, 2)) * jnp.array([1., -1.])).reshape(o.shape[:-1] + (6,))
    d = (o[..., 11:20].reshape(o.shape[:-1] + (3, 3)) * jnp.array([1., -1., -1.])).reshape(o.shape[:-1] + (9,))
    lid = o[..., 20:].reshape(o.shape[:-1] + (3, NUM_RAYS))[..., ::-1].reshape(o.shape[:-1] + (3 * NUM_RAYS,))
    return jnp.concatenate([kin, goal, d, lid], -1)


def mean_sym(params, obs):
    m1 = L.NET.apply({"params": params}, obs)[0]
    m2 = L.NET.apply({"params": params}, mirror_obs(obs))[0]
    return jnp.stack([0.5 * (m1[..., 0] + m2[..., 0]), 0.5 * (m1[..., 1] - m2[..., 1])], -1)


def front_gap(obs, half_width=0.35, cone=jnp.pi / 2):
    """Free distance (m) straight ahead inside the robot's swept corridor, from the newest LiDAR frame."""
    inv = obs[..., 20 + 2 * NUM_RAYS:]
    r = MAX_LIDAR_DIST - inv * (MAX_LIDAR_DIST - ROBOT_RADIUS)
    fwd, lat = r * jnp.cos(PHI), r * jnp.sin(PHI)
    m = (jnp.abs(PHI) < cone) & (jnp.abs(lat) < half_width) & (fwd > 0)
    return jnp.min(jnp.where(m, fwd, 50.0), -1) - ROBOT_RADIUS


def make_shield(tau, half_width=0.35, sym=False, v_floor=0.0):
    def mean_fn(params, obs):
        m = mean_sym(params, obs) if sym else L.NET.apply({"params": params}, obs)[0]
        max_v = jnp.clip(obs[..., 2] * 1.8 + 0.2, 0.2, 2.0)
        cap = jnp.clip(front_gap(obs, half_width) / tau, v_floor, None) / max_v      # as fraction of max_v
        return jnp.stack([jnp.minimum(m[..., 0], cap), m[..., 1]], -1)
    return mean_fn


def act_from_mean(mean_fn):
    return lambda params, obs, max_v: L.scale_actions_batched(mean_fn(params, obs), max_v)
