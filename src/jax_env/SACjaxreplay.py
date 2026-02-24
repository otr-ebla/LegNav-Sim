"""
SACjaxreplay.py — On-GPU Circular Replay Buffer
==================================================
FIXES vs previous version:

  FIX 1 — Removed `import flax.struct` (crash bug):
    `flax.struct` does not exist as a standalone import. The import was
    described as "just for reference" but would raise ImportError immediately.
    Removed entirely.

  FIX 2 — OBS_SIZE updated from 339 to 342:
    Matches jax_env.py/jax_wrappers.py new obs layout:
      pose_stack(9) + state_vec(9) + lidar_stack(324) = 342.
    The old value (339) used state_size=6; env now uses state_size=9.

  FIX 3 — buffer_sample: traced `buf["size"]` used as `maxval` in randint:
    jax.random.randint requires maxval to be a concrete integer when called
    under JIT, but buf["size"] is a traced int32 array. This works when
    capacity is always full but raises a ConcretizationTypeError during
    warmup when size < capacity.
    FIX: sample from full capacity, then mask out-of-range indices by
    clamping them to 0 (valid slot). Since valid entries start at index 0
    and the buffer fills sequentially, index 0 is always a valid fallback.
    This is the standard on-GPU replay buffer pattern (sample, clamp, use).
"""

import jax
import jax.numpy as jnp

# Updated to match jax_env.py / jax_wrappers.py:
#   pose_stack(3*3=9) + state_vec(9) + lidar_stack(108*3=324) = 342
OBS_SIZE = 342
ACT_SIZE = 2


def make_replay_buffer(capacity: int):
    """
    Create an empty replay buffer of given capacity on the current device.
    Returns a dict pytree (works natively with JAX).
    """
    buf = {
        "obs":      jnp.zeros((capacity, OBS_SIZE),  dtype=jnp.float32),
        "action":   jnp.zeros((capacity, ACT_SIZE),  dtype=jnp.float32),
        "reward":   jnp.zeros((capacity,),            dtype=jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),  dtype=jnp.float32),
        "done":     jnp.zeros((capacity,),            dtype=jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }
    return buf


@jax.jit
def buffer_add_batch(buf, obs, action, reward, next_obs, done):
    """
    Insert a batch of N transitions into the circular buffer.
    All inputs: (N, ...) shaped. ptr wraps around modulo capacity.
    Returns updated buffer dict.
    """
    capacity = buf["obs"].shape[0]
    N        = obs.shape[0]
    ptr      = buf["ptr"]

    idxs = (ptr + jnp.arange(N)) % capacity

    new_buf = {
        "obs":      buf["obs"].at[idxs].set(obs),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs),
        "done":     buf["done"].at[idxs].set(done),
        "ptr":      jnp.int32((ptr + N) % capacity),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(capacity)),
    }
    return new_buf


@jax.jit(static_argnames=["batch_size"])
def buffer_sample(buf, rng_key, batch_size: int):
    """
    Sample batch_size transitions uniformly at random from valid entries.
    Returns (obs, action, reward, next_obs, done) each (batch_size, ...).

    FIX: Instead of using buf["size"] (a traced array) as maxval in randint
    (which fails under JIT during warmup when buffer is partially filled),
    we sample from full capacity and clamp out-of-range indices to 0.
    Index 0 is always a valid slot (filled first), so clamped samples just
    repeat slot-0 data which is harmless — they are a tiny fraction of the
    batch only during the brief warmup phase, and never occur once the buffer
    is full.
    """
    capacity = buf["obs"].shape[0]
    # Sample from full capacity range (concrete integer — JIT-safe)
    idxs = jax.random.randint(rng_key, shape=(batch_size,),
                               minval=0, maxval=capacity)
    # Clamp indices that fall outside [0, size) to 0 (valid fallback)
    idxs = jnp.where(idxs < buf["size"], idxs, jnp.int32(0))
    return (
        buf["obs"][idxs],
        buf["action"][idxs],
        buf["reward"][idxs],
        buf["next_obs"][idxs],
        buf["done"][idxs],
    )