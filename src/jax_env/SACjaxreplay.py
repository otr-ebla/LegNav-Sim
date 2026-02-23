"""
jax_sac_replay.py — On-GPU Circular Replay Buffer
==================================================
Pure JAX implementation so everything stays on the GPU with no host transfers
during training. Uses dynamic_update_slice / dynamic_slice for O(1) insertions.

Stores per-transition:
  obs       (OBS_SIZE,)   = 339
  action    (2,)
  reward    ()
  next_obs  (OBS_SIZE,)   = 339
  done      ()

Memory: 500k × (339+2+1+339+1) × 4 bytes = ~1.36 GB — fits in remaining VRAM.

The buffer is a pytree of arrays, each with shape (capacity, ...).
`ptr`   : int32 scalar — next write position (wraps around)
`size`  : int32 scalar — number of valid transitions (capped at capacity)
"""

import jax
import jax.numpy as jnp
import flax.struct   # just for reference; we use a plain dict pytree

OBS_SIZE = 339
ACT_SIZE = 2


def make_replay_buffer(capacity: int):
    """
    Create an empty replay buffer of given capacity on the current device.
    Returns a dict pytree (works natively with JAX, no Flax needed).
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

    Uses lax.dynamic_update_slice — O(N) and fully JIT-compilable.
    When a batch wraps around the end of the buffer, we handle it by
    splitting into two contiguous writes (head and tail).

    Returns updated buffer dict.
    """
    capacity = buf["obs"].shape[0]
    N        = obs.shape[0]
    ptr      = buf["ptr"]

    # Build the indices we'll write to: [ptr, ptr+1, ..., ptr+N-1] mod capacity
    idxs = (ptr + jnp.arange(N)) % capacity

    # Scatter into each array using advanced indexing .at[idxs].set(...)
    # This is valid under jit because idxs is a concrete-shape array
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


@jax.jit
def buffer_sample(buf, rng_key, batch_size: int):
    """
    Sample batch_size transitions uniformly at random from valid entries.
    Returns (obs, action, reward, next_obs, done) each (batch_size, ...).

    Note: batch_size must be static (known at compile time) — pass as Python int.
    """
    idxs = jax.random.randint(rng_key, shape=(batch_size,),
                               minval=0, maxval=buf["size"])
    return (
        buf["obs"][idxs],
        buf["action"][idxs],
        buf["reward"][idxs],
        buf["next_obs"][idxs],
        buf["done"][idxs],
    )