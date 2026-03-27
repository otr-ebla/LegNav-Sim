"""
jax_buffer.py — Vectorized Sequence Replay Buffer for DreamerV3
"""

import functools
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class ReplayBufferState:
    obs: jnp.ndarray       # [Capacity, Num_Envs, Obs_Size]
    actions: jnp.ndarray   # [Capacity, Num_Envs, Action_Dim]
    rewards: jnp.ndarray   # [Capacity, Num_Envs]
    dones: jnp.ndarray     # [Capacity, Num_Envs]
    insert_idx: jnp.int32
    is_full: jnp.bool_

def init_buffer(capacity: int, num_envs: int, obs_size: int, action_dim: int) -> ReplayBufferState:
    """
    Allocates the static buffer memory on the GPU.
    """
    return ReplayBufferState(
        obs=jnp.zeros((capacity, num_envs, obs_size), dtype=jnp.float32),
        actions=jnp.zeros((capacity, num_envs, action_dim), dtype=jnp.float32),
        rewards=jnp.zeros((capacity, num_envs), dtype=jnp.float32),
        dones=jnp.zeros((capacity, num_envs), dtype=jnp.bool_),
        insert_idx=jnp.int32(0),
        is_full=jnp.bool_(False)
    )

@jax.jit
def add_batch(buffer_state: ReplayBufferState, obs: jnp.ndarray, actions: jnp.ndarray, 
              rewards: jnp.ndarray, dones: jnp.ndarray) -> ReplayBufferState:
    """
    Inserts a full batch of environment transitions in-place.
    """
    idx = buffer_state.insert_idx
    
    # dynamic_update_slice avoids full array copies, keeping memory operations O(1)
    new_obs = jax.lax.dynamic_update_slice(buffer_state.obs, obs[None, ...], (idx, 0, 0))
    new_actions = jax.lax.dynamic_update_slice(buffer_state.actions, actions[None, ...], (idx, 0, 0))
    new_rewards = jax.lax.dynamic_update_slice(buffer_state.rewards, rewards[None, ...], (idx, 0))
    new_dones = jax.lax.dynamic_update_slice(buffer_state.dones, dones[None, ...], (idx, 0))

    capacity = buffer_state.obs.shape[0]
    next_idx = (idx + 1) % capacity
    is_full = buffer_state.is_full | (next_idx == 0)

    return buffer_state.replace(
        obs=new_obs,
        actions=new_actions,
        rewards=new_rewards,
        dones=new_dones,
        insert_idx=next_idx,
        is_full=is_full
    )

@functools.partial(jax.jit, static_argnums=(2, 3))
def sample_sequences(rng_key: jnp.ndarray, buffer_state: ReplayBufferState, 
                     batch_size: int, seq_len: int):
    """
    Samples sequences of length `seq_len` uniformly across time and environments.
    """
    capacity = buffer_state.obs.shape[0]
    num_envs = buffer_state.obs.shape[1]
    
    # If buffer is not full, we can only sample up to (insert_idx - seq_len).
    # If full, we can sample up to (capacity - seq_len).
    max_valid_idx = jnp.where(
        buffer_state.is_full, 
        capacity - seq_len, 
        jnp.maximum(1, buffer_state.insert_idx - seq_len)
    )

    k_time, k_env = jax.random.split(rng_key)
    
    # Sample starting time indices and environment indices
    start_t = jax.random.randint(k_time, (batch_size,), minval=0, maxval=max_valid_idx)
    env_idx = jax.random.randint(k_env, (batch_size,), minval=0, maxval=num_envs)

    # vmap over the sampled indices to extract the sequences
    def extract_seq(t, env_i):
        # dynamic_slice grabs a chunk of size (seq_len, 1, Feature_Dim)
        seq_obs = jax.lax.dynamic_slice(buffer_state.obs, (t, env_i, 0), (seq_len, 1, buffer_state.obs.shape[2]))
        seq_actions = jax.lax.dynamic_slice(buffer_state.actions, (t, env_i, 0), (seq_len, 1, buffer_state.actions.shape[2]))
        seq_rewards = jax.lax.dynamic_slice(buffer_state.rewards, (t, env_i), (seq_len, 1))
        seq_dones = jax.lax.dynamic_slice(buffer_state.dones, (t, env_i), (seq_len, 1))
        
        # Squeeze out the env dimension to yield (seq_len, Feature_Dim)
        return jnp.squeeze(seq_obs, axis=1), jnp.squeeze(seq_actions, axis=1), jnp.squeeze(seq_rewards, axis=1), jnp.squeeze(seq_dones, axis=1)

    batch_obs, batch_actions, batch_rewards, batch_dones = jax.vmap(extract_seq)(start_t, env_idx)
    
    return batch_obs, batch_actions, batch_rewards, batch_dones