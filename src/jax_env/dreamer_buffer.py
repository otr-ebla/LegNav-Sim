"""
dreamer_buffer.py — Vectorized Sequence Replay Buffer for DreamerV3

"""

import functools
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ReplayBufferState:
    obs:        jnp.ndarray   # [Capacity, Num_Envs, Obs_Size]
    actions:    jnp.ndarray   # [Capacity, Num_Envs, Action_Dim]
    rewards:    jnp.ndarray   # [Capacity, Num_Envs]
    dones:      jnp.ndarray   # [Capacity, Num_Envs]  bool
    insert_idx: jnp.int32
    is_full:    jnp.bool_


def init_buffer(capacity: int, num_envs: int,
                obs_size: int, action_dim: int) -> ReplayBufferState:
    """Allocates static buffer memory on the GPU."""
    return ReplayBufferState(
        obs        = jnp.zeros((capacity, num_envs, obs_size),   dtype=jnp.float32),
        actions    = jnp.zeros((capacity, num_envs, action_dim), dtype=jnp.float32),
        rewards    = jnp.zeros((capacity, num_envs),             dtype=jnp.float32),
        dones      = jnp.zeros((capacity, num_envs),             dtype=jnp.bool_),
        insert_idx = jnp.int32(0),
        is_full    = jnp.bool_(False),
    )


def buffer_ready(buffer_state: ReplayBufferState, seq_len: int) -> bool:

    insert_idx = int(buffer_state.insert_idx)
    is_full    = bool(buffer_state.is_full)
    return is_full or (insert_idx >= seq_len)



@jax.jit
def add_batch(buffer_state: ReplayBufferState,
              obs: jnp.ndarray, actions: jnp.ndarray,
              rewards: jnp.ndarray, dones: jnp.ndarray) -> ReplayBufferState:
    """
    Inserts one timestep for all envs. Uses dynamic_update_slice — O(1) memory.
    """
    idx = buffer_state.insert_idx

    new_obs     = jax.lax.dynamic_update_slice(buffer_state.obs,     obs[None],     (idx, 0, 0))
    new_actions = jax.lax.dynamic_update_slice(buffer_state.actions, actions[None], (idx, 0, 0))
    new_rewards = jax.lax.dynamic_update_slice(buffer_state.rewards, rewards[None], (idx, 0))
    new_dones   = jax.lax.dynamic_update_slice(buffer_state.dones,   dones[None],   (idx, 0))

    capacity = buffer_state.obs.shape[0]
    next_idx = (idx + 1) % capacity
    is_full  = buffer_state.is_full | (next_idx == 0)

    return buffer_state.replace(
        obs=new_obs, actions=new_actions,
        rewards=new_rewards, dones=new_dones,
        insert_idx=next_idx, is_full=is_full,
    )


@functools.partial(jax.jit, static_argnums=(2, 3))
def sample_sequences(rng_key: jnp.ndarray, buffer_state: ReplayBufferState,
                     batch_size: int, seq_len: int):
    """
    Samples `batch_size` sequences of length `seq_len`.

    Episode boundary fix: a forward scan builds a causal alive-mask from the
    done flags. Rewards and done flags are zeroed from the first terminal step
    onward so the RSSM never sees cross-episode transitions.

    Assumes buffer_ready(buffer_state, seq_len) == True (Python gate in loop).
    """
    capacity = buffer_state.obs.shape[0]
    num_envs = buffer_state.obs.shape[1]
    obs_dim  = buffer_state.obs.shape[2]
    act_dim  = buffer_state.actions.shape[2]

    # BUG 4 FIX: floor at seq_len (not 1) so we always have a valid window
    max_valid_idx = jnp.where(
        buffer_state.is_full,
        capacity - seq_len,
        jnp.maximum(seq_len, buffer_state.insert_idx) - seq_len,
    )

    k_time, k_env = jax.random.split(rng_key)
    start_t = jax.random.randint(k_time, (batch_size,), minval=0, maxval=max_valid_idx)
    env_idx  = jax.random.randint(k_env,  (batch_size,), minval=0, maxval=num_envs)

    def extract_seq(t, env_i):
        seq_obs  = jnp.squeeze(
            jax.lax.dynamic_slice(buffer_state.obs,     (t, env_i, 0), (seq_len, 1, obs_dim)), axis=1)
        seq_act  = jnp.squeeze(
            jax.lax.dynamic_slice(buffer_state.actions, (t, env_i, 0), (seq_len, 1, act_dim)), axis=1)
        seq_rew  = jnp.squeeze(
            jax.lax.dynamic_slice(buffer_state.rewards, (t, env_i),    (seq_len, 1)), axis=1)
        seq_done = jnp.squeeze(
            jax.lax.dynamic_slice(buffer_state.dones,   (t, env_i),    (seq_len, 1)), axis=1)

        # --- BUG 3 FIX: causal alive-mask ---
        # alive[t] = True iff no done has fired in [0, t-1].
        # The done at step t ends the episode; the obs at t+1 is a new episode.
        def mask_step(still_alive, done_t):
            alive_now   = still_alive           # alive before this done fires
            still_alive = still_alive & ~done_t
            return still_alive, alive_now

        _, alive_mask = jax.lax.scan(mask_step, jnp.bool_(True), seq_done)
        alive_f = alive_mask.astype(jnp.float32)   # [seq_len]

        seq_rew_masked  = seq_rew  * alive_f
        seq_done_masked = seq_done & alive_mask     # keep bool dtype

        return seq_obs, seq_act, seq_rew_masked, seq_done_masked

    batch_obs, batch_act, batch_rew, batch_done = jax.vmap(extract_seq)(start_t, env_idx)
    return batch_obs, batch_act, batch_rew, batch_done