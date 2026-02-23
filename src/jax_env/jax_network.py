"""
jax_network.py — Actor-Critic Neural Network
=============================================
CRITICAL FIX vs previous version:

  ENTROPY SATURATION (the reason training plateaued at update ~90):
    The previous version used a single GLOBAL log_std parameter (shape [2],
    shared across all states). The entropy loss gradient (coef=0.05) pushed
    this scalar directly into the +0.5 ceiling of the clamp, where it stuck
    permanently at H=3.838. The policy became maximally noisy and could never
    tighten regardless of how good the mean became — the advantage signal was
    completely overwhelmed by the noise.

    FIX: log_std is now a STATE-DEPENDENT Dense head (same as SAC style).
      • Different states get different noise levels (more expressive)
      • The entropy gradient is spread across the full Dense(2) layer params
        instead of a 2-element vector → cannot saturate at the ceiling
      • Initialised with bias=-1.0 → initial std=exp(-1)≈0.37, not 1.0
      • Clamp extended to [-4, +0.5]: same upper bound, wider floor so the
        policy can commit to good actions as it improves
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.5


class EndToEndActorCritic(nn.Module):
    """
    Architecture:
      LiDAR stack  -> Conv1D(32,7) -> Conv1D(64,5) -> Conv1D(64,3) -> LayerNorm
      Pose+State   -> Dense(128)   -> Dense(64)
      Concat       -> Dense(256)   -> Dense(128)             [shared trunk]
      Actor mean   -> Dense(action_dim)
      Actor logstd -> Dense(action_dim)  [state-dependent, bias init -1.0]
      Critic       -> Dense(128)   -> Dense(64) -> Dense(1)  [separate head]
    """
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 6

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # 1-D CNN on stacked LiDAR
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        # Global state MLP
        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        # Shared trunk
        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        # Actor mean
        actor_mean = nn.Dense(self.action_dim)(shared)

        # Actor log_std — state-dependent Dense head, bias init -1.0
        actor_logstd = nn.Dense(
            self.action_dim,
            bias_init=nn.initializers.constant(-1.0),
        )(shared)
        actor_logstd = jnp.clip(actor_logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Critic head (separate from actor)
        critic = nn.relu(nn.Dense(128)(fused))
        critic = nn.relu(nn.Dense(64)(critic))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)


def sample_action(
    rng_key: jnp.ndarray,
    mean:    jnp.ndarray,
    logstd:  jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reparameterised Gaussian sample in raw (pre-squash) space."""
    std      = jnp.exp(logstd)
    noise    = jax.random.normal(rng_key, shape=mean.shape)
    action   = mean + noise * std
    log_prob = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return action, log_prob


def scale_action_to_env(raw_action: jnp.ndarray, max_v: float) -> jnp.ndarray:
    """raw -> env action: sigmoid*max_v for v, tanh for w."""
    v = jax.nn.sigmoid(raw_action[..., 0]) * max_v
    w = jnp.tanh(raw_action[..., 1])
    return jnp.stack([v, w], axis=-1)


def scale_actions_batched(raw_actions: jnp.ndarray, max_v: jnp.ndarray) -> jnp.ndarray:
    """Batched: raw_actions (N,2), max_v (N,) -> env_actions (N,2)."""
    v = jax.nn.sigmoid(raw_actions[:, 0]) * max_v
    w = jnp.tanh(raw_actions[:, 1])
    return jnp.stack([v, w], axis=-1)


def get_deterministic_action(mean: jnp.ndarray, max_v: float = 1.5) -> jnp.ndarray:
    """Evaluation-time deterministic action (no noise)."""
    return scale_action_to_env(mean, max_v)