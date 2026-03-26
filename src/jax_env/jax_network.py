"""
jax_network.py — Actor-Critic Neural Network for PPO (Shared Trunk)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0

_STATE_VEC_SIZE = 9

class EndToEndActorCritic(nn.Module):
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pose_size  = 3 * self.stack_dim          # 9
        state_size = _STATE_VEC_SIZE             # 9

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

        # Shared trunk (Questo garantisce i 135k FPS!)
        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        # Actor mean
        actor_mean = nn.Dense(self.action_dim)(shared)

        # Actor log_std — Parametro globale
        logstd_param = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        actor_logstd = jnp.broadcast_to(logstd_param, actor_mean.shape)
        actor_logstd = jnp.clip(actor_logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Critic head
        critic = nn.relu(nn.Dense(128)(shared))
        critic = nn.relu(nn.Dense(64)(critic))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)

def sample_action(rng_key: jnp.ndarray, mean: jnp.ndarray, logstd: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    std      = jnp.exp(logstd)
    noise    = jax.random.normal(rng_key, shape=mean.shape)
    action   = mean + noise * std
    log_prob = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return action, log_prob

def scale_action_to_env(raw_action: jnp.ndarray, max_v: float) -> jnp.ndarray:
    v = jax.nn.sigmoid(raw_action[..., 0]) * max_v
    w = jnp.tanh(raw_action[..., 1])
    return jnp.stack([v, w], axis=-1)

def scale_actions_batched(raw_actions: jnp.ndarray, max_v: jnp.ndarray) -> jnp.ndarray:
    v = jax.nn.sigmoid(raw_actions[:, 0]) * max_v
    w = jnp.tanh(raw_actions[:, 1])
    return jnp.stack([v, w], axis=-1)

def get_deterministic_action(mean: jnp.ndarray, max_v: float = 1.5) -> jnp.ndarray:
    return scale_action_to_env(mean, max_v)