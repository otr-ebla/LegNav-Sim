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
        #global_feat = nn.relu(nn.Dense(128)(global_in))
        #global_feat = nn.relu(nn.Dense(64)(global_feat))

        # Shared trunk (Questo garantisce i 135k FPS!)
        fused  = jnp.concatenate([cnn_feat, global_in], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        # Actor mean
        actor_mean = nn.Dense(self.action_dim)(shared)

        # Actor log_std — Global parameter with smooth bounded mapping.
        # FIX 3: Replace jnp.clip (dead gradient outside bounds) with tanh squashing.
        # jnp.clip has zero gradient when logstd hits LOG_STD_MIN or LOG_STD_MAX,
        # permanently freezing the parameter. tanh keeps gradients alive everywhere.
        logstd_param = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        actor_logstd_raw = jnp.broadcast_to(logstd_param, actor_mean.shape)
        # Maps (-inf, +inf) → (LOG_STD_MIN, LOG_STD_MAX) with non-zero gradient everywhere
        actor_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (jnp.tanh(actor_logstd_raw) + 1.0)

        # Critic head
        #critic = nn.relu(nn.Dense(128)(shared))
        critic = nn.relu(nn.Dense(64)(shared))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)

def _squash_log_jacobian(raw_actions: jnp.ndarray, max_v: float = 1.0) -> jnp.ndarray:
    """
    FIX 1: Log-determinant of the Jacobian of the squashing functions.
    v = sigmoid(raw[0]) * max_v  →  d/d(raw[0]) = sigmoid * (1 - sigmoid) * max_v
    w = tanh(raw[1])             →  d/d(raw[1]) = 1 - tanh²
    Returns shape (...,) — to be subtracted from the Gaussian log_prob.
    """
    v_squash = jax.nn.sigmoid(raw_actions[..., 0])
    w_squash = jnp.tanh(raw_actions[..., 1])
    log_dv = jnp.log(v_squash * (1.0 - v_squash) * max_v + 1e-6)
    log_dw = jnp.log(1.0 - w_squash ** 2 + 1e-6)
    return log_dv + log_dw


def squash_corrected_log_prob(raw_actions: jnp.ndarray, mean: jnp.ndarray,
                               logstd: jnp.ndarray, max_v: float = 1.0) -> jnp.ndarray:
    """
    FIX 1: Log-probability of raw_actions under Gaussian(mean, exp(logstd)),
    corrected for the sigmoid/tanh squashing applied to produce env actions.
    Both rollout collection and PPO loss must use this function so the
    importance sampling ratio exp(new_log_prob - old_log_prob) is valid.
    max_v is the mean robot max speed (used to scale the sigmoid output).
    In practice we use a per-env max_v; pass the batch mean or a fixed scalar
    here — the Jacobian correction absorbs most of the variance.
    """
    std = jnp.exp(logstd)
    z   = (raw_actions - mean) / (std + 1e-8)
    base_log_prob = jnp.sum(-0.5 * (z ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return base_log_prob - _squash_log_jacobian(raw_actions, max_v) # it uses above function to correct the log_prob


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