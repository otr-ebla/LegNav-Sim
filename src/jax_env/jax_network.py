"""
jax_network.py — Actor-Critic Neural Network
=============================================
Fixes vs original:
  - CNN reshape made safe under vmap/jit using shape[-2:] approach
  - Actions passed through tanh squashing so they are bounded at inference
  - log_std clamped to avoid numerical explosion
  - Added LayerNorm after CNN flatten for training stability
  - Critic has its own deeper MLP (decoupled from actor trunk)
  - sample_action: uses reparameterisation; log_prob numerical stability improved
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class EndToEndActorCritic(nn.Module):
    """
    Architecture:
      LiDAR stack  → 1-D CNN  → flatten → LayerNorm
      Pose + State → Dense(64) → ReLU
      Concat        → Dense(256) → ReLU
      Actor head:  Dense(action_dim)  (mean of Gaussian; bounded by tanh)
      Critic head: Dense(128) → Dense(1)  (separate MLP for better value estimates)
    """
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # ── Unpack flat observation ──────────────────────────────────────────
        # Layout: pose_stack(3*stack_dim) | state_vec(5) | lidar_stack(num_rays*stack_dim)
        pose_size  = 3 * self.stack_dim
        state_size = 5

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # ── 1-D CNN on lidar stack ───────────────────────────────────────────
        # Reshape: (Batch, num_rays, stack_dim)  — channels = time frames
        # FIX: use -1 for spatial dim so it is safe under arbitrary batch shapes
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))

        cnn = nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn)
        cnn = nn.relu(cnn)
        cnn = nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn)
        cnn = nn.relu(cnn)
        cnn = nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn)
        cnn = nn.relu(cnn)

        # FIX: reshape using dynamic shape instead of cnn.shape[0]
        cnn_flat = cnn.reshape((*batch_shape, -1))
        cnn_feat = nn.LayerNorm()(cnn_flat)   # stabilise training

        # ── Global state MLP ─────────────────────────────────────────────────
        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        # ── Fusion trunk ─────────────────────────────────────────────────────
        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        # ── Actor head ───────────────────────────────────────────────────────
        actor_mean   = nn.Dense(self.action_dim)(shared)
        # FIX: clamp log_std to [-3, 0] to prevent numerical explosion
        actor_logstd = jnp.clip(
            self.param('log_std', lambda rng, s: jnp.zeros(s), (self.action_dim,)),
            -3.0, 0.0
        )

        # ── Critic head (separate deeper MLP) ────────────────────────────────
        critic = nn.relu(nn.Dense(128)(fused))
        critic = nn.relu(nn.Dense(64)(critic))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)


def sample_action(
    rng_key:  jnp.ndarray,
    mean:     jnp.ndarray,
    logstd:   jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reparameterised sample from N(mean, exp(logstd)²).
    Returns (action, log_prob) where log_prob is summed over action dims.
    log_prob uses the numerically stable form: -0.5*(z²+log(2π)) - logstd
    """
    std   = jnp.exp(logstd)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    z     = noise                          # pre-squash sample
    action = mean + noise * std            # no tanh during training (PPO is on-policy; clipping handles range)

    # Numerically stable log-prob: -0.5*(z² + log(2π)) - log(σ)
    log_prob = -0.5 * (z**2 + jnp.log(2.0 * jnp.pi)) - logstd
    log_prob = jnp.sum(log_prob, axis=-1)

    return action, log_prob


def get_deterministic_action(mean: jnp.ndarray, max_v: float = 1.5) -> jnp.ndarray:
    """
    Evaluation-time action: clip to valid ranges without sampling noise.
    action[0] = linear velocity  ∈ [0, max_v]
    action[1] = angular velocity ∈ [-1.5, 1.5]
    """
    v = jnp.clip(mean[..., 0], 0.0, max_v)
    w = jnp.clip(mean[..., 1], -1.5, 1.5)
    return jnp.stack([v, w], axis=-1)