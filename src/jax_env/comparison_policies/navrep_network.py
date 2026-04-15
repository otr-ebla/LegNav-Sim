"""
navrep_network.py — Faithful NavRep (V + M + C) for indoor-rl-nav.

Modules:
  V  LidarEncoder + LidarDecoder (VAE): 1D-CNN encoder → (z_mean, z_logvar),
     ConvTranspose decoder → reconstructed 216-ray scan.
  M  Causal 2-layer Transformer over z-sequences, trained with MSE(z_{t+1}).
  C  2-layer MLP controller [64, 64] consuming [z_t, h_t, state_r(14)].

Pretraining:
  V is trained on shuffled single-frame LiDARs (reconstruction + β·KL).
  M is trained on per-env z-sequences with predictive MSE loss.

PPO (controller-only) training:
  The forward pass applies stop_gradient to z_t and h_t so gradients flow
  only through the controller parameters; V and M weights remain frozen.

Obs layout (662D from make_stacked_env, stack_dim=3):
  obs[:9]    pose_stack   (3 × 3)
  obs[9:14]  state_vec    (5)
  obs[14:]   lidar_stack  (3 × 216)  frame 0 = oldest, frame 2 = newest
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal, constant

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0

_POSE_STACK_END = 9
_STATE_END      = 14
NUM_RAYS        = 216
STACK_DIM       = 3

Z_DIM       = 32
H_DIM       = 64
STATE_R_DIM = 14


# ══════════════════════════════════════════════════════════════════════════════
# Module V — VAE
# ══════════════════════════════════════════════════════════════════════════════

class LidarEncoder(nn.Module):
    z_dim: int = Z_DIM

    @nn.compact
    def __call__(self, lidar: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_shape = lidar.shape[:-1]
        x = lidar[..., None]                                      # (..., 216, 1)
        x = nn.relu(nn.Conv(32, (8,), strides=(4,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 53, 32)
        x = nn.relu(nn.Conv(64, (4,), strides=(2,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 25, 64)
        x = nn.relu(nn.Conv(64, (4,), strides=(2,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 11, 64)
        x = x.reshape(*batch_shape, -1)                           # (..., 704)
        h = nn.relu(nn.Dense(128,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        z_mean = nn.Dense(self.z_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0), name="z_mean")(h)
        z_logvar = nn.Dense(self.z_dim,
                            kernel_init=orthogonal(0.01),
                            bias_init=constant(0.0), name="z_logvar")(h)
        return z_mean, z_logvar


class LidarDecoder(nn.Module):
    num_rays: int = NUM_RAYS

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch_shape = z.shape[:-1]
        x = nn.relu(nn.Dense(128,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(z))
        x = nn.relu(nn.Dense(27 * 32,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        x = x.reshape(*batch_shape, 27, 32)                       # (..., 27, 32)
        x = nn.relu(nn.ConvTranspose(32, (4,), strides=(2,), padding='SAME',
                                     kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.0))(x))  # (..., 54, 32)
        x = nn.relu(nn.ConvTranspose(16, (4,), strides=(2,), padding='SAME',
                                     kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.0))(x))  # (..., 108, 16)
        x = nn.ConvTranspose(1, (4,), strides=(2,), padding='SAME',
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)           # (..., 216, 1)
        x = nn.sigmoid(x)
        return x[..., 0]                                           # (..., 216)


class VAE(nn.Module):
    """Wraps LidarEncoder + LidarDecoder for pretraining Module V."""
    z_dim: int = Z_DIM

    @nn.compact
    def __call__(self, lidar: jnp.ndarray, rng: jax.Array):
        encoder = LidarEncoder(z_dim=self.z_dim, name="encoder")
        decoder = LidarDecoder(name="decoder")
        z_mean, z_logvar = encoder(lidar)
        eps = jax.random.normal(rng, z_mean.shape)
        z   = z_mean + jnp.exp(0.5 * z_logvar) * eps
        recon = decoder(z)
        return recon, z_mean, z_logvar


# ══════════════════════════════════════════════════════════════════════════════
# Module M — Causal Transformer over z-sequences
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=False,
        )(h, h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        return x + h


class TransformerM(nn.Module):
    """Causal Transformer M: (..., T, Z) → next-z predictions + hidden states."""
    d_model:  int = H_DIM
    n_layers: int = 2
    n_heads:  int = 4
    max_len:  int = 64
    z_dim:    int = Z_DIM

    @nn.compact
    def __call__(self, z_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        T = z_seq.shape[-2]

        h = nn.Dense(self.d_model,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0), name="in_proj")(z_seq)

        pos = self.param("pos_emb", constant(0.0), (self.max_len, self.d_model))
        h = h + pos[:T]

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]

        for i in range(self.n_layers):
            h = TransformerBlock(d_model=self.d_model, n_heads=self.n_heads,
                                 mlp_dim=4 * self.d_model,
                                 name=f"block_{i}")(h, causal)

        h = nn.LayerNorm(name="out_ln")(h)
        next_z = nn.Dense(self.z_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0),
                          name="next_z_head")(h)
        return next_z, h                       # (..., T, Z), (..., T, d_model)


class MWrapper(nn.Module):
    """Pretraining wrapper for M so saved params live under key 'M'."""
    z_dim: int = Z_DIM

    @nn.compact
    def __call__(self, z_seq: jnp.ndarray):
        return TransformerM(z_dim=self.z_dim, name="M")(z_seq)


# ══════════════════════════════════════════════════════════════════════════════
# Full NavRep Actor-Critic (V + M + C, stop_gradient on V and M outputs)
# ══════════════════════════════════════════════════════════════════════════════

class NavRepActorCritic(nn.Module):
    action_dim: int = 2
    hidden_dim: int = 64
    z_dim:      int = Z_DIM

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        batch_shape = obs.shape[:-1]

        pose_flat  = obs[..., :_POSE_STACK_END]
        state_vec  = obs[..., _POSE_STACK_END:_STATE_END]
        lidar_flat = obs[..., _STATE_END:]

        lidar_stack = lidar_flat.reshape(*batch_shape, STACK_DIM, NUM_RAYS)

        # V: encode each of 3 frames (deterministic: use z_mean) → (..., 3, Z)
        z_mean, _z_logvar = LidarEncoder(z_dim=self.z_dim, name="encoder")(lidar_stack)

        # M: causal Transformer over 3 frames → (..., 3, d_model)
        _next_z, h_seq = TransformerM(z_dim=self.z_dim, name="M")(z_mean)

        # Most recent latent / hidden, frozen
        z_t = jax.lax.stop_gradient(z_mean[..., -1, :])            # (..., Z)
        h_t = jax.lax.stop_gradient(h_seq[..., -1, :])             # (..., H)

        state_r = jnp.concatenate([pose_flat, state_vec], axis=-1) # (..., 14)
        feat    = jnp.concatenate([z_t, h_t, state_r], axis=-1)    # (..., Z+H+14)

        # Controller (C): policy head
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(pi))
        mean = nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(pi)

        raw_logstd = self.param("log_std", constant(-1.0), (self.action_dim,))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(raw_logstd) + 1.0
        )
        logstd = jnp.broadcast_to(logstd, mean.shape)

        # Controller (C): value head
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(vf))
        value = nn.Dense(1,
                         kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(vf)
        value = jnp.squeeze(value, axis=-1)

        return mean, logstd, value
