"""
navrep_network.py — NavRep-style Actor-Critic (Flax, CPU-safe to import).

Architecture (NavRep E2E variant):
  Obs layout (662D from make_stacked_env):
    obs[:9]   = 3-frame pose stack (3×3)
    obs[9:14] = current state_vec (5)
    obs[14:]  = 3-frame lidar stack (3×216 = 648), treated as (216, 3) channels

  LiDAR branch  → 1D CNN → 32-dim latent  (analogous to NavRep V, _Z=32)
  State branch  → MLP    → 32-dim
  Combined (64) → separate [64,64] policy and value MLPs  (NavRep 2-layer policy)

Initialisation mirrors jax_ppo.py (orthogonal kernel, zero bias).
"""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal, constant

LOG_STD_MIN = -4.0
LOG_STD_MAX  =  0.0

# Obs layout constants (must match jax_wrappers.py)
_POSE_STACK_END  = 9       # 3 frames × 3D pose
_STATE_END       = 14      # + 5D state_vec
_NUM_RAYS        = 216
_STACK_DIM       = 3


class NavRepActorCritic(nn.Module):
    """
    1D CNN + MLP actor-critic.

    Parameters
    ----------
    action_dim : int   Dimensionality of the action space (default 2: [v, w]).
    lidar_z    : int   Lidar latent dimension, analogous to NavRep _Z (default 32).
    hidden_dim : int   Hidden dim for policy/value MLPs (default 64).
    """
    action_dim: int = 2
    lidar_z:    int = 32
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """
        Parameters
        ----------
        obs : (..., 662) float32

        Returns
        -------
        mean   : (..., action_dim)
        logstd : (..., action_dim)
        value  : (...)
        """
        batch_shape = obs.shape[:-1]

        # ── Split obs ─────────────────────────────────────────────────────────
        pose_flat  = obs[..., :_POSE_STACK_END]            # (..., 9)
        state_vec  = obs[..., _POSE_STACK_END:_STATE_END]  # (..., 5)
        lidar_flat = obs[..., _STATE_END:]                  # (..., 648)

        # ── LiDAR branch: 1D CNN ──────────────────────────────────────────────
        # Reshape to (B, 216, 3): length=216 samples, 3 time-frame channels.
        # Flax Conv1D operates on (..., L, C_in) → (..., L_out, C_out).
        #
        # Convolution output lengths for L_in=216 (no padding):
        #   Conv(32, k=8, s=4): floor((216-8)/4)+1 = 53
        #   Conv(64, k=4, s=2): floor(( 53-4)/2)+1 = 25
        #   Conv(64, k=4, s=2): floor(( 25-4)/2)+1 = 11
        #   Flatten: 11×64 = 704  →  Linear(32)
        lidar_3f = lidar_flat.reshape(*batch_shape, _NUM_RAYS, _STACK_DIM)

        x = nn.Conv(32, kernel_size=(8,), strides=(4,),
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(lidar_3f)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4,), strides=(2,),
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4,), strides=(2,),
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = nn.relu(x)
        z_lidar = x.reshape(*batch_shape, -1)                    # (..., 704)
        z_lidar = nn.Dense(self.lidar_z,
                           kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(z_lidar)
        z_lidar = nn.relu(z_lidar)                               # (..., lidar_z=32)

        # ── State branch: MLP ─────────────────────────────────────────────────
        state_input = jnp.concatenate([pose_flat, state_vec], axis=-1)  # (..., 14)
        z_state = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(state_input)
        z_state = nn.relu(z_state)
        z_state = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(z_state)
        z_state = nn.relu(z_state)                               # (..., 32)

        # ── Combined features ─────────────────────────────────────────────────
        features = jnp.concatenate([z_lidar, z_state], axis=-1)  # (..., 64)

        # ── Actor head (2-layer MLP, NavRep policy architecture) ──────────────
        pi = nn.Dense(self.hidden_dim,
                      kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(features)
        pi = nn.tanh(pi)
        pi = nn.Dense(self.hidden_dim,
                      kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(pi)
        pi = nn.tanh(pi)
        mean = nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(pi)

        raw_logstd = self.param("log_std", constant(-1.0), (self.action_dim,))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(raw_logstd) + 1.0
        )

        # ── Critic head (2-layer MLP) ─────────────────────────────────────────
        vf = nn.Dense(self.hidden_dim,
                      kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(features)
        vf = nn.tanh(vf)
        vf = nn.Dense(self.hidden_dim,
                      kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(vf)
        vf = nn.tanh(vf)
        value = nn.Dense(1,
                         kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(vf)
        value = jnp.squeeze(value, axis=-1)

        return mean, logstd, value
