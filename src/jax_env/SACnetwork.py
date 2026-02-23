"""
jax_sac_network.py — SAC Actor and Twin-Critic Networks
========================================================
Design:
  - Actor:   CNN(obs) → shared trunk → mean(2) + log_std(2)  [state-dependent std]
  - Critic1: CNN(obs) → shared trunk → concat(action) → Q(1)
  - Critic2: identical architecture, separate parameters
  - Actor and critics are FULLY SEPARATE (no shared weights) — standard SAC.

Action space:
  The environment expects:
    v ∈ [0, max_v]  (linear speed)
    w ∈ [-1, 1]     (angular speed)
  where max_v is per-episode and encoded in obs (state_vec[2] = max_v / 2.0).

  SAC squashes via tanh internally. The squashed action sent to the env is:
    a_v = (tanh(u_v) + 1) / 2 * max_v     → [0, max_v]
    a_w = tanh(u_w)                        → [-1, 1]

  Log-prob correction for change of variables (mandatory for correct SAC):
    log π(a|s) = log π_gaussian(u|s)
               − log(max_v / 2)                      [Jacobian of v-dim scaling]
               − log(1 − tanh²(u_v) + ε)            [Jacobian of tanh on v]
               − log(1 − tanh²(u_w) + ε)            [Jacobian of tanh on w]

  The numerically stable form of log(1 − tanh²(u)) is used throughout:
    log(1 − tanh²(u)) = 2*(log 2 − u − softplus(−2u))
                       ≡ −2 * log_cosh_stable(u) + log 4   [equivalent]
  We use: −2 * jnp.log(jnp.cosh(u) + ε) for clarity and stability.
  Actually the cleanest stable form used here:
    log(1 − tanh²(u) + ε)  with ε = 1e-6
  is fine for |u| < ~10, which is guaranteed by log_std clamping.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# ── Shared CNN + trunk encoder (used by both actor and critics) ────────────────

class ObsEncoder(nn.Module):
    """
    Encodes the stacked obs into a single feature vector.
    Obs layout: [pose_stack(9) | state_vec(6) | lidar_stack(324)]
    Output: feature vector of size 128.
    """
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 6

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # ── 1-D CNN on stacked LiDAR ─────────────────────────────────────────
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))

        cnn = nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn)
        cnn = nn.relu(cnn)
        cnn = nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn)
        cnn = nn.relu(cnn)
        cnn = nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn)
        cnn = nn.relu(cnn)

        cnn_flat = cnn.reshape((*batch_shape, -1))
        cnn_feat = nn.LayerNorm()(cnn_flat)

        # ── Global state MLP ─────────────────────────────────────────────────
        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        # ── Fusion ────────────────────────────────────────────────────────────
        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        return shared  # (batch, 128)


# ── Actor ─────────────────────────────────────────────────────────────────────

class SACActorNetwork(nn.Module):
    """
    Gaussian actor with state-dependent log_std.
    Outputs (mean, log_std) both of shape (batch, action_dim).
    log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX] for stability.
    """
    action_dim: int = 2
    stack_dim:  int = 3
    num_rays:   int = 108

    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        feat    = ObsEncoder(stack_dim=self.stack_dim, num_rays=self.num_rays)(obs)
        mean    = nn.Dense(self.action_dim)(feat)
        log_std = nn.Dense(self.action_dim)(feat)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std


# ── Critic (Q-network) ────────────────────────────────────────────────────────

class SACCriticNetwork(nn.Module):
    """
    Twin Q-network. Takes (obs, action) and returns (Q1, Q2).
    The action is concatenated AFTER the obs encoder (not before the CNN).
    Both Q heads share the CNN encoder but have separate dense layers.
    """
    action_dim: int = 2
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Each Q-head has its OWN encoder (separate parameters, crucial for SAC)
        feat1 = ObsEncoder(stack_dim=self.stack_dim, num_rays=self.num_rays,
                           name='encoder_q1')(obs)
        feat2 = ObsEncoder(stack_dim=self.stack_dim, num_rays=self.num_rays,
                           name='encoder_q2')(obs)

        # Concatenate action AFTER encoding obs
        q1_in = jnp.concatenate([feat1, action], axis=-1)
        q2_in = jnp.concatenate([feat2, action], axis=-1)

        # Q1 head
        q1 = nn.relu(nn.Dense(256, name='q1_l1')(q1_in))
        q1 = nn.relu(nn.Dense(128, name='q1_l2')(q1))
        q1 = nn.Dense(1, name='q1_out')(q1)
        q1 = jnp.squeeze(q1, axis=-1)  # (batch,)

        # Q2 head
        q2 = nn.relu(nn.Dense(256, name='q2_l1')(q2_in))
        q2 = nn.relu(nn.Dense(128, name='q2_l2')(q2))
        q2 = nn.Dense(1, name='q2_out')(q2)
        q2 = jnp.squeeze(q2, axis=-1)  # (batch,)

        return q1, q2


# ── Action squashing and log-prob ─────────────────────────────────────────────

LOG_STD_EPS = 1e-6   # numerical stability for log(1 - tanh^2(u))


def squash_and_log_prob(
    u: jnp.ndarray,          # pre-squash action (batch, 2)
    log_std: jnp.ndarray,    # (batch, 2) or broadcast (2,)
    max_v: jnp.ndarray,      # per-sample max speed (batch,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given pre-squash u ~ N(mean, std):
      1. Compute Gaussian log-prob in u-space.
      2. Apply tanh squashing to both dims.
      3. Shift v-dim from [-1,1] to [0,1] and scale by max_v.
      4. Correct log-prob for the change of variables.

    Returns:
      env_action : (batch, 2) — [v ∈ [0,max_v], w ∈ [-1,1]]
      log_pi     : (batch,)  — corrected log-probability
    """
    std = jnp.exp(log_std)

    # Gaussian log-prob: -0.5*(u^2/std^2 + log(2π) + 2*log_std)
    log_prob_gaussian = -0.5 * (
        ((u - 0.0) / std) ** 2          # (mean is passed separately; u = mean + noise*std)
        + jnp.log(2.0 * jnp.pi)
        + 2.0 * log_std
    )
    # Sum over action dims
    log_prob_gaussian = jnp.sum(log_prob_gaussian, axis=-1)   # (batch,)

    # Squash both dims through tanh
    tanh_u = jnp.tanh(u)   # (batch, 2), both in (-1, 1)

    # v-dim: shift to [0, 1] then scale by max_v
    # a_v = (tanh(u_v) + 1) / 2 * max_v
    a_v = (tanh_u[..., 0] + 1.0) * 0.5 * max_v   # (batch,)

    # w-dim: stays as tanh(u_w) in [-1, 1]
    a_w = tanh_u[..., 1]                           # (batch,)

    env_action = jnp.stack([a_v, a_w], axis=-1)   # (batch, 2)

    # ── Log-prob correction for change of variables ───────────────────────────
    # d(tanh(u))/du = 1 - tanh^2(u), so the Jacobian correction is:
    #   -log(1 - tanh^2(u) + eps)  per dimension
    # For v-dim, there's an extra scaling by max_v/2 (from the (tanh+1)/2*max_v):
    #   d(a_v)/d(u_v) = max_v/2 * (1 - tanh^2(u_v))
    #   correction_v = -log(max_v/2) - log(1 - tanh^2(u_v) + eps)
    # For w-dim:
    #   correction_w = -log(1 - tanh^2(u_w) + eps)

    corr_v = (
        -jnp.log(max_v * 0.5 + LOG_STD_EPS)
        - jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )   # (batch,)

    corr_w = -jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)   # (batch,)

    log_pi = log_prob_gaussian + corr_v + corr_w   # (batch,)

    return env_action, log_pi


def sample_action_sac(
    rng_key:  jnp.ndarray,
    mean:     jnp.ndarray,    # (batch, 2) or (2,)
    log_std:  jnp.ndarray,    # (batch, 2) or (2,)
    max_v:    jnp.ndarray,    # (batch,)  or scalar
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reparameterised sample from the squashed Gaussian policy.
    Returns:
      env_action : (batch, 2)  — scaled action for the environment
      log_pi     : (batch,)   — corrected log-probability
      u          : (batch, 2) — pre-squash sample (for debugging)
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std   # pre-squash sample

    # Gaussian log-prob using noise (numerically cleaner than using u directly)
    log_prob_gaussian = jnp.sum(
        -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std,
        axis=-1
    )   # (batch,)

    # Squash and correct
    tanh_u = jnp.tanh(u)
    a_v    = (tanh_u[..., 0] + 1.0) * 0.5 * max_v
    a_w    = tanh_u[..., 1]
    env_action = jnp.stack([a_v, a_w], axis=-1)

    # Log-prob correction
    corr_v = (
        -jnp.log(max_v * 0.5 + LOG_STD_EPS)
        - jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )
    corr_w = -jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    log_pi = log_prob_gaussian + corr_v + corr_w

    return env_action, log_pi, u


def get_deterministic_action_sac(
    mean:  jnp.ndarray,   # (2,) or (batch, 2)
    max_v: jnp.ndarray,   # scalar or (batch,)
) -> jnp.ndarray:
    """
    Evaluation-time deterministic action (no noise, use mean directly).
    """
    tanh_mean = jnp.tanh(mean)
    a_v = (tanh_mean[..., 0] + 1.0) * 0.5 * max_v
    a_w = tanh_mean[..., 1]
    return jnp.stack([a_v, a_w], axis=-1)