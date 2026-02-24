"""
SACnetwork.py — SAC Actor and Twin-Critic Networks
========================================================
FIXES vs previous version:

  FIX 1 — OBS_SIZE / state_size updated from 6 to 9 (critical):
    ObsEncoder hardcoded state_size=6 but jax_env.py now emits STATE_VEC_SIZE=9
    (v, w, max_v_norm, goal_dist, goal_align, rear_prox×4). This caused wrong
    slice boundaries: lidar_flat started at wrong offset, silently feeding
    garbled data into the CNN. Updated state_size=9 throughout.
    New obs layout: pose_stack(9) + state_vec(9) + lidar_stack(324) = 342.

  FIX 2 — squash_and_log_prob missing `mean` parameter (wrong log-prob):
    The function computed the Gaussian log-prob as:
        -0.5 * ((u - 0.0) / std)**2  ← assumed mean=0 everywhere!
    `mean` was never passed in, so the log-prob was only correct at init
    when mean≈0. During training this gave arbitrarily wrong values and would
    corrupt both the actor loss and the alpha (temperature) tuning.
    FIX: added `mean` as an explicit parameter; compute noise=(u-mean)/std
    cleanly, matching the numerics used in sample_action_sac.

  FIX 3 — squash_and_log_prob and sample_action_sac now share identical
    log-prob arithmetic (via a shared helper _tanh_log_prob_correction) to
    prevent any future drift between the two code paths.

  UNCHANGED — Architecture, CNN, trunk, log_std clamping, action squashing
    math, get_deterministic_action_sac are all correct and unchanged.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

# Obs layout: pose_stack(3*3=9) + state_vec(9) + lidar_stack(108*3=324) = 342
# state_size MUST match STATE_VEC_SIZE in jax_env.py
_STATE_VEC_SIZE = 9   # v, w, max_v_norm, goal_dist, goal_align, rear_prox×4

LOG_STD_EPS = 1e-6   # numerical stability for log(1 - tanh²(u))


# ── Shared CNN + trunk encoder ────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """
    Encodes the stacked obs into a single feature vector.
    Obs layout: [pose_stack(9) | state_vec(9) | lidar_stack(324)]
    Output: feature vector of size 128.

    FIX: state_size corrected from 6 → 9 to match jax_env.py STATE_VEC_SIZE.
    """
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pose_size  = 3 * self.stack_dim   # 9
        state_size = _STATE_VEC_SIZE      # 9  ← FIX: was 6

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
    Both Q heads have separate encoders (no shared weights — standard SAC).
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


# ── Shared log-prob correction helper ────────────────────────────────────────

def _tanh_log_prob_correction(
    tanh_u: jnp.ndarray,   # (batch, 2) — tanh(u), already computed
    max_v:  jnp.ndarray,   # (batch,) or scalar — per-sample max linear speed
) -> jnp.ndarray:
    """
    Compute the total Jacobian log-prob correction for the squashing transform:

      v-dim: a_v = (tanh(u_v) + 1) / 2 * max_v
        da_v/du_v = max_v/2 * (1 - tanh^2(u_v))
        correction = log(max_v/2) + log(1 - tanh^2(u_v) + eps)

      w-dim: a_w = tanh(u_w)
        da_w/du_w = 1 - tanh^2(u_w)
        correction = log(1 - tanh^2(u_w) + eps)

    Returns the NEGATIVE sum (to subtract from Gaussian log-prob):
        -(log|da_v/du_v| + log|da_w/du_w|)

    This shared helper guarantees squash_and_log_prob and sample_action_sac
    use identical arithmetic.
    """
    corr_v = (
        jnp.log(max_v * 0.5 + LOG_STD_EPS)
        + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )   # (batch,)

    corr_w = jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)   # (batch,)

    return -(corr_v + corr_w)   # subtract from log-prob


# ── Action squashing and log-prob ─────────────────────────────────────────────

def squash_and_log_prob(
    u:       jnp.ndarray,    # pre-squash action (batch, 2)
    mean:    jnp.ndarray,    # (batch, 2) — actor mean (FIX: was missing)
    log_std: jnp.ndarray,    # (batch, 2)
    max_v:   jnp.ndarray,    # per-sample max speed (batch,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given pre-squash sample u ~ N(mean, std):
      1. Compute Gaussian log-prob in u-space using (u - mean) / std.
      2. Apply tanh squashing to both dims.
      3. Shift v-dim from [-1,1] to [0,1] and scale by max_v.
      4. Correct log-prob for the change of variables via the shared helper.

    FIX: `mean` is now an explicit parameter. Previously the function assumed
    mean=0, giving wrong log-probs throughout training.

    Returns:
      env_action : (batch, 2) — [v ∈ [0,max_v], w ∈ [-1,1]]
      log_pi     : (batch,)  — corrected log-probability
    """
    std   = jnp.exp(log_std)
    noise = (u - mean) / std   # FIX: was (u - 0.0) / std

    # Gaussian log-prob via noise (numerically cleaner, matches sample_action_sac)
    log_prob_gaussian = jnp.sum(
        -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std,
        axis=-1
    )   # (batch,)

    # Squash both dims through tanh
    tanh_u = jnp.tanh(u)   # (batch, 2)

    # v-dim: shift to [0, 1] then scale by max_v
    a_v = (tanh_u[..., 0] + 1.0) * 0.5 * max_v   # (batch,)
    # w-dim: stays as tanh(u_w) in [-1, 1]
    a_w = tanh_u[..., 1]                           # (batch,)

    env_action = jnp.stack([a_v, a_w], axis=-1)   # (batch, 2)

    # Jacobian correction (shared helper — identical to sample_action_sac)
    log_pi = log_prob_gaussian + _tanh_log_prob_correction(tanh_u, max_v)

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
      u          : (batch, 2) — pre-squash sample (for debugging / squash_and_log_prob)
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std   # pre-squash sample

    # Gaussian log-prob via noise (numerically cleaner than via u directly)
    log_prob_gaussian = jnp.sum(
        -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std,
        axis=-1
    )   # (batch,)

    # Squash and build env action
    tanh_u = jnp.tanh(u)
    a_v    = (tanh_u[..., 0] + 1.0) * 0.5 * max_v
    a_w    = tanh_u[..., 1]
    env_action = jnp.stack([a_v, a_w], axis=-1)

    # Jacobian correction (shared helper — identical to squash_and_log_prob)
    log_pi = log_prob_gaussian + _tanh_log_prob_correction(tanh_u, max_v)

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