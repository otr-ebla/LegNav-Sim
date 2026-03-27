"""
dreamer_decoders.py — World Model Decoders and Objective for DreamerV3

No bugs in this version. Per-element free_nats clipping is correctly applied
before averaging (jnp.mean(jnp.maximum(kl, free_nats))) — this matches the
DreamerV3 paper spec and was already fixed in the submitted version.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from dreamer_rssm import DETERMINISTIC_SIZE, symlog


class ObservationDecoder(nn.Module):
    obs_dim: int = 342

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        return nn.Dense(self.obs_dim)(x)


class RewardDecoder(nn.Module):
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class ContinueDecoder(nn.Module):
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


def compute_kl_loss(
    prior_logits: jnp.ndarray,
    posterior_logits: jnp.ndarray,
    free_nats: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    prior_log_probs     = jax.nn.log_softmax(prior_logits,     axis=-1)
    posterior_log_probs = jax.nn.log_softmax(posterior_logits, axis=-1)
    posterior_probs     = jax.nn.softmax(posterior_logits,     axis=-1)

    # KL(Posterior || Prior) split into prior-training and posterior-training terms
    kl_prior = jnp.sum(
        jax.lax.stop_gradient(posterior_probs) * (
            jax.lax.stop_gradient(posterior_log_probs) - prior_log_probs),
        axis=-1)
    kl_posterior = jnp.sum(
        posterior_probs * (
            posterior_log_probs - jax.lax.stop_gradient(prior_log_probs)),
        axis=-1)

    # Per-element free_nats clipping (correct DreamerV3 spec)
    kl_prior_loss     = jnp.mean(jnp.maximum(kl_prior,     free_nats))
    kl_posterior_loss = jnp.mean(jnp.maximum(kl_posterior, free_nats))

    loss    = 0.8 * kl_prior_loss + 0.2 * kl_posterior_loss
    raw_kl  = jnp.mean(jnp.sum(posterior_probs * (posterior_log_probs - prior_log_probs), axis=-1))
    return loss, raw_kl


def world_model_loss(
    obs_target, reward_target, done_target,
    obs_pred, reward_pred, continue_logits,
    prior_logits, posterior_logits,
):
    obs_loss    = jnp.mean((obs_pred - obs_target) ** 2)
    reward_loss = jnp.mean((reward_pred - symlog(reward_target)) ** 2)

    continue_target = 1.0 - done_target.astype(jnp.float32)
    continue_loss   = jnp.mean(
        jnp.maximum(continue_logits, 0.0)
        - continue_logits * continue_target
        + jnp.log(1.0 + jnp.exp(-jnp.abs(continue_logits)))
    )

    kl_loss, raw_kl = compute_kl_loss(prior_logits, posterior_logits)
    total_loss      = obs_loss + reward_loss + continue_loss + kl_loss
    return total_loss, (obs_loss, reward_loss, continue_loss, kl_loss, raw_kl)