"""
dreamer_decoders.py — World Model Decoders and Objective for DreamerV3

"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from dreamer_rssm import DETERMINISTIC_SIZE, symlog, apply_unimix


class ObservationDecoder(nn.Module):
    obs_dim: int = 662

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        return nn.Dense(self.obs_dim)(x)


class RewardDecoder(nn.Module):
    num_bins: int = 255

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        # V3 zero-initialization to prevent early reward hallucination
        return nn.Dense(self.num_bins, kernel_init=jax.nn.initializers.zeros)(x)


class ContinueDecoder(nn.Module):
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        x = nn.swish(nn.RMSNorm()(nn.Dense(DETERMINISTIC_SIZE)(x)))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)

def two_hot_encode(target: jnp.ndarray, min_val: float = -20.0, max_val: float = 20.0, num_bins: int = 255) -> jnp.ndarray:
    """
    Projects a continuous target into a two-hot categorical distribution.
    Optimized for JAX XLA compilation with zero branching.
    """
    target = jnp.clip(target, min_val, max_val)
    bin_width = (max_val - min_val) / (num_bins - 1)
    bin_index_continuous = (target - min_val) / bin_width
    
    lower_bin = jnp.floor(bin_index_continuous)
    upper_bin = jnp.ceil(bin_index_continuous)
    
    upper_bin = jnp.where(lower_bin == upper_bin, lower_bin + 1, upper_bin)
    upper_bin = jnp.clip(upper_bin, 0, num_bins - 1)
    
    upper_weight = bin_index_continuous - lower_bin
    lower_weight = 1.0 - upper_weight
    
    lower_one_hot = jax.nn.one_hot(lower_bin.astype(jnp.int32), num_bins)
    upper_one_hot = jax.nn.one_hot(upper_bin.astype(jnp.int32), num_bins)
    
    return lower_one_hot * lower_weight[..., None] + upper_one_hot * upper_weight[..., None]

def two_hot_loss(logits: jnp.ndarray, target_scalar: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross-entropy loss between network logits and symlog two-hot targets.
    """
    symlog_target = symlog(target_scalar)
    two_hot_target = two_hot_encode(symlog_target, num_bins=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(two_hot_target * log_probs, axis=-1)


def compute_kl_loss(
    prior_logits: jnp.ndarray,
    posterior_logits: jnp.ndarray,
    free_nats: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    prior_log_probs     = apply_unimix(prior_logits)
    posterior_log_probs = apply_unimix(posterior_logits)
    # posterior_probs derived from the same regularized distribution so the
    # KL expectation E_q[log q - log p] is internally consistent.
    posterior_probs     = jnp.exp(posterior_log_probs)

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

    # V3 Dynamics and Representation loss weights: beta_dyn=1.0, beta_rep=0.1
    loss   = 1.0 * kl_prior_loss + 0.1 * kl_posterior_loss
    raw_kl = jnp.mean(jnp.sum(posterior_probs * (posterior_log_probs - prior_log_probs), axis=-1))
    return loss, raw_kl


def world_model_loss(
    obs_target, reward_target, done_target,
    obs_pred, reward_pred, continue_logits,
    prior_logits, posterior_logits,
):
    obs_loss    = 0.5 * jnp.mean((obs_pred - symlog(obs_target)) ** 2)
    reward_loss = jnp.mean(two_hot_loss(reward_pred, reward_target))

    continue_target = 1.0 - done_target.astype(jnp.float32)
    continue_loss   = jnp.mean(
        jnp.maximum(continue_logits, 0.0)
        - continue_logits * continue_target
        + jnp.log(1.0 + jnp.exp(-jnp.abs(continue_logits)))
    )

    kl_loss, raw_kl = compute_kl_loss(prior_logits, posterior_logits)
    total_loss      = obs_loss + reward_loss + continue_loss + kl_loss
    return total_loss, (obs_loss, reward_loss, continue_loss, kl_loss, raw_kl)