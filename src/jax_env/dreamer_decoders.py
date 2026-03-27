"""
dreamer_decoders.py — World Model Decoders and Objective for DreamerV3
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple  # <-- ADD THIS LINE
from dreamer_rssm import DETERMINISTIC_SIZE, symlog

class ObservationDecoder(nn.Module):
    """
    Reconstructs the flat 342-dim observation vector from the latent state.
    """
    obs_dim: int = 342

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        # No activation on the final layer; we predict raw values
        obs_pred = nn.Dense(self.obs_dim)(x)
        return obs_pred

class RewardDecoder(nn.Module):
    """
    Predicts the symlog-scaled reward from the latent state.
    """
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        reward_pred = nn.Dense(1)(x)
        return jnp.squeeze(reward_pred, axis=-1)

class ContinueDecoder(nn.Module):
    """
    Predicts whether the episode continues (1.0) or terminates/times out (0.0).
    """
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        # Output logits for Binary Cross Entropy
        continue_logits = nn.Dense(1)(x)
        return jnp.squeeze(continue_logits, axis=-1)

def compute_kl_loss(prior_logits: jnp.ndarray, posterior_logits: jnp.ndarray, free_nats: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the KL divergence between the Posterior (which saw the observation) 
    and the Prior (which guessed the future). 
    Uses DreamerV3 KL Balancing to train the prior faster than the posterior.
    """
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)
    posterior_probs = jax.nn.softmax(posterior_logits, axis=-1)

    # Convert to log probabilities for numerical stability
    prior_log_probs = jax.nn.log_softmax(prior_logits, axis=-1)
    posterior_log_probs = jax.nn.log_softmax(posterior_logits, axis=-1)

    # KL(Posterior || Prior) = sum( Posterior * (log(Posterior) - log(Prior)) )
    # 1. Train the Prior toward the Posterior (weight 0.8)
    # stop_gradient prevents the prior from pulling the posterior down to its level
    kl_prior = jnp.sum(
        jax.lax.stop_gradient(posterior_probs) * (jax.lax.stop_gradient(posterior_log_probs) - prior_log_probs), 
        axis=-1
    )
    
    # 2. Train the Posterior toward the Prior (weight 0.2)
    # This acts as a regularizer to keep the latent space compact
    kl_posterior = jnp.sum(
        posterior_probs * (posterior_log_probs - jax.lax.stop_gradient(prior_log_probs)), 
        axis=-1
    )

    # Apply free nats (free bits) clipping to prevent over-regularization early in training
    kl_prior_loss = jnp.maximum(jnp.mean(kl_prior), free_nats)
    kl_posterior_loss = jnp.maximum(jnp.mean(kl_posterior), free_nats)

    # DreamerV3 balancing formulation
    loss = 0.8 * kl_prior_loss + 0.2 * kl_posterior_loss
    return loss, jnp.mean(kl_prior) # Return unclipped KL for logging

def world_model_loss(obs_target, reward_target, done_target, 
                     obs_pred, reward_pred, continue_logits, 
                     prior_logits, posterior_logits):
    """
    Aggregates the reconstruction and dynamics losses.
    """
    # 1. Observation Reconstruction Loss (MSE)
    obs_loss = jnp.mean((obs_pred - obs_target) ** 2)

    # 2. Reward Reconstruction Loss (MSE on Symlog scaled target)
    symlog_reward_target = symlog(reward_target)
    reward_loss = jnp.mean((reward_pred - symlog_reward_target) ** 2)

    # 3. Continue Loss (Binary Cross Entropy)
    # target == 1.0 if episode continues, 0.0 if done
    continue_target = 1.0 - done_target
    continue_loss = jnp.mean(
        jnp.maximum(continue_logits, 0) - continue_logits * continue_target + 
        jnp.log(1 + jnp.exp(-jnp.abs(continue_logits)))
    )

    # 4. Dynamics / KL Loss
    kl_loss, raw_kl = compute_kl_loss(prior_logits, posterior_logits)

    # Total loss (you can tune these coefficients, but equal weighting is standard starting point)
    total_loss = obs_loss + reward_loss + continue_loss + kl_loss
    
    return total_loss, (obs_loss, reward_loss, continue_loss, kl_loss, raw_kl)