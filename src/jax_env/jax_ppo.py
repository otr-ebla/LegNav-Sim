import os
import time
import jax
import jax.numpy as jnp
import optax
import flax.serialization

from jax_network import EndToEndActorCritic
from jax_train import collect_rollouts, NUM_ENVS, ROLLOUT_STEPS

# --- Hyperparameters ---
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
TOTAL_UPDATES = 2000

# --- Checkpoint Utils ---
def save_checkpoint(params, filepath="checkpoints/ppo_model_best.msgpack"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bytes_data = flax.serialization.to_bytes(params)
    with open(filepath, "wb") as f:
        f.write(bytes_data)

def load_checkpoint(dummy_params, filepath="checkpoints/ppo_model_best.msgpack"):
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    return flax.serialization.from_bytes(dummy_params, bytes_data)

# --- Initialization ---
network = EndToEndActorCritic(action_dim=2)
optimizer = optax.adam(learning_rate=LR)

# --- Math: Generalized Advantage Estimation (GAE) ---
@jax.jit
def compute_gae(rewards, values, dones, last_val, last_done):
    """
    Computes GAE using a backward scan for maximum GPU efficiency.
    """
    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        r, v, d = transition
        delta = r + GAMMA * next_value * (1.0 - d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * gae
        return (gae, v, d), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns

# --- Math: PPO Loss ---
@jax.jit
def ppo_loss_fn(params, apply_fn, obs, actions, advantages, returns, old_log_probs):
    mean, logstd, values = apply_fn({"params": params}, obs)
    
    var = jnp.exp(2 * logstd)
    log_prob = -0.5 * (((actions - mean) ** 2) / var + jnp.log(var) + jnp.log(2 * jnp.pi))
    log_prob = jnp.sum(log_prob, axis=-1)
    
    ratio = jnp.exp(log_prob - old_log_probs)
    p_loss1 = ratio * advantages
    p_loss2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
    policy_loss = -jnp.mean(jnp.minimum(p_loss1, p_loss2))
    
    value_loss = VF_COEF * jnp.mean((returns - values) ** 2)
    entropy_loss = -ENTROPY_COEF * jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e * var))
    
    return policy_loss + value_loss + entropy_loss

# --- Optimization Step ---
@jax.jit
def update_step(train_state, rollout_data, advantages, returns):
    params, opt_state = train_state
    
    # Flatten everything: (ROLLOUT_STEPS, NUM_ENVS, ...) -> (ROLLOUT_STEPS * NUM_ENVS, ...)
    obs_flat = rollout_data["obs"].reshape(-1, rollout_data["obs"].shape[-1])
    actions_flat = rollout_data["actions"].reshape(-1, rollout_data["actions"].shape[-1])
    adv_flat = advantages.reshape(-1)
    ret_flat = returns.reshape(-1)
    old_lp_flat = rollout_data["log_probs"].reshape(-1)
    
    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    
    grad_fn = jax.value_and_grad(ppo_loss_fn)
    loss, grads = grad_fn(params, network.apply, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return (new_params, new_opt_state), loss

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    print("🚀 Initializing Pure JAX Engine...")
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Dummy observation to initialize the network architecture
    dummy_obs = jnp.zeros((1, 9 + 5 + 324)) 
    params = network.init(init_rng, dummy_obs)["params"]
    
    # (Optional) Load previous weights
    # if os.path.exists("checkpoints/ppo_model_best.msgpack"):
    #     params = load_checkpoint(params)
    #     print("📥 Loaded existing checkpoint.")
    
    opt_state = optimizer.init(params)
    train_state = (params, opt_state)
    
    print(f"🔥 Training Started! Generating {NUM_ENVS * ROLLOUT_STEPS} steps per update.")
    start_time = time.time()
    
    for update in range(TOTAL_UPDATES):
        rng, rollout_rng = jax.random.split(rng)
        
        # 1. Rollout Collection
        rollout_history, final_carry = collect_rollouts(rollout_rng, train_state[0], network.apply)
        
        # 2. Bootstrap Value for GAE
        final_obs = final_carry[1]
        _, _, last_val = network.apply({"params": train_state[0]}, final_obs)
        last_done = jnp.zeros(NUM_ENVS, dtype=jnp.bool_) # Bootstrap masks
        
        # 3. Compute GAE
        advantages, returns = compute_gae(
            rollout_history["rewards"], 
            rollout_history["values"], 
            rollout_history["dones"], 
            last_val, 
            last_done
        )
        
        # 4. Neural Network Update
        train_state, loss = update_step(train_state, rollout_history, advantages, returns)
        
        # 5. Logging & Saving
        if update % 10 == 0:
            avg_reward = rollout_history["rewards"].mean()
            elapsed = time.time() - start_time
            fps = (NUM_ENVS * ROLLOUT_STEPS * 10) / elapsed
            print(f"Update {update:04d} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.4f} | FPS: {int(fps)}")
            start_time = time.time() # Reset timer for next batch
            
            save_checkpoint(train_state[0])

    print("✅ Training Complete!")