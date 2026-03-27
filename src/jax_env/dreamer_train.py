"""
dreamer_train.py — Main Training Loop for DreamerV3
"""

import jax
import jax.numpy as jnp
import optax
import time
import argparse
from functools import partial

# Import your custom Dreamer modules
from dreamer_buffer import init_buffer, add_batch, sample_sequences
from dreamer_rssm import RSSM, DreamerEncoder
from dreamer_decoders import ObservationDecoder, RewardDecoder, ContinueDecoder, world_model_loss
from dreamer_behavior import DreamerActor, DreamerCritic, compute_lambda_returns, unroll_imagination, sample_action

# Import your vectorized environment
from jax_train import init_env_state
from jax_network import scale_actions_batched

# --- Hyperparameters ---
NUM_ENVS = 64
SEQ_LEN = 64
BATCH_SIZE = 16
HORIZON = 15
BUFFER_CAPACITY = 100000 
PREFILL_STEPS = 1000

LR_WM = 1e-4
LR_ACTOR = 3e-5
LR_CRITIC = 3e-5
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_COEF = 3e-4

# --- Network Instantiation ---
encoder = DreamerEncoder()
rssm = RSSM(action_dim=2)
obs_decoder = ObservationDecoder(obs_dim=342)
rew_decoder = RewardDecoder()
cont_decoder = ContinueDecoder()
actor = DreamerActor(action_dim=2)
critic = DreamerCritic()

def scan_rssm(wm_params: dict, obs_seq: jnp.ndarray, act_seq: jnp.ndarray) -> tuple:
    """
    Unrolls the RSSM over the L=64 temporal sequence.
    """
    def step(carry, inputs):
        h_prev, z_prev = carry
        obs_t, act_prev = inputs
        
        # 1. Deterministic step
        h_t = rssm.apply({'params': wm_params['rssm']}, h_prev, z_prev, act_prev, method=rssm.step_gru)
        
        # 2. Encode observation
        obs_embed = encoder.apply({'params': wm_params['encoder']}, obs_t)
        
        # 3. Compute Posterior (uses observation)
        z_t, post_logits = rssm.apply({'params': wm_params['rssm']}, h_t, obs_embed, method=rssm.posterior)
        
        # 4. Compute Prior (guesses without observation)
        _, prior_logits = rssm.apply({'params': wm_params['rssm']}, h_t, method=rssm.prior)
        
        return (h_t, z_t), (h_t, z_t, prior_logits, post_logits)

    # Initialize h_0 and z_0 with zeros
    batch_sz = obs_seq.shape[0]
    h_0 = jnp.zeros((batch_sz, 512))
    z_0 = jnp.zeros((batch_sz, 32 * 32))
    
    # We shift actions by 1 to represent act_{t-1}. Pad the first step with zero action.
    act_prev = jnp.concatenate([jnp.zeros((batch_sz, 1, 2)), act_seq[:, :-1, :]], axis=1)
    
    # Scan over the sequence length dimension (axis 1)
    # Swap axes so time is the leading dimension for jax.lax.scan
    obs_seq_t = jnp.swapaxes(obs_seq, 0, 1)
    act_prev_t = jnp.swapaxes(act_prev, 0, 1)
    
    _, (h_states, z_states, prior_logits, post_logits) = jax.lax.scan(
        step, (h_0, z_0), (obs_seq_t, act_prev_t)
    )
    
    # Swap axes back to [Batch, Seq, Feature]
    return (
        jnp.swapaxes(h_states, 0, 1),
        jnp.swapaxes(z_states, 0, 1),
        jnp.swapaxes(prior_logits, 0, 1),
        jnp.swapaxes(post_logits, 0, 1)
    )

@partial(jax.jit, static_argnums=(4,))
def train_step(rng_key, buffer_state, params, opt_states, max_goal_dist):
    """
    The monolithic DreamerV3 update loop. Three separate optimizers step inside one JIT.
    """
    k_sample, k_actor, k_wm = jax.random.split(rng_key, 3)
    
    # 1. Sample Sequence
    obs_seq, act_seq, rew_seq, done_seq = sample_sequences(k_sample, buffer_state, BATCH_SIZE, SEQ_LEN)
    
    # --- A. WORLD MODEL UPDATE ---
    def wm_loss_fn(wm_params):
        h_states, z_states, prior_logits, post_logits = scan_rssm(wm_params, obs_seq, act_seq)
        
        obs_pred = obs_decoder.apply({'params': wm_params['obs']}, h_states, z_states)
        rew_pred = rew_decoder.apply({'params': wm_params['rew']}, h_states, z_states)
        cont_logits = cont_decoder.apply({'params': wm_params['cont']}, h_states, z_states)
        
        loss, aux = world_model_loss(
            obs_seq, rew_seq, done_seq, 
            obs_pred, rew_pred, cont_logits, 
            prior_logits, post_logits
        )
        return loss, (aux, h_states, z_states)

    (wm_loss, (wm_aux, h_states, z_states)), wm_grads = jax.value_and_grad(wm_loss_fn, has_aux=True)(params['wm'])
    wm_updates, new_wm_opt = opt_wm.update(wm_grads, opt_states['wm'])
    new_wm_params = optax.apply_updates(params['wm'], wm_updates)
    
    # --- B. ACTOR UPDATE ---
    # Reshape states to collapse Batch and Seq dimensions: [Batch * Seq, Dim]
    # We imagine trajectories starting from EVERY state in the historical sequence
    flat_h = h_states.reshape(-1, h_states.shape[-1])
    flat_z = z_states.reshape(-1, z_states.shape[-1])
    
    # Stop gradients so Actor does not change the World Model physics
    start_h = jax.lax.stop_gradient(flat_h)
    start_z = jax.lax.stop_gradient(flat_z)
    
    def actor_loss_fn(actor_params):
        # Unroll imagination using the frozen World Model
        traj, _, _ = unroll_imagination(
            k_actor, 
            lambda p, h, z, a: rssm.apply({'params': p}, h, z, a, method=rssm.step_gru),
            lambda p, h, k: rssm.apply({'params': p}, h, method=rssm.prior),
            lambda p, h, z: actor.apply({'params': p}, h, z),
            {'rssm': jax.lax.stop_gradient(new_wm_params['rssm']), 'actor': actor_params},
            start_h, start_z, HORIZON
        )
        
        # Predict values and rewards on the imagined trajectory
        values = critic.apply({'params': jax.lax.stop_gradient(params['critic'])}, traj['h'], traj['z'])
        rewards = rew_decoder.apply({'params': jax.lax.stop_gradient(new_wm_params['rew'])}, traj['h'], traj['z'])
        continues = jax.nn.sigmoid(cont_decoder.apply({'params': jax.lax.stop_gradient(new_wm_params['cont'])}, traj['h'], traj['z']))
        
        # Bootstrap value for the last step
        bootstrap = values[-1]
        
        # Compute Lambda Returns
        lambda_returns = compute_lambda_returns(rewards[:-1], values[:-1], continues[:-1], bootstrap, GAMMA, LAMBDA)
        
        # REINFORCE policy gradient
        advantages = jax.lax.stop_gradient(lambda_returns - values[:-1])
        actor_loss = -jnp.mean(traj['log_prob'][:-1] * advantages)
        
        # Entropy bonus to encourage exploration
        entropy_loss = -ENTROPY_COEF * jnp.mean(-traj['log_prob'][:-1]) 
        
        return actor_loss + entropy_loss, (traj, lambda_returns)

    (act_loss, (traj, lambda_returns)), act_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(params['actor'])
    act_updates, new_act_opt = opt_actor.update(act_grads, opt_states['actor'])
    new_act_params = optax.apply_updates(params['actor'], act_updates)

    # --- C. CRITIC UPDATE ---
    def critic_loss_fn(critic_params):
        # Target is the lambda returns computed during the Actor update (stopped gradient)
        target = jax.lax.stop_gradient(lambda_returns)
        values = critic.apply({'params': critic_params}, jax.lax.stop_gradient(traj['h'][:-1]), jax.lax.stop_gradient(traj['z'][:-1]))
        critic_loss = jnp.mean((values - target)**2)
        return critic_loss

    critic_loss, crit_grads = jax.value_and_grad(critic_loss_fn)(params['critic'])
    crit_updates, new_crit_opt = opt_critic.update(crit_grads, opt_states['critic'])
    new_crit_params = optax.apply_updates(params['critic'], crit_updates)

    # Pack updated parameters
    new_params = {
        'wm': new_wm_params,
        'actor': new_act_params,
        'critic': new_crit_params
    }
    new_opt_states = {
        'wm': new_wm_opt,
        'actor': new_act_opt,
        'critic': new_crit_opt
    }
    
    metrics = {
        'wm_loss': wm_loss,
        'actor_loss': act_loss,
        'critic_loss': critic_loss,
        'kl_loss': wm_aux[3]
    }
    
    return new_params, new_opt_states, metrics

@partial(jax.jit, static_argnums=(4,))
def act_step(rng_key, wm_params, actor_params, obs, prev_h, prev_z, prev_action, explore: bool = True):
    """
    Tracks the latent state and samples an action from the trained Actor.
    Runs entirely on the GPU to keep inference blazingly fast.
    """
    # 1. Encode the current observation
    obs_embed = encoder.apply({'params': wm_params['encoder']}, obs)
    
    # 2. Advance the deterministic state (GRU)
    h_t = rssm.apply({'params': wm_params['rssm']}, prev_h, prev_z, prev_action, method=rssm.step_gru)
    
    # 3. Compute the posterior stochastic state
    z_t, _ = rssm.apply({'params': wm_params['rssm']}, h_t, obs_embed, method=rssm.posterior)
    
    # 4. Get the action distribution from the Actor
    mean, std = actor.apply({'params': actor_params}, h_t, z_t)
    
    # 5. Sample the action (with exploration noise) or take the mean (for evaluation)
    if explore:
        action, _ = sample_action(rng_key, mean, std)
    else:
        action = jnp.tanh(mean)
        
    return action, h_t, z_t

# --- Initialization ---
opt_wm = optax.adam(LR_WM, eps=1e-8)
opt_actor = optax.adam(LR_ACTOR, eps=1e-5)
opt_critic = optax.adam(LR_CRITIC, eps=1e-5)

if __name__ == "__main__":
    print("Initializing DreamerV3 Training...")
    rng = jax.random.PRNGKey(42)
    rng, env_rng, init_rng = jax.random.split(rng, 3)
    
    # 1. Initialize Environments
    env_obs, env_state, vmap_step = init_env_state(env_rng, max_goal_dist=3.0, ghost_prob=1.0, scenario_idx=-1)
    
    # 2. Initialize Replay Buffer
    buffer_state = init_buffer(BUFFER_CAPACITY, NUM_ENVS, obs_size=342, action_dim=2)
    
    # 3. Initialize Parameters
    dummy_obs = jnp.zeros((1, 342))
    dummy_h = jnp.zeros((1, 512))
    dummy_z = jnp.zeros((1, 1024))
    dummy_act = jnp.zeros((1, 2))
    
    params = {
        'wm': {
            'encoder': encoder.init(init_rng, dummy_obs)['params'],
            'rssm': rssm.init(init_rng, dummy_h, dummy_z, dummy_act, method=rssm.step_gru)['params'],
            'obs': obs_decoder.init(init_rng, dummy_h, dummy_z)['params'],
            'rew': rew_decoder.init(init_rng, dummy_h, dummy_z)['params'],
            'cont': cont_decoder.init(init_rng, dummy_h, dummy_z)['params'],
        },
        'actor': actor.init(init_rng, dummy_h, dummy_z)['params'],
        'critic': critic.init(init_rng, dummy_h, dummy_z)['params']
    }
    
    opt_states = {
        'wm': opt_wm.init(params['wm']),
        'actor': opt_actor.init(params['actor']),
        'critic': opt_critic.init(params['critic'])
    }
    
    # Track the latent state for the environments
    current_h = jnp.zeros((NUM_ENVS, 512))
    current_z = jnp.zeros((NUM_ENVS, 1024))
    current_action = jnp.zeros((NUM_ENVS, 2))
    
    print("Pre-filling buffer with random actions...")
    for _ in range(PREFILL_STEPS):
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        raw_actions = jax.random.uniform(act_rng, (NUM_ENVS, 2), minval=-1.0, maxval=1.0)
        env_actions = scale_actions_batched(raw_actions, env_state.env_state.max_v)
        
        next_obs, next_state, rewards, dones, _ = vmap_step(step_rng, env_state, env_actions, 3.0, -1)
        buffer_state = add_batch(buffer_state, env_obs, raw_actions, rewards, dones)
        
        env_obs = next_obs
        env_state = next_state

    # Reset latent states after pre-fill
    current_h = jnp.zeros((NUM_ENVS, 512))
    current_z = jnp.zeros((NUM_ENVS, 1024))
    current_action = jnp.zeros((NUM_ENVS, 2))
        
    print("Starting Main Training Loop...")
    for step in range(50000):
        t0 = time.time()
        rng, act_rng, step_rng, train_rng = jax.random.split(rng, 4)
        
        # 1. ACT: Ask the Actor what to do based on the current latent reality
        raw_actions, current_h, current_z = act_step(
            act_rng, params['wm'], params['actor'], 
            env_obs, current_h, current_z, current_action, explore=True
        )
        current_action = raw_actions
        
        # 2. STEP: Execute the actions in the physics simulator
        env_actions = scale_actions_batched(raw_actions, env_state.env_state.max_v)
        next_obs, next_state, rewards, dones, _ = vmap_step(step_rng, env_state, env_actions, 3.0, -1)
        
        # 3. STORE: Add to memory
        buffer_state = add_batch(buffer_state, env_obs, raw_actions, rewards, dones)
        
        # 4. RESET MEMORY: Wipe GRU state for environments that terminated
        # If done is True (1.0), (1.0 - done) becomes 0.0, zeroing out the state.
        mask = (1.0 - dones[:, None])
        current_h = current_h * mask
        current_z = current_z * mask
        current_action = current_action * mask
        
        env_obs = next_obs
        env_state = next_state
        
        # 5. TRAIN: Execute the monolithic JIT update
        params, opt_states, metrics = train_step(train_rng, buffer_state, params, opt_states, 3.0)
        
        if step % 100 == 0:
            fps = (NUM_ENVS) / (time.time() - t0)
            print(f"Step {step:05d} | WM Loss: {metrics['wm_loss']:.3f} | Act Loss: {metrics['actor_loss']:.3f} | Crit Loss: {metrics['critic_loss']:.3f} | FPS: {fps:.0f}")

    print("Training Complete.")