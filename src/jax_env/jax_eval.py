import os
import time
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

from jax_env import reset_env, step_env, ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic

# =============================================================================
# EVALUATION SETUP
# =============================================================================

WINDOW_SIZE = 800
SCALE = WINDOW_SIZE / max(ROOM_W, ROOM_H)

def load_checkpoint(dummy_params, filepath="checkpoints/ppo_model_best.msgpack"):
    """Loads the trained weights from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    return flax.serialization.from_bytes(dummy_params, bytes_data)

def draw_env(screen, state, lidar_flat):
    """
    Renders the environment using PyGame.
    Expects state variables to be standard CPU Numpy arrays.
    """
    screen.fill((255, 255, 255))
    
    # Helper to convert simulation coordinates to screen pixels
    def to_screen(x, y):
        return int(x * SCALE), int(WINDOW_SIZE - (y * SCALE))

    # 1. Draw Goal
    gx, gy = to_screen(state.goal_x, state.goal_y)
    pygame.draw.circle(screen, (255, 165, 0), (gx, gy), int(0.3 * SCALE))
    
    # 2. Draw People
    for i in range(state.people.shape[0]):
        px, py = state.people[i, 0], state.people[i, 1]
        cx, cy = to_screen(px, py)
        pygame.draw.circle(screen, (0, 200, 0), (cx, cy), int(PEOPLE_RADIUS * SCALE))
        
        # Draw heading
        angle = state.people[i, 4]
        hx, hy = to_screen(px + 0.3 * np.cos(angle), py + 0.3 * np.sin(angle))
        pygame.draw.line(screen, (0, 100, 0), (cx, cy), (hx, hy), 2)
        
    # 3. Draw Robot
    rx, ry = to_screen(state.x, state.y)
    pygame.draw.circle(screen, (0, 0, 255), (rx, ry), int(ROBOT_RADIUS * SCALE))
    hx, hy = to_screen(state.x + (ROBOT_RADIUS * 1.5) * np.cos(state.theta), 
                       state.y + (ROBOT_RADIUS * 1.5) * np.sin(state.theta))
    pygame.draw.line(screen, (0, 0, 100), (rx, ry), (hx, hy), 3)
    
    pygame.display.flip()

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================
def main():
    print("🚀 Initializing JAX Evaluation Engine...")
    
    # 1. Setup Network and load weights
    network = EndToEndActorCritic(action_dim=2)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_obs = jnp.zeros((1, 9 + 5 + 324)) 
    params = network.init(init_rng, dummy_obs)["params"]
    
    try:
        params = load_checkpoint(params, "checkpoints/ppo_model_best.msgpack")
        print("✅ Loaded trained policy weights.")
    except FileNotFoundError as e:
        print(f"⚠️ {e}. Running with untrained random weights.")

    # 2. Setup Environment
    # Note: We use the stacked env, but NOT the autoreset env. 
    # For evaluation, we want to know when the episode actually ends.
    reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
    
    # We use jit to compile single-environment step for fast inference
    fast_reset = jax.jit(reset_stacked)
    fast_step = jax.jit(step_stacked)

    # 3. Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("JAX Trained Policy Evaluation")
    clock = pygame.time.Clock()

    # 4. Episode Loop
    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)
    
    ep_reward = 0.0
    done = False
    
    print("🎮 Starting simulation. Close the window to stop.")
    while True:
        # Handle PyGame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # --- JAX GPU INFERENCE ---
        # Add batch dimension since network expects (Batch, Features)
        obs_batch = jnp.expand_dims(obs, axis=0)
        
        # We only take the mean (deterministic action) for evaluation. No sampling noise!
        mean, _, _ = network.apply({"params": params}, obs_batch)
        action = jnp.squeeze(mean, axis=0) # Remove batch dimension
        
        # Step environment on GPU
        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, action)
        ep_reward += reward

        # --- DATA TRANSFER: VRAM -> RAM ---
        # Pull the current physical state from the GPU to the CPU for PyGame
        # We only extract the base env_state, not the entire historical stack
        cpu_state = jax.device_get(stacked_state.env_state)
        cpu_obs = jax.device_get(obs)
        
        # Render on CPU
        draw_env(screen, cpu_state, cpu_obs)
        
        # Control playback speed (e.g., 30 FPS)
        clock.tick(30)
        
        if done:
            print(f"🏁 Episode Finished! Reward: {ep_reward:.2f}")
            time.sleep(1.0) # Pause for a second to see the result
            
            # Reset for a new episode
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            ep_reward = 0.0
            done = False

if __name__ == "__main__":
    main()