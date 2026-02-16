import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
import argparse
import os
import torch
import numpy as np
# DummyVecEnv is fundamental for rendering
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Import environment and wrapper
from src.envs.gym_nav_env import GymNavEnv       
from src.envs.vec_lidar_stack import VecTemporalStack as VecLidarStack 
from src.config import RobotConfig, LidarConfig

# Import Custom Algorithm
from models.adaptive_tqc import AdaptiveTQC

class StatePrintCallback(BaseCallback):
    """
    Custom callback to print the UNNORMALIZED 5-element state vector
    at every step to verify physical constants like v_max.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Fetch the raw, unnormalized observation directly from the VecNormalize wrapper
        original_obs = self.training_env.get_original_obs()
        
        if isinstance(original_obs, dict) and "state" in original_obs:
            # Extract the 5-element state vector for the first (and only) environment
            state_vec = original_obs["state"][0]
            
            # Format the array for clean terminal output (3 decimal places)
            formatted_state = np.round(state_vec, 3)
            
            # State maps strictly to: [v_t, w_t, v_max, goal_distance, goal_error_alignment]
            print(f"Step {self.num_timesteps:05d} | PHYSICAL State: {formatted_state}")
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="TQC", choices=["SAC", "TQC", "PPO", "AR_TQC"])
    parser.add_argument("--steps", type=int, default=100000, help="Short training loop")
    
    # Env Config
    parser.add_argument("--num_people", type=int, default=6)
    parser.add_argument("--num_obstacles", type=int, default=12)
    parser.add_argument("--use_legs", action="store_true", default=False)
    parser.add_argument("--real_lidar", action="store_true", default=False)
    parser.add_argument("--lidar_noise", action="store_true", default=False)
    parser.add_argument("--distraction_prob", type=float, default=0.3)
    
    args = parser.parse_args()

    print(f"👀 DEBUG VISUAL TRAINING: {args.algo} | Render Mode: HUMAN")
    print("Press CTRL+C to stop.")

    # 1. Env Factory (Single Env for rendering)
    def make_env():
        env = GymNavEnv(
            render_mode="human", # <--- ENABLES RENDERING
            num_rays=LidarConfig.NUM_RAYS, 
            num_people=args.num_people, 
            num_obstacles=args.num_obstacles,
            use_legs=args.use_legs,
            distraction_prob=args.distraction_prob,
            render_skip=1, # Render every frame
            lidar_noise_enable=args.lidar_noise,
            real_lidar_specs=args.real_lidar,
            stack_dim=5 if args.real_lidar else 3 
        )
        return env

    # 2. Setup Environment (DummyVecEnv is mandatory to see the Pygame window)
    env = DummyVecEnv([make_env]) 
    
    # Wrapper Stacking
    stack_dim_actual = 5 if args.real_lidar else 3
    env = VecLidarStack(env, stack_dim=stack_dim_actual)
    
    # Normalization (creating a new one without loading old states)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 3. Model Setup
    if args.algo == "AR_TQC": ModelClass = AdaptiveTQC
    elif args.algo == "TQC": ModelClass = TQC
    elif args.algo == "PPO": ModelClass = PPO
    elif args.algo == "SAC": ModelClass = SAC
    
    # Lightweight architecture for testing
    policy_kwargs = dict(net_arch=[64, 64])
    if args.algo in ["TQC", "AR_TQC", "SAC"]:
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    model = ModelClass(
        "MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=10000, # Small buffer to start immediately
        batch_size=256, 
        gamma=0.99,
        tau=0.005,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )

    # Instantiate the custom debug callback
    debug_callback = StatePrintCallback()

    # 4. Run Training with Rendering and Callback
    try:
        # Pass the callback to the learn method
        model.learn(total_timesteps=args.steps, log_interval=1, callback=debug_callback)
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    main()