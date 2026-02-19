import time
import argparse
import numpy as np
import sys
import os

# Append root to path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

# Imports from your project
from src.envs.gym_nav_env import GymNavEnv
from src.envs.vec_lidar_stack import VecTemporalStack as VecLidarStack
from src.config import LidarConfig, RobotConfig

# Import Baseline Planners
from baselines.dwa_planner import DWAPlanner

def make_env_factory(num_people, num_obstacles, render_mode, use_legs, rank):
    def _init():
        env = GymNavEnv(
            render_mode=render_mode,
            num_rays=LidarConfig.NUM_RAYS,
            num_people=num_people,
            num_obstacles=num_obstacles,
            render_skip=1,
            use_legs=use_legs,
            distraction_prob=0.3,
            real_lidar_specs=False,
            lidar_noise_enable=False,
            stack_dim=3 # DWA doesn't really use stack, but env needs it
        )
        env.reset(seed=rank + int(time.time()))
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="DWA", choices=["DWA", "MPC"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--no_render", action="store_true", default=False)
    parser.add_argument("--num_people", type=int, default=10)
    
    args = parser.parse_args()
    
    render_mode = None if args.no_render else "human"
    
    print(f"--- 📊 Baseline Evaluation: {args.method} ---")
    
    # 1. Initialize Planner
    if args.method == "DWA":
        planner = DWAPlanner(RobotConfig)
    else:
        raise NotImplementedError("MPC not implemented yet.")

    # 2. Env Setup (Single Env for Baselines usually)
    env = GymNavEnv(
        render_mode=render_mode,
        num_rays=LidarConfig.NUM_RAYS,
        num_people=args.num_people,
        num_obstacles=5,
        render_skip=1,
        use_legs=False, # Simplify for baseline first
        real_lidar_specs=False,
        lidar_noise_enable=False,
        stack_dim=3 
    )
    
    # 3. Eval Loop
    obs_dict, _ = env.reset()
    pbar = tqdm(total=args.episodes)
    
    ep_count = 0
    successes = 0
    collisions = 0
    timeouts = 0
    
    try:
        while ep_count < args.episodes:
            # --- PLANNING STEP ---
            # Instead of model.predict(obs), we call planner.plan(obs)
            # We need to pass the raw dict, GymNavEnv returns (obs, info) on reset/step usually
            # But inside loop we handle the tuple.
            
            action = planner.plan(obs_dict)
            
            # --- ENV STEP ---
            # Environment expects normalized action? Or raw? 
            # Your GymNavEnv usually expects raw velocities if you didn't wrap it in normalization
            # but usually RL outputs [-1, 1]. DWA outputs real m/s.
            # Check GymNavEnv:
            # self.v = np.clip(action[0], 0.0, MAX_LINEAR_VEL) 
            # So it expects Real Units. Excellent.
            
            obs_dict, reward, done, truncated, info = env.step(action)
            
            if not args.no_render:
                env.render()
                # time.sleep(0.01)

            if done or truncated:
                ep_count += 1
                pbar.update(1)
                
                reason = info.get("termination_reason", "unknown")
                if reason == "goal_reached": successes += 1
                elif "collision" in reason: collisions += 1
                elif "max_steps" in reason: timeouts += 1
                
                obs_dict, _ = env.reset()
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        
    print("\n--- 📝 Results ---")
    print(f"Success Rate: {successes/ep_count*100:.1f}%")
    print(f"Collision Rate: {collisions/ep_count*100:.1f}%")
    print(f"Timeout Rate: {timeouts/ep_count*100:.1f}%")

if __name__ == "__main__":
    main()