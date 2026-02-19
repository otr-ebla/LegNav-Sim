import numpy as np
import gymnasium as gym
import multiprocessing
import time
import itertools
from tqdm import tqdm
import pandas as pd

# Project imports
from src.envs.gym_nav_env import GymNavEnv
from baselines.dwa_planner import DWAPlanner
from src.config import SimConfig, RobotConfig

# ==============================================================================
# 🎯 REFINED SEARCH SPACE (FAST & FOCUSED)
# ==============================================================================
PARAM_GRID = {
    'to_goal_cost_gain': [0.5, 1.5, 3.0],    # Focus on low to mid sensitivity
    'speed_cost_gain':   [1.0, 10.0, 25.0],  # Test slow vs aggressive
    'obstacle_cost_gain':[1.5, 4.0, 8.0],    # Safety levels
    'predict_time':      [1.0, 1.8],         # Dynamic horizon
    'dt':                [0.1, 0.2]          # Frequency
}

N_TEST_EPISODES = 20 # Balanced for statistical significance and speed
NUM_WORKERS = 32      # Matches your CPU architecture

def evaluate_config(params):
    """
    Core simulation loop optimized for raw throughput.
    """
    # Create environment (Hard-disable any UI overhead)
    env = GymNavEnv(
        render_mode=None, 
        max_steps=SimConfig.MAX_STEPS, 
        num_obstacles=SimConfig.NUM_OBSTACLES,
        num_people=SimConfig.NUM_HUMANS,
        lidar_noise_enable=False 
    )
    
    # Initialize Planner with correct Robot limits
    planner = DWAPlanner(RobotConfig)
    
    # Inject parameters
    for key, value in params.items():
        setattr(planner, key, value)
    
    success_count = 0
    total_steps = 0
    
    for _ in range(N_TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_steps = 0
        
        while not (done or truncated):
            action = planner.plan(obs)
            obs, _, done, truncated, info = env.step(action)
            episode_steps += 1
            
        if info.get("termination_reason") == "goal_reached":
            success_count += 1
            total_steps += episode_steps
            
    env.close()
    
    return {
        **params,
        'success_rate': success_count / N_TEST_EPISODES,
        'avg_steps': total_steps / success_count if success_count > 0 else 1000.0
    }

def main():
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🚀 Running optimized sweep: {len(combinations)} configurations.")
    start_time = time.time()

    # Using imap_unordered with a chunksize for better 32-core scaling
    results = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(evaluate_config, combinations, chunksize=2), total=len(combinations)):
            results.append(res)

    # Save and Analyze
    df = pd.DataFrame(results).sort_values(by=['success_rate', 'avg_steps'], ascending=[False, True])
    df.to_csv("fast_dwa_results.csv", index=False)
    
    print(f"\n✅ Done in {(time.time() - start_time)/60:.2f} minutes.")
    print("\n🏆 BEST PARAMETERS FOUND:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()