import time
import numpy as np

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Ensure these imports point to your actual file structure
from src.envs.gym_nav_env import GymNavEnv, NUM_PEOPLE
from .train_ppo import NUM_RAYS, PEOPLE_SPEED   

training_name = "10M_nocollisions"

def main():
    # Setup environment
    env_fn = lambda: GymNavEnv(render_mode="human", num_rays=NUM_RAYS,
                            num_people=NUM_PEOPLE, people_speed=PEOPLE_SPEED)
    env = DummyVecEnv([env_fn])

    # Load Normalization and Model
    try:
        env = VecNormalize.load(training_name + "_vecnormalize.pkl", env)
        env.training = False 
        env.norm_reward = False
    except Exception as e:
        print(f"Warning: Could not load VecNormalize: {e}")

    print(f"Loading model: ./checkpoints/{training_name}")
    model = SAC.load("./checkpoints/" + training_name, env=env)

    obs = env.reset()
    
    # --- STATISTICS VARIABLES ---
    total_episodes = 0
    success_count = 0
    timeout_count = 0
    collision_count = 0 
    
    coll_people = 0
    coll_obstacle = 0

    # Accumulators for averages
    total_path_length = 0.0
    total_time_to_goal = 0.0 # Only for successful episodes
    total_jerk = 0.0
    total_spl = 0.0

    print("\n--- Starting Inference ---\n")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        print("Taking action:", action)  # Debug
        print
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            inf = info[0] # info is a list of dicts in VecEnv
            reason = inf.get("termination_reason", "unknown")
            total_episodes += 1

            # Extract new metrics from environment
            ep_path_len = inf.get("path_length", 0.0)
            ep_time = inf.get("total_time", 0.0)
            ep_jerk = inf.get("mean_jerk", 0.0)
            opt_len = inf.get("optimal_length", 1.0) # avoid div/0

            # Update General Stats
            is_success = 0
            if reason == "goal_reached":
                success_count += 1
                is_success = 1
                total_time_to_goal += ep_time
            elif reason == "max_steps_reached":
                timeout_count += 1
            elif reason in ["people_collision", "obstacle_collision", "wall_collision"]:
                collision_count += 1
                if reason == "people_collision":
                    coll_people += 1
                else:
                    coll_obstacle += 1
            
            # Update Averages
            total_path_length += ep_path_len
            total_jerk += ep_jerk

            # Calculate SPL (Success weighted by Path Length) for this episode
            # SPL = Success * (Optimal_Path / max(Actual_Path, Optimal_Path))
            current_spl = is_success * (opt_len / max(ep_path_len, opt_len))
            total_spl += current_spl

            # Compute Rates
            sr = (success_count / total_episodes) * 100
            tr = (timeout_count / total_episodes) * 100
            cr = (collision_count / total_episodes) * 100
            
            # Compute Means
            avg_path = total_path_length / total_episodes
            avg_jerk = total_jerk / total_episodes
            avg_spl = total_spl / total_episodes
            avg_time = total_time_to_goal / success_count if success_count > 0 else 0.0

            # Print Table
            print("-" * 60)
            print(f"EPISODE {total_episodes} ENDED: {reason.upper()}")
            print(f"Outcomes:")
            print(f"  > Success Rate:   {sr:.1f}%  ({success_count}/{total_episodes})")
            print(f"  > Collision Rate: {cr:.1f}%  ({collision_count}/{total_episodes}) [Ppl: {coll_people}, Obs: {coll_obstacle}]")
            print(f"  > Timeout Rate:   {tr:.1f}%  ({timeout_count}/{total_episodes})")
            print(f"Metrics (Avg):")
            print(f"  > SPL:            {avg_spl:.3f}  (0.0 = Fail, 1.0 = Optimal)")
            print(f"  > Time to Goal:   {avg_time:.2f} s")
            print(f"  > Path Length:    {avg_path:.2f} m")
            print(f"  > Ang. Jerk:      {avg_jerk:.2f} rad/s³")
            print("-" * 60)
            print( )

            time.sleep(0.5)
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()