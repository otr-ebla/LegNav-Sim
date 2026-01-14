import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
import argparse
import os
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Importa il tuo environment
from src.envs.gym_nav_env import GymNavEnv
from src.config import RobotConfig, LidarConfig

# --- COSTANTI DEFAULT ---
NUM_RAYS = LidarConfig.NUM_RAYS
N_ENVS = 100 # Adjust based on CPU

class TerminationStatsCallback(BaseCallback):
    """Logga metriche custom su Tensorboard"""
    def __init__(self, verbose=0, training_name="default"):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir="./logs/" + training_name + "_tb")
        self.success = 0; self.timeout = 0; self.obstacle = 0; self.human = 0; self.stuck = 0
        self.episode_id = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                reason = info.get("termination_reason", "unknown")
                if reason == "goal_reached": self.success += 1
                elif reason == "max_steps_reached": self.timeout += 1
                elif reason == "collision_static": self.obstacle += 1
                elif reason == "people_collision": self.human += 1
                elif reason == "stuck": self.stuck += 1
                
                total = self.success + self.timeout + self.obstacle + self.human + self.stuck
                if total > 0:
                    self.writer.add_scalar("metrics/success_rate", self.success/total, self.episode_id)
                    self.writer.add_scalar("metrics/human_collision_rate", self.human/total, self.episode_id)
                self.episode_id += 1
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="TQC", choices=["SAC", "TQC", "PPO"])
    parser.add_argument("--load_model", type=str, default=None, help="Model to resume from")
    parser.add_argument("--training_name", type=str, required=True, help="Stage Name")
    parser.add_argument("--num_people", type=int, default=0)
    parser.add_argument("--num_obstacles", type=int, default=0) # <--- NEW ARGUMENT
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()

    # Create logs directory specifically for this stage (for Monitor CSVs)
    log_dir = f"./logs/{args.training_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"./checkpoints", exist_ok=True)

    print(f"🛠️  ENV SETUP: {args.num_obstacles} Obstacles | {args.num_people} Humans")
    
    # Define environment factory
    def make_env(rank: int):
        def _init():
            # Pass num_obstacles to GymNavEnv
            env = GymNavEnv(
                num_rays=NUM_RAYS, 
                num_people=args.num_people, 
                num_obstacles=args.num_obstacles # <--- Pass argument
            )
            # Setup Monitor to save CSV in the correct folder
            return Monitor(env, os.path.join(log_dir, str(rank)))
        return _init

    # Create Vectorized Environment
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Setup VecNormalize
    if args.load_model:
        # Try to find corresponding vecnorm
        vec_path = args.load_model.replace(".zip", "") + "_vecnormalize.pkl"
        if not os.path.exists(vec_path):
             # Fallback logic
             base_name = os.path.splitext(os.path.basename(args.load_model))[0]
             vec_path = f"./checkpoints/{base_name}_vecnormalize.pkl"

        if os.path.exists(vec_path):
            print(f"📥 Loading VecNormalize from: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = True # Continue updating stats!
            env.norm_reward = False
        else:
            print("⚠️ VecNormalize not found. Creating NEW one.")
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    else:
        print("🆕 Creating NEW VecNormalize.")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Setup Model
    ModelClass = TQC if args.algo == "TQC" else SAC
    
    if args.load_model:
        print(f"🧠 Loading Weights from: {args.load_model}")
        custom_objects = {
            "learning_rate": 3e-4, 
            "lr_schedule": lambda _: 3e-4,
            "clip_range": lambda _: 0.2
        }
        model = ModelClass.load(args.load_model, env=env, custom_objects=custom_objects)
    else:
        print(f"✨ Initializing NEW {args.algo} Agent")
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir + "_tb",
            buffer_size=1_000_000, 
            learning_rate=3e-4, 
            batch_size=256
        )

    print(f"🚀 Launching Training: {args.training_name}")
    
    # Save checkpoint periodically (safety net)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f"./checkpoints/{args.training_name}_ckpts",
        name_prefix="ckpt"
    )
    stats_callback = TerminationStatsCallback(training_name=args.training_name)

    try:
        model.learn(
            total_timesteps=args.steps,
            tb_log_name=args.training_name,
            callback=[stats_callback, checkpoint_callback],
            reset_num_timesteps=(args.load_model is None)
        )
    except KeyboardInterrupt:
        print("🛑 Interrupted by Curriculum Master (or User).")
    finally:
        # ALWAYS SAVE ON EXIT
        save_path = f"./checkpoints/{args.training_name}"
        print(f"💾 Saving Final Model to {save_path}")
        model.save(save_path)
        env.save(f"{save_path}_vecnormalize.pkl")
        env.close()

if __name__ == "__main__":
    main()