import argparse
import os
import gymnasium as gym
import torch
import numpy as np
import random

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Import custom environment
# Run from root with: python -m scripts.TrainMaster_Prime ...
from src.envs.jhsfm_nav_env import SimpleNavEnv

"""
COMMAND FOR CURRICULUM STAGE 2 (Mixed Scenarios + Load Pre-trained):
python3 -m scripts.TrainMaster_Prime \
    --name Curriculum_Stage2_Mixed \
    --scenario mixed \
    --algo tqc \
    --steps 1000000 \
    --n_envs 16 \
    --use_legs \
    --load_model checkpoints/Curriculum_Stage1_MixedStatic/TrainMaster_Final.zip \
    --load_vecnorm checkpoints/Curriculum_Stage1_MixedStatic/TrainMaster_VecNorm.pkl

COMMAND FOR EVALUATION:  
python3 -m scripts.TrainMaster_Prime \
    --eval --render \
    --name Eval_Stage2 \
    --scenario mixed \
    --load_model checkpoints/Curriculum_Stage2_Mixed/TrainMaster_Final.zip \
    --load_vecnorm checkpoints/Curriculum_Stage2_Mixed/TrainMaster_VecNorm.pkl \
    --use_legs
"""

class TrainMasterMetrics(BaseCallback):
    """Callback to monitor TrainMaster_Prime performance during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.stats = {"success": 0, "collision": 0, "timeout": 0}
        self.total = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for done, info in zip(self.locals.get("dones", []), infos):
            if done:
                self.total += 1
                reason = info.get("termination_reason", "none")
                if reason == "goal_reached": self.stats["success"] += 1
                elif "collision" in reason: self.stats["collision"] += 1
                elif reason == "timeout": self.stats["timeout"] += 1
                
                if self.total % 20 == 0:
                    for k, v in self.stats.items():
                        self.logger.record(f"train_master/{k}_rate", v / self.total)
        return True

def make_env(rank, seed, scenario, num_people, use_legs, is_training=True, render_mode=None, force_static=False, render_skip=1):
    def _init():
        env = SimpleNavEnv(
            scenario_type=scenario, 
            num_people=num_people, 
            allow_keyboard_skip=True,
            training=is_training,
            use_legs=use_legs,
            render_mode=render_mode,
            force_static=force_static,
            render_skip=render_skip
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def evaluate_master(model, env, num_episodes, render=False, name="eval"):
    """
    Executes evaluation of NavigatorPrime.
    """
    print(f"\n--- 🧐 EVALUATION TrainMaster_Prime ({num_episodes} episodes) ---")
    if render:
        print("📺 Rendering ACTIVE (Press Ctrl+C in terminal to stop)")

    stats = {"SR": 0, "CR": 0, "TR": 0, "Time": []}
    episodes_finished = 0
    
    obs = env.reset()
    steps_per_env = np.zeros(env.num_envs)

    try:
        while episodes_finished < num_episodes:
            action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)

            if hasattr(env.envs[0].unwrapped, 'manual_skip_triggered') and env.envs[0].unwrapped.manual_skip_triggered:
                obs = env.reset()

            steps_per_env += 1
            
            if render:
                env.render()

            for i, done in enumerate(dones):
                if done:
                    if episodes_finished < num_episodes:
                        episodes_finished += 1
                        res = infos[i].get("termination_reason", "none")
                        
                        if res == "goal_reached":
                            stats["SR"] += 1
                            stats["Time"].append(steps_per_env[i] * 0.1) 
                        elif "collision" in res:
                            stats["CR"] += 1
                        else:
                            stats["TR"] += 1
                        
                        if episodes_finished % 10 == 0:
                            print(f"   -> Episodes completed: {episodes_finished}/{num_episodes}")

                    steps_per_env[i] = 0
                    
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted manually.")

    if episodes_finished > 0:
        print(f"\n--- 📊 EVALUATION REPORT ({episodes_finished} EPISODES) ---")
        print(f"Success Rate (SR): {stats['SR']/episodes_finished:.2%}")
        print(f"Collision Rate (CR): {stats['CR']/episodes_finished:.2%}")
        if stats['Time']: 
            print(f"Mean Time: {np.mean(stats['Time']):.2f}s")
        else:
            print("Mean Time: N/A")
    else:
        print("\n⚠️ No episodes completed.")
    print("-------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="TrainMaster_Prime: Advanced RL Training Suite")
    
    # Training Params
    parser.add_argument("--algo", type=str, default="tqc", choices=["ppo", "sac", "tqc"], help="RL Algorithm")
    parser.add_argument("--steps", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    
    # Environment Params
    parser.add_argument("--scenario", type=str, default="static_groups", help="Scenario: static_groups, random, mixed, etc.")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--use_legs", action="store_true", help="Enable legs simulation in Lidar")
    parser.add_argument("--num_people", type=int, default=None, help="Override number of people")
    parser.add_argument("--force_static", action="store_true", help="Freeze humans (Stage 1)")

    # Load / Resume Params
    parser.add_argument("--load_model", type=str, default=None, help="Path to .zip model to load")
    parser.add_argument("--load_vecnorm", type=str, default=None, help="Path to .pkl stats to load")
    parser.add_argument("--continue_training", action="store_true", help="If set, continues tensorboard step counter. If not, resets steps to 0 (New Stage).")
    
    # Evaluation / Render Params
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Visual render (forces 1 env)")
    parser.add_argument("--render_skip", type=int, default=1, help="Render skip frames")

    args = parser.parse_args()

    base_save_path = f"./checkpoints/{args.name}"
    log_path = f"./logs/{args.name}"
    os.makedirs(base_save_path, exist_ok=True)

    print(f"🛠️  TrainMaster_Prime Started: {args.algo.upper()} | Mission: {args.name}")
    print(f"🦵 Legs Mode: {'ACTIVE' if args.use_legs else 'INACTIVE'}")
    
    if args.scenario == "mixed":
        print(f"🔄 Scenario MIXED: Randomization active (including static groups)!")

    is_training_env = not args.eval
    random_seed_number = random.randint(0, 100)

    # 1. Environment Setup
    if args.render:
        print("📺 Visual Mode: Forcing single environment...")
        n_envs = 1
        env = DummyVecEnv([
            make_env(0, random_seed_number, args.scenario, args.num_people, args.use_legs, 
                     is_training=is_training_env, 
                     render_mode="human", 
                     force_static=args.force_static, 
                     render_skip=args.render_skip)
        ])
    else:
        n_envs = args.n_envs
        env = SubprocVecEnv([
            make_env(i, random_seed_number, args.scenario, args.num_people, args.use_legs, 
                     is_training=is_training_env,
                     force_static=args.force_static) 
            for i in range(n_envs)
        ])
    
    # 2. VecNormalize Handling
    if args.load_vecnorm:
        print(f"📥 Loading VecNormalize from {args.load_vecnorm}...")
        env = VecNormalize.load(args.load_vecnorm, env)
        
        if args.eval or args.render:
            print("   -> Mode: EVAL (Stats frozen)")
            env.training = False 
            env.norm_reward = False 
        else:
            print("   -> Mode: TRAINING (Stats will continue to update)")
            env.training = True 
            env.norm_reward = True
    else:
        if args.eval:
            print("⚠️ WARNING: Evaluation without loading VecNormalize! Performance might be poor.")
        print("🆕 Creating new VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 3. Model Setup
    model_cls = {"ppo": PPO, "sac": SAC, "tqc": TQC}[args.algo]
    policy_kwargs = dict(net_arch=[256, 256])

    if args.load_model:
        print(f"🧠 Loading Brain from: {args.load_model}")
        
        # When moving to Stage 2, we usually reset the Learning Rate to the default
        custom_objects = {"learning_rate": 3e-4} 
        
        model = model_cls.load(
            args.load_model, 
            env=env, 
            tensorboard_log=log_path,
            custom_objects=custom_objects
        )
    else:
        print(f"✨ Initializing NEW {args.algo.upper()} model...")
        use_sde = (args.algo != "ppo")
        if args.algo == "ppo":
             model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
        else:
             model = model_cls(
                 "MlpPolicy", env, verbose=1, tensorboard_log=log_path, 
                 use_sde=use_sde, policy_kwargs=policy_kwargs
             )

    # 4. Execution
    if args.eval:
        evaluate_master(model, env, args.eval_episodes, render=args.render, name=args.name)
    else:
        callbacks = [TrainMasterMetrics(), CheckpointCallback(save_freq=50000, save_path=base_save_path, name_prefix="tm_ckpt")]
        
        # Determine whether to reset timesteps (New Stage) or continue (Resume)
        reset_timesteps = True
        if args.load_model is not None and args.continue_training:
            reset_timesteps = False
            print("⏩ Resuming training (Timesteps continued)...")
        else:
            print("🆕 Starting NEW Training Stage (Timesteps reset to 0)...")

        try:
            model.learn(
                total_timesteps=args.steps, 
                callback=callbacks, 
                reset_num_timesteps=reset_timesteps
            )
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted manually.")

        # 5. Save
        print(f"💾 Saving model to {base_save_path}...")
        model.save(f"{base_save_path}/TrainMaster_Final")
        env.save(f"{base_save_path}/TrainMaster_VecNorm.pkl")
        print("✅ Done.")

    env.close()

if __name__ == "__main__":
    main()