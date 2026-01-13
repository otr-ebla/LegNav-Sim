import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
import argparse
import os
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from src.envs.gym_nav_env import GymNavEnv
from src.config import RobotConfig, LidarConfig

# --- COSTANTI DEFAULT (usate solo se non sovrascritte da args) ---
NUM_RAYS = LidarConfig.NUM_RAYS
TRAINING_STEPS = 15_000_000
N_ENVS = 100 # Riduci se la CPU soffre (es. 16-32)

class RatesCallback(BaseCallback):
    def __init__(self, verbose=1, window_size=100):
        super().__init__(verbose)
        self.history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # 'infos' is a list of info dicts from the vectorized environment
        for info in self.locals['infos']:
            # We only care if the episode actually ended
            if "termination_reason" in info:
                self.history.append(info["termination_reason"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.history) == 0: return
        
        # Calculate percentages based on the history buffer
        n = len(self.history)
        success_rate = sum(1 for r in self.history if r == "goal_reached") / n
        col_rate = sum(1 for r in self.history if "collision" in r) / n
        timeout_rate = sum(1 for r in self.history if r == "max_steps_reached" or r == "stuck") / n

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/collision_rate", col_rate)
        self.logger.record("rollout/timeout_rate", timeout_rate)

class TerminationStatsCallback(BaseCallback):
    def __init__(self, verbose=0, training_name="default"):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir="./logs/" + training_name)
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
                    self.writer.add_scalar("metrics/stuck_rate", self.stuck/total, self.episode_id)
                self.episode_id += 1
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="TQC", choices=["SAC", "TQC", "PPO"])
    parser.add_argument("--load_model", type=str, default=None, help="Path .zip da caricare per resume")
    parser.add_argument("--training_name", type=str, required=True, help="Nome del nuovo training")
    parser.add_argument("--num_people", type=int, default=15, help="Num persone ambiente")
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS, help="Steps totali")
    args = parser.parse_args()

    # 1. SETUP AMBIENTE (Prima di tutto!)
    print(f"🛠️ Creating {N_ENVS} environments with {args.num_people} people...")
    
    def make_env(rank: int):
        def _init():
            return Monitor(GymNavEnv(num_rays=NUM_RAYS, num_people=args.num_people))
        return _init

    # Usa SubprocVecEnv per parallelizzazione reale
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # 2. SETUP VECNORMALIZE
    if args.load_model:
        # Cerca il file .pkl associato al modello caricato
        vec_path = args.load_model.replace(".zip", "") + "_vecnormalize.pkl"
        # Se non lo trova lì, prova nella root con nome standard
        if not os.path.exists(vec_path):
             vec_path = args.load_model.split("/")[-1].replace(".zip", "") + "_vecnormalize.pkl"

        if os.path.exists(vec_path):
            print(f"📥 Loading VecNormalize from: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = True # IMPORTANTE: Continua ad aggiornare le statistiche!
            env.norm_reward = False # TQC/SAC spesso preferiscono reward raw
        else:
            print(f"⚠️ VecNormalize pkl not found at {vec_path}. Creating NEW one.")
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    else:
        print("🆕 Creating NEW VecNormalize (Training from Scratch).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 3. SETUP MODELLO
    ModelClass = TQC if args.algo == "TQC" else SAC
    
    if args.load_model:
        print(f"🧠 Loading Agent from: {args.load_model}")
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
            tensorboard_log="./logs/" + args.training_name,
            buffer_size=1_000_000, 
            learning_rate=3e-4, 
            batch_size=256
        )

    # 4. AVVIO TRAINING
    print(f"🚀 Starting training: {args.training_name}")
    callback = TerminationStatsCallback(training_name=args.training_name)

    rates_callback = RatesCallback()

    callbacks = [callback, rates_callback]

    try:
        model.learn(
            total_timesteps=args.steps,
            tb_log_name=args.training_name,
            callback=callbacks,
            reset_num_timesteps=(args.load_model is None) # Reset step solo se è nuovo training
        )
    except KeyboardInterrupt:
        print("🛑 Training interrupted manually.")

    # 5. SALVATAGGIO FINALE
    save_path = f"./checkpoints/{args.training_name}"
    print(f"💾 Saving final model to {save_path}")
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    print("✅ Done.")
    env.close()

if __name__ == "__main__":
    main()