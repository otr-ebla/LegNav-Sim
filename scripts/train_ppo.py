from email import parser
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

from models.custon_cnn import Lidar1DCNN
from models.hybrid_cnn_mlp import HybridCnnMlp   

from typing import Callable
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining va da 1.0 (inizio) a 0.0 (fine)
        return progress_remaining * initial_value
    return func

# --- COSTANTI DEFAULT ---
NUM_RAYS = LidarConfig.NUM_RAYS
N_ENVS = 100 # [CONSIGLIO] Riduci a 64 o 32 se usi la CNN, 100 potrebbe saturare la CPU

class TerminationStatsCallback(BaseCallback):
    """
    Custom callback to log stats and optionally stop training early if success rate is high.
    """
    def __init__(self, verbose=0, window_size=100, stop_at_success_rate=None):
        super().__init__(verbose)
        self.window_size = window_size
        self.stop_at_success_rate = stop_at_success_rate
        self.history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                reason = info.get("termination_reason", "unknown")
                self.history.append(reason)

        if len(self.history) > 0:
            total = len(self.history)
            
            # Counts
            success_count = self.history.count("goal_reached")
            timeout_count = self.history.count("max_steps_reached")
            obstacle_count = self.history.count("collision_static")
            human_count = sum(1 for r in self.history if "people_collision" in r)
            
            # Rates
            success_rate = success_count / total
            timeout_rate = timeout_count / total
            total_collision_rate = (obstacle_count + human_count) / total
            human_coll_rate = human_count / total

            # Logging
            self.logger.record("metrics/success_rate", success_rate)
            self.logger.record("metrics/timeout_rate", timeout_rate)
            self.logger.record("metrics/collision_rate", total_collision_rate)
            self.logger.record("metrics/human_only_rate", human_coll_rate)

            # --- EARLY STOPPING CHECK ---
            # Only check if we have enough data (history full) to be statistically significant
            if self.stop_at_success_rate is not None and len(self.history) >= self.window_size:
                if success_rate >= self.stop_at_success_rate:
                    if self.verbose > 0:
                        print(f"\n🛑 EARLY STOPPING TRIGGERED: Success Rate ({success_rate:.2f}) reached threshold ({self.stop_at_success_rate})!")
                    return False  # returning False stops the training loop

        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="TQC", choices=["SAC", "TQC", "PPO"])
    parser.add_argument("--load_model", type=str, default=None, help="Model to resume from")
    parser.add_argument("--training_name", type=str, required=True, help="Stage Name")
    parser.add_argument("--num_people", type=int, default=0)
    parser.add_argument("--num_obstacles", type=int, default=20)
    parser.add_argument("--rew_progress", type=float, default=5.0, help="Reward factor for progress towards goal")  # [CORREZIONE 1]
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--early_stop", type=float, default=0.95, help="Stop training if success rate reaches this value (e.g., 0.95)")
    parser.add_argument("--use_legs", action="store_true", default=False, help="Enable realistic leg physics for humans") # <--- NUOVO FLAG    
    # [CORREZIONE 2] Usa store_true per i flag booleani
    parser.add_argument("--custom_nn", action="store_true", default=False, help="Use custom NN architecture")
    
    args = parser.parse_args()

    # Create logs directory specifically for this stage
    log_dir = f"./logs/{args.training_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"./checkpoints", exist_ok=True)

    print(f"🛠️  ENV SETUP: {args.num_obstacles} Obstacles | {args.num_people} Humans")
    if args.custom_nn:
        #print("🧠 ARCHITETTURA: Custom 1D-CNN (Lidar Processing)")
        print("🧠 ARCHITETTURA: Hybrid CNN-MLP")
    else:
        print("🧠 ARCHITETTURA: Standard MLP")

    # Define environment factory
    def make_env(rank: int):
        def _init():
            env = GymNavEnv(
                num_rays=NUM_RAYS, 
                num_people=args.num_people, 
                num_obstacles=args.num_obstacles,
                reward_factor_progress=args.rew_progress,  # [CORREZIONE 3]
                use_legs=args.use_legs,  # <--- NUOVO PARAMETRO
            )
            return Monitor(env, os.path.join(log_dir, str(rank)))
        return _init

    # Create Vectorized Environment
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Setup VecNormalize
    if args.load_model:
        vec_path = args.load_model.replace(".zip", "") + "_vecnormalize.pkl"
        if not os.path.exists(vec_path):
             base_name = os.path.splitext(os.path.basename(args.load_model))[0]
             vec_path = f"./checkpoints/{base_name}_vecnormalize.pkl"

        if os.path.exists(vec_path):
            print(f"📥 Loading VecNormalize from: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = True 
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
        # Configurazione Policy Keywords
        policy_kwargs = {}
        if args.custom_nn:
            policy_kwargs = dict(
            features_extractor_class=HybridCnnMlp,
            features_extractor_kwargs=dict(
                num_rays=108,
                stack_dim=3,
                hidden_dim=128,
            ),
        )
                
        print(f"✨ Initializing NEW {args.algo} Agent")
        
        # Inizializzazione unica
        model = ModelClass(
            "MlpPolicy",
            env,
            buffer_size=1_000_000,
            learning_rate=3e-4,
            batch_size=256,
            train_freq=1,
            gradient_steps=4,  # ⬅️ IMPORTANTE
            gamma=0.99,
            tau=0.005,
            top_quantiles_to_drop_per_net=2,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir + "_tb",
            verbose=1,
            device="cuda"
        )

        if args.algo == "SAC":
            model = SAC(
                    "MlpPolicy",
                    env,
                    tensorboard_log=log_dir,
                    learning_rate=3e-4,
                    buffer_size=int(1e6),
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    ent_coef="auto",
                    target_update_interval=1,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

    print(f"🚀 Launching Training: {args.training_name}")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f"./checkpoints/{args.training_name}_ckpts",
        name_prefix="ckpt"
    )
    stats_callback = TerminationStatsCallback(
        window_size=200, 
        stop_at_success_rate=args.early_stop,
        verbose=1 
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            tb_log_name=args.training_name,
            callback=[stats_callback, checkpoint_callback],
            reset_num_timesteps=(args.load_model is None)
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrotto manualmente (CTRL+C).")
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        raise e  # Rilancia l'errore per vedere il traceback
    finally:
        # Controlliamo che il modello esista prima di salvare
        if 'model' in locals() and model is not None:
            save_path = f"./checkpoints/{args.training_name}"
            print(f"💾 Salvataggio di emergenza in corso su: {save_path}")
            model.save(save_path)
            
            # Salviamo anche la normalizzazione se esiste
            if 'env' in locals() and env is not None:
                env.save(f"{save_path}_vecnormalize.pkl")
                try:
                    env.close()
                except Exception:
                    pass # Ignora errori di chiusura processi se già morti
            
            print("✅ Salvataggio completato.")
        else:
            print("⚠️ Nessun modello da salvare (interruzione avvenuta prima dell'inizializzazione).")

if __name__ == "__main__":
    main()