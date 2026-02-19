import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
import argparse
import os
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from collections import deque
import shutil  # <--- Fondamentale per pulire

# Importa l'environment e il wrapper
from src.envs.gym_nav_env import GymNavEnv       
from src.envs.vec_lidar_stack import VecTemporalStack as VecLidarStack 
from src.config import RobotConfig, LidarConfig

# Import Custom Algorithm
from models.adaptive_tqc import AdaptiveTQC
from models.custom_cnn import EndToEndNavExtractor 

# Costanti
NUM_RAYS = LidarConfig.NUM_RAYS
DEFAULT_N_ENVS = 64 

class TerminationStatsCallback(BaseCallback):
    """
    Callback che calcola statistiche e salva il BEST model.
    """
    def __init__(self, verbose=0, window_size=100, best_model_save_path=None):
        super().__init__(verbose)
        self.window_size = window_size
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -float('inf') 
        self.reward_history = deque(maxlen=window_size)
        self.history = deque(maxlen=window_size)
        self.global_ep_count = 0
        self.global_success = 0
        self.global_collision = 0 
        self.global_passive = 0
        self.global_timeout = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.reward_history.append(ep_reward)

                reason = info.get("termination_reason", "unknown")
                self.history.append(reason)
                self.global_ep_count += 1
                
                if reason == "goal_reached": self.global_success += 1
                elif reason in ["collision_static", "people_collision_active"]: self.global_collision += 1
                elif reason == "people_collision_passive": self.global_passive += 1
                elif reason == "max_steps_reached": self.global_timeout += 1

        if self.global_ep_count > 0 and len(self.history) > 0:
            sr_total = self.global_success / self.global_ep_count
            cr_total = self.global_collision / self.global_ep_count
            pr_total = self.global_passive / self.global_ep_count
            tr_total = self.global_timeout / self.global_ep_count
            
            window_len = len(self.history)
            win_success = self.history.count("goal_reached")
            win_coll = self.history.count("collision_static") + self.history.count("people_collision_active")
            win_passive = self.history.count("people_collision_passive")
            win_timeout = self.history.count("max_steps_reached")
            
            sr_window = win_success / window_len
            cr_window = win_coll / window_len
            pr_window = win_passive / window_len
            tr_window = win_timeout / window_len
            mean_reward = np.mean(self.reward_history) if len(self.reward_history) > 0 else -100.0
            
            self.logger.record("metrics_total/success_rate", sr_total)
            self.logger.record("metrics_total/collision_rate", cr_total)
            self.logger.record("metrics_total/passive_rate", pr_total)
            self.logger.record("metrics_total/timeout_rate", tr_total)
            self.logger.record("metrics_window/success_rate", sr_window)
            self.logger.record("metrics_window/collision_rate", cr_window)
            self.logger.record("metrics_window/passive_rate", pr_window)
            self.logger.record("metrics_window/timeout_rate", tr_window)
            self.logger.record("metrics_window/mean_reward", mean_reward)

            if self.best_model_save_path and len(self.reward_history) >= self.window_size:
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0: 
                        print(f"\n🔥 NEW BEST MODEL! Reward: {mean_reward:.2f} | Window SR: {sr_window:.2f}")
                    self.best_mean_reward = mean_reward
                    self.model.save(self.best_model_save_path)
                    if self.training_env: 
                        self.training_env.save(f"{self.best_model_save_path}_vecnormalize.pkl")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="AR_TQC", choices=["SAC", "TQC", "PPO", "AR_TQC"])
    parser.add_argument("--training_name", type=str, required=True)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5_000_000)
    
    parser.add_argument("--num_people", type=int, default=6)
    parser.add_argument("--num_obstacles", type=int, default=12)
    parser.add_argument("--use_legs", action="store_true", default=False)
    parser.add_argument("--real_lidar", action="store_true", default=False)
    parser.add_argument("--lidar_noise", action="store_true", default=False)
    parser.add_argument("--distraction_prob", type=float, default=0.3)
    
    parser.add_argument("--custom_nn", action="store_true", default=False)
    parser.add_argument("--fast_mlp", action="store_true", default=False)

    parser.add_argument("--n_envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()

    # --- MODIFICA 1: Percorso Log Temporaneo ---
    # Creiamo comunque la cartella (alcuni logger ne hanno bisogno), ma la svuoteremo alla fine.
    log_dir_temp = f"./logs/{args.training_name}_monitor_temp"
    os.makedirs(log_dir_temp, exist_ok=True)
    
    os.makedirs(f"./checkpoints", exist_ok=True)
    # Tensorboard log separato (questo lo vogliamo tenere!)
    tb_log_dir = f"./logs/{args.training_name}_tb"

    arch_name = "Default SB3"
    if args.custom_nn: arch_name = "Hybrid CNN 1D"
    if args.fast_mlp: arch_name = "Fast Flat MLP"

    print(f"🚀 TRAINING: {args.algo} | {args.n_envs} Envs | Arch: {arch_name}")

    final_path_zip = f"./checkpoints/{args.training_name}.zip"
    final_path_vec = f"./checkpoints/{args.training_name}_vecnormalize.pkl"
    
    best_path_base = f"./checkpoints/{args.training_name}_BEST"
    best_path_zip = f"{best_path_base}.zip"
    best_path_vec = f"{best_path_base}_vecnormalize.pkl"

    # 1. Env Factory
    def make_env(rank: int):
        def _init():
            env = GymNavEnv(
                num_rays=NUM_RAYS, 
                num_people=args.num_people, 
                num_obstacles=args.num_obstacles,
                use_legs=args.use_legs,
                distraction_prob=args.distraction_prob,
                render_skip=1,
                lidar_noise_enable=args.lidar_noise,
                real_lidar_specs=args.real_lidar,
                stack_dim=5 if args.real_lidar else 3 
            )
            # --- MODIFICA 2: filename=None ---
            # Monitor calcola le statistiche (Reward, EpLen) MA NON SCRIVE FILE SU DISCO.
            return Monitor(env, filename=None)
        return _init

    # 2. Vec Env + Stacking + Norm
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    
    stack_dim_actual = 5 if args.real_lidar else 3
    env = VecLidarStack(env, stack_dim=stack_dim_actual)
    
    if args.load_model:
        vec_path = args.load_model.replace(".zip", "") + "_vecnormalize.pkl"
        if os.path.exists(vec_path):
            print(f"📥 Loaded VecNormalize: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # 3. Model Setup
    if args.algo == "AR_TQC": ModelClass = AdaptiveTQC
    elif args.algo == "TQC": ModelClass = TQC
    elif args.algo == "PPO": ModelClass = PPO
    elif args.algo == "SAC": ModelClass = SAC
    
    policy_kwargs = {}
    if args.custom_nn:
        print("🧠 Using Custom NN model")
        policy_kwargs = dict(
            features_extractor_class=EndToEndNavExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], qf=[128, 128]) if args.algo != "PPO" else [128, 128]
        )
    else:
        print("🧠 Using Default SB3 MLP")
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]) if args.algo != "PPO" else [256, 256]
        )

    model = ModelClass(
        "MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=2048, 
        gamma=0.99,
        tau=0.005,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        verbose=1,
        device=args.device
    )

    if args.load_model:
        print(f"📥 Loading Model Weights: {args.load_model}")
        try:
            old_model = ModelClass.load(args.load_model, device=args.device)
            model.policy.load_state_dict(old_model.policy.state_dict())
            if hasattr(model, "critic"): 
                model.critic.load_state_dict(old_model.critic.state_dict())
                model.critic_target.load_state_dict(old_model.critic_target.state_dict())
            print("✅ Weights Loaded.")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f"./checkpoints/{args.training_name}_ckpts", name_prefix="ckpt")
    
    stats_callback = TerminationStatsCallback(
        best_model_save_path=best_path_base,
        window_size=100
    )

    try:
        model.learn(total_timesteps=args.steps, tb_log_name=args.training_name, callback=[stats_callback, checkpoint_callback])
    except KeyboardInterrupt:
        print("\n🛑 Training interrotto manualmente (CTRL+C).")
    finally:
        env.close()
        print("\n💾 Finalizing Model...")

        # Gestione BEST Model
        if os.path.exists(best_path_zip):
            print(f"🏆 Found BEST model! Renaming '{best_path_zip}' to '{final_path_zip}'")
            shutil.move(best_path_zip, final_path_zip)
            if os.path.exists(best_path_vec):
                shutil.move(best_path_vec, final_path_vec)
        else:
            print(f"⚠️ Nessun modello BEST trovato. Salvataggio stato corrente.")
            model.save(final_path_zip)
            if hasattr(env, "save"):
                env.save(final_path_vec)

        # --- MODIFICA 3: PULIZIA FINALE ---
        # Rimuove la cartella temporanea dei monitor (anche se vuota)
        if os.path.exists(log_dir_temp):
            try:
                shutil.rmtree(log_dir_temp)
                print(f"🧹 Cartella temporanea monitor rimossa: {log_dir_temp}")
            except Exception as e:
                print(f"⚠️ Impossibile rimuovere cartella temp: {e}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()