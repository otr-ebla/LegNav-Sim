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
import shutil

# Importa l'environment e il wrapper
from src.envs.gym_nav_env import GymNavEnv       
from src.envs.vec_lidar_stack import VecTemporalStack as VecLidarStack 
from src.config import RobotConfig, LidarConfig

# Import Custom Algorithm (se usi AR_TQC)
from models.adaptive_tqc import AdaptiveTQC

# Import Custom Neural Networks (CNN 1D e Fast MLP)
from models.custom_cnn import EndToEndNavExtractor 
#from models.fast_mlp import FastFlatMlp

# Costanti
NUM_RAYS = LidarConfig.NUM_RAYS
DEFAULT_N_ENVS = 64 

class TerminationStatsCallback(BaseCallback):
    """
    Callback personalizzato Completo.
    - Calcola metriche WINDOW (ultimi N episodi) per vedere il trend attuale.
    - Calcola metriche TOTAL (dall'inizio) per vedere la storia.
    - Salva il BEST MODEL basandosi sulla REWARD MEDIA (Window).
    """
    def __init__(self, verbose=0, window_size=100, best_model_save_path=None):
        super().__init__(verbose)
        self.window_size = window_size
        self.best_model_save_path = best_model_save_path
        
        # Criterio di salvataggio: Reward
        self.best_mean_reward = -float('inf') 
        self.reward_history = deque(maxlen=window_size)

        # History per i Rate (Termination Reasons) - Finestra Mobile
        self.history = deque(maxlen=window_size)

        # Contatori Globali (per i Rate totali)
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
                
                # --- 1. Rilevamento Reward (fornito da Monitor) ---
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.reward_history.append(ep_reward)

                # --- 2. Rilevamento Termination Reason ---
                reason = info.get("termination_reason", "unknown")
                
                # Aggiorna finestra mobile
                self.history.append(reason)
                
                # Aggiorna totali globali
                self.global_ep_count += 1
                
                if reason == "goal_reached":
                    self.global_success += 1
                elif reason in ["collision_static", "people_collision_active"]:
                    self.global_collision += 1
                elif reason == "people_collision_passive":
                    self.global_passive += 1
                elif reason == "max_steps_reached":
                    self.global_timeout += 1

        # Logging periodico
        if self.global_ep_count > 0 and len(self.history) > 0:
            
            # --- A. Calcolo Metriche TOTALI (Storiche) ---
            sr_total = self.global_success / self.global_ep_count
            cr_total = self.global_collision / self.global_ep_count
            pr_total = self.global_passive / self.global_ep_count
            tr_total = self.global_timeout / self.global_ep_count
            
            # --- B. Calcolo Metriche WINDOW (Dinamiche - Ultime 100) ---
            window_len = len(self.history)
            
            # Conta occorrenze nella finestra
            win_success = self.history.count("goal_reached")
            win_coll = self.history.count("collision_static") + self.history.count("people_collision_active")
            win_passive = self.history.count("people_collision_passive")
            win_timeout = self.history.count("max_steps_reached")
            
            sr_window = win_success / window_len
            cr_window = win_coll / window_len
            pr_window = win_passive / window_len
            tr_window = win_timeout / window_len
            
            # Media Reward (Window)
            mean_reward = np.mean(self.reward_history) if len(self.reward_history) > 0 else -100.0
            
            # --- Log su Tensorboard ---
            # 1. Totali
            self.logger.record("metrics_total/success_rate", sr_total)
            self.logger.record("metrics_total/collision_rate", cr_total)
            self.logger.record("metrics_total/passive_rate", pr_total)
            self.logger.record("metrics_total/timeout_rate", tr_total)
            
            # 2. Window (Dinamici - Più utili per il debug immediato)
            self.logger.record("metrics_window/success_rate", sr_window)
            self.logger.record("metrics_window/collision_rate", cr_window)
            self.logger.record("metrics_window/passive_rate", pr_window)
            self.logger.record("metrics_window/timeout_rate", tr_window)
            self.logger.record("metrics_window/mean_reward", mean_reward)

            # --- SALVATAGGIO BEST MODEL (Basato su REWARD) ---
            if self.best_model_save_path and len(self.reward_history) >= self.window_size:
                
                # Se la reward media attuale è migliore del record precedente
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
    
    # Env Config
    parser.add_argument("--num_people", type=int, default=6)
    parser.add_argument("--num_obstacles", type=int, default=12)
    parser.add_argument("--use_legs", action="store_true", default=True)
    parser.add_argument("--real_lidar", action="store_true", default=True)
    parser.add_argument("--lidar_noise", action="store_true", default=True)
    parser.add_argument("--distraction_prob", type=float, default=0.3)
    
    # Scelta Rete Neurale
    parser.add_argument("--custom_nn", action="store_true", default=True, help="Usa CNN 1D o FastMLP")
    parser.add_argument("--fast_mlp", action="store_true", default=False, help="Usa FastFlatMlp invece di CNN")

    # Hardware
    parser.add_argument("--n_envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()

    # Directories
    log_dir = f"./logs/{args.training_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"./checkpoints", exist_ok=True)

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
            # Monitor è essenziale qui: calcola i reward dell'episodio per noi
            return Monitor(env, os.path.join(log_dir, str(rank)))
        return _init

    # 2. Vec Env + Stacking + Norm
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    
    stack_dim_actual = 5 if args.real_lidar else 3
    # Wrapper Stacking
    env = VecLidarStack(env, stack_dim=stack_dim_actual)
    
    # VecNormalize
    if args.load_model:
        vec_path = args.load_model.replace(".zip", "") + "_vecnormalize.pkl"
        if os.path.exists(vec_path):
            print(f"📥 Loaded VecNormalize: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = True
            env.norm_reward = True
        else:
            print("⚠️ VecNormalize not found, creating new one.")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # 3. Model Setup
    if args.algo == "AR_TQC": ModelClass = AdaptiveTQC
    elif args.algo == "TQC": ModelClass = TQC
    elif args.algo == "PPO": ModelClass = PPO
    elif args.algo == "SAC": ModelClass = SAC
    
    # Selezione Architettura
    policy_kwargs = {}
    
    if args.custom_nn:
        print("🧠 Using Costum NN model")
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
        tensorboard_log=log_dir + "_tb",
        verbose=1,
        device=args.device
    )

    # 4. Load Weights
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
            print(f"⚠️ Load failed (Architecture mismatch?): {e}")

    # 5. Learn
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f"./checkpoints/{args.training_name}_ckpts", name_prefix="ckpt")
    
    # Callback aggiornato
    stats_callback = TerminationStatsCallback(
        best_model_save_path=best_path_base, # Salverà qui durante il training,
        window_size=100
    )

    try:
        model.learn(total_timesteps=args.steps, tb_log_name=args.training_name, callback=[stats_callback, checkpoint_callback])
    except KeyboardInterrupt:
        print("\n🛑 Training interrotto manualmente (CTRL+C).")
    finally:
        # Chiudiamo l'ambiente per liberare risorse
        env.close()
        
        print("\n💾 Finalizing Model...")

        # LOGICA DI SOSTITUZIONE:
        # Se esiste un modello BEST salvato dalla callback, quello diventa il modello FINALE.
        if os.path.exists(best_path_zip):
            print(f"🏆 Found BEST model! Renaming '{best_path_zip}' to '{final_path_zip}'")
            
            # Sposta (Rinomina) il file .zip
            shutil.move(best_path_zip, final_path_zip)
            
            # Sposta (Rinomina) il file VecNormalize se esiste
            if os.path.exists(best_path_vec):
                shutil.move(best_path_vec, final_path_vec)
                
            print(f"✅ Il modello finale salvato è la versione MIGLIORE ottenuta.")
            
        else:
            # Caso di fallback: Se il training è durato troppo poco per generare un BEST (meno di 100 ep),
            # salviamo lo stato attuale come finale.
            print(f"⚠️ Nessun modello BEST trovato (training troppo breve?). Salvataggio stato corrente.")
            model.save(final_path_zip)
            if hasattr(env, "save"):
                env.save(final_path_vec)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()