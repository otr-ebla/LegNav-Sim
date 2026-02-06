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
from models.slot_attention import LidarSlotAttentionExtractor

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
    Custom callback to log stats, save the best model based on success rate,
    and optionally stop training early.
    """
    def __init__(self, verbose=0, window_size=100, stop_at_success_rate=None, best_model_save_path=None):
        super().__init__(verbose)
        self.window_size = window_size
        self.stop_at_success_rate = stop_at_success_rate
        self.history = deque(maxlen=window_size)
        
        # Variabili per il salvataggio del modello migliore
        self.best_model_save_path = best_model_save_path
        self.best_success_rate = -1.0 # Inizializzato a un valore basso

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                reason = info.get("termination_reason", "unknown")
                self.history.append(reason)

        # Calcoliamo le statistiche solo se abbiamo un po' di storico
        if len(self.history) > 0:
            total = len(self.history)
            
            # Counts
            # [MODIFICATION] Success now includes goal reached AND passive collisions
            success_count = sum(1 for r in self.history if r in ["goal_reached", "people_collision_passive"])
            
            timeout_count = self.history.count("max_steps_reached")
            obstacle_count = self.history.count("collision_static")
            
            # We track active human collisions as failures
            active_human_crash = self.history.count("people_collision_active")
            
            # Rates
            success_rate = success_count / total
            timeout_rate = timeout_count / total
            
            # Metric for dangerous collisions only
            collision_rate = (obstacle_count + active_human_crash) / total

            # Logging
            self.logger.record("metrics/success_rate", success_rate)
            self.logger.record("metrics/timeout_rate", timeout_rate)
            self.logger.record("metrics/collision_rate", collision_rate)
            
            # If you want to track specifically how many were "passive" successes
            passive_rate = self.history.count("people_collision_passive") / total
            self.logger.record("metrics/passive_success_rate", passive_rate)

            # --- BEST MODEL SAVING ---
            # Salviamo solo se la finestra è piena (per avere una statistica affidabile)
            if self.best_model_save_path is not None and len(self.history) >= self.window_size:
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    if self.verbose > 0:
                        print(f"\n🔥 NEW BEST MODEL FOUND! Success Rate: {success_rate:.2f} (Prev: {self.best_success_rate:.2f})")
                        print(f"💾 Saving best model to {self.best_model_save_path}...")
                    
                    # Salva il modello
                    self.model.save(self.best_model_save_path)
                    
                    # Salva anche VecNormalize (CRUCIALE per far funzionare il modello dopo il caricamento)
                    if self.training_env is not None:
                        self.training_env.save(f"{self.best_model_save_path}_vecnormalize.pkl")

            # --- EARLY STOPPING CHECK ---
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
    parser.add_argument("--rew_progress", type=float, default=5.0, help="Reward factor for progress towards goal")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--early_stop", type=float, default=0.95, help="Stop training if success rate reaches this value")
    parser.add_argument("--use_legs", action="store_true", default=False, help="Enable realistic leg physics for humans")
    parser.add_argument("--custom_nn", action="store_true", default=False, help="Use custom NN architecture")
    parser.add_argument("--lidar_noise", action="store_true", default=False, help="Enable LIDAR noise")
    parser.add_argument("--real_lidar", action="store_true", default=False, help="Enable 1080 rays (Real Specs)")
    parser.add_argument("--distraction_prob", type=float, default=0.3, help="Probability of human distraction behavior")
    
    args = parser.parse_args()

    # Create logs directory
    log_dir = f"./logs/{args.training_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"./checkpoints", exist_ok=True)

    print(f"🛠️  ENV SETUP: {args.num_obstacles} Obstacles | {args.num_people} Humans")
    if args.custom_nn:
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
                reward_factor_progress=args.rew_progress,
                use_legs=args.use_legs,
                distraction_prob=args.distraction_prob,
                render_skip=1,
                lidar_noise_enable=args.lidar_noise,
                real_lidar_specs=args.real_lidar 
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

    # Setup Model Class
    ModelClass = TQC if args.algo == "TQC" else SAC
    
    # 1. CONFIGURAZIONE ARCHITETTURA (Sempre Nuova)
    policy_kwargs = {}
    if args.custom_nn:
        print("🧠 ARCHITETTURA: Hybrid CNN-MLP (Unified)")
        policy_kwargs = dict(
            features_extractor_class=HybridCnnMlp,
            features_extractor_kwargs=dict(
                # Usa 1080 se real_lidar è attivo, altrimenti 108
                num_rays=1080 if getattr(args, "real_lidar", False) else LidarConfig.NUM_RAYS,
                stack_dim=5 if getattr(args, "real_lidar", False) else 3,
                hidden_dim=256,
            ),
            net_arch=[256, 128] 
        )

    # 2. INIZIALIZZA NUOVO AGENTE (Fresh Optimizer)
    print(f"✨ Initializing NEW {args.algo} Agent (Fresh Optimizer)")
    
    # Parametri comuni
    common_args = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4 if args.algo == "TQC" else 1, # TQC usa più gradient steps solitamente
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir + "_tb",
        verbose=1,
        device="cuda"
    )
    
    # Aggiungi parametri specifici per TQC
    if args.algo == "TQC":
        common_args["top_quantiles_to_drop_per_net"] = 2
        
    # Istanzia il modello
    model = ModelClass(**common_args)

    # 3. CARICAMENTO PESI (Curriculum Learning - FRESH OPTIMIZER MODE)
    if args.load_model:
        print(f"📥 Curriculum Learning: Injecting Weights from {args.load_model}")
        print("🛡️  Safe Mode: Loading weights ONLY (discarding old optimizer state)...")
        
        try:
            # 1. Carichiamo il vecchio modello in memoria (solo per leggere i pesi)
            old_model = ModelClass.load(args.load_model, device="cuda")
            
            # 2. Copiamo i pesi della Policy (Actor)
            # Questo include: Feature Extractor (CNN), Latent layers, Mean/LogStd heads
            model.policy.load_state_dict(old_model.policy.state_dict())
            print("   ✅ Policy (Actor) weights transferred.")
            
            # 3. Copiamo i pesi del Critic e del Target
            # Per TQC/SAC è fondamentale trasferire anche la "critica" allenata
            if hasattr(model, "critic") and hasattr(old_model, "critic"):
                model.critic.load_state_dict(old_model.critic.state_dict())
                print("   ✅ Critic weights transferred.")
            
            if hasattr(model, "critic_target") and hasattr(old_model, "critic_target"):
                model.critic_target.load_state_dict(old_model.critic_target.state_dict())
                print("   ✅ Critic Target weights transferred.")
                
            print("🚀 Ready to train! Optimizer is fresh (Learning Rate reset).")

        except Exception as e:
            print(f"❌ Error during manual weight transfer: {e}")
            print("🔄 Attempting Surgical Load (Partial Match)...")
            
            # FALLBACK: Se la copia esatta fallisce (es. architettura diversa),
            # usiamo il filtro chirurgico che avevamo scritto prima.
            try:
                current_state = model.policy.state_dict()
                old_state = old_model.policy.state_dict()
                
                compatible_state = {
                    k: v for k, v in old_state.items() 
                    if k in current_state and v.shape == current_state[k].shape
                }
                
                model.policy.load_state_dict(compatible_state, strict=False)
                print(f"   ✅ Surgical Load: {len(compatible_state)} layers transferred.")
            except Exception as e2:
                print(f"   ❌ Surgical load failed too: {e2}")
                print("   ⚠️  WARNING: Starting from SCRATCH.")

    print(f"🚀 Launching Training: {args.training_name}")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f"./checkpoints/{args.training_name}_ckpts",
        name_prefix="ckpt"
    )
    best_model_path = f"./checkpoints/{args.training_name}_BEST"

    stats_callback = TerminationStatsCallback(
        window_size=200, 
        stop_at_success_rate=args.early_stop,
        best_model_save_path=best_model_path, 
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
        raise e 
    finally:
        if 'model' in locals() and model is not None:
            save_path = f"./checkpoints/{args.training_name}"
            print(f"💾 Salvataggio di emergenza in corso su: {save_path}")
            model.save(save_path)
            
            if 'env' in locals() and env is not None:
                env.save(f"{save_path}_vecnormalize.pkl")
                try:
                    env.close()
                except Exception:
                    pass 
            
            print("✅ Salvataggio completato.")
        else:
            print("⚠️ Nessun modello da salvare.")

if __name__ == "__main__":
    main()