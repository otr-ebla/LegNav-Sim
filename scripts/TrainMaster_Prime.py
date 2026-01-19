import argparse
import os
import gymnasium as gym
import torch
import numpy as np
import random
import time

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Import custom environment
# Run from root with: python -m scripts.TrainMaster_Prime ...
from src.envs.jhsfm_nav_env import SimpleNavEnv
from src.config import LidarConfig

# Import Custom Architectures (per caricare modelli da train_ppo)
try:
    from models.hybrid_cnn_mlp import HybridCnnMlp
except ImportError:
    print("⚠️ Warning: Could not import HybridCnnMlp. Custom NN loading might fail.")
    HybridCnnMlp = None

"""
COMMAND FOR FULL EVALUATION (All Scenarios) using a Train_PPO model:
python3 -m scripts.TrainMaster_Prime \
    --eval --render \
    --name Eval_Global_Test \
    --scenario all \
    --algo tqc \
    --custom_nn \
    --load_model checkpoints/Stage2_Model_BEST.zip \
    --load_vecnorm checkpoints/Stage2_Model_BEST_vecnormalize.pkl \
    --use_legs
"""

class SaveBestSuccessCallback(BaseCallback):
    """
    Callback personalizzata per valutare l'agente periodicamente
    e salvare il modello SOLO se il Success Rate migliora.
    """
    def __init__(self, eval_env, eval_freq=20000, save_path="./checkpoints/", name_prefix="best_model", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.best_success_rate = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n--- 🔍 Auto-Evaluating at step {self.num_timesteps} ---")
            
            # 1. Sincronizza VecNormalize (Fondamentale!)
            # Copia le statistiche (media/var) dall'env di training a quello di eval
            # altrimenti l'agente vede "numeri sbagliati".
            if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.training = False # Non aggiornare stats durante eval
                self.eval_env.norm_reward = False 
                self.eval_env.obs_rms = self.training_env.obs_rms
            
            # 2. Esegui valutazione usando la tua funzione esistente
            # Usiamo un numero ridotto di episodi (es. 20) per non rallentare troppo il training
            current_sr, current_spl = evaluate_master(
                self.model, 
                self.eval_env, 
                num_episodes=20, 
                render=False, 
                name="AutoEval"
            )
            
            # 3. Confronta e Salva
            if current_sr > self.best_success_rate:
                if self.verbose > 0:
                    print(f"🔥 NEW RECORD! Success Rate: {current_sr:.2%} (Was: {self.best_success_rate:.2%})")
                    print(f"💾 Saving model to {self.save_path}/{self.name_prefix}_BEST.zip")
                
                self.best_success_rate = current_sr
                self.model.save(os.path.join(self.save_path, f"{self.name_prefix}_BEST"))
                
                # Salviamo anche il VecNormalize associato al miglior modello
                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env.save(os.path.join(self.save_path, f"{self.name_prefix}_BEST_vecnormalize.pkl"))
            else:
                print(f"❄️  No improvement (Current: {current_sr:.2%} <= Best: {self.best_success_rate:.2%})")
                
        return True

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

def make_env(rank, seed, scenario, num_people, use_legs, is_training=True, render_mode=None, force_static=False, render_skip=1, distraction_prob=0.0):
    def _init():
        env = SimpleNavEnv(
            scenario_type=scenario, 
            num_people=num_people, 
            allow_keyboard_skip=True,
            training=is_training,
            use_legs=use_legs,
            render_mode=render_mode,
            force_static=force_static,
            render_skip=render_skip,
            distraction_prob=distraction_prob
        )
        env = Monitor(env)
        env.reset(seed=seed + rank + int(time.time()))
        return env
    return _init

def evaluate_master(model, env, num_episodes, render=False, name="eval"):
    """
    Executes evaluation of NavigatorPrime.
    """
    print(f"\n--- 🧐 EVALUATION: {name} ({num_episodes} episodes) ---")
    
    stats = {
        "SR": 0, "CR": 0, "TR": 0, "Time": [], "SPL": [], "Jerk": [], 
        "Coll_Ppl": 0, "Coll_Obs": 0
    }
    episodes_finished = 0
    
    obs = env.reset()
    steps_per_env = np.zeros(env.num_envs)

    try:
        while episodes_finished < num_episodes:
            action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)

            # Supporto per skip manuale da tastiera (definito in jhsfm_nav_env)
            if hasattr(env.envs[0].unwrapped, 'manual_skip_triggered') and env.envs[0].unwrapped.manual_skip_triggered:
                obs = env.reset()

            steps_per_env += 1
            
            if render:
                env.render()
                # time.sleep(0.01) # Scommentare per rallentare il render

            for i, done in enumerate(dones):
                if done:
                    # Raccogliamo i dati solo se non abbiamo finito i test richiesti
                    if episodes_finished < num_episodes:
                        episodes_finished += 1
                        
                        # Parsing Info
                        inf = infos[i]
                        res = inf.get("termination_reason", "none")
                        ep_len = inf.get("path_length", 0.0)
                        
                        # Stats Base
                        if res == "goal_reached":
                            stats["SR"] += 1
                            stats["Time"].append(inf.get("total_time", 0.0))
                            
                            # SPL Calculation
                            start_goal_dist = np.linalg.norm(np.array([inf.get("start_x",0), inf.get("start_y",0)]) - np.array([inf.get("goal_x",0), inf.get("goal_y",0)]))
                            denom = max(ep_len, start_goal_dist)
                            if denom > 0: stats["SPL"].append(start_goal_dist / denom)

                        elif "collision" in res:
                            stats["CR"] += 1
                            if "people" in res: stats["Coll_Ppl"] += 1
                            else: stats["Coll_Obs"] += 1
                            stats["SPL"].append(0.0)
                        else:
                            stats["TR"] += 1
                            stats["SPL"].append(0.0)
                        
                        stats["Jerk"].append(inf.get("mean_jerk", 0.0))
                        
                        # Print Progress
                        if episodes_finished % 5 == 0 or render:
                            icon = "✅" if res=="goal_reached" else "💥" if "collision" in res else "⏳"
                            print(f"   -> Ep {episodes_finished}/{num_episodes}: {icon} {res} (SPL: {stats['SPL'][-1]:.2f})")

                    steps_per_env[i] = 0
                    
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted manually.")

    # Report Finale
    if episodes_finished > 0:
        sr = stats['SR']/episodes_finished
        cr = stats['CR']/episodes_finished
        tr = stats['TR']/episodes_finished
        mean_time = np.mean(stats['Time']) if stats['Time'] else 0.0
        mean_spl = np.mean(stats['SPL']) if stats['SPL'] else 0.0
        
        print(f"\n" + "="*50)
        print(f"📊 REPORT: {name}")
        print(f"="*50)
        print(f"✅ Success Rate:   {sr:.2%}")
        print(f"💥 Collision Rate: {cr:.2%}")
        print(f"   ├─ People:      {stats['Coll_Ppl']}")
        print(f"   └─ Obstacles:   {stats['Coll_Obs']}")
        print(f"⏳ Timeout Rate:   {tr:.2%}")
        print(f"📏 Mean SPL:       {mean_spl:.3f}")
        print(f"⏱️  Mean Time:      {mean_time:.2f} s")
        print(f"="*50 + "\n")
        return sr, mean_spl
    else:
        print("\n⚠️ No episodes completed.")
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="TrainMaster_Prime: Universal Evaluator")
    
    # Training Params
    parser.add_argument("--algo", type=str, default="tqc", choices=["ppo", "sac", "tqc"], help="RL Algorithm")
    parser.add_argument("--steps", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    
    # Environment Params
    parser.add_argument("--scenario", type=str, default="static_groups", help="Scenario: 'all', 'mixed', 'bottleneck', 'intersection', etc.")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--use_legs", action="store_true", help="Enable legs simulation in Lidar")
    parser.add_argument("--num_people", type=int, default=None, help="Override number of people")
    parser.add_argument("--force_static", action="store_true", help="Freeze humans (Stage 1)")
    parser.add_argument("--distraction_prob", type=float, default=0.0, help="Probability humans are distracted")

    # Load / Resume Params
    parser.add_argument("--load_model", type=str, default=None, help="Path to .zip model to load")
    parser.add_argument("--load_vecnorm", type=str, default=None, help="Path to .pkl stats to load")
    parser.add_argument("--continue_training", action="store_true", help="If set, continues tensorboard step counter.")
    parser.add_argument("--custom_nn", action="store_true", help="Set if model uses HybridCnnMlp")
    
    # Evaluation / Render Params
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of test episodes per scenario")
    parser.add_argument("--render", action="store_true", help="Visual render (forces 1 env)")
    parser.add_argument("--render_skip", type=int, default=1, help="Render skip frames")

    args = parser.parse_args()

    base_save_path = f"./checkpoints/{args.name}"
    log_path = f"./logs/{args.name}"
    os.makedirs(base_save_path, exist_ok=True)

    print(f"🛠️  TrainMaster_Prime Started: {args.algo.upper()} | Mission: {args.name}")

    # Determina lista scenari
    if args.scenario == "all":
        scenarios_to_run = ["parallel", "intersection", "bottleneck", "static_groups", "random"]
        print(f"🌍 Mode: GLOBAL EVALUATION (Running on: {scenarios_to_run})")
    else:
        scenarios_to_run = [args.scenario]

    # Parametri modello
    ModelClass = {"ppo": PPO, "sac": SAC, "tqc": TQC}[args.algo]
    
    # Preparazione Custom Objects per caricamento
    custom_objects = {}
    if args.custom_nn and HybridCnnMlp:
        custom_objects["HybridCnnMlp"] = HybridCnnMlp
        # Fix per parametri scheduler salvati
        custom_objects["learning_rate"] = 0.0 
        custom_objects["lr_schedule"] = lambda _: 0.0
        custom_objects["clip_range"] = lambda _: 0.0

    # --- CICLO SCENARI ---
    for current_scenario in scenarios_to_run:
        print(f"\n🔄 SETUP SCENARIO: {current_scenario.upper()}")
        
        # 1. Configurazione Env
        is_training_env = not args.eval
        random_seed = random.randint(0, 1000)
        
        # Override num_people per scenario se non forzato
        curr_people = args.num_people
        if curr_people is None:
            if current_scenario == "bottleneck": curr_people = 5
            elif current_scenario == "intersection": curr_people = 6
            elif current_scenario == "parallel": curr_people = 7
            else: curr_people = 10 # Default

        if args.render:
            n_envs = 1
            env = DummyVecEnv([
                make_env(0, random_seed, current_scenario, curr_people, args.use_legs, 
                        is_training=is_training_env, render_mode="human", 
                        force_static=args.force_static, render_skip=args.render_skip,
                        distraction_prob=args.distraction_prob)
            ])
        else:
            n_envs = args.n_envs
            env = SubprocVecEnv([
                make_env(i, random_seed, current_scenario, curr_people, args.use_legs, 
                        is_training=is_training_env, force_static=args.force_static,
                        distraction_prob=args.distraction_prob) 
                for i in range(n_envs)
            ])

        # 2. Caricamento VecNormalize (Cruciale!)
        if args.load_vecnorm:
            # print(f"📥 Loading VecNormalize stats...")
            env = VecNormalize.load(args.load_vecnorm, env)
            env.training = not args.eval # Stop updating stats in eval
            env.norm_reward = not args.eval
        else:
            if args.eval: print("⚠️ WARNING: Eval without VecNormalize!")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # 3. Caricamento/Creazione Modello
        if args.load_model:
            model = ModelClass.load(args.load_model, env=env, custom_objects=custom_objects)
        else:
            # Init nuovo modello (solo training mode)
            policy_kwargs = dict(net_arch=[256, 256])
            model = ModelClass("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_kwargs)

        # 4. Esecuzione (Training o Eval)
        if args.eval:
            evaluate_master(model, env, args.eval_episodes, render=args.render, name=f"{args.name}_{current_scenario}")
        else:
            # --- SETUP CALLBACKS ---
            # A. Creiamo un Env separato per la validazione
            val_env = SubprocVecEnv([
                make_env(999, random_seed+999, current_scenario, curr_people, args.use_legs, 
                        is_training=False, 
                        force_static=args.force_static,
                        distraction_prob=args.distraction_prob)
            ])
            
            if args.load_vecnorm:
                val_env = VecNormalize.load(args.load_vecnorm, val_env)
            else:
                val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False)
            
            val_env.training = False     
            val_env.norm_reward = False 

            save_best_cb = SaveBestSuccessCallback(
                eval_env=val_env,
                eval_freq=20000,
                save_path=base_save_path,
                name_prefix=args.name
            )
            
            checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=base_save_path, name_prefix="tm_ckpt")
            metrics_cb = TrainMasterMetrics()
            
            print(f"🚀 Training Started... (Press Ctrl+C to save and exit)")
            
            # --- BLOCCO TRY-EXCEPT PER GESTIRE CTRL+C ---
            try:
                model.learn(
                    total_timesteps=args.steps, 
                    callback=[metrics_cb, checkpoint_cb, save_best_cb], 
                    reset_num_timesteps=(args.load_model is None)
                )
                
                # Salvataggio standard se finisce i passi
                print("✅ Training Completed. Saving final model...")
                model.save(f"{base_save_path}/TrainMaster_Final")
                if isinstance(env, VecNormalize):
                    env.save(f"{base_save_path}/TrainMaster_VecNorm.pkl")

            except KeyboardInterrupt:
                print("\n\n⚠️  INTERRUPTED MANUALLY! Saving current state before exiting...")
                
                # Salvataggio di emergenza
                model.save(f"{base_save_path}/TrainMaster_INTERRUPTED")
                
                if isinstance(env, VecNormalize):
                    env.save(f"{base_save_path}/TrainMaster_VecNorm_INTERRUPTED.pkl")
                
                print(f"💾 Emergency save completed at: {base_save_path}/TrainMaster_INTERRUPTED")
            
            val_env.close() 
        
        env.close()
        # Se siamo in training, usciamo dopo il primo scenario (la logica "all" è solo per eval)
        if not args.eval:
            break

if __name__ == "__main__":
    # Fix multiprocessing start method
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()