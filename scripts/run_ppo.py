import time
import argparse
import os
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import environment e config
from src.envs.gym_nav_env import GymNavEnv
from src.config import LidarConfig

"""
COMANDO PER EVALUATION: python3 -m scripts.run_ppo --algo TQC --name New_Training_random --num_people 0



"""




def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Run inference for PPO/SAC/TQC agents")
    parser.add_argument("--algo", type=str, default="TQC", choices=["PPO", "SAC", "TQC"], help="RL Algorithm")
    parser.add_argument("--name", type=str, required=True, help="Model name (without .zip)")
    parser.add_argument("--num_people", type=int, default=0, help="Number of humans for testing")
    parser.add_argument("--render_skip", type=int, default=1, help="Render every N frames to speed up visualization")

    args = parser.parse_args()

    # 2. Selezione Classe Modello
    ModelClass = {"PPO": PPO, "SAC": SAC, "TQC": TQC}[args.algo]
    print(f"--- 🚀 Setup: Algo={args.algo} | Model={args.name} | Humans={args.num_people} ---")

    # 3. Setup Ambiente
    # Usiamo DummyVecEnv per compatibilità con VecNormalize e Rendering
    env = DummyVecEnv([lambda: GymNavEnv(
        render_mode="human",
        num_rays=LidarConfig.NUM_RAYS,
        num_people=args.num_people,
        render_skip=args.render_skip
    )])

    # 4. Caricamento Normalizzazione (VecNormalize)
    # Cerca prima nella root, poi in checkpoints
    pkl_filename = f"{args.name}_vecnormalize.pkl"
    paths_to_check = [pkl_filename, f"./checkpoints/{pkl_filename}"]
    
    loaded_norm = False
    for p in paths_to_check:
        if os.path.exists(p):
            print(f"📥 Loading VecNormalize stats from: {p}")
            env = VecNormalize.load(p, env)
            env.training = False     # STOP aggiornamento statistiche
            env.norm_reward = False  # Vogliamo vedere il reward reale
            loaded_norm = True
            break
    
    if not loaded_norm:
        print("⚠️  WARNING: VecNormalize .pkl not found! The agent might fail if trained with normalization.")

    # 5. Caricamento Modello
    model_path = f"./checkpoints/{args.name}"
    print(f"🧠 Loading Model from: {model_path}")
    
    try:
        model = ModelClass.load(model_path, env=env)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 6. Variabili Statistiche
    obs = env.reset()
    stats = {
        "episodes": 0, "success": 0, "timeout": 0, "collision": 0,
        "coll_ppl": 0, "coll_obs": 0,
        "path_len": 0.0, "time": 0.0, "jerk": 0.0, "spl": 0.0
    }

    print("\n--- 🎬 Starting Inference ---\n")

    try:
        while True:
            # Predict deterministico
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            #print("Action taken:", action)

            if done[0]:
                inf = info[0]
                reason = inf.get("termination_reason", "unknown")
                
                # [NUOVO] Gestione Skip Manuale
                if reason == "manual_skip":
                    print("-" * 60)
                    print(f"⏩ EPISODE SKIPPED MANUALLY (Not counted in stats)")
                    print("-" * 60)
                    time.sleep(0.5)
                    obs = env.reset()
                    continue # <--- Salta l'aggiornamento delle statistiche sotto

                # --- DA QUI IN POI IL CODICE RIMANE PER GLI EPISODI VALIDI ---
                stats["episodes"] += 1

                # Estrazione metriche episodio
                ep_len = inf.get("path_length", 0.0)
                ep_time = inf.get("total_time", 0.0)
                ep_jerk = inf.get("mean_jerk", 0.0)
                opt_len = inf.get("optimal_length", 1.0) # Se non presente usa 1.0 per evitare div by zero

                # Aggiornamento contatori
                is_success = 0
                if reason == "goal_reached":
                    stats["success"] += 1
                    is_success = 1
                    stats["time"] += ep_time
                elif reason == "max_steps_reached" or reason == "stuck":
                    stats["timeout"] += 1
                elif "collision" in reason:
                    stats["collision"] += 1
                    if "people" in reason: stats["coll_ppl"] += 1
                    else: stats["coll_obs"] += 1
                
                # Accumulo medie
                stats["path_len"] += ep_len
                stats["jerk"] += ep_jerk
                
                # Calcolo SPL (Success weighted by Path Length)
                # SPL = Success * (Shortest_Path / Max(Actual, Shortest))
                # Nota: optimal_length non è calcolato in nav_env originale, 
                # se non ce l'hai, SPL sarà approssimato o dovrai calcolarlo (distanza euclidea start-goal)
                dist_start_goal = np.linalg.norm(np.array([inf.get("start_x",0), inf.get("start_y",0)]) - np.array([inf.get("goal_x",0), inf.get("goal_y",0)]))
                curr_spl = is_success * (dist_start_goal / max(ep_len, dist_start_goal))
                stats["spl"] += curr_spl

                # Stampa Report
                tot = stats["episodes"]
                succ_rate = (stats["success"] / tot) * 100
                coll_rate = (stats["collision"] / tot) * 100
                timeout_rate = (stats["timeout"] / tot) * 100
                avg_spl = stats["spl"] / tot
                avg_time = stats["time"] / stats["success"] if stats["success"] > 0 else 0.0

                print("-" * 60)
                print(f"🏁 EPISODE {tot} ENDED: {reason.upper()}")
                print(f"📊 Stats:")
                print(f"  > Success Rate:   {succ_rate:.1f}%")
                print(f"  > Collision Rate: {coll_rate:.1f}% (Ppl: {stats['coll_ppl']}, Obs: {stats['coll_obs']})")
                print(f"  > Timeout Rate:   {timeout_rate:.1f}%")
                print(f"  > Avg SPL:        {avg_spl:.3f}")
                print(f"  > Avg Time:       {avg_time:.2f} s")
                print(f"  > Ang. Jerk:      {ep_jerk:.2f} rad/s³")
                print("-" * 60)
                
                time.sleep(0.5)
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 Inference stopped by user.")
        env.close()

if __name__ == "__main__":
    main()