import time
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv

# Import environment e config
from src.envs.gym_nav_env import GymNavEnv
from src.config import LidarConfig

# Importazione per Custom NN
from models.hybrid_cnn_mlp_previous import HybridCnnMlp

"""
COMMAND FOR FAST PARALLEL EVALUATION:
python3 -m scripts.run_ppo --name Stage2_Model --num_people 10 --eval_episodes 1000 --n_envs 10
"""

def make_env_factory(num_people, num_obstacles, render_mode, use_legs, render_skip, distraction_prob, rank):
    """Factory function for creating environments in parallel processes."""
    def _init():
        env = GymNavEnv(
            render_mode=render_mode,
            num_rays=LidarConfig.NUM_RAYS,
            num_people=num_people,
            num_obstacles=num_obstacles,
            render_skip=render_skip,
            use_legs=use_legs,
            distraction_prob=distraction_prob,
            # Importante: seed diversi per ogni processo per evitare episodi identici
        )
        env.reset(seed=rank + int(time.time())) 
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Run inference/evaluation for Indoor Navigation")
    parser.add_argument("--algo", type=str, default="TQC", choices=["PPO", "SAC", "TQC"], help="RL Algorithm")
    parser.add_argument("--name", type=str, required=True, help="Model name (without .zip)")
    parser.add_argument("--num_people", type=int, default=0, help="Number of humans")
    parser.add_argument("--num_obstacles", type=int, default=15, help="Number of static obstacles")
    parser.add_argument("--render_skip", type=int, default=1, help="Skip N frames in render mode")
    parser.add_argument("--custom_nn", action="store_true", default=False, help="Use custom Hybrid CNN architecture")
    parser.add_argument("--use_legs", action="store_true", default=False, help="Use leg detection in LIDAR simulation")

    parser.add_argument("--distraction_prob", type=float, default=0.6, help="Prob. humans are distracted (0.0=All Cooperative, 1.0=All Blind)")
    
    # Flags Evaluation
    parser.add_argument("--eval_episodes", type=int, default=0, help="If > 0, runs fast evaluation without rendering")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments for fast eval")

    args = parser.parse_args()
    
    FAST_EVAL = args.eval_episodes > 0
    
    # --- MODIFICA RICHIESTA ---
    # Se FAST_EVAL è True, usa il numero passato da riga di comando.
    # Se è False (rendering attivo), impostiamo il limite fisso a 300.
    if FAST_EVAL:
        target_episodes = args.eval_episodes
    else:
        target_episodes = 300
    # --------------------------
    
    # Forziamo n_envs a 1 se dobbiamo renderizzare (non si può renderizzare in parallelo facilmente)
    if not FAST_EVAL and args.n_envs > 1:
        print("⚠️  Rendering enabled: Forcing n_envs=1 (Visualization works best in single process)")
        args.n_envs = 1

    render_mode = None if FAST_EVAL else "human"
    
    print(f"--- 🚀 Setup: Algo={args.algo} | Model={args.name} | Humans={args.num_people} ---")
    if FAST_EVAL:
        print(f"⚡ FAST PARALLEL EVAL: {target_episodes} Episodes on {args.n_envs} CPU Cores")
    else:
        print(f"📺 RENDER EVAL: Running {target_episodes} episodes with visualization.")

    # 1. Selezione Classe Modello
    ModelClass = {"PPO": PPO, "SAC": SAC, "TQC": TQC}[args.algo]

    # 2. Setup Ambiente (Vectorized)
    env_fns = [make_env_factory(
        args.num_people,
        args.num_obstacles, 
        render_mode, 
        args.use_legs,
        args.render_skip, 
        args.distraction_prob, 
        i) for i in range(args.n_envs)]
    
    if args.n_envs > 1:
        # Usa SubprocVecEnv per vero parallelismo su CPU multiple
        env = SubprocVecEnv(env_fns)
    else:
        # Usa DummyVecEnv per debug o rendering (stesso processo)
        env = DummyVecEnv(env_fns)

    # 3. Caricamento Normalizzazione
    pkl_filename = f"{args.name}_vecnormalize.pkl"
    paths_to_check = [pkl_filename, f"./checkpoints/{pkl_filename}"]
    
    loaded_norm = False
    for p in paths_to_check:
        if os.path.exists(p):
            print(f"📥 Loading VecNormalize stats from: {p}")
            env = VecNormalize.load(p, env)
            env.training = False     
            env.norm_reward = False  
            loaded_norm = True
            break
    
    if not loaded_norm:
        print("⚠️  WARNING: VecNormalize .pkl not found! Agent performance might be degraded.")

    # 4. Caricamento Modello
    model_path = f"./checkpoints/{args.name}"
    print(f"🧠 Loading Model from: {model_path}")
    
    custom_objects = {}
    if args.custom_nn:
        custom_objects = {"HybridCnnMlp": HybridCnnMlp}

    try:
        model = ModelClass.load(model_path, env=env, custom_objects=custom_objects)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 5. Inizializzazione Metriche
    stats = {
        "episodes": 0, 
        "success": 0, "timeout": 0, "collision": 0,
        "coll_ppl": 0, "coll_ppl_active": 0, "coll_ppl_passive": 0, "coll_obs": 0,
        "yield_violations": 0, "total_yield_count": 0,
        "path_len": [], "time": [], "jerk": [], "spl": []
    }

    obs = env.reset()
    
    # Progress bar solo se fast eval, altrimenti sporca il render output
    pbar = tqdm(total=target_episodes) if FAST_EVAL else None

    try:
        while True:
            # --- MODIFICA: Controllo universale sul numero di episodi ---
            if stats["episodes"] >= target_episodes:
                break
            # ------------------------------------------------------------

            # Predict (su N ambienti contemporaneamente)
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # Rendering (Solo Single Core)
            if not FAST_EVAL:
                env.render()
                time.sleep(0.001)

            # Gestione Parallelizzata dei risultati
            # 'dones' è un array di booleans [True, False, True...]
            for i, done in enumerate(dones):
                if done:
                    # SB3 mette le info dell'episodio terminato in infos[i] o infos[i]['terminal_observation']
                    # Ma dato che usiamo GymNavEnv custom, le info sono direttamente accessibili
                    inf = infos[i]
                    reason = inf.get("termination_reason", "unknown")
                    
                    # Skip manuale (solo in visual mode)
                    if reason == "manual_skip": 
                        if not FAST_EVAL: print("⏩ Skipped"); obs[i] = env.envs[i].reset()[0] # Reset manuale su Dummy
                        continue

                    # --- AGGIORNAMENTO STATISTICHE ---
                    stats["episodes"] += 1
                    if FAST_EVAL: pbar.update(1)

                    # Estrazione dati
                    ep_len = inf.get("path_length", 0.0)
                    ep_time = inf.get("total_time", 0.0)
                    ep_jerk = inf.get("mean_jerk", 0.0)
                    yield_viols = inf.get("yield_violations", 0)

                    stats["path_len"].append(ep_len)
                    stats["jerk"].append(ep_jerk)
                    stats["total_yield_count"] += yield_viols
                    if yield_viols > 0: stats["yield_violations"] += 1

                    # Classificazione
                    is_success = 0
                    if reason == "goal_reached":
                        stats["success"] += 1
                        is_success = 1
                        stats["time"].append(ep_time)
                    elif reason == "max_steps_reached":
                        stats["timeout"] += 1
                    elif "collision" in reason:
                        stats["collision"] += 1
                        if "people" in reason:
                            stats["coll_ppl"] += 1
                            c_type = inf.get("collision_type", "unknown")
                            if c_type == "active": stats["coll_ppl_active"] += 1
                            elif c_type == "passive": stats["coll_ppl_passive"] += 1
                        else:
                            stats["coll_obs"] += 1

                    # SPL
                    dist_start_goal = np.linalg.norm(np.array([inf.get("start_x",0), inf.get("start_y",0)]) - np.array([inf.get("goal_x",0), inf.get("goal_y",0)]))
                    denominator = max(ep_len, dist_start_goal)
                    if denominator == 0: denominator = 1.0
                    curr_spl = is_success * (dist_start_goal / denominator)
                    stats["spl"].append(curr_spl)

                    # Print solo in single mode
                    if not FAST_EVAL:
                        print(f"🏁 Ep {stats['episodes']}/{target_episodes}: {reason} | SPL: {curr_spl:.2f}")

                    # Se abbiamo raggiunto il target durante questo ciclo for (dovuto a env paralleli), usciamo
                    if stats["episodes"] >= target_episodes:
                        break

        # --- REPORT FINALE ---
        if FAST_EVAL:
            pbar.close()
            
        tot = stats["episodes"]
        if tot == 0: return

        sr = (stats["success"] / tot) * 100
        cr = (stats["collision"] / tot) * 100
        tr = (stats["timeout"] / tot) * 100
        
        avg_spl = np.mean(stats["spl"]) if stats["spl"] else 0.0
        avg_time = np.mean(stats["time"]) if stats["time"] else 0.0
        avg_len = np.mean(stats["path_len"]) if stats["path_len"] else 0.0
        avg_jerk = np.mean(stats["jerk"]) if stats["jerk"] else 0.0
        
        avg_yield_per_ep = stats["total_yield_count"] / tot
        ep_with_yield_viol_rate = (stats["yield_violations"] / tot) * 100
        
        print("\n" + "="*60)
        print(f"📊 FINAL REPORT ({tot} Episodes | {args.num_people} Humans | {args.n_envs} Parallel Envs)")
        print("="*60)
        print(f"✅ Success Rate:        {sr:.2f}%")
        print(f"💥 Collision Rate:      {cr:.2f}%")
        print(f"   ├─ Obstacles:        {stats['coll_obs']} ({(stats['coll_obs']/tot)*100:.1f}%)")
        print(f"   ├─ People (Total):   {stats['coll_ppl']} ({(stats['coll_ppl']/tot)*100:.1f}%)")
        print(f"   │   ├─ Active:       {stats['coll_ppl_active']}")
        print(f"   │   └─ Passive:      {stats['coll_ppl_passive']}")
        print(f"⏳ Timeout Rate:        {tr:.2f}%")
        print("-" * 60)
        print(f"📏 Avg SPL (Efficienza):{avg_spl:.3f}  (1.0 = Perfect)")
        print(f"⏱️  Avg Time (Success):  {avg_time:.2f} s")
        print(f"👣 Avg Path Length:     {avg_len:.2f} m")
        print(f"📉 Avg Ang. Jerk:       {avg_jerk:.2f} rad/s³")
        print("-" * 60)
        print(f"🤖 SOCIAL METRICS:")
        print(f"⚠️  Yield Violations/Ep: {avg_yield_per_ep:.2f}")
        print(f"🚫 Episodes w/ Violat.:  {ep_with_yield_viol_rate:.1f}%")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
        if args.n_envs > 1:
            env.close() # Importante chiudere i sottoprocessi

if __name__ == "__main__":
    # Fix per multiprocessing su alcuni OS
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()