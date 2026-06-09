import time
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv

# Import environment e config
from src.envs.gym_nav_env import GymNavEnv
from src.envs.vec_lidar_stack import VecTemporalStack as VecLidarStack 
from src.config import LidarConfig

# Importazione per Custom NN
from models.custom_cnn import EndToEndNavExtractor as HybridCnnMlp

def make_env_factory(num_people, num_obstacles, render_mode, use_legs, render_skip, distraction_prob, real_lidar, lidar_noise, rank):
    def _init():
        env = GymNavEnv(
            render_mode=render_mode,
            num_rays=LidarConfig.NUM_RAYS,
            num_people=num_people,
            num_obstacles=num_obstacles,
            render_skip=render_skip,
            use_legs=use_legs,
            distraction_prob=distraction_prob,
            real_lidar_specs=real_lidar,
            lidar_noise_enable=lidar_noise,
            stack_dim=3 if real_lidar else 3
        )
        env.reset(seed=rank + int(time.time())) 
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="TQC", choices=["PPO", "SAC", "TQC", "AR_TQC"])
    parser.add_argument("--name", type=str, required=True, help="Model name")
    
    # --- MODIFICA FONDAMENTALE ---
    # Questo numero definisce QUANDO FERMARSI, indipendentemente se guardi o no.
    parser.add_argument("--eval_episodes", type=int, default=100, help="Numero di episodi da testare (Default: 100)")
    
    # Flag per disattivare la grafica se vuoi fare test muti
    parser.add_argument("--no_render", action="store_true", default=False, help="Se presente, DISATTIVA la grafica (per test veloci)")

    # Parametri Env
    parser.add_argument("--num_people", type=int, default=5)
    parser.add_argument("--num_obstacles", type=int, default=15)
    parser.add_argument("--use_legs", action="store_true", default=False)
    parser.add_argument("--real_lidar", action="store_true", default=False)
    parser.add_argument("--custom_nn", action="store_true", default=False) 
    parser.add_argument("--distraction_prob", type=float, default=0.3)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--lidar_noise", action="store_true", default=False)

    args = parser.parse_args()
    
    # 1. Logica Rendering
    # Se NON c'è il flag no_render, attiviamo "human".
    render_mode = "human" if not args.no_render else None
    
    # Se renderizziamo, forziamo n_envs=1 (Pygame non gestisce multi-finestra bene)
    if render_mode == "human" and args.n_envs > 1:
        print("⚠️ Rendering attivo: forzato n_envs = 1")
        args.n_envs = 1

    print(f"--- 🚀 Eval Setup: {args.name} ---")
    print(f"👁️  Render Mode: {'ON' if render_mode == 'human' else 'OFF (Fast)'}")
    print(f"🏁 Max Episodes: {args.eval_episodes}")

    # 2. Modello
    if args.algo == "AR_TQC":
        from models.adaptive_tqc import AdaptiveTQC
        ModelClass = AdaptiveTQC
    else:
        ModelClass = {"PPO": PPO, "SAC": SAC, "TQC": TQC}[args.algo]

    # 3. Env Base
    env_fns = [make_env_factory(
        args.num_people, args.num_obstacles, render_mode, args.use_legs, 1, 
        args.distraction_prob, args.real_lidar, args.lidar_noise, i
    ) for i in range(args.n_envs)]
    
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # 4. Wrapper Stacking
    stack_dim = 3 if args.real_lidar else 3
    print(f"📦 Applying VecLidarStack (Dict Mode)")
    env = VecLidarStack(env, stack_dim=stack_dim)

    # 5. Normalizzazione
    pkl_filename = f"{args.name}_vecnormalize.pkl"
    path_norm = pkl_filename if os.path.exists(pkl_filename) else f"./checkpoints/{pkl_filename}"
    if os.path.exists(path_norm):
        print(f"📥 Loading VecNormalize: {path_norm}")
        env = VecNormalize.load(path_norm, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ VecNormalize not found!")

    # 6. Load Model
    model_path = f"./checkpoints/{args.name}"
    print(f"🧠 Loading Model: {model_path}")
    
    custom_objects = {}
    if args.custom_nn:
        custom_objects = {"HybridCnnMlp": HybridCnnMlp}

    try:
        model = ModelClass.load(model_path, env=env, custom_objects=custom_objects)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 7. Loop Valutazione
    obs = env.reset()
    
    # Barra di progresso attiva (mostra 0/100)
    pbar = tqdm(total=args.eval_episodes)
    
    ep_count = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # Il render avviene automaticamente dentro env.step() se render_mode="human"
            # Ma aggiungiamo uno sleep minimo per non far volare via la simulazione se il PC è troppo potente
            if render_mode == "human":
                time.sleep(0.002) # 500 FPS cap circa

            for i, done in enumerate(dones):
                if done:
                    ep_count += 1
                    pbar.update(1)
                    
                    # Log opzionale nel terminale se vuoi vedere il motivo
                    # info = infos[i]
                    # print(f"Ep {ep_count}: {info.get('termination_reason')}")

            # Stop Condition: Fermati se arrivi a 100
            if args.eval_episodes > 0 and ep_count >= args.eval_episodes:
                print(f"\n✅ Raggiunti {args.eval_episodes} episodi. Stop.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente.")
    finally:
        pbar.close()
        env.close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()