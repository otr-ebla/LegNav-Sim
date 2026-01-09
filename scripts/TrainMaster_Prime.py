import argparse
import os
import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Import dell'ambiente custom
# Assicurati di lanciare lo script dalla root con: python -m scripts.TrainMaster_Prime ...
from src.envs.jhsfm_nav_env import SimpleNavEnv


"""
COMANDO PER ESEGUIRE IL TRAINING: 

COMANDO PER ESEGUIRE EVALUATION:  python3 -m scripts.TrainMaster_Prime      --eval      --render      --name eval_dt01      --scenario static_groups      --load_model checkpoints/DT01_25M.zip      --load_vecnorm DT01_vecnormalize.pkl      --eval_episodes 10      --use_legs

"""


class TrainMasterMetrics(BaseCallback):
    """Callback per monitorare le performance di TrainMaster_Prime durante il training."""
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

# Aggiungi force_static=False alla firma e passalo a SimpleNavEnv
def make_env(rank, seed, scenario, num_people, use_legs, render_mode=None, force_static=False):
    def _init():
        env = SimpleNavEnv(
            scenario_type=scenario, 
            num_people=num_people, 
            training=True, 
            use_legs=use_legs,
            render_mode=render_mode,
            force_static=force_static  # <--- AGGIUNGI QUI
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def evaluate_master(model, env, num_episodes, render=False):
    """
    Esegue la valutazione del NavigatorPrime.
    Gestisce correttamente sia ambienti vettorizzati multipli che singoli (rendering).
    """
    print(f"\n--- 🧐 VALUTAZIONE TrainMaster_Prime ({num_episodes} episodi) ---")
    if render:
        print("📺 Rendering ATTIVO (Premi Ctrl+C nel terminale per stoppare se necessario)")

    stats = {"SR": 0, "CR": 0, "TR": 0, "Time": []}
    episodes_finished = 0
    
    # Reset iniziale
    obs = env.reset()
    # Array per contare i passi di ogni ambiente parallelo
    steps_per_env = np.zeros(env.num_envs)

    try:
        while episodes_finished < num_episodes:
            # Predizione deterministica (migliore azione possibile senza rumore)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step nell'ambiente
            obs, rewards, dones, infos = env.step(action)
            steps_per_env += 1
            
            # Rendering grafico (funziona solo se env è DummyVecEnv con 1 processo)
            if render:
                env.render()

            # Analisi risultati per ogni ambiente
            for i, done in enumerate(dones):
                if done:
                    # Se abbiamo già raggiunto il numero richiesto, ignoriamo gli extra
                    if episodes_finished < num_episodes:
                        episodes_finished += 1
                        res = infos[i].get("termination_reason", "none")
                        
                        if res == "goal_reached":
                            stats["SR"] += 1
                            # dt = 0.1s (tempo simulato)
                            stats["Time"].append(steps_per_env[i] * 0.1) 
                        elif "collision" in res:
                            stats["CR"] += 1
                        else:
                            stats["TR"] += 1
                        
                        # Log rapido progressi
                        if episodes_finished % 10 == 0:
                            print(f"   -> Episodi completati: {episodes_finished}/{num_episodes}")

                    # Reset contatore passi per questo ambiente specifico
                    steps_per_env[i] = 0
                    
    except KeyboardInterrupt:
        print("\n⚠️ Valutazione interrotta manualmente.")

    # Calcolo Metriche Finali
    if episodes_finished > 0:
        print(f"\n--- 📊 REPORT EVALUATION ({episodes_finished} EPISODI) ---")
        print(f"Success Rate (SR): {stats['SR']/episodes_finished:.2%}")
        print(f"Collision Rate (CR): {stats['CR']/episodes_finished:.2%}")
        if stats['Time']: 
            print(f"Tempo Medio: {np.mean(stats['Time']):.2f}s")
        else:
            print("Tempo Medio: N/A (Nessun successo)")
    else:
        print("\n⚠️ Nessun episodio completato.")
    print("-------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="TrainMaster_Prime: Advanced RL Training Suite")
    
    # Parametri Training
    parser.add_argument("--algo", type=str, default="tqc", choices=["ppo", "sac", "tqc"], help="Algoritmo RL")
    parser.add_argument("--steps", type=int, default=1000000, help="Step totali di training")
    parser.add_argument("--n_envs", type=int, default=8, help="Numero di ambienti paralleli")
    
    # Parametri Ambiente
    parser.add_argument("--scenario", type=str, default="static_groups", help="Scenario: static_groups, random, parallel, ecc.")
    parser.add_argument("--name", type=str, required=True, help="Nome identificativo della missione")
    parser.add_argument("--use_legs", action="store_true", help="Abilita la simulazione delle gambe nel Lidar")
    parser.add_argument("--num_people", type=int, default=None, help="Override numero persone")

    # Parametri Caricamento / Evaluation
    parser.add_argument("--load_model", type=str, default=None, help="Path modello .zip da caricare")
    parser.add_argument("--load_vecnorm", type=str, default=None, help="Path statistiche .pkl da caricare")
    
    # Flag Valutazione e Rendering
    parser.add_argument("--eval", action="store_true", help="Attiva modalità valutazione (no training)")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Numero episodi di test")
    parser.add_argument("--render", action="store_true", help="Visualizza graficamente (forza 1 ambiente)")
    parser.add_argument("--save_gif", action="store_true", help="Salva un video GIF della valutazione")

    parser.add_argument("--force_static", action="store_true", help="Congela gli umani (Velocità 0) per il Curriculum Stage 1")

    args = parser.parse_args()

    # Setup percorsi
    base_save_path = f"./checkpoints/{args.name}"
    log_path = f"./logs/{args.name}"
    os.makedirs(base_save_path, exist_ok=True)

    print(f"🛠️  TrainMaster_Prime avviato: {args.algo.upper()} | Missione: {args.name}")
    print(f"🦵 Modalità Gambe: {'ATTIVA' if args.use_legs else 'DISATTIVA (Cilindri statici)'}")

    # 1. Setup Ambiente
    # Se dobbiamo renderizzare, usiamo DummyVecEnv (singolo processo) altrimenti SubprocVecEnv (parallelo)
    if args.render or args.save_gif:
        # ... (codice dummy env)
        env = DummyVecEnv([make_env(0, 42, args.scenario, args.num_people, args.use_legs, render_mode="human", force_static=args.force_static)])
    else:
        n_envs = args.n_envs
        # Aggiorna SubprocVecEnv
        env = SubprocVecEnv([make_env(i, 42, args.scenario, args.num_people, args.use_legs, force_static=args.force_static) for i in range(n_envs)])
    
    # 2. Gestione VecNormalize (Normalizzazione input Lidar)
    if args.load_vecnorm:
        print(f"📥 Caricamento VecNormalize da {args.load_vecnorm}...")
        env = VecNormalize.load(args.load_vecnorm, env)
        # In eval o render non vogliamo aggiornare le statistiche (media/varianza), le usiamo e basta
        if args.eval or args.render:
            env.training = False 
            env.norm_reward = False # Non serve normalizzare reward in test
        else:
            env.training = True # In Curriculum training continuiamo ad aggiornare
    else:
        if args.eval:
            print("⚠️ ATTENZIONE: Stai facendo eval senza caricare vecnorm! I risultati potrebbero essere scarsi.")
        print("🆕 Creazione nuova normalizzazione ambiente...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 3. Setup Modello (Policy)
    model_cls = {"ppo": PPO, "sac": SAC, "tqc": TQC}[args.algo]
    policy_kwargs = dict(net_arch=[256, 256])

    if args.load_model:
        print(f"🧠 Ricaricamento cervello da: {args.load_model}")
        model = model_cls.load(
            args.load_model, 
            env=env, 
            tensorboard_log=log_path,
            custom_objects={"learning_rate": 3e-4} # Reset LR opzionale se riprendi training
        )
    else:
        print(f"✨ Inizializzazione nuovo modello {args.algo.upper()}...")
        use_sde = (args.algo != "ppo") # SDE consigliato per SAC/TQC
        if args.algo == "ppo":
             model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
        else:
             model = model_cls(
                 "MlpPolicy", env, verbose=1, tensorboard_log=log_path, 
                 use_sde=use_sde, policy_kwargs=policy_kwargs
             )

    # 4. Esecuzione: EVALUATION o TRAINING
    if args.eval:
        evaluate_master(model, env, args.eval_episodes, render=args.render, save_gif=args.save_gif, name=args.name)
    else:
        callbacks = [TrainMasterMetrics(), CheckpointCallback(save_freq=50000, save_path=base_save_path, name_prefix="tm_ckpt")]
        try:
            model.learn(
                total_timesteps=args.steps, 
                callback=callbacks, 
                reset_num_timesteps=(args.load_model is None)
            )
        except KeyboardInterrupt:
            print("\n🛑 Training interrotto manualmente.")

        # 5. Salvataggio Finale
        print(f"💾 Salvataggio modello in {base_save_path}...")
        model.save(f"{base_save_path}/TrainMaster_Final")
        env.save(f"{base_save_path}/TrainMaster_VecNorm.pkl")
        print("✅ Operazione completata.")

    env.close()

if __name__ == "__main__":
    main()