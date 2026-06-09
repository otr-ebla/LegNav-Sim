import gymnasium as gym
import numpy as np
import os
import sys
import time
import random
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import Ambiente e Wrapper
from src.envs.jhsfm_nav_env import JhsfmGymNavEnv
from src.envs.vec_lidar_stack import VecTemporalStack
from src.config import LidarConfig, SimConfig, RobotConfig

# =============================================================================
# SETUP OGGETTI CUSTOM
# =============================================================================
try:
    from models.custom_cnn import EndToEndNavExtractor as HybridCnnMlp
    CUSTOM_OBJECTS = {"HybridCnnMlp": HybridCnnMlp}
    print("✅ Custom CNN (HybridCnnMlp) rilevata.")
except ImportError:
    CUSTOM_OBJECTS = {}

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
MODEL_PATH = "checkpoints/DAJEFERMI2.zip"
VECNORM_PATH = "checkpoints/DAJEFERMI2_vecnormalize.pkl"

# Lista da cui estrarre casualmente ad ogni episodio
SCENARIOS_POOL = ["random", "bottleneck", "intersection", "parallel", "static_groups"]

TOTAL_EPISODES = 20  # Numero totale di episodi misti da eseguire
RENDER = True
STACK_DIM = RobotConfig.LIDAR_STACK_DIM

# =============================================================================
# MAIN
# =============================================================================
def run_evaluation():
    # 1. Caricamento Modello
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Errore: Modello non trovato in {MODEL_PATH}")
        return

    print(f"🔄 Caricamento Modello da: {MODEL_PATH}")
    try:
        try:
            model = SAC.load(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print("✅ Modello caricato come: SAC")
        except:
            model = PPO.load(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print("✅ Modello caricato come: PPO")
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

   # 2. Creazione Ambiente
    print("\n🛠️  Inizializzazione Ambiente Base...")
    
    # Funzione factory semplice (come make_env_factory in run_ppo.py)
    def make_env():
        return JhsfmGymNavEnv(
            render_mode="human" if RENDER else None,
            scenario_type="random", 
            num_people=SimConfig.NUM_HUMANS,
            use_legs=True,
            render_skip=1,
            lidar_noise_enable=False
        )

    # A. Creiamo il DummyVecEnv (Livello Base)
    raw_env = DummyVecEnv([make_env])

    # B. APPLICHIAMO LO STACKING (Come riga 80 di run_ppo.py)
    # Transforms observation from (216,) to (5, 216)
    print(f"📦 Applicazione VecTemporalStack (dim={STACK_DIM})")
    stacked_env = VecTemporalStack(raw_env, stack_dim=STACK_DIM)

    # C. APPLICHIAMO LA NORMALIZZAZIONE (Come riga 86 di run_ppo.py)
    # Pass 'stacked_env' with the correct shape (5, 216) expected by the pkl.
    if os.path.exists(VECNORM_PATH):
        print(f"📥 Caricamento VecNormalize da {VECNORM_PATH}")
        env = VecNormalize.load(VECNORM_PATH, stacked_env)
        # Fondamentale per eval: bloccare l'aggiornamento delle statistiche
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ VecNormalize non trovato! Uso env non normalizzato.")
        env = stacked_env

    # D. Accesso all'ambiente "profondo" per cambiare scenario
    # La catena è: VecNormalize -> VecTemporalStack -> DummyVecEnv -> JhsfmGymNavEnv
    if os.path.exists(VECNORM_PATH):
        # env è VecNormalize, il suo .venv è VecTemporalStack, il cui .venv è DummyVecEnv
        base_env = env.venv.venv.envs[0]
    else:
        # env è VecTemporalStack, il suo .venv è DummyVecEnv
        base_env = env.venv.envs[0]

    # 3. Loop Episodi Misti
    print(f"\n🚀 AVVIO VALUTAZIONE ({TOTAL_EPISODES} episodi con scenari misti)\n")
    
    stats = {s: {"success": 0, "collision": 0, "timeout": 0, "count": 0} for s in SCENARIOS_POOL}
    total_success = 0

    for ep in range(TOTAL_EPISODES):
        # A. Estrai Nuova Tipologia di Scenario
        current_scenario = random.choice(SCENARIOS_POOL)
        
        # B. "Iniettiamo" lo scenario nell'ambiente base prima del reset
        base_env.scenario_type = current_scenario
        
        print(f"▶ Ep {ep+1}/{TOTAL_EPISODES} | Scenario: {current_scenario.upper()} ...", end="", flush=True)

        # C. Reset (Questo ora userà il nuovo scenario_type impostato)
        obs = env.reset()
        
        done = False
        ep_len = 0
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            ep_reward += reward[0]
            ep_len += 1
            
            if RENDER:
                env.render()
                time.sleep(0.002)

            if done[0]:
                info = infos[0]
                outcome = info.get("termination_reason", "unknown")
                
                # Aggiornamento Statistiche
                stats[current_scenario]["count"] += 1
                
                if outcome == "goal_reached":
                    total_success += 1
                    stats[current_scenario]["success"] += 1
                    print(f" ✅ WIN ({ep_len} steps)")
                elif "collision" in outcome:
                    stats[current_scenario]["collision"] += 1
                    print(f" 💥 CRASH ({outcome})")
                elif outcome == "max_steps_reached" or outcome == "timeout":
                    stats[current_scenario]["timeout"] += 1
                    print(f" ⏰ TIMEOUT")
                else:
                    print(f" ❓ END ({outcome})")

    # 4. Report Finale
    print("\n" + "="*60)
    print("📊 REPORT FINALE PER SCENARIO")
    print("="*60)
    
    for scn in SCENARIOS_POOL:
        data = stats[scn]
        count = data["count"]
        if count > 0:
            sr = (data["success"] / count) * 100
            print(f"🔹 {scn.upper():<15}: {sr:5.1f}% Success | {data['success']} Win | {data['collision']} Crash | {data['timeout']} Time ({count} tot)")
        else:
            print(f"🔹 {scn.upper():<15}: N/A (0 episodi)")
            
    total_sr = (total_success / TOTAL_EPISODES) * 100
    print("-"*60)
    print(f"🏆 TOTAL SUCCESS RATE: {total_sr:.2f}%")
    print("="*60)

    env.close()

if __name__ == "__main__":
    run_evaluation()