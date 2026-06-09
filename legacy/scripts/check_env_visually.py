import time
import random
from src.envs.jhsfm_nav_env import SimpleNavEnv

# --- CONFIGURAZIONE ---
USE_RANDOM_SCENARIO = True  # <--- Metti False per usare 'FIXED_SCENARIO'
FIXED_SCENARIO = "static_groups" 

# Lista dei 6 scenari implementati
SCENARIOS_LIST = [
    "parallel", 
    "perpendicular", 
    "circular", 
    "bottleneck", 
    "static_groups", 
    "intersection"
]

def run_visual_test():
    print("Premi FRECCIA DESTRA per saltare allo scenario successivo.")
    
    while True: # Ciclo infinito: ogni giro è un NUOVO scenario
        
        # 1. Scelta dello Scenario
        if USE_RANDOM_SCENARIO:
            selected_type = random.choice(SCENARIOS_LIST)
            print(f"\n🎲 NUOVO EPISODIO: Caricamento scenario '{selected_type.upper()}'...")
        else:
            selected_type = FIXED_SCENARIO
            print(f"\n🔒 SCENARIO FISSO: '{selected_type.upper()}'")

        # 2. Creazione Environment (necessaria per resettare dimensioni e numero persone)
        # Nota: allow_keyboard_skip=True ti permette di premere 'destra' per cambiare
        try:
            env = SimpleNavEnv(scenario_type=selected_type, training=False, allow_keyboard_skip=True, use_legs=True)
            obs, _ = env.reset()
            
            # 3. Loop Episodio
            running = True
            while running:
                # Azione nulla (robot fermo o lento)
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                
                env.render()
                #time.sleep(0.03) # 30ms delay
                
                # Check fine episodio o skip manuale
                if terminated or truncated:
                    running = False
                    reason = info.get("termination_reason", "unknown")
                    print(f"   -> Terminato: {reason}")

            # Chiudiamo l'ambiente vecchio per liberare memoria/finestre prima di crearne uno nuovo
            env.close()
            
        except KeyboardInterrupt:
            print("\nInterruzione manuale. Uscita.")
            if 'env' in locals(): env.close()
            break
        except Exception as e:
            print(f"Errore: {e}")
            break

if __name__ == "__main__":
    run_visual_test()