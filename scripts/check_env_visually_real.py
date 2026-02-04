import time
import random
import traceback
# [CORRETTO] Importiamo il wrapper Gymnasium come richiesto
from src.envs.gym_nav_env import GymNavEnv 

def run_visual_lidar_test():
    print("==================================================")
    print("TEST VISIVO: GYM WRAPPER + REAL LIDAR + MOVIMENTO CIRCOLARE")
    print("==================================================")
    print("Premi Ctrl+C per terminare.")
    
    # Istanziamo GymNavEnv invece di Simple2DEnv
    # Questo wrapper gestisce automaticamente la compatibilità Gym e lo stacking
    env = GymNavEnv(
        render_mode="human",       # Abilita il rendering a schermo
        real_lidar_specs=True,     # Attiva modalità 1080 raggi + noise
        render_skip=5,             # Renderizza ogni frame
        use_legs=True,
        num_people=10,
        lidar_noise_enable=True,
    )

    try:
        while True: 
            print("\n🔄 Reset ambiente... Generazione umani random.")
            
            # [CORRETTO] GymNavEnv.reset() restituisce (obs, info)
            # Nota: GymNavEnv usa reset(seed=...) internamente se necessario
            obs, info = env.reset() 
            
            running = True
            step_cnt = 0
            
            # Parametri Cerchio
            CMD_V = 0.25  
            CMD_W = 0.5   

            while running:
                # [CORRETTO] GymNavEnv.step() restituisce 5 valori (Standard Gymnasium)
                # obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = env.step([CMD_V, CMD_W])
                
                # GymNavEnv gestisce il render se render_mode="human", 
                # ma chiamiamo esplicitamente render() per sicurezza se lo script è standalone
                env.render()
                
                step_cnt += 1

                # Check fine episodio (Gymnasium style)
                if terminated or truncated:
                    running = False
                    reason = info.get("termination_reason", "unknown")
                    print(f"   -> Episodio terminato: {reason} (Step: {step_cnt})")
                    #time.sleep(1.0) # Pausa breve per vedere l'effetto

    except KeyboardInterrupt:
        print("\n🛑 Interruzione manuale ricevuta.")
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {e}")
        traceback.print_exc()
    finally:
        env.close()
        print("Environment chiuso correttamente.")

if __name__ == "__main__":
    run_visual_lidar_test()