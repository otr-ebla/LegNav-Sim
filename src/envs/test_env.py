import numpy as np
from src.envs.gym_nav_env import GymNavEnv

def test_environment():
    print("Inizializzazione Environment...")
    # Crea l'ambiente
    env = GymNavEnv(render_mode=None) # Metti "human" se vuoi vederlo a video
    
    print("Testing Reset...")
    obs, info = env.reset()
    
    # Verifica dimensioni osservazione
    # 4 scalari + (108 raggi * 3 stack) = 328
    expected_dim = 4 + (108 * 3)
    assert obs.shape == (expected_dim,), f"Errore dimensione obs: {obs.shape} vs {expected_dim}"
    print(f"Reset OK. Shape: {obs.shape}")

    print("Testing Step Loop...")
    for i in range(20):
        # Azione casuale
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 5 == 0:
            print(f"Step {i}: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
            
        if terminated or truncated:
            obs, info = env.reset()
            print("Environment resettato.")

    env.close()
    print("\n✅ TUTTO OK! L'ambiente è pronto per il training.")

if __name__ == "__main__":
    test_environment()