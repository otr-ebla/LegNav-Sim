# import time
# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# from src.envs.jhsfm_nav_env import SimpleNavEnv

# MODEL_NAME = "2_Mno_obstacles_part2" # Il nome del tuo checkpoint

# # 1. Init Environment wrapper in DummyVecEnv (Required by SB3)
# env = DummyVecEnv([lambda: SimpleNavEnv(num_people=5, scenario_type="cross")])
# # 2. Load Stats (VecNormalize) from the trained agent
# # This ensures the new observation (lidar/dist) is scaled exactly like the old one
# env = VecNormalize.load(f"{MODEL_NAME}_vecnormalize.pkl", env)
# env.training = False; env.norm_reward = False # Inference mode (no update)

# # 3. Load the SAC Model
# model = SAC.load(f"checkpoints/{MODEL_NAME}", env=env)
# # 4. Inference Loop
# obs = env.reset()
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, _, done, _ = env.step(action)
    
#     # Access inner env to call the custom render method
#     env.envs[0].render() 
#     #time.sleep(0.05)

import time
from src.envs.jhsfm_nav_env import SimpleNavEnv

# Testiamo 'parallel' per vedere il respawn infinito (Umani scendono, Robot sale)
env = SimpleNavEnv(scenario_type="circular") 
obs, _ = env.reset(seed=None)

print("Check: Osserva se gli umani 'teletrasportano' in alto quando arrivano in basso.")

for i in range(2000):
    # Robot fermo o lento in avanti ([0.1, 0.0]) per farsi sorpassare dal flusso
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    
    env.render()
    time.sleep(0.03) # 30ms per frame
    
    if terminated or truncated:
        env.reset()