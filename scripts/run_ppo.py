import time

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC

# Updated import to find env in src
from src.envs.gym_nav_env import GymNavEnv
# Updated import to find NUM_RAYS in the sibling script
from .train_ppo import NUM_RAYS   

def main():
    env = GymNavEnv(render_mode="human", num_rays=NUM_RAYS, num_people=20)

    # Updated path to point to the new checkpoints folder
    model = TQC.load("./checkpoints/30MTQC", env=env)

    obs, info = env.reset()
    done = False
    terminated = False
    truncated = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        # Eseguiamo prima lo step per ottenere la reward
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Ora possiamo stampare sia l'azione che la reward ottenuta
        print(f"Action: v={action[0]:.2f}, w={action[1]:.2f} | Step Reward: {reward:.2f}")

        if terminated or truncated:
            print("Episode ended:", info.get("termination_reason"))
            time.sleep(1.0)
            obs, info = env.reset()
            terminated = truncated = False

    env.close()


if __name__ == "__main__":
    main()