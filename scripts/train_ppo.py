import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC

from src.envs.gym_nav_env import GymNavEnv  # Updated import path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch
from stable_baselines3.common.callbacks import BaseCallback

NUM_RAYS = 108
NUM_PEOPLE = 0
NUM_OBSTACLES = 10
N_ENVS = 100
PEOPLE_SPEED = 0.0
TRAINING_NAME = "DT01_25M"
LOAD_MODEL = False  # Set to True to load existing model
STACK_DIM = 3


PREV_MODEL_PATH = "checkpoints/25MSAC_jack.zip"
# Assicurati che questo percorso sia corretto (usa il percorso assoluto se serve)
PREV_ENV_STATS_PATH = "25MSAC_jack_vecnormalize.pkl"



class TerminationStatsCallback(BaseCallback):
    def __init__(self, verbose=0, training_name="default"):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir="./logs/" + training_name)
        self.success = 0
        self.timeout = 0
        self.obstacle = 0
        self.human = 0
        self.episode_id = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                # NOTA: Qui usiamo le chiavi esatte definite in simple_env.py
                reason = info.get("termination_reason", "unknown")

                if reason == "goal_reached":
                    self.success += 1
                elif reason == "max_steps_reached":
                    self.timeout += 1
                elif reason == "obstacle_collision" or reason == "wall_collision":
                    self.obstacle += 1
                elif reason == "people_collision":
                    self.human += 1

                total = self.success + self.timeout + self.obstacle + self.human
                if total > 0:
                    self.writer.add_scalar("metrics/success_rate", self.success / total, self.episode_id)
                    self.writer.add_scalar("metrics/timeout_rate", self.timeout / total, self.episode_id)
                    self.writer.add_scalar("metrics/obstacle_collision_rate", self.obstacle / total, self.episode_id)
                    self.writer.add_scalar("metrics/human_collision_rate", self.human / total, self.episode_id)

                self.episode_id += 1

        return True

def make_env(rank: int):
    def _init():
        env = GymNavEnv(
            render_mode=None,
            num_rays=NUM_RAYS,
            num_people=NUM_PEOPLE,
        )
        env = Monitor(env)
        return env
    return _init

def main():
    use_subproc = True
    total_timesteps = 25_000_000  # O quanti ne vuoi fare in più

    # 1. Creazione dell'ambiente base (uguale a prima)
    print("Creating environment...")
    if use_subproc:
        env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

    # 2. Gestione VecNormalize (Caricamento vs Nuova Creazione)
    if LOAD_MODEL and PREV_ENV_STATS_PATH:
        print(f"Loading VecNormalize stats from: {PREV_ENV_STATS_PATH}")
        # Carica le statistiche (media/var) salvate
        env = VecNormalize.load(PREV_ENV_STATS_PATH, env)
        # Importante: rimetti in modalità training (aggiorna le statistiche)
        env.training = True 
        # Assicurati che la config sia coerente (nel tuo script originale norm_reward=False)
        env.norm_reward = False
        env.norm_obs = True
    else:
        print("Creating new VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 3. Gestione Modello (Caricamento vs Nuova Creazione)
    if LOAD_MODEL and PREV_MODEL_PATH:
        print(f"Loading model from: {PREV_MODEL_PATH}")
        model = SAC.load(
            PREV_MODEL_PATH,
            env=env,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            # Sovrascriviamo il path dei log per puntare al nuovo TRAINING_NAME
            tensorboard_log="./logs/" + TRAINING_NAME, 
            # Opzionale: Se vuoi cambiare learning rate o altro durante il fine-tuning
            # custom_objects={"learning_rate": 1e-4} 
        )
    else:
        print("Initializing new SAC model...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/" + TRAINING_NAME,
            buffer_size=int(1e6),
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

    print(f"Starting training: {TRAINING_NAME} with {N_ENVS} envs.")

    callback = TerminationStatsCallback(training_name=TRAINING_NAME)

    # 4. Avvio Training
    # reset_num_timesteps=False fa sì che i log continuino da dove erano rimasti (es. step 25M -> 25M+1)
    # Se vuoi ripartire da 0 nei grafici, metti True.
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="./logs/sac_nav_" + TRAINING_NAME,
        callback=callback,
        reset_num_timesteps=False 
    )

    # 5. Salvataggio finale
    save_path = "./checkpoints/" + TRAINING_NAME
    print(f"Saving model to {save_path}")
    model.save(save_path)
    env.save(TRAINING_NAME + "_vecnormalize.pkl")

    print("Training completed.")

if __name__ == "__main__":
    main()
