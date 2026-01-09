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
NUM_OBSTACLES = 10 # NOOOOOOO, questo non si cambia da qui ma si cambia in gym_nav_env.py!!!!!!
N_ENVS = 100
PEOPLE_SPEED = 0.7
STACK_DIM = 3

TRAINING_NAME = "2_Mno_obstacles_part2"
TRAINING_STEPS = 7_000_000



LOAD_MODEL = True  # Set to True to load existing model
PREV_MODEL_PATH = "checkpoints/2_Mno_obstacles.zip"
# Assicurati che questo percorso sia corretto (usa il percorso assoluto se serve)
PREV_ENV_STATS_PATH = "2_Mno_obstacles_vecnormalize.pkl"



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


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LidarAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Branch LiDAR: 1 canale input (distanza), esce con 32 canali
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        # Self-Attention: permette di pesare l'importanza dei vari segmenti lidar
        # embed_dim=64 deve corrispondere all'output della seconda Conv1d
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        # Branch Scalari: processa (DistGoal, AngGoal, V, W)
        self.scalar_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        
        # Fusione finale: 64 (feat lidar) * 27 (lunghezza seq) + 32 (scalari)
        self.final_fc = nn.Linear(64 * 27 + 32, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # I primi 4 valori sono scalari (Goal + Vel), il resto è Lidar
        scalars = observations[:, :4]
        lidar_raw = observations[:, 4:]
        
        # Reshape Lidar per la CNN: [Batch, Channel=1, Length=108]
        lidar = lidar_raw.unsqueeze(1)
        # Estrazione feature geometriche
        cnn_out = self.cnn(lidar) # Output: [Batch, 64, 27]
        
        # Permute per Attention: serve [Batch, Sequence, Features] -> [B, 27, 64]
        cnn_permuted = cnn_out.permute(0, 2, 1)
        
        # Applica Self-Attention (query, key, value sono lo stesso tensore)
        attn_out, _ = self.attn(cnn_permuted, cnn_permuted, cnn_permuted)
        # Flatten dell'output attenzione: da [B, 27, 64] a [B, 1728]
        lidar_flat = attn_out.reshape(attn_out.size(0), -1)
        
        # Processamento scalari e concatenazione
        scalar_feat = self.scalar_net(scalars)
        combined = torch.cat((lidar_flat, scalar_feat), dim=1)
        
        # Output finale verso l'Actor/Critic
        return self.final_fc(combined)







































def make_env(rank: int):
    def _init():
        env = GymNavEnv(
            render_mode=None,
            num_rays=NUM_RAYS,
            num_people=NUM_PEOPLE,
            num_obstacles=NUM_OBSTACLES,
        )
        env = Monitor(env)
        return env
    return _init

def main():
    use_subproc = True
    total_timesteps = TRAINING_STEPS  # O quanti ne vuoi fare in più

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
