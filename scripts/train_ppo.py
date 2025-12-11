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
NUM_PEOPLE = 25
N_ENVS = 64
PEOPLE_SPEED = 0.0
TRAINING_NAME = "25MSAC_jack"
LOAD_MODEL = None  # Set to True to load existing model
STACK_DIM = 3


PREV_MODEL = None
PREV_ENV_STATS = None



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
            people_speed=PEOPLE_SPEED,  
        )
        env = Monitor(env)
        return env
    return _init

def main():
    use_subproc = True

    training_name = TRAINING_NAME
    total_timesteps = 25_000_000

    if use_subproc:
        env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)  # Normalizza osservazioni e reward

    # model = TQC(
    #    "MlpPolicy",
    #    env,
    #    #device=("cuda" if torch.cuda.is_available() else "cpu"),
    #    device="cpu",   
    #    verbose=1,
    #    tensorboard_log="./logs/" + training_name,  # Updated path
    #    buffer_size=int(1e5),   
    #    learning_rate=3e-4,
    #    learning_starts=20_000,
    #    batch_size=512,
    #    gamma=0.99,
    #    tau=0.005,
    #    train_freq=1,
    #    gradient_steps=3,
    #    target_entropy=-2,
    #    target_update_interval=1,
    # )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=-2,
        )

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     device="cpu",
    #     verbose=1,
    #     tensorboard_log="./logs/ppo_nav",  # Updated path
    #     learning_rate=1e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     target_kl=0.03,
    #     policy_kwargs=policy_kwargs,
    # )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/" + training_name,  # Updated path
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

    print(f"Training SAC with {N_ENVS} parallel envs, total_timesteps={total_timesteps}.")

    callback = TerminationStatsCallback(training_name=training_name)

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="./logs/sac_nav" + training_name,
        callback=callback,
    )

    # Updated save path to checkpoints folder
    model.save("./checkpoints/" + training_name)
    env.save(training_name + "_vecnormalize.pkl")

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
