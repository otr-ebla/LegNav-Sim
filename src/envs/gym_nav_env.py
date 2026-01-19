import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from .nav_env import Simple2DEnv, MAX_LIN_VEL, MAX_ANG_VEL
from src.config import RobotConfig, LidarConfig, SimConfig 

NUM_PEOPLE = 15
NUM_RAYS = LidarConfig.NUM_RAYS
STACK_DIM = RobotConfig.LIDAR_STACK_DIM
MAX_STEPS = SimConfig.MAX_STEPS
NUM_OBSTACLES = 12
PEOPLE_SPEED = 1.0

class GymNavEnv(gym.Env):
    """
    Gymnasium-compatible wrapper con stacking temporale LIDAR.
    
    Observation Space:
        [norm_dist_goal, norm_heading, norm_v, norm_w, 
         lidar_frame_t-2, lidar_frame_t-1, lidar_frame_t]
    
    Tutti i valori sono normalizzati in range definiti.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = NUM_RAYS,
        num_people: int = 0,
        num_obstacles: int = NUM_OBSTACLES,
        max_steps: int = MAX_STEPS,
        stack_dim: int = STACK_DIM,
        reward_factor_progress: float = 5.0,
        use_legs: bool = False,
        distraction_prob: float = 0.0,  
        render_skip: int = 1
    ):  
        super().__init__()

        self.env = Simple2DEnv(
            num_rays=num_rays,
            max_steps=max_steps,
            num_people=num_people,
            room_width=12.0,
            room_height=12.0,
            num_obstacles=num_obstacles,
            reward_factor_progress=reward_factor_progress,
            people_speed=PEOPLE_SPEED,
            use_legs=use_legs,
            human_distraction_prob=distraction_prob,
            render_skip=render_skip,
        )
        self.render_mode = render_mode
        self.num_rays = num_rays
        self.stack_dim = stack_dim
        
        # [MIGLIORATO] Usa deque per efficienza
        self.lidar_stack = deque(maxlen=stack_dim)

        # [MIGLIORATO] Observation space con bounds corretti
        obs_dim = 4 + (self.num_rays * self.stack_dim)
        
        # Bounds per scalari: [dist(0-1), heading(-1,1), v(0-1), w(-1,1)]
        scalar_low = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        scalar_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        # Bounds per LIDAR stack (tutti 0-1)
        lidar_low = np.zeros(self.num_rays * self.stack_dim, dtype=np.float32)
        lidar_high = np.ones(self.num_rays * self.stack_dim, dtype=np.float32)
        
        obs_low = np.concatenate([scalar_low, lidar_low])
        obs_high = np.concatenate([scalar_high, lidar_high])
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # Action space
        self.action_space = spaces.Box(
            low=np.array([0.0, -RobotConfig.MAX_W]), 
            high=np.array([RobotConfig.MAX_LINEAR_VEL, RobotConfig.MAX_W]), 
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        env_obs = self.env.reset()
        
        # Estrai LIDAR corrente
        current_lidar = env_obs[4:]
        
        # [MIGLIORATO] Inizializza stack con deque
        self.lidar_stack.clear()
        for _ in range(self.stack_dim):
            self.lidar_stack.append(current_lidar.copy())
        
        info = {
            "episode_start": True,
            "stack_dim": self.stack_dim,
        }
        
        return self._compose_obs(env_obs), info

    def step(self, action):
        # Converti e clippa azione
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, RobotConfig.MAX_LINEAR_VEL))
        w = float(np.clip(action[1], -RobotConfig.MAX_W, RobotConfig.MAX_W))

        # Step nell'ambiente
        env_obs, reward, done, info = self.env.step((v, w))
        
        # [MIGLIORATO] Aggiorna stack (auto-pop con maxlen)
        current_lidar = env_obs[4:]
        self.lidar_stack.append(current_lidar)
        
        obs = self._compose_obs(env_obs)

        # Gestione terminazione
        terminated = False
        truncated = False
        
        reason = info.get("termination_reason", "unknown")
        if done:
            if reason == "max_steps_reached":
                truncated = True
            else:
                terminated = True

        if self.render_mode == "human":
            self.env.render()

        return obs, reward, terminated, truncated, info

    def _compose_obs(self, env_obs):
        """
        Combina scalari + stack LIDAR.
        
        Returns:
            Array shape: (4 + num_rays * stack_dim,)
        """
        scalars = env_obs[:4]  # [norm_dist, norm_heading, norm_v, norm_w]
        
        # Concatena stack in ordine cronologico [oldest → newest]
        flat_lidar = np.concatenate(list(self.lidar_stack), axis=0)
        
        return np.concatenate([scalars, flat_lidar]).astype(np.float32)
    
    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()