import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# Assicurati che l'import punti al file corretto dove hai fatto le modifiche
from .nav_env import Simple2DEnv 
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
    
    Se real_lidar_specs=True, la dimensione del lidar aumenta drasticamente (1080 raggi).
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
        render_skip: int = 1,
        lidar_noise_enable: bool = False,
        real_lidar_specs: bool = False,  # <--- [NUOVO PARAMETRO]
    ):  
        super().__init__()

        # --- [LOGICA DIMENSIONALE] ---
        # Se attiviamo il Lidar Reale, dobbiamo ignorare num_rays della config
        # e forzare 1080, altrimenti gym andrà in crash per mismatch di dimensioni.
        if real_lidar_specs:
            self.num_rays = 1080
        else:
            self.num_rays = num_rays

        self.stack_dim = stack_dim
        self.render_mode = render_mode

        # Inizializzazione Ambiente Base
        self.env = Simple2DEnv(
            num_rays=self.num_rays, # Passiamo il numero corretto aggiornato
            max_steps=max_steps,
            num_people=num_people,
            room_width=SimConfig.ROOM_SIDE_LENGTH,
            room_height=SimConfig.ROOM_SIDE_LENGTH,
            num_obstacles=num_obstacles,
            reward_factor_progress=reward_factor_progress,
            people_speed=PEOPLE_SPEED,
            use_legs=use_legs,
            human_distraction_prob=distraction_prob,
            render_skip=render_skip,
            lidar_noise_enable=lidar_noise_enable,
            real_lidar_specs=real_lidar_specs, # <--- Passaggio al motore fisico
        )
        
        # [MIGLIORATO] Usa deque per efficienza
        self.lidar_stack = deque(maxlen=stack_dim)

        # [CALCOLO SPAZIO OSSERVAZIONI]
        # Dimensione totale = 4 scalari + (Raggi * Stack)
        # Esempio Reale: 4 + (1080 * 3) = 3244 float
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
        
        # Nota: Simple2DEnv.reset() ora restituisce solo 'obs' (array numpy)
        # Se hai modificato Simple2DEnv per restituire (obs, info), adatta questa riga.
        # Basandomi sull'ultimo codice che mi hai dato, restituiva solo obs.
        env_obs = self.env.reset()
        
        # Estrai LIDAR corrente (gli ultimi N elementi sono il lidar)
        # env_obs è: [dist, head, v, w, ...lidar...]
        current_lidar = env_obs[4:]
        
        # Check di sicurezza per evitare bug silenziosi sulle dimensioni
        if len(current_lidar) != self.num_rays:
            raise ValueError(f"Lidar dimension mismatch! Expected {self.num_rays}, got {len(current_lidar)}. Check real_lidar_specs flag.")

        # Inizializza stack con deque
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
        # Simple2DEnv.step restituisce: obs, reward, done, info
        env_obs, reward, done, info = self.env.step((v, w))
        
        # Aggiorna stack (auto-pop con maxlen)
        current_lidar = env_obs[4:]
        self.lidar_stack.append(current_lidar)
        
        obs = self._compose_obs(env_obs)

        # Gestione terminazione (Gymnasium API: terminated vs truncated)
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
        Returns: Array shape: (4 + num_rays * stack_dim,)
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