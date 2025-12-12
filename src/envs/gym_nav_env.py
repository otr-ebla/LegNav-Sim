import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .nav_env import Simple2DEnv

# [MODIFICATO] Aggiornati ai limiti reali impostati in nav_env (TurtleBot4)
MAX_LIN_VEL = 0.3   # m/s (era 0.46)
MAX_ANG_VEL = 2.3   # rad/s (era 2.0)
NUM_PEOPLE = 0      # Default
NUM_RAYS = 108
STACK_DIM = 3
MAX_STEPS = 1000

class GymNavEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around Simple2DEnv.
    
    Il nuovo Simple2DEnv restituisce già un'osservazione normalizzata:
    [dist_goal (0-1), angle_goal (-1, 1), v (0-1), w (-1, 1), ...lidar (0-1)...]
    
    Questo wrapper gestisce principalmente:
    1. La conversione delle azioni in float32
    2. Lo stacking temporale del Lidar (se STACK_DIM > 1)
    3. La gestione di Terminated vs Truncated
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = NUM_RAYS,
        num_people: int = NUM_PEOPLE,
        max_steps: int = MAX_STEPS,
        stack_dim: int = STACK_DIM, # Aggiunto parametro esplicito
    ):  
        super().__init__()
    
        self.env = Simple2DEnv(
            num_rays=num_rays,
            max_steps=max_steps,
            num_people=num_people,
            room_width=12.0,
            room_height=12.0,
            # I limiti di velocità sono interni a Simple2DEnv, ma usiamo le costanti qui per l'action space
        )
        self.render_mode = render_mode
        self.num_rays = num_rays
        self.stack_dim = stack_dim
        
        # Buffer per lo stacking dei lidar
        self.lidar_stack = []

        # Calcolo dimensione osservazione:
        # 4 scalari (dist, angle, v, w) + (num_rays * stack_dim)
        obs_dim = 4 + (self.num_rays * self.stack_dim)

        self.observation_space = spaces.Box(
            low=-1.0, # Alcuni valori sono 0-1, altri -1 a 1. Il bound sicuro è -1, 1
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space normalizzato ai limiti del robot
        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANG_VEL]), 
            high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), 
            dtype = np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # 1. Ottieni l'osservazione iniziale dall'environment
        # env_obs è già un np.array: [dist, angle, v, w, l1, l2, ... l108]
        env_obs = self.env.reset() 

        # 2. Inizializza lo stack
        # Estrai la parte lidar (dal 5° elemento in poi, indice 4)
        current_lidar = env_obs[4:] # shape (108,)
        
        if self.stack_dim > 1:
            # Riempi lo stack con copie del primo frame
            self.lidar_stack = [current_lidar.copy() for _ in range(self.stack_dim)]
        else:
            self.lidar_stack = [current_lidar]

        # 3. Costruisci l'osservazione completa
        return self._compose_obs(env_obs), {}

    def step(self, action):
        # Conversione e clipping azione (ridondante col clip interno, ma buona pratica gym)
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, MAX_LIN_VEL))
        w = float(np.clip(action[1], -MAX_ANG_VEL, MAX_ANG_VEL))

        # Esegui step nell'ambiente
        # env_obs è già normalizzato
        env_obs, reward, done, info = self.env.step((v, w))
        
        # Aggiorna lo stack Lidar
        current_lidar = env_obs[4:]
        if self.stack_dim > 1:
            self.lidar_stack.pop(0) # Rimuovi il più vecchio
            self.lidar_stack.append(current_lidar) # Aggiungi il nuovo
        else:
            self.lidar_stack = [current_lidar]

        # Costruisci l'osservazione finale per la rete
        obs = self._compose_obs(env_obs)

        # Gestione Terminated vs Truncated (Standard Gymnasium API)
        terminated = False
        truncated = False
        
        reason = info.get("termination_reason", "unknown")

        if done:
            if reason == "max_steps_reached":
                truncated = True
            else:
                # goal_reached, collision_static, people_collision
                terminated = True 

        if self.render_mode == "human":
            self.env.render()

        return obs, reward, terminated, truncated, info

    def _compose_obs(self, env_obs):
        """
        Combina gli scalari (dist, angle, v, w) con lo stack dei lidar.
        env_obs: [dist, angle, v, w, ...lidar_corrente...]
        """
        # I primi 4 valori sono gli scalari di stato
        scalars = env_obs[:4] 
        
        # Concatena lo stack dei lidar in un unico array piatto
        # Se stack_dim=3 e rays=108, flat_lidar avrà dimensione 324
        flat_lidar = np.concatenate(self.lidar_stack, axis=0)
        
        # Unisci tutto
        return np.concatenate([scalars, flat_lidar]).astype(np.float32)
    
    def render(self):
        self.env.render()   

    def close(self):
        self.env.close()