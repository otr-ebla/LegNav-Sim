import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .nav_env import Simple2DEnv
#from scripts.train_ppo import NUM_PEOPLE, NUM_RAYS, NUM_PEOPLE, PEOPLE_SPEED, STACK_DIM

MAX_LIN_VEL = 0.46  # m/s
MAX_ANG_VEL = 2.0  # rad/s
NUM_PEOPLE = 10
NUM_RAYS = 108
STACK_DIM = 3
PEOPLE_SPEED = 0.7
MAX_STEPS = 400


class GymNavEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around Simple2DEnv.
    Observation:
      - x, y (normalized to [-1, 1])
      - cos(theta), sin(theta)
      - goal_dx, goal_dy (relative goal, normalized to [-1, 1])
      - lidar distances normalized to [0, 1]
    Action:
      - [v, w] in [0, 1] x [-1, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = NUM_RAYS,
        num_people: int = NUM_PEOPLE,
        people_speed: float = PEOPLE_SPEED,
        max_steps: int = MAX_STEPS,
    ):  
        super().__init__()
    
        # Create internal simulator
        self.env = Simple2DEnv(
            num_rays=num_rays,
            max_steps=max_steps,
            num_people=num_people,
            people_speed=people_speed,
            room_height=12.0,
            room_width=12.0,
        )
        self.render_mode = render_mode

        self.num_rays = num_rays
        self.num_people = num_people

        self.stack_dim = STACK_DIM
        # Pre-allocate a fixed-size flattened lidar buffer (num_rays * stack_dim,)
        obs_dim = 4 + self.num_rays*self.stack_dim

        self.lidar_stack = None

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANG_VEL]), 
            high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), 
            dtype = np.float32,
        )

    # ... dentro gym_nav_env.py ...

    def _get_obs(self, env_obs=None) -> np.ndarray:
        # Se env_obs è None (caso raro/fallback), prendiamo i valori dall'oggetto env
        if env_obs is None:
            # Fallback manuale
            x, y, theta = self.env.x, self.env.y, self.env.theta
            last_v, last_w = self.env.last_v, self.env.last_w
            lidar_distances = self.env._compute_lidar()
        else:
            # Unpack dal tuple restituito da step/reset
            # (x, y, theta, last_v, last_w, lidar)
            x = env_obs[0]
            y = env_obs[1]
            theta = env_obs[2]
            last_v = env_obs[3]
            last_w = env_obs[4]
            lidar_distances = env_obs[5]

        # --- 1. Goal Info ---
        goal_dx = self.env.goal_x - x
        goal_dy = self.env.goal_y - y
        dist_to_goal = np.sqrt(goal_dx**2 + goal_dy**2)
        max_room_dist = np.sqrt(self.env.room_width**2 + self.env.room_height**2)
        
        rho_norm = np.clip(dist_to_goal / max_room_dist, 0.0, 1.0)

        theta_goal = np.arctan2(goal_dy, goal_dx)
        angle_diff = theta_goal - theta
        wrapped_angle = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        alpha_norm = wrapped_angle / np.pi # Normalizzato tra -1 e 1

        # --- 2. Velocity Info (NUOVO) ---
        # Normalizziamo le velocità precedenti per aiutare la rete neurale
        last_v_norm = np.clip(last_v / MAX_LIN_VEL, 0.0, 1.0)
        last_w_norm = np.clip(last_w / MAX_ANG_VEL, -1.0, 1.0)

        # --- 3. Lidar Stack ---
        lidar_arr = np.array(lidar_distances, dtype=np.float32)
        lidar_norm = np.clip(lidar_arr / self.env.max_lidar_distance, 0.0, 1.0)

        if self.stack_dim > 1:
            if self.lidar_stack is None:
                self.lidar_stack = [lidar_norm.copy() for _ in range(self.stack_dim)]
            else:
                self.lidar_stack.pop(0)
                self.lidar_stack.append(lidar_norm.copy())
            lidar_input = np.concatenate(self.lidar_stack, axis=0)
        else:
            lidar_input = lidar_norm

        # --- 4. Costruzione Osservazione Finale ---
        # Aggiungiamo last_v_norm e last_w_norm al vettore
        obs = np.concatenate([
            [rho_norm, alpha_norm, last_v_norm, last_w_norm], 
            lidar_input,
        ]).astype(np.float32)

        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, MAX_LIN_VEL))
        w = float(np.clip(action[1], -MAX_ANG_VEL, MAX_ANG_VEL))

        # Recuperiamo l'osservazione raw dall'env per evitare di ricalcolarla
        raw_obs, reward, done, info = self.env.step((v, w))
        
        # Passiamo raw_obs a _get_obs
        obs = self._get_obs(env_obs=raw_obs)

        terminated = False
        truncated = False # Inizia False
        
        # 3. FIX TYPO KEY
        reason = info.get("termination_reason", None) # Era "terminated_reason"

        if done:
            if reason == "max_steps_reached":
                truncated = True
            else:
                terminated = True # Collisione o Goal

        if self.render_mode == "human":
            self.env.render()

        return obs, reward, terminated, truncated, info
    

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # 1. Cattura l'osservazione iniziale dall'ambiente
        raw_obs = self.env.reset() 

        # 2. Usa i lidar già calcolati per riempire lo stack iniziale
        if self.stack_dim > 1:
            # raw_obs è (x, y, theta, last_v, last_w, lidar) -> lidar è indice 5
            lidar = np.array(raw_obs[5], dtype=np.float32) 
            lidar_norm = np.clip(lidar / self.env.max_lidar_distance, 0.0, 1.0)
            
            # Riempi lo stack con copie dello stesso frame iniziale
            self.lidar_stack = [lidar_norm.copy() for _ in range(self.stack_dim)]   

        # 3. Passa raw_obs a _get_obs
        obs = self._get_obs(env_obs=raw_obs)
        
        info = {}
        return obs, info
    
    def render(self):
        if self.render_mode == "human":
            self.env.render()   

    def close(self):
        self.env.close()



 