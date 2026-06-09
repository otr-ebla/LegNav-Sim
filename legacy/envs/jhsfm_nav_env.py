import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import jax
import jax.numpy as jnp
import random

# Import standard config and the Fast Numba Engine
from src.config import RobotConfig, LidarConfig, SimConfig
from .fast_nav_env import Simple2DEnv, fast_update_legs_batch

# Import JAX HSFM components
try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    from jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

# Import Scenarios
from src.envs.scenarios import Scenarios

class JhsfmGymNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = LidarConfig.NUM_RAYS,
        num_people: int = SimConfig.NUM_HUMANS,
        scenario_type: str = "random",
        max_steps: int = SimConfig.MAX_STEPS,
        use_legs: bool = False,
        render_skip: int = 1,
        lidar_noise_enable: bool = False,
        real_lidar_specs: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.num_rays = num_rays
        self.scenario_type = scenario_type
        self.use_legs = use_legs
        self.num_people = num_people
        
        # 1. Initialize the Fast Numba Environment (Background Engine)
        self.env = Simple2DEnv(
            max_steps=max_steps,
            dt=RobotConfig.DT,
            room_width=SimConfig.ROOM_SIDE_LENGTH,
            room_height=SimConfig.ROOM_SIDE_LENGTH,
            robot_radius=RobotConfig.RADIUS,
            num_rays=num_rays,
            max_lidar_distance=LidarConfig.MAX_DISTANCE,
            num_people=num_people,
            people_radius=SimConfig.HUMANS_RADIUS,
            people_speed=0.0, 
            reward_factor_progress=0.0, 
            num_obstacles=0, 
            render_skip=render_skip,
            use_legs=use_legs,
            human_distraction_prob=0.0,
            lidar_noise_enable=lidar_noise_enable,
            real_lidar_specs=real_lidar_specs
        )

        # 2. JAX / HSFM Setup
        self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        self.hsfm_step_fn = jax.jit(hsfm_step)
        
        # JAX State Placeholders
        self.humans_state_jax = None
        self.humans_goal_jax = None
        self.static_obstacles_jax = None
        self.humans_goals_mem_jax = None
        self.humans_goal_indices = None
        
        self.smooth_v = np.zeros(self.num_people, dtype=np.float32)

        # 3. Observation Space
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32),
            "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

        self.action_space = spaces.Box(
            low=np.array([0.0, -RobotConfig.MAX_W]),
            high=np.array([RobotConfig.MAX_LINEAR_VEL, RobotConfig.MAX_W]),
            dtype=np.float32,
        )

    def _sync_scenarios_to_fast_env(self, w, h, obstacles_list):
        """Invia gli ostacoli al motore Numba per Lidar e Collisioni."""
        self.env.room_width = w
        self.env.room_height = h
        self.env.num_active_obstacles = 0
        self.env.obstacles = [] 
        
        for obs in obstacles_list:
            if self.env.num_active_obstacles >= self.env.max_obstacles: break
            idx = self.env.num_active_obstacles
            
            if obs["type"] == "circle":
                self.env.obstacles_arr[idx] = [0.0, obs["cx"], obs["cy"], obs["radius"], 0.0]
            elif obs["type"] == "rect":
                self.env.obstacles_arr[idx] = [1.0, obs["xmin"], obs["xmax"], obs["ymin"], obs["ymax"]]
            
            self.env.obstacles.append(obs)
            self.env.num_active_obstacles += 1

    def _convert_all_obs_to_jax(self, w, h, obs_list):
        """
        Converte MURI + OSTACOLI RETTANGOLARI in segmenti per HSFM.
        Formato richiesto: (N_Humans, N_Obstacles, N_Edges, 2_Points, 2_Coords)
        Gestisce il padding con NaN per evitare crash di dimensionalità.
        """
        # Lista di tutti gli ostacoli. Ogni elemento è una lista di edges [p1, p2].
        # 1. Il Muro esterno è un unico ostacolo fatto da 4 edges
        room_edges = [
            [[0, 0], [w, 0]], # Bottom
            [[w, 0], [w, h]], # Right
            [[w, h], [0, h]], # Top
            [[0, h], [0, 0]]  # Left
        ]
        
        all_obstacles = [room_edges] # Inizia con la stanza

        # 2. Aggiungi gli ostacoli dalla lista (rettangoli)
        for obs in obs_list:
            if obs["type"] == "rect":
                xmin, xmax, ymin, ymax = obs["xmin"], obs["xmax"], obs["ymin"], obs["ymax"]
                # Un rettangolo ha 4 edges
                rect_edges = [
                    [[xmin, ymin], [xmax, ymin]], # Bottom
                    [[xmax, ymin], [xmax, ymax]], # Right
                    [[xmax, ymax], [xmin, ymax]], # Top
                    [[xmin, ymax], [xmin, ymin]]  # Left
                ]
                all_obstacles.append(rect_edges)
        
        # 3. Creazione Tensore con Padding (NaN)
        # HSFM vuole una forma fissa (N_Obs, Max_Edges, 2, 2)
        # Qui Max_Edges è 4 (perché abbiamo solo rettangoli e muri).
        max_edges = 4 
        num_total_obs = len(all_obstacles)
        
        # Inizializza con NaN (Dummy edges ignorati da HSFM)
        obs_array = np.full((num_total_obs, max_edges, 2, 2), np.nan, dtype=np.float32)
        
        for i, obstacle_edges in enumerate(all_obstacles):
            for k, edge in enumerate(obstacle_edges):
                if k < max_edges:
                    obs_array[i, k] = edge

        # Converti in JAX e Tile per ogni agente
        # Shape finale: (N_Humans+1, N_Obs, Max_Edges, 2, 2)
        n_agents = self.num_people + 1 
        jax_obs = jnp.array(obs_array, dtype=jnp.float32)
        return jnp.tile(jax_obs[None, ...], (n_agents, 1, 1, 1, 1))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 1. RESET BASE (Pulizia)
        self.env.reset() 

        # 2. Setup Scenario
        w, h = SimConfig.ROOM_SIDE_LENGTH, SimConfig.ROOM_SIDE_LENGTH
        if self.scenario_type == "parallel": w, h = 4.0, 14.0
        elif self.scenario_type == "perpendicular": w, h = 8.0, 9.0
        elif self.scenario_type == "intersection": w, h = 10.0, 10.0

        if self.scenario_type == "random":
            data = Scenarios.random_static(w, h, self.num_people)
            if len(data) == 5:
                rob_start, rob_goal, states, raw_goals, obs_list = data
            else:
                rob_start, rob_goal, states, raw_goals = data
                obs_list = [] 

        elif self.scenario_type == "bottleneck":
            gap_x = random.uniform(2.0, w - 2.0)
            rob_start, rob_goal, states, raw_goals = Scenarios.bottleneck(w, h, self.num_people, gap_x)
            wall_y = h / 2; gap_size = 1.8
            obs_list = [
                {"type": "rect", "xmin": 0, "xmax": gap_x - gap_size/2, "ymin": wall_y - 0.2, "ymax": wall_y + 0.2},
                {"type": "rect", "xmin": gap_x + gap_size/2, "xmax": w, "ymin": wall_y - 0.2, "ymax": wall_y + 0.2}
            ]

        elif self.scenario_type == "parallel":
            rob_start, rob_goal, states, raw_goals = Scenarios.parallel_traffic(w, h, self.num_people)
            obs_list = []
        elif self.scenario_type == "intersection":
            rob_start, rob_goal, states, raw_goals = Scenarios.intersection(w, h, self.num_people)
            obs_list = []
        else:
            data = Scenarios.random_static(w, h, self.num_people)
            rob_start, rob_goal, states, raw_goals = data[:4]
            obs_list = []

        # 3. Sync FastEnv (LiDAR Robot)
        self._sync_scenarios_to_fast_env(w, h, obs_list)
        
        # Setup Robot
        self.env.x, self.env.y, self.env.theta = rob_start[0], rob_start[1], rob_start[2]
        self.env.goal_x, self.env.goal_y = rob_goal[0], rob_goal[1]
        self.env.room_width = w; self.env.room_height = h
        self.env.max_possible_dist = math.hypot(w, h)
        self.env.prev_dist_to_goal = math.hypot(self.env.x - self.env.goal_x, self.env.y - self.env.goal_y)

        # 4. Sync JAX (Human Physics)
        self.humans_state_jax = jnp.array(states, dtype=jnp.float32)
        
        goals_mem_np = np.zeros((self.num_people, 3, 2), dtype=np.float32)
        indices = []
        for i, g_data in enumerate(raw_goals):
            if isinstance(g_data[0], (list, tuple, np.ndarray)):
                count = len(g_data)
                for k in range(3): goals_mem_np[i, k] = g_data[min(k, count - 1)]
                indices.append(0)
            else:
                goals_mem_np[i, :] = g_data
                indices.append(0)
                
        self.humans_goals_mem_jax = jnp.array(goals_mem_np)
        self.humans_goal_indices = jnp.array(indices, dtype=jnp.int32)
        self.humans_goal_jax = self.humans_goals_mem_jax[:, 0, :]
        
        # [FIX] Converti TUTTI gli ostacoli (muri + rettangoli) per JAX
        self.static_obstacles_jax = self._convert_all_obs_to_jax(w, h, obs_list)

        self._sync_jax_to_fast()
        self.env.lidar_readings = self.env._compute_lidar_fast()
        
        return self._process_obs(0.0, 0.0), {}

    def _sync_jax_to_fast(self):
        states = np.array(self.humans_state_jax)
        for i in range(self.num_people):
            self.env.people_arr[i, 0] = states[i, 0]
            self.env.people_arr[i, 1] = states[i, 1]
            self.env.people_arr[i, 2] = states[i, 2]
            self.env.people_arr[i, 3] = states[i, 3]
            self.env.people_arr[i, 4] = states[i, 4]
            v_mag = math.hypot(states[i, 2], states[i, 3])
            self.smooth_v[i] = 0.8 * self.smooth_v[i] + 0.2 * v_mag

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, RobotConfig.MAX_LINEAR_VEL))
        w = float(np.clip(action[1], -RobotConfig.MAX_W, RobotConfig.MAX_W))

        v = min(v, self.env.max_v)

        dt = self.env.dt
        self.env.theta = (self.env.theta + w * dt + math.pi) % (2 * math.pi) - math.pi
        self.env.x += v * dt * math.cos(self.env.theta)
        self.env.y += v * dt * math.sin(self.env.theta)
        
        self.env.v = v; self.env.w = w
        self.env.step_count += 1

        n_substeps = int(dt / SimConfig.HSFM_DT)
        robot_state = jnp.array([self.env.x, self.env.y, v * math.cos(self.env.theta), v * math.sin(self.env.theta), self.env.theta, w])
        robot_goal = jnp.array([self.env.goal_x, self.env.goal_y])
        
        curr_params = self.hsfm_params
        
        for _ in range(n_substeps):
            humans_state_ext = jnp.concatenate([self.humans_state_jax, robot_state[None, :]], axis=0)
            goals_ext = jnp.concatenate([self.humans_goal_jax, robot_goal[None, :]], axis=0)
            next_state_ext = self.hsfm_step_fn(humans_state_ext, goals_ext, curr_params, self.static_obstacles_jax, SimConfig.HSFM_DT)
            self.humans_state_jax = next_state_ext[:-1]

        # [FIX HARD COLLISION] Clamping delle posizioni all'interno della stanza
        # Questo impedisce agli umani di uscire dai muri anche se la forza repulsiva fallisce
        margin = 0.1
        w, h = self.env.room_width, self.env.room_height
        self.humans_state_jax = self.humans_state_jax.at[:, 0].set(
            jnp.clip(self.humans_state_jax[:, 0], margin, w - margin)
        )
        self.humans_state_jax = self.humans_state_jax.at[:, 1].set(
            jnp.clip(self.humans_state_jax[:, 1], margin, h - margin)
        )

        states_np = np.array(self.humans_state_jax)
        goals_np = np.array(self.humans_goal_jax)
        indices_np = np.array(self.humans_goal_indices)
        all_goals_np = np.array(self.humans_goals_mem_jax)
        
        for i in range(self.num_people):
            if math.hypot(states_np[i,0]-goals_np[i,0], states_np[i,1]-goals_np[i,1]) < 0.5:
                if indices_np[i] < 2: indices_np[i] += 1
        
        self.humans_goal_indices = jnp.array(indices_np)
        self.humans_goal_jax = jnp.array([all_goals_np[i, indices_np[i]] for i in range(self.num_people)])

        self._sync_jax_to_fast()
        if self.use_legs: fast_update_legs_batch(self.env.people_arr, self.env.legs_coords, self.env.humans_leg_phase, self.num_people, dt)
        
        self.env.lidar_readings = self.env._compute_lidar_fast()
        
        terminated = False; truncated = False; reward = 0.0; info = {}
        from .fast_nav_env import fast_check_static_collision, fast_scan_closest_human
        
        if fast_check_static_collision(self.env.x, self.env.y, self.env.robot_radius - 0.02, self.env.obstacles_arr, self.env.num_active_obstacles, self.env.room_width, self.env.room_height):
            terminated = True; reward = -200.0; info["termination_reason"] = "collision_static"
        
        closest_dist, _ = fast_scan_closest_human(self.env.people_arr, self.num_people, self.env.x, self.env.y, self.env.theta)
        if closest_dist < (self.env.robot_radius + self.env.people_radius):
            terminated = True; reward = -200.0; info["termination_reason"] = "collision_people"
            
        dist_to_goal = math.hypot(self.env.x - self.env.goal_x, self.env.y - self.env.goal_y)
        if dist_to_goal < 0.3:
            terminated = True; reward = 200.0; info["termination_reason"] = "goal_reached"
        elif self.env.step_count >= self.env.max_steps:
            truncated = True; info["termination_reason"] = "max_steps_reached"

        if not terminated:
            reward = 5.0 * (self.env.prev_dist_to_goal - dist_to_goal) - 0.01
        self.env.prev_dist_to_goal = dist_to_goal
        
        return self._process_obs(v, w), reward, terminated, truncated, info

    def _process_obs(self, v, w):
        lidar = self.env.lidar_readings
        inv_lidar = np.clip((self.env.max_lidar_distance - lidar) / (self.env.max_lidar_distance - 0.12), 0.0, 1.0).astype(np.float32)
        pose_vec = np.array([self.env.x/SimConfig.ROOM_SIDE_LENGTH, self.env.y/SimConfig.ROOM_SIDE_LENGTH, self.env.theta/math.pi], dtype=np.float32)
        dx = self.env.goal_x - self.env.x; dy = self.env.goal_y - self.env.y
        alignment = (math.atan2(dy, dx) - self.env.theta + math.pi) % (2 * math.pi) - math.pi
        state_vec = np.array([v, w, self.env.max_v, math.hypot(dx, dy), alignment], dtype=np.float32)
        return {"lidar": inv_lidar, "pose": pose_vec, "state": state_vec}

    def render(self): self.env.render()
    def close(self): self.env.close()