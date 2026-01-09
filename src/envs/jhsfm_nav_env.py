import math
import random
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces
import gymnasium as gym

from collections import deque
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# Configurazione
from src.config import RobotConfig, LidarConfig, SimConfig

# Importiamo la classe padre per riutilizzare render e matematica
from src.envs.nav_env import Simple2DEnv
from src.envs.scenarios import Scenarios

# --- IMPORT JAX/HSFM ---
try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    import sys
    print("Warning: Import JHSFM standard fallito, tentativo path relativo...")
    from jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

class SimpleNavEnv(Simple2DEnv, gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # Aggiorna metadata

    def __init__(self, 
                num_people=None, 
                scenario_type="random",
                allow_keyboard_skip=False,
                training=True,
                use_legs=True,
                render_mode=None,
                force_static=False):
        # 1. Configurazione Scenario
        self.base_scenario_type = scenario_type
        self.scenario_type = scenario_type
        self.force_static = force_static

        self.training = training
        self.allow_keyboard_skip = allow_keyboard_skip
        self.use_legs = use_legs

        self.render_mode = render_mode

        # 2. Configurazione Numero Persone
        if self.scenario_type == "bottleneck":
            self.num_people = 5
        elif self.scenario_type == "intersection": # <--- NUOVO
            self.num_people = 9
        else:
            self.num_people = num_people if num_people is not None else SimConfig.NUM_HUMANS

        # 3. Configurazione Velocità (Solo per logica padre)
        if self.force_static:
            self.current_people_speed = 0.0
        elif self.scenario_type == "static_groups":
            self.current_people_speed = 0.0
        else:
            self.current_people_speed = SimConfig.HUMANS_VELOCITY

        # 4. Dimensioni Stanza
        if self.scenario_type == "parallel":
            eff_width, eff_height = 4.0, 14.0 
        elif self.scenario_type == "perpendicular":
            eff_width, eff_height = 8.0, 9.0 
        elif self.scenario_type == "circular":
            eff_width, eff_height = 12.0, 12.0 
        elif self.scenario_type in ["bottleneck", "static_groups", "intersection"]:
            eff_width, eff_height = 10.0, 10.0
        else:
            eff_width, eff_height = SimConfig.ROOM_SIZE[0], SimConfig.ROOM_SIZE[1]

        # 5. Init Padre
        super().__init__(
            max_steps=SimConfig.MAX_STEPS,
            dt=RobotConfig.DT,
            room_width=eff_width,
            room_height=eff_height,
            robot_radius=RobotConfig.RADIUS,
            num_rays=LidarConfig.NUM_RAYS,
            max_lidar_distance=LidarConfig.MAX_DISTANCE,
            num_people=self.num_people,
            people_radius=SimConfig.HUMANS_RADIUS, 
            people_speed=self.current_people_speed, # Passiamo la velocità (utile per debug/render)
        )
        
        # Override parametri fisici robot
        self.max_v = RobotConfig.MAX_LINEAR_VEL
        self.max_w = RobotConfig.MAX_W
        self.lidar_min_distance = LidarConfig.MIN_DIST
        self.lidar_offset = RobotConfig.LIDAR_OFFSET

        # 6. SETUP HSFM (Motore Fisico)
        # Genera i parametri base leggendo dal Config globale
        self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        
        # [FIX CRUCIALE] Sovrascrivi velocità JAX se siamo in static_groups
        if self.scenario_type == "static_groups":
            # Indice 1 dei parametri HSFM è la Desired Velocity (v0).
            # Lo impostiamo a 0.0 per tutti gli umani ([:-1]), escluso il robot.
            self.hsfm_params = self.hsfm_params.at[:-1, 1].set(0.2)

        self.hsfm_step_fn = jax.jit(hsfm_step)
        
        # Setup Stato
        self.humans_state_jax = None 
        self.humans_goal_jax = None
        self.static_obstacles_jax = None
        self.humans_goals_mem_jax = None
        self.humans_goal_indices = None
        self.last_human_pos = None

        self.smooth_v = None
        self.humans_leg_phase = None

        self.n_obstacles = SimConfig.NUM_OBSTACLES  
        self.n_stack = RobotConfig.LIDAR_STACK_DIM
        self.lidar_stack = deque(maxlen=self.n_stack)
        
        # Spazi Gym
        obs_dim = 4 + self.num_rays * self.n_stack
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_v, -self.max_w]), 
            high=np.array([self.max_v, self.max_w]), 
            dtype=np.float32
        )

        self.manual_skip_triggered = False
        self._listener_attached = False

    def _get_room_size(self, s_type):
        """Helper per ottenere dimensioni in base al tipo."""
        if s_type == "parallel": return 4.0, 14.0 
        elif s_type == "perpendicular": return 8.0, 9.0 
        elif s_type == "circular": return 12.0, 12.0 
        elif s_type in ["bottleneck", "static_groups", "intersection"]: return 10.0, 10.0
        else: return SimConfig.ROOM_SIZE[0], SimConfig.ROOM_SIZE[1]

    def reset(self, seed=None, options=None):
        self.manual_skip_triggered = False
        self.step_count = 0
        self.episode_path_length = 0.0
        self.trajectory = []
        self.last_v, self.last_w = 0.0, 0.0
        self.last_termination_reason = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1. LOGICA MIXED: Scegliamo uno scenario a caso
        if self.base_scenario_type == "mixed":
            candidates = ["parallel", "perpendicular", "intersection", "bottleneck", "circular", "random"]
            self.scenario_type = random.choice(candidates)
            # Aggiorniamo dimensioni stanza dinamicamente
            self.room_width, self.room_height = self._get_room_size(self.scenario_type)
            # Bottleneck e Intersection richiedono num_people specifico
            if self.scenario_type == "bottleneck": self.num_people = 5
            elif self.scenario_type == "intersection": self.num_people = 9
            else: self.num_people = SimConfig.NUM_HUMANS
            
            # Re-inizializziamo parametri HSFM se il numero è cambiato
            if len(self.hsfm_params) != self.num_people + 1:
                self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        else:
            self.scenario_type = self.base_scenario_type

        # 2. Reset Variabili
        self.smooth_v = [0.0] * self.num_people
        self.humans_leg_phase = [0.0] * self.num_people

        self._generate_static_obstacles()
        self._setup_scenario()
        self.static_obstacles_jax = self._convert_obs_to_jax()

        # 3. LOGICA FORCE STATIC (Curriculum Phase 1)
        # Resetta ai parametri standard
        self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        
        if self.force_static:
            # Forza velocità a 0.0 per tutti gli umani
            self.hsfm_params = self.hsfm_params.at[:-1, 1].set(0.0)
        elif self.scenario_type == "static_groups":
            self.hsfm_params = self.hsfm_params.at[:-1, 1].set(0.0)

        self.start_x, self.start_y = self.x, self.y
        self.trajectory.append((self.x, self.y))
        
        current_states = np.array(self.humans_state_jax)
        self.last_human_pos = current_states[:, :2]

        return self._get_obs(reset_stack=True), {}
    




    def _get_obs(self, reset_stack=False):
        # 1. Calcoli scalari
        dist = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        angle = (math.atan2(self.goal_y - self.y, self.goal_x - self.x) - self.theta + math.pi) % (2 * math.pi) - math.pi
        norm_angle = angle / math.pi # Normalizziamo tra -1 e 1 come nav_env.py
        
        # 2. Calcolo Lidar Inverso (Logic from nav_env.py)
        raw_lidar = self._compute_lidar()
        lidar_processed = []
        for d in raw_lidar:
            if d >= self.max_lidar_distance:
                lidar_processed.append(0.0)
            else:
                # Inversa cubica come nel tuo nav_env
                inv = (self.lidar_min_distance / d) ** (1/3)
                lidar_processed.append(min(inv, 1.0))
        
        lidar_array = np.array(lidar_processed, dtype=np.float32)

        # 3. Gestione Stacking
        if reset_stack:
            # Riempie la coda con lo stesso frame iniziale
            self.lidar_stack.clear()
            for _ in range(self.n_stack):
                self.lidar_stack.append(lidar_array)
        else:
            # Aggiunge nuovo frame, il più vecchio viene rimosso automaticamente
            self.lidar_stack.append(lidar_array)
            
        # 4. Concatenazione finale
        # [Inv_Dist, Angle, V, W, Lidar_t, Lidar_t-1, Lidar_t-2]
        scalars = np.array([dist, norm_angle, self.last_v, self.last_w], dtype=np.float32)
        stacked_lidar = np.concatenate(list(self.lidar_stack))
        
        return np.concatenate([scalars, stacked_lidar])

    def step(self, action):
        # 1. Unpack & Clamp Action (Gymnasium passes a numpy array)
        v_cmd, w_cmd = float(action[0]), float(action[1])

        v_cmd, w_cmd = self._apply_differential_drive_constraints(v_cmd, w_cmd)

        v = np.clip(v_cmd, -self.max_v, self.max_v)
        w = np.clip(w_cmd, -self.max_w, self.max_w)
        
        # 2. Physics & Logic (Robot Kinematics + HSFM Humans)
        dt = self.dt
        prev_x, prev_y = self.x, self.y
        dist_before = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        
        # --- HSFM Update (Humans) ---
        # --- HSFM Update (Humans) ---
        n_substeps = int(dt / SimConfig.HSFM_DT)
        
        # Prepara dati Robot (servono sempre per aggiornare la fisica del robot dopo)
        rob_state = jnp.array([self.x, self.y, v*math.cos(self.theta), v*math.sin(self.theta), self.theta, w])
        rob_goal  = jnp.array([self.goal_x, self.goal_y])

        # LOGICA CONDIZIONALE: Robot Visibile o Invisibile
        if not self.training:
            # --- TESTING / VISUALIZATION (Robot Visibile) ---
            # Il robot viene aggiunto alla simulazione: gli umani lo evitano.
            c_states = jnp.concatenate([self.humans_state_jax, rob_state[None, :]], axis=0)
            c_goals  = jnp.concatenate([self.humans_goal_jax, rob_goal[None, :]], axis=0)
            
            # Usiamo tutti i parametri (N umani + 1 Robot)
            c_params = self.hsfm_params 
            # Usiamo tutti gli ostacoli (N + 1)
            c_obs    = self.static_obstacles_jax 
        else:
            # --- TRAINING (Robot Invisibile / Ghost) ---
            # Gli umani ignorano il robot. Simuliamo solo N agenti.
            c_states = self.humans_state_jax
            c_goals  = self.humans_goal_jax
            
            # Slice dei parametri: prendiamo solo i primi N (escludiamo l'ultimo che è il robot)
            c_params = self.hsfm_params[:-1]
            # Slice degli ostacoli: prendiamo solo i primi N layer
            c_obs    = self.static_obstacles_jax[:-1]

        # Esegui step HSFM
        for _ in range(n_substeps):
            c_states = self.hsfm_step_fn(c_states, c_goals, c_params, c_obs, SimConfig.HSFM_DT)
            
        # Aggiorna lo stato degli umani
        if not self.training:
            self.humans_state_jax = c_states[:-1] # Rimuovi robot dal risultato
        else:
            self.humans_state_jax = c_states      # Risultato contiene solo umani

        self._sync_people_list()

        # --- Robot Update (Kinematics) ---
        self.theta += w * dt
        self.x += v * dt * math.cos(self.theta)
        self.y += v * dt * math.sin(self.theta)
        self.last_v, self.last_w = v, w
        
        # --- Collision Handling ---
        if self._is_collision_with_walls() or self._is_collision_with_obstacles():
             self.x, self.y = prev_x, prev_y; v = 0.0 # Stop on wall hit

        # 3. Compute Return Values
        self.step_count += 1
        dist_after = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        self.episode_path_length += math.hypot(self.x - prev_x, self.y - prev_y)
        
        obs = self._get_obs(reset_stack=False)
        reward = self._compute_reward_custom(dist_before, dist_after, v, w, obs[4:]) # obs[4:] is Lidar part
        terminated, info = self._check_termination_custom(obs[4:4+self.num_rays])
        if terminated and info["termination_reason"] == "collision_lidar": 
            reward -= 100.0
        truncated = False # Can be used for timeouts if preferred over terminated

        return obs, reward, terminated, truncated, info
    # --- METODI CUSTOM ---
    
    def _sync_people_list(self):
        self.people = [] 
        states = np.array(self.humans_state_jax)
        current_time = self.step_count * self.dt

        # Inizializzazione memoria posizioni se assente
        if self.last_human_pos is None or len(self.last_human_pos) != len(states):
            self.last_human_pos = states[:, :2]
        
        goals_mem = np.array(self.humans_goals_mem_jax)
        indices = np.array(self.humans_goal_indices)
        
        # 1. Logica Gestione Goal e Waypoint
        for i, s in enumerate(states):
            current_idx = int(indices[i])
            target_pos = goals_mem[i, current_idx]
            dist = math.hypot(s[0]-target_pos[0], s[1]-target_pos[1])
            
            if dist < 0.6: 
                if self.scenario_type == "bottleneck":
                    if current_idx == 1: 
                        new_idx = 2
                        self.humans_goal_indices = self.humans_goal_indices.at[i].set(new_idx)
                        self.humans_goal_jax = self.humans_goal_jax.at[i].set(self.humans_goals_mem_jax[i, new_idx])
                    elif current_idx == 2:
                        start_pos = goals_mem[i, 0]
                        new_s = jnp.array([start_pos[0], start_pos[1], 0.0, 0.0, s[4], 0.0])
                        self.humans_state_jax = self.humans_state_jax.at[i].set(new_s)
                        new_idx = 1
                        self.humans_goal_indices = self.humans_goal_indices.at[i].set(new_idx)
                        self.humans_goal_jax = self.humans_goal_jax.at[i].set(self.humans_goals_mem_jax[i, new_idx])
                elif self.scenario_type in ["parallel", "perpendicular"]:
                    start_pos = goals_mem[i, 0]
                    new_s = jnp.array([start_pos[0], start_pos[1], 0.0, 0.0, s[4], 0.0])
                    self.humans_state_jax = self.humans_state_jax.at[i].set(new_s)
                elif self.scenario_type not in ["static_groups"]:
                    target_idx = 1 if current_idx == 0 else 0
                    self.humans_goal_indices = self.humans_goal_indices.at[i].set(target_idx)
                    self.humans_goal_jax = self.humans_goal_jax.at[i].set(self.humans_goals_mem_jax[i, target_idx])

        # 2. Ricostruzione lista e Calcolo Velocità REALE
        current_states = np.array(self.humans_state_jax)
        for i, s in enumerate(current_states):
             x, y, vx_jax, vy_jax, theta = float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])
             
             prev_x, prev_y = self.last_human_pos[i]
             actual_dist = math.hypot(x - prev_x, y - prev_y)
             
             # [FIX TELETRASPORTO] Se la distanza è > 1.0m, è un respawn.
             # Non calcolare la velocità su questo salto.
             if actual_dist > 1.0:
                 v_real = 0.0
             else:
                 v_real = actual_dist / self.dt 
             
             # [FIX AMPLIEZZA GIGANTE] Cap velocità a 1.5 m/s (corsa umana standard)
             v_real = min(v_real, 1.5)
             
             vis_v = self._get_smooth_speed(i, v_real)
             
             person_data = {"x": x, "y": y, "vx": vx_jax, "vy": vy_jax, "angle": theta}
             
             # Aggiorna fase: 8.0-10.0 è un valore più naturale per la camminata
             self.humans_leg_phase[i] += vis_v * self.dt * 4.0 
             
             if self.use_legs:
                 person_data["legs"] = self._calculate_leg_positions(x, y, vis_v, theta, self.humans_leg_phase[i])

             self.people.append(person_data)
        
        self.last_human_pos = current_states[:, :2]





    def _generate_static_obstacles(self):
        self.obstacles = []
        
        # --- SCENARIO 1: Parallel (Corridoio con ostacoli sparsi) ---
        if self.scenario_type == "parallel":
            num_obs = random.randint(1, 4)
            for _ in range(num_obs):
                cx = random.uniform(0.5, self.room_width - 0.5)
                cy = random.uniform(3.0, self.room_height - 3.0)
                if random.random() < 0.5:
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.2, 0.4)})
                else:
                    w, h = random.uniform(0.4, 0.8), random.uniform(0.4, 0.8)
                    self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})

        # --- SCENARIO 2: Perpendicular (Stanza Quadrata con ostacoli centrali) ---
        elif self.scenario_type == "perpendicular":
            num_obs = random.randint(1, 4)
            for _ in range(num_obs):
                cx = random.uniform(1.5, self.room_width - 1.5)
                cy = random.uniform(2.5, self.room_height - 2.5)
                if random.random() < 0.5:
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.1, 0.4)})
                else:
                    w, h = random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)
                    self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})

        # --- SCENARIO 3: Circular (4 Pilastri agli angoli) ---
        elif self.scenario_type == "circular":
            offset = 1.5
            corners = [
                (offset, offset), (self.room_width-offset, offset),
                (offset, self.room_height-offset), (self.room_width-offset, self.room_height-offset)
            ]
            for cx, cy in corners:
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": 0.6})

        # --- SCENARIO 4: Bottleneck (Muro con buco centrale) ---
        elif self.scenario_type == "bottleneck":
            wall_y = self.room_height / 2
            gap_size = 1.5
            
            # Posizione random del varco (con margine dai bordi laterali)
            # Salviamo in self per passarlo dopo a _setup_scenario
            min_x = 2.0
            max_x = self.room_width - 2.0
            self.bottleneck_gap_x = random.uniform(min_x, max_x)
            
            # Muro Sinistro (da 0 a inizio varco)
            self.obstacles.append({
                "type": "rect", 
                "xmin": 0, "xmax": self.bottleneck_gap_x - gap_size/2, 
                "ymin": wall_y - 0.2, "ymax": wall_y + 0.2
            })
            # Muro Destro (da fine varco a larghezza stanza)
            self.obstacles.append({
                "type": "rect", 
                "xmin": self.bottleneck_gap_x + gap_size/2, "xmax": self.room_width, 
                "ymin": wall_y - 0.2, "ymax": wall_y + 0.2
            })
            return
        # --- SCENARIO 5: Empty Scenarios (Solo folla, niente muri) ---
        elif self.scenario_type == "static_groups":
            # Genera 3-6 ostacoli piccoli (tipo fioriere o sedie)
            num_obs = random.randint(1, 4)
            for _ in range(num_obs):
                cx = random.uniform(1.0, self.room_width - 1.0)
                cy = random.uniform(1.0, self.room_height - 1.0)
                
                # Evita zona Start/Goal Robot
                if math.hypot(cx - self.room_width/2, cy - 1.0) < 1.5: continue
                if math.hypot(cx - self.room_width/2, cy - (self.room_height-1.0)) < 1.5: continue

                # Ostacoli piccoli (raggio 0.2-0.3)
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.2, 0.35)})

        elif self.scenario_type == "intersection":
            # Definiamo la larghezza dei corridoi (incrocio)
            gap = 3.0 
            
            # Calcoliamo le coordinate di divisione
            mid_x = self.room_width / 2
            mid_y = self.room_height / 2
            
            # Dimensioni dei blocchi angolari
            # Devono coprire da 0 fino a (centro - metà corridoio)
            block_w = (self.room_width - gap) / 2
            block_h = (self.room_height - gap) / 2
            
            # 1. Blocco Basso-Sinistra
            self.obstacles.append({
                "type": "rect", 
                "xmin": 0, "xmax": mid_x - gap/2, 
                "ymin": 0, "ymax": mid_y - gap/2
            })
            # 2. Blocco Basso-Destra
            self.obstacles.append({
                "type": "rect", 
                "xmin": mid_x + gap/2, "xmax": self.room_width, 
                "ymin": 0, "ymax": mid_y - gap/2
            })
            # 3. Blocco Alto-Sinistra
            self.obstacles.append({
                "type": "rect", 
                "xmin": 0, "xmax": mid_x - gap/2, 
                "ymin": mid_y + gap/2, "ymax": self.room_height
            })
            # 4. Blocco Alto-Destra
            self.obstacles.append({
                "type": "rect", 
                "xmin": mid_x + gap/2, "xmax": self.room_width, 
                "ymin": mid_y + gap/2, "ymax": self.room_height
            })
            return

        # --- SCENARIO DEFAULT: Random (Generazione procedurale classica) ---
        else:
            while len(self.obstacles) < self.n_obstacles:
                cx, cy = random.uniform(1, self.room_width-1), random.uniform(1, self.room_height-1)
                # Protezione zona start robot
                if math.hypot(cx-2, cy-2) < 2.5: continue 
                
                if random.random() < 0.5: 
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.15, 0.5)})
                else: 
                    w, h = random.uniform(0.6, 1.2), random.uniform(0.6, 1.2)
                    self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})


    def _setup_scenario(self):
        w, h = self.room_width, self.room_height
        states, raw_goals = [], [] # raw_goals può contenere punti singoli o liste
        
        # --- Helper: Check Valid Point with Margin ---
        def is_safe_point(x, y, margin=0.4):
            # 1. Check room bounds (with margin)
            if x < margin or x > w - margin or y < margin or y > h - margin:
                return False
            
            # 2. Check obstacles (with margin)
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    dist_sq = (x - obs["cx"])**2 + (y - obs["cy"])**2
                    min_dist = obs["radius"] + margin
                    if dist_sq < min_dist**2: return False
                elif obs["type"] == "rect":
                    # Expand rect by margin for safety check
                    if (obs["xmin"] - margin <= x <= obs["xmax"] + margin) and \
                       (obs["ymin"] - margin <= y <= obs["ymax"] + margin):
                        return False
                
            return True

        def get_valid_point(x, y, search_radius=2.0, margin=0.4):
            """If (x,y) is unsafe, find a neighbor point that is safe."""
            if is_safe_point(x, y, margin):
                return x, y
            
            # Retry loop: search for a safe spot nearby
            for _ in range(100):
                nx = x + random.uniform(-search_radius, search_radius)
                ny = y + random.uniform(-search_radius, search_radius)
                if is_safe_point(nx, ny, margin):
                    return nx, ny
            return x, y # Fallback (should be extremely rare)

        #1. Chiamata a Scenarios
        if self.scenario_type == "bottleneck":
            # Passiamo il gap dinamico calcolato in _generate_static_obstacles
            r_s, r_g, states, raw_goals = Scenarios.bottleneck(w, h, self.num_people, getattr(self, 'bottleneck_gap_x', w/2))
        elif self.scenario_type == "parallel":
            r_s, r_g, states, raw_goals = Scenarios.parallel_traffic(w, h, self.num_people)
        elif self.scenario_type == "perpendicular":
            r_s, r_g, states, raw_goals = Scenarios.perpendicular_crossing(w, h, self.num_people)
        elif self.scenario_type == "circular":
            r_s, r_g, states, raw_goals = Scenarios.circular_crossing(w, h, self.num_people)
        elif self.scenario_type == "static_groups":
            r_s, r_g, states, raw_goals = Scenarios.static_groups(w, h, self.num_people)
        elif self.scenario_type == "intersection":
            r_s, r_g, states, raw_goals = Scenarios.intersection(w, h, self.num_people)
        else:
            r_s, r_g, states, raw_goals = Scenarios.random_static(w, h, self.num_people)

        # 2. Setup Robot (Validazione)
        rx, ry, rt = r_s
        gx, gy = r_g
        rx, ry = get_valid_point(rx, ry, margin=0.35)
        self.x, self.y, self.theta = rx, ry, rt
        gx, gy = get_valid_point(gx, gy, margin=0.35)
        self.goal_x, self.goal_y = gx, gy

        # 3. Setup Umani e Memoria Waypoints (Standardizzazione a 3 stadi)
        # Struttura Memoria: [START, TARGET_1, TARGET_2]
        goals_mem = []
        
        safe_margin = 0.4
        for i in range(len(states)):
            # Validazione Start Position (Umano)
            states[i][0], states[i][1] = get_valid_point(states[i][0], states[i][1], margin=safe_margin)
            s_pos = states[i][:2] # Start effettivo (middle)
            
            g_data = raw_goals[i]
            
            # --- CASO A: BOTTLENECK (Waypoints: Start -> Gap -> End) ---
            if self.scenario_type == "bottleneck":
                gap_pt = g_data[0] 
                end_pt = list(get_valid_point(g_data[1][0], g_data[1][1], margin=safe_margin))
                goals_mem.append([s_pos, gap_pt, end_pt]) # Usa s_pos per il respawn
            
            # --- CASO B: INTERSECTION (Waypoints: Estremo A <-> Estremo B) ---
            elif self.scenario_type == "intersection":
                # g_data è [Estremo_A, Estremo_B]
                pt_a = list(get_valid_point(g_data[0][0], g_data[0][1], margin=safe_margin))
                pt_b = list(get_valid_point(g_data[1][0], g_data[1][1], margin=safe_margin))
                
                # Memoria: [Estremo_A, Estremo_B, Estremo_B]
                # Nota: Non usiamo s_pos qui, perché in patrol si rimbalza tra A e B
                goals_mem.append([pt_a, pt_b, pt_b])

            # --- CASO C: ALTRI (Start -> End) ---
            else:
                # g_data è [End] o singolo punto
                if isinstance(g_data[0], list) or isinstance(g_data[0], tuple): 
                     final_pt = list(get_valid_point(g_data[-1][0], g_data[-1][1], margin=safe_margin))
                else:
                     final_pt = list(get_valid_point(g_data[0], g_data[1], margin=safe_margin))
                
                # Memoria: [Start, End, End]
                goals_mem.append([s_pos, final_pt, final_pt])

        # 4. Finalize JAX Arrays
        self.humans_goals_mem_jax = jnp.array(goals_mem, dtype=jnp.float32)
        
        # Tutti iniziano puntando all'Indice 1 (Target B / End / Gap)
        self.humans_goal_indices = jnp.ones(self.num_people, dtype=jnp.int32)
        self.humans_goal_jax = self.humans_goals_mem_jax[:, 1]
        self.humans_state_jax = jnp.array(states, dtype=jnp.float32)
        
        self._sync_people_list()

    def _convert_obs_to_jax(self):
        """Converte la lista self.obstacles (usata dal padre) in tensori JAX per HSFM"""
        edges = []
        w, h = self.room_width, self.room_height
        # Muri
        edges.extend([[[0,0], [w,0]], [[w,0], [w,h]], [[w,h], [0,h]], [[0,h], [0,0]]])
        
        for obs in self.obstacles:
            if obs["type"] == "rect":
                p1, p2 = [obs["xmin"], obs["ymin"]], [obs["xmax"], obs["ymin"]]
                p3, p4 = [obs["xmax"], obs["ymax"]], [obs["xmin"], obs["ymax"]]
                edges.extend([[p1, p2], [p2, p3], [p3, p4], [p4, p1]])
            elif obs["type"] == "circle":
                cx, cy, r = obs["cx"], obs["cy"], obs["radius"]
                sides = 8
                pts = [[cx + r*math.cos(2*math.pi*i/sides), cy + r*math.sin(2*math.pi*i/sides)] for i in range(sides)]
                for i in range(sides):
                    edges.append([pts[i], pts[(i+1)%sides]])
                    
        n_agents = self.num_people + 1
        obs_array = jnp.array(edges, dtype=jnp.float32)
        return jnp.tile(obs_array[None, None, ...], (n_agents, 1, 1, 1, 1))

    def _compute_reward_custom(self, dist_before, dist_after, v, w, lidar_stack):
        reward = 0.0
        
        # 1. Progress Reward (Molto importante)
        reward += 20.0 * (dist_before - dist_after) 
        reward -= 0.05 # Time penalty
        
        # 2. Alignment Reward (solo se si muove)
        if v > 0.05:
            to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
            err = (to_goal - self.theta + math.pi) % (2*math.pi) - math.pi
            reward += 0.5 * v * math.cos(err)

        # 3. Lidar Safety Reward (CORRETTO PER LIDAR INVERTITO)
        # lidar_stack contiene tutto lo storico. Prendiamo il frame attuale (primi num_rays)
        # Valori: 0.0 = Lontano, 1.0 = Vicinissimo (0.12m)
        current_lidar = lidar_stack[:self.num_rays] 
        max_val = np.max(current_lidar) # Cerchiamo il punto più pericoloso (più alto)

        # Se siamo troppo vicini (es. > 0.5 che corrisponde a circa 1 metro in scala inversa cubica)
        if max_val > 0.5:
            # Penalità esponenziale: piccola a 0.5, enorme a 1.0
            reward -= np.exp(3.0 * (max_val - 0.5)) 

        # 4. Human Distance Safety (Backup geometrico)
        # Calcoliamo la distanza reale dall'umano più vicino
        min_human_dist = min([math.hypot(self.x - h["x"], self.y - h["y"]) for h in self.people])
        
        # Se entri nella "zona intima" (0.5m), penalità forte
        if min_human_dist < 0.5:
             reward -= 5.0 * (0.5 - min_human_dist)

        return reward

    def _check_termination_custom(self, inv_lidar):
        # 1.0 (inverted) = 0.12m (real). Check max value in current frame
        if self.manual_skip_triggered:
            return True, {"termination_reason": "manual_skip"}

        if np.max(inv_lidar) >= 0.999: return True, {"termination_reason": "collision_lidar"}
        if math.hypot(self.x-self.goal_x, self.y-self.goal_y) < 0.3: return True, {"termination_reason": "goal_reached"}
        return (self.step_count >= self.max_steps), {"termination_reason": "timeout" if self.step_count >= self.max_steps else "none"}
    
    def _on_key_press(self, event):
        """Callback: se premi freccia destra, attiva il flag di skip"""
        if event.key == 'right':
            print("\n>>> SKIP MANUALE ATTIVATO: Passo al prossimo episodio! <<<\n")
            self.manual_skip_triggered = True

    def _calculate_leg_positions(self, x, y, v, theta, leg_phase):
        HIP_SPACING = 0.20
        
        # --- CALCOLO MATEMATICO ANTI-SCIVOLAMENTO ---
        # Perchè il piede stia fermo, la velocità all'indietro della gamba deve
        # annullare la velocità in avanti del corpo.
        # Formula: Ampiezza = Pi_Greco / (2 * K_Phase)
        # Usiamo K=6.0 (definito in _sync_people_list). 
        # Quindi Amp = 3.14 / 12 = ~0.26m (passo totale ~0.52m)
        K_PHASE = 6.0 
        target_amp = math.pi / (2 * K_PHASE)
        
        # Fade-in dell'ampiezza per evitare scatti da fermo
        if v < 0.05:
            stride_amp = 0.0
        else:
            # Interpolazione dolce verso l'ampiezza target
            stride_amp = target_amp * min(1.0, v / 0.2)

        # --- FUNZIONE D'ONDA IBRIDA (Linear Stance + Cosine Swing) ---
        def get_offset(phi):
            # Normalizza phi tra 0 e 2*pi
            p = phi % (2 * math.pi)
            
            if p < math.pi:
                # FASE 1: STANCE (Piede a terra)
                # Il corpo avanza, quindi il piede deve arretrare LINEARMENTE
                # Va da +1 (fronte) a -1 (retro)
                return 1.0 - (2.0 * p / math.pi)
            else:
                # FASE 2: SWING (Piede in aria)
                # Recupero veloce in avanti (curva morbida)
                # Va da -1 (retro) a +1 (fronte) usando -coseno
                swing_progress = p - math.pi # da 0 a pi
                return -math.cos(swing_progress)

        # Calcolo offset per le due gambe (sfasate di 180° o pi)
        off_l_val = get_offset(leg_phase)
        off_r_val = get_offset(leg_phase + math.pi)

        # Applica l'ampiezza corretta
        offset_l = stride_amp * off_l_val
        offset_r = stride_amp * off_r_val

        # --- TRASFORMAZIONE GEOMETRICA ---
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px, py = -sin_t, cos_t 

        # Gamba Sinistra
        lx = x - (px * HIP_SPACING / 2) + (cos_t * offset_l)
        ly = y - (py * HIP_SPACING / 2) + (sin_t * offset_l)
        
        # Gamba Destra
        rx = x + (px * HIP_SPACING / 2) + (cos_t * offset_r)
        ry = y + (py * HIP_SPACING / 2) + (sin_t * offset_r)
        
        return [(lx, ly), (rx, ry)]
    

    def _dist_point_to_segment(self, px, py, x1, y1, x2, y2):
        """Calcola la distanza minima tra il punto (px,py) e il segmento (x1,y1)-(x2,y2)."""
        # Vettore Segmento
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0: 
            return math.hypot(px - x1, py - y1)

        # Proiezione del punto sulla retta (parametro t)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # Clamping del segmento (il piede ha una lunghezza finita)
        t = max(0, min(1, t))

        # Punto più vicino sul segmento
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.hypot(px - closest_x, py - closest_y)
    
    def _is_collision_with_people(self):
        rr = self.robot_radius
        if not self.use_legs: # Caso Standard (Padre)
            min_sq = (rr + self.people_radius)**2
            for p in self.people:
                if (self.x-p["x"])**2+(self.y-p["y"])**2 < min_sq: return True
            return False

        # Caso Avanzato (Gambe + Piedi)
        LEG_R, FOOT_L = 0.09, 0.30
        for p in self.people:
            # Check sicurezza corpo (Ghost)
            if math.hypot(self.x-p["x"], self.y-p["y"]) < (rr + 0.15): return True
            if "legs" in p:
                ct, st = math.cos(p["angle"]), math.sin(p["angle"])
                for lx, ly in p["legs"]:
                    # Collisione Stinco
                    if math.hypot(self.x-lx, self.y-ly) < (rr + LEG_R): return True
                    # Collisione Piede (Segmento asse scarpa)
                    x_s, y_s = lx - LEG_R*ct, ly - LEG_R*st
                    x_e, y_e = lx + (FOOT_L-LEG_R)*ct, ly + (FOOT_L-LEG_R)*st
                    if self._dist_point_to_segment(self.x, self.y, x_s, y_s, x_e, y_e) < (rr + 0.05):
                        return True
        return False
    






























    

    def render(self):
        """Render condizionale: Standard (Padre) o Custom (Gambe + Vettori Coerenti)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        
        # --- CASO A: Render Semplice (Cerchi Verdi) ---
        if not self.use_legs:
            super().render() 
            
            if self.allow_keyboard_skip and self.fig is not None and not self._listener_attached:
                self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
                self._listener_attached = True
            return

        # --- CASO B: Render Realistico (Gambe Nere + Ghost + Vettori) ---
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
        
        self.ax.clear()

        # 1. Stanza (Muri)
        self.ax.plot([0, self.room_width, self.room_width, 0, 0], [0, 0, self.room_height, self.room_height, 0], 'k-')

        # 2. Ostacoli Statici
        for obs in self.obstacles:
            if obs["type"] == "circle":
                circle = Circle((obs["cx"], obs["cy"]), obs["radius"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(circle)
            elif obs["type"] == "rect":
                rect = Rectangle((obs["xmin"], obs["ymin"]), obs["xmax"] - obs["xmin"], obs["ymax"] - obs["ymin"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(rect)

        # 3. Robot
        robot_circle = Circle((self.x, self.y), self.robot_radius, color='blue', fill=True)
        self.ax.add_patch(robot_circle)
        arrow_len = self.robot_radius * 1.5
        x_head = self.x + arrow_len * math.cos(self.theta)
        y_head = self.y + arrow_len * math.sin(self.theta)
        self.ax.plot([self.x, x_head], [self.y, y_head], 'b-', linewidth=2)

        # 4. UMANI (Disegno Custom)
        LEG_RADIUS = 0.09
        for p in self.people:
            # A. Corpo Fantasma (Rosso tratteggiato)
            body = Circle((p["x"], p["y"]), SimConfig.HUMANS_RADIUS, fill=False, color='red', linestyle='--', alpha=0.5, linewidth=1)
            self.ax.add_patch(body)
            
            # B. Gambe (Nere) e Scarpe (Rettangoli)
            if "legs" in p:
                # [MODIFICA] Allineamento Forzato alla Freccia Verde
                # La freccia verde usa p["angle"], quindi anche le scarpe devono usare p["angle"].
                foot_theta = p["angle"]

                # 2. Geometria Scarpa
                foot_width = LEG_RADIUS * 2.0  
                foot_len   = 0.30              
                
                # Offset locale dell'angolo in basso a sinistra del rettangolo
                local_anchor_x = -LEG_RADIUS
                local_anchor_y = -LEG_RADIUS

                # Pre-calcolo rotazione
                cos_t = math.cos(foot_theta)
                sin_t = math.sin(foot_theta)

                for lx, ly in p["legs"]:
                    # --- DISEGNO SCARPA (Rettangolo) ---
                    anchor_x = lx + (local_anchor_x * cos_t - local_anchor_y * sin_t)
                    anchor_y = ly + (local_anchor_x * sin_t + local_anchor_y * cos_t)

                    rect = Rectangle(
                        (anchor_x, anchor_y), width=foot_len, height=foot_width,
                        angle=math.degrees(foot_theta), 
                        fill=True,             # [MODIFICATO] Riempimento attivo
                        facecolor='lightgray', # [MODIFICATO] Colore grigio chiaro
                        edgecolor='black',     # Bordo nero per definizione
                        linewidth=0.8,
                        alpha=0.6,             # Trasparenza per non coprire troppo
                        zorder=1               # Disegnato sotto la gamba
                    )
                    self.ax.add_patch(rect)
                    
                    # --- DISEGNO GAMBA (Cerchio Nero) ---
                    leg_circle = Circle((lx, ly), LEG_RADIUS, color='black', fill=True)
                    self.ax.add_patch(leg_circle)

            # C. Vettore Velocità (Freccia Verde)
            speed = math.hypot(p["vx"], p["vy"])
            
            if speed > 0.05:
                # Proiezione velocità su direzione corpo (Codice confermato da te)
                vis_vx = speed * math.cos(p["angle"]) * 0.6
                vis_vy = speed * math.sin(p["angle"]) * 0.6

                self.ax.arrow(
                    p["x"], p["y"],
                    vis_vx, vis_vy,
                    head_width=0.15,
                    head_length=0.1,
                    fc='green', ec='green',
                    alpha=0.8,
                    length_includes_head=True
                )

        # 5. LIDAR (Raggi)
        lidar = self._compute_lidar() 
        angles = [self.theta + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]
        lidar_x_origin = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y_origin = self.y + self.lidar_offset * math.sin(self.theta)

        if len(self.trajectory) >= 2:
            traj_xs, traj_ys = zip(*self.trajectory)
            self.ax.plot(traj_xs, traj_ys, 'b--', linewidth=1)

        base_grey = (0.6, 0.6, 0.6)
        base_red = (1.0, 0.0, 0.0)
        
        for i, (dist, ang) in enumerate(zip(lidar, angles)): 
            x_end = lidar_x_origin + dist * math.cos(ang)
            y_end = lidar_y_origin + dist * math.sin(ang)

            color_ray_dist = 5.0
            norm_d = min(dist, color_ray_dist) / color_ray_dist
            proximity = (1.0 - norm_d) 
            r = base_grey[0] + proximity * (base_red[0] - base_grey[0])
            g = base_grey[1] + proximity * (base_red[1] - base_grey[1])
            b = base_grey[2] + proximity * (base_red[2] - base_grey[2])
            
            if i == 0: 
                rgba = (0.5, 0.7, 1.0, 1.0)
                linewidth = 1.0             
            else:
                rgba = (r, g, b, 0.9 * proximity)
                linewidth = 0.5

            self.ax.plot([lidar_x_origin, x_end], [lidar_y_origin, y_end], color=rgba, linewidth=linewidth)
            if i != 0:
                self.ax.plot(x_end, y_end, marker='o', markersize=2, color=rgba)

        # 6. Goal
        if self.goal_x is not None:
            self.ax.plot(self.goal_x, self.goal_y, marker='*', markersize=7, color='orange')

        self.ax.set_xlim(-1, self.room_width + 1)
        self.ax.set_ylim(-1, self.room_height + 1)
        self.ax.set_aspect('equal', adjustable='box')

        status_text = f"Step: {self.step_count} | Scenario: {self.scenario_type.upper()}"
        self.ax.text(0.01, 0.99, status_text, verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        plt.pause(0.0001)
        
        if self.allow_keyboard_skip and self.fig is not None and not self._listener_attached:
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            self._listener_attached = True






































    # --- OVERRIDE FISICA: Il Lidar colpisce le gambe, non il corpo ---
    def _cast_ray(self, angle, x0, y0):
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []
        
        # 1. Muri (Sempre uguali)
        if abs(dx) > 1e-6:
            t = (0 - x0) / dx; distances.append(t) if t >= 0 and 0 <= y0 + t * dy <= self.room_height else None
            t = (self.room_width - x0) / dx; distances.append(t) if t >= 0 and 0 <= y0 + t * dy <= self.room_height else None
        if abs(dy) > 1e-6:
            t = (0 - y0) / dy; distances.append(t) if t >= 0 and 0 <= x0 + t * dx <= self.room_width else None
            t = (self.room_height - y0) / dy; distances.append(t) if t >= 0 and 0 <= x0 + t * dx <= self.room_width else None

        # 2. Ostacoli Statici (Sempre uguali)
        for obs in self.obstacles:
            if obs["type"] == "circle":
                t = self._ray_circle_intersection(angle, obs["cx"], obs["cy"], obs["radius"], x0, y0)
                if t is not None: distances.append(t)
            elif obs["type"] == "rect":
                t = self._ray_rect_intersection(angle, obs["xmin"], obs["xmax"], obs["ymin"], obs["ymax"], x0, y0)
                if t is not None: distances.append(t)

        # 3. UMANI (Logica Condizionale)
        if self.use_legs:
            # --- MODALITÀ REALISTICA (Gambe) ---
            LEG_RADIUS = 0.09
            for p in self.people:
                if "legs" in p:
                    # Gamba Sinistra
                    t_l = self._ray_circle_intersection(angle, p["legs"][0][0], p["legs"][0][1], LEG_RADIUS, x0, y0)
                    if t_l is not None: distances.append(t_l)
                    # Gamba Destra
                    t_r = self._ray_circle_intersection(angle, p["legs"][1][0], p["legs"][1][1], LEG_RADIUS, x0, y0)
                    if t_r is not None: distances.append(t_r)
                else:
                    # Fallback di sicurezza
                    t = self._ray_circle_intersection(angle, p["x"], p["y"], self.people_radius, x0, y0)
                    if t is not None: distances.append(t)
        else:
            # --- MODALITÀ STANDARD (Cerchio Verde) ---
            for p in self.people:
                t = self._ray_circle_intersection(angle, p["x"], p["y"], self.people_radius, x0, y0)
                if t is not None: distances.append(t)

        if not distances: return self.max_lidar_distance
        
        real_dist = min(distances)
        if real_dist > self.max_lidar_distance: return self.max_lidar_distance
        if real_dist < self.lidar_min_distance: return self.lidar_min_distance
            
        return real_dist
    
    def _get_smooth_speed(self, human_index, current_v_mag):
        """
        Filtro Passa-Basso (EMA) per simulare l'inerzia delle gambe.
        Input: current_v_mag (Velocità reale istantanea calcolata da delta pos)
        """
        ALPHA = 0.15  # Fattore di smorzamento (basso = molta inerzia/fluidità)
        
        # Recupera vecchia velocità filtrata
        old_v = self.smooth_v[human_index]
        
        # Aggiorna
        new_v = (ALPHA * current_v_mag) + ((1.0 - ALPHA) * old_v)
        
        # Salva e ritorna
        self.smooth_v[human_index] = new_v
        return new_v