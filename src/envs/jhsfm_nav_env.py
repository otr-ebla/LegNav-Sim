import math
import random
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces
import gymnasium as gym

from collections import deque

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
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario_type="random"):
        # 1. Imposta lo scenario
        self.scenario_type = scenario_type

        # 2. Dimensioni Dinamiche: Corridoio stretto per "parallel"
        if self.scenario_type == "parallel":
            eff_width, eff_height = 4.0, 14.0 
        elif self.scenario_type == "perpendicular":
            eff_width, eff_height = 9.0, 9.0 # <--- NUOVO: Stanza quadrata
        elif self.scenario_type == "circular":
            eff_width, eff_height = 12.0, 12.0 # <--- NUOVO: Spazio ampio per il cerchio
        else:
            eff_width, eff_height = SimConfig.ROOM_SIZE[0], SimConfig.ROOM_SIZE[1]

        self.num_people = SimConfig.NUM_HUMANS 

        # 3. Init padre con le dimensioni calcolate
        super().__init__(
            max_steps=SimConfig.MAX_STEPS,
            dt=RobotConfig.DT,
            room_width=eff_width,   # Usa dimensione dinamica
            room_height=eff_height, # Usa dimensione dinamica
            robot_radius=RobotConfig.RADIUS,
            num_rays=LidarConfig.NUM_RAYS,
            max_lidar_distance=LidarConfig.MAX_DISTANCE,
            num_people=self.num_people,
            people_radius=SimConfig.HUMANS_RADIUS, 
            people_speed=SimConfig.HUMANS_VELOCITY,     
        )
        
        # Override parametri fisici
        self.max_v = RobotConfig.MAX_LINEAR_VEL
        self.max_w = RobotConfig.MAX_W
        self.lidar_min_distance = LidarConfig.MIN_DIST
        self.lidar_offset = RobotConfig.LIDAR_OFFSET

        # Setup HSFM
        self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        self.hsfm_step_fn = jax.jit(hsfm_step)
        
        # Stato JAX
        self.humans_state_jax = None 
        self.humans_goal_jax = None
        self.static_obstacles_jax = None
        self.humans_goals_mem_jax = None
        self.humans_goal_indices = None

        self.n_obstacles = SimConfig.NUM_OBSTACLES  
        self.n_stack = RobotConfig.LIDAR_STACK_DIM
        self.lidar_stack = deque(maxlen=self.n_stack)
        
        obs_dim = 4 + self.num_rays * self.n_stack
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_v, -self.max_w]), 
            high=np.array([self.max_v, self.max_w]), 
            dtype=np.float32
        )


        self.allow_keyboard_skip = True  # Abilita salto con tastiera nel render
        self.manual_skip_triggered = False
        self._listener_attached = False


    def reset(self, seed=None, options=None):
        # Reset variabili base
        self.manual_skip_triggered = False
        self.step_count = 0
        self.episode_path_length = 0.0
        self.trajectory = []
        self.last_v = 0.0
        self.last_w = 0.0
        self.last_termination_reason = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 1. Genera Ostacoli (sovrascrive il metodo del padre o usa quello custom qui sotto)
        self._generate_static_obstacles()
        
        # 2. Setup Scenario (Posiziona Robot e Umani)
        self._setup_scenario()
        
        # 3. Prepara ostacoli per JAX
        self.static_obstacles_jax = self._convert_obs_to_jax()
        
        # Inizializza traiettoria
        self.start_x, self.start_y = self.x, self.y
        self.trajectory.append((self.x, self.y))
        self.step_count = 0
        self.episode_path_length = 0.0
        self.last_v, self.last_w = 0.0, 0.0

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
        # [Dist, Angle, V, W, Lidar_t, Lidar_t-1, Lidar_t-2]
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
        n_substeps = int(dt / SimConfig.HSFM_DT)
        rob_state = jnp.array([self.x, self.y, v*math.cos(self.theta), v*math.sin(self.theta), self.theta, w])
        rob_goal  = jnp.array([self.goal_x, self.goal_y])
        
        c_states = jnp.concatenate([self.humans_state_jax, rob_state[None, :]], axis=0)
        c_goals  = jnp.concatenate([self.humans_goal_jax, rob_goal[None, :]], axis=0)
        
        for _ in range(n_substeps):
            c_states = self.hsfm_step_fn(c_states, c_goals, self.hsfm_params, self.static_obstacles_jax, SimConfig.HSFM_DT)
            
        self.humans_state_jax = c_states[:-1] # Update humans only
        self._sync_people_list()              # Update self.people for Lidar/Render

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
        """Converte stato JAX nella lista e gestisce il Respawn/Patrol"""
        self.people = [] 
        states = np.array(self.humans_state_jax)
        goals = np.array(self.humans_goal_jax)

        for i, s in enumerate(states):
            # Check arrivo al goal (distanza < 0.5m)
            if math.hypot(s[0]-goals[i][0], s[1]-goals[i][1]) < 0.5:
                
                if self.scenario_type in ["parallel", "perpendicular"]: # <--- MODIFICA QUI
                    start_pos = self.humans_goals_mem_jax[i, 0]
                    # Reset posizione + Velocità Zero
                    new_s = jnp.array([start_pos[0], start_pos[1], 0.0, 0.0, s[4], 0.0])
                    self.humans_state_jax = self.humans_state_jax.at[i].set(new_s)
                
                else:
                    # --- PATROLLING (Ping-Pong A <-> B) ---
                    new_idx = 1 - int(self.humans_goal_indices[i])
                    self.humans_goal_indices = self.humans_goal_indices.at[i].set(new_idx)
                    self.humans_goal_jax = self.humans_goal_jax.at[i].set(self.humans_goals_mem_jax[i, new_idx])

        # Ricostruiamo la lista self.people per il render e il lidar
        # Nota: Dobbiamo rileggere humans_state_jax perché potremmo averlo appena modificato (teletrasporto)
        current_states = np.array(self.humans_state_jax)
        
        for s in current_states:
             self.people.append({"x": float(s[0]), "y": float(s[1]), "vx": float(s[2]), "vy": float(s[3]), "angle": float(s[4])})


    def _generate_static_obstacles(self):
        self.obstacles = []
        
        # --- CASO 1: Scenario Parallel (Corridoio) ---
        if self.scenario_type == "parallel":
            # ... (codice esistente: 1-4 ostacoli piccoli nel corridoio) ...
            num_obs = random.randint(1, 4)
            for _ in range(num_obs):
                cx = random.uniform(0.5, self.room_width - 0.5)
                cy = random.uniform(3.0, self.room_height - 3.0)
                if random.random() < 0.5:
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.2, 0.4)})
                else:
                    w, h = random.uniform(0.4, 0.8), random.uniform(0.4, 0.8)
                    self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})
            return

        # --- CASO 2: Scenario Perpendicular (Stanza Quadrata con Ostacoli) ---
        if self.scenario_type == "perpendicular":
            # Genera da 3 a 6 ostacoli statici (la stanza è grande 11x11)
            num_obs = random.randint(1, 4)
            
            for _ in range(num_obs):
                # Logica di posizionamento:
                # Evita la zona di partenza Robot (Basso, Y < 2.0)
                # Evita la zona di arrivo Robot (Alto, Y > h-2.0)
                # X: Ovunque tranne troppo vicino ai muri laterali (dove spawnano gli umani)
                
                cx = random.uniform(1.5, self.room_width - 1.5)
                cy = random.uniform(2.5, self.room_height - 2.5) # Zona centrale sicura
                
                # Forme miste
                if random.random() < 0.5:
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.3, 0.7)})
                else:
                    w, h = random.uniform(0.2, 0.7), random.uniform(0.2, 0.7)
                    self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})
            return 

        if self.scenario_type == "circular":
            # Mettiamo 4 pilastri agli angoli della stanza per delimitare l'arena
            # ma lasciando libero il centro e gli assi principali
            offset = 1.5
            corners = [
                (offset, offset), (self.room_width-offset, offset),
                (offset, self.room_height-offset), (self.room_width-offset, self.room_height-offset)
            ]
            for cx, cy in corners:
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": 0.6})
            return

        # --- CASO 3: Random Scenario (Default) ---
        # ... (codice esistente del while loop) ...
        while len(self.obstacles) < self.n_obstacles:
            cx, cy = random.uniform(1, self.room_width-1), random.uniform(1, self.room_height-1)
            if math.hypot(cx-2, cy-2) < 2.5: continue 
            if random.random() < 0.5: 
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.15, 0.5)})
            else: 
                w, h = random.uniform(0.6, 1.2), random.uniform(0.6, 1.2)
                self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})

    def _setup_scenario(self):
        w, h = self.room_width, self.room_height
        states, goals = [], []
        
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

        # --- 1. Generate Raw Coordinates from Scenarios Class ---
        # This replaces the old hardcoded logic. 
        # Scenarios class returns: robot_start, robot_goal, human_states, human_goals
        if self.scenario_type == "parallel":
            r_s, r_g, states, goals = Scenarios.parallel_traffic(w, h, self.num_people)
        elif self.scenario_type == "perpendicular":
            r_s, r_g, states, goals = Scenarios.perpendicular_crossing(w, h, self.num_people)
        elif self.scenario_type == "circular": # <--- NUOVO
            r_s, r_g, states, goals = Scenarios.circular_crossing(w, h, self.num_people)
        else:
            # Default to random static points
            r_s, r_g, states, goals = Scenarios.random_static(w, h, self.num_people)

        # Unpack Robot data
        rx, ry, rt = r_s
        gx, gy = r_g

        # --- 2. Validate and Fix Humans (Start AND Goal) ---
        # Margin = Radius (0.3) + Safety (0.1) = 0.4
        safe_margin = 0.4 
        
        for i in range(len(states)):
            # Fix Start Position (Human)
            sx, sy = get_valid_point(states[i][0], states[i][1], margin=safe_margin)
            states[i][0], states[i][1] = sx, sy
            
            # Fix Goal Position (Human)
            # The scenario gives a desired goal, but we ensure it's not inside an obstacle
            g_x, g_y = get_valid_point(goals[i][0], goals[i][1], margin=safe_margin)
            goals[i][0], goals[i][1] = g_x, g_y

        # --- 3. Validate Robot Position and Goal ---
        # Robot Radius ~0.2 -> Margin 0.35
        rx, ry = get_valid_point(rx, ry, margin=0.35)
        self.x, self.y, self.theta = rx, ry, rt
        
        # Fix Robot Goal
        gx, gy = get_valid_point(gx, gy, margin=0.35)
        self.goal_x, self.goal_y = gx, gy

        # --- 4. Finalize JAX Memory ---
        # Create memory [Start(A), Target(B)] for patrolling/respawning
        goals_mem = [[s[:2], g] for s, g in zip(states, goals)]
        self.humans_goals_mem_jax = jnp.array(goals_mem, dtype=jnp.float32)

        # Init indices to 1 (Agents move to Target B initially)
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

    def render(self):
        """Override: Chiama il render del padre e attacca il listener se necessario"""
        super().render() # Disegna usando la logica di nav_env.py
        
        # Attacca il listener solo se la figura esiste e non l'abbiamo già fatto
        if self.allow_keyboard_skip and self.fig is not None and not self._listener_attached:
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            self._listener_attached = True