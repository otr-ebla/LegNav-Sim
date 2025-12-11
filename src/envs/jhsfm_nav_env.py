import math
import random
import numpy as np
import jax
import jax.numpy as jnp

# Configurazione
from src.config import RobotConfig, LidarConfig, SimConfig

# Importiamo la classe padre per riutilizzare render e matematica
from src.envs.nav_env import Simple2DEnv

# --- IMPORT JAX/HSFM ---
try:
    from src.jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from src.jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
except ImportError:
    import sys
    print("Warning: Import JHSFM standard fallito, tentativo path relativo...")
    from jhsfm_utils.JHSFM.jhsfm.hsfm import step as hsfm_step
    from jhsfm_utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

class SimpleNavEnv(Simple2DEnv):
    def __init__(self, num_people=5, scenario_type="random"):
        # 1. Inizializza la classe padre con i valori dal Config
        # Questo setta room_width, lidar parameters, variabili di rendering, ecc.
        super().__init__(
            max_steps=SimConfig.MAX_STEPS,
            dt=RobotConfig.DT,
            room_width=SimConfig.ROOM_SIZE[0],
            room_height=SimConfig.ROOM_SIZE[1],
            robot_radius=RobotConfig.RADIUS,
            num_rays=LidarConfig.NUM_RAYS,
            max_lidar_distance=LidarConfig.MAX_DISTANCE,
            num_people=num_people,
            people_radius=0.3, # Standard per HSFM
            people_speed=0.0   # Non usato, ci pensa HSFM
        )
        
        # Sovrascriviamo i limiti fisici del padre con quelli precisi del config
        self.max_v = RobotConfig.MAX_LINEAR_VEL
        self.max_w = RobotConfig.MAX_W
        self.lidar_min_distance = LidarConfig.MIN_DIST
        self.lidar_offset = RobotConfig.LIDAR_OFFSET

        # 2. Setup Specifico per JAX / HSFM
        self.scenario_type = scenario_type
        # +1 includiamo il robot nella simulazione sociale
        self.hsfm_params = get_standard_humans_parameters(self.num_people + 1)
        self.hsfm_step_fn = jax.jit(hsfm_step)
        
        # Stato JAX (Tensori)
        self.humans_state_jax = None 
        self.humans_goal_jax = None
        self.static_obstacles_jax = None
        
        # La lista self.people[] è ereditata dal padre e usata per render/lidar.
        # La terremo sincronizzata con self.humans_state_jax.

    def reset(self):
        # Reset variabili base
        self.step_count = 0
        self.episode_path_length = 0.0
        self.trajectory = []
        self.last_v = 0.0
        self.last_w = 0.0
        self.last_termination_reason = None
        
        # 1. Genera Ostacoli (sovrascrive il metodo del padre o usa quello custom qui sotto)
        self._generate_static_obstacles()
        
        # 2. Setup Scenario (Posiziona Robot e Umani)
        self._setup_scenario()
        
        # 3. Prepara ostacoli per JAX
        self.static_obstacles_jax = self._convert_obs_to_jax()
        
        # Inizializza traiettoria
        self.start_x, self.start_y = self.x, self.y
        self.trajectory.append((self.x, self.y))
        
        # Calcola Lidar usando il metodo EREDITATO (che legge self.x, self.obstacles, self.people)
        lidar = self._compute_lidar()
        
        return (self.x, self.y, self.theta, self.last_v, self.last_w, lidar)

    def step(self, action_v, action_w):
        dt = self.dt # 0.25 dal config
        
        # 1. Clamp Azioni Robot
        v = max(-self.max_v, min(action_v, self.max_v))
        w = max(-self.max_w, min(action_w, self.max_w))
        
        prev_x, prev_y = self.x, self.y
        dist_before = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        self.episode_jerk_sum += abs((w - self.last_w) / dt)

        # 2. Fisica Umana (HSFM Sub-stepping)
        n_substeps = int(dt / SimConfig.HSFM_DT)
        
        # Prepara stato robot per JAX
        rob_vx = v * math.cos(self.theta)
        rob_vy = v * math.sin(self.theta)
        robot_state_jax = jnp.array([self.x, self.y, rob_vx, rob_vy, self.theta, w])
        robot_goal_jax = jnp.array([self.goal_x, self.goal_y])
        
        curr_states = jnp.concatenate([self.humans_state_jax, robot_state_jax[None, :]], axis=0)
        curr_goals = jnp.concatenate([self.humans_goal_jax, robot_goal_jax[None, :]], axis=0)
        
        for _ in range(n_substeps):
            new_states = self.hsfm_step_fn(
                curr_states, curr_goals, self.hsfm_params, 
                self.static_obstacles_jax, SimConfig.HSFM_DT
            )
            curr_states = new_states
            
        self.humans_state_jax = curr_states[:-1]
        
        # IMPORTANTE: Sincronizza self.people (del padre) con JAX (nuovo)
        # Così _compute_lidar() e render() del padre funzionano sui dati nuovi!
        self._sync_people_list()

        # 3. Aggiorna Robot (Cinematica)
        self.theta += w * dt
        self.x += v * dt * math.cos(self.theta)
        self.y += v * dt * math.sin(self.theta)
        
        # Collision Check (usa metodi EREDITATI che leggono self.obstacles)
        will_collide = False
        if self._is_collision_with_walls() or self._is_collision_with_obstacles():
             will_collide = True
        
        if will_collide:
            self.x, self.y = prev_x, prev_y
            v = 0.0
            
        self.last_v, self.last_w = v, w
        self.trajectory.append((self.x, self.y))
        self.step_count += 1
        self.episode_path_length += math.hypot(self.x - prev_x, self.y - prev_y)

        # 4. Output
        # Riutilizziamo il raycasting C-like del padre che è già ottimizzato
        lidar = self._compute_lidar()
        dist_after = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        
        reward = self._compute_reward_custom(dist_before, dist_after, v, w, lidar)
        
        # Riutilizziamo parzialmente la logica di terminazione del padre o custom
        done, info = self._check_termination_custom()
        if done:
            info["path_length"] = self.episode_path_length
            
        return (self.x, self.y, self.theta, v, w, lidar), reward, done, info

    # --- METODI CUSTOM ---
    
    def _sync_people_list(self):
        """Converte stato JAX nella lista di dizionari che Simple2DEnv si aspetta"""
        self.people = [] # Pulisce la lista del padre
        states = np.array(self.humans_state_jax)
        goals = np.array(self.humans_goal_jax)
        w, h = self.room_width, self.room_height

        for i, s in enumerate(states):
            # Reset Goal semplice per umani
            if math.hypot(s[0]-goals[i][0], s[1]-goals[i][1]) < 0.5:
                 new_g = np.random.uniform(1.0, w-1.0, 2)
                 self.humans_goal_jax = self.humans_goal_jax.at[i].set(new_g)
            
            # Popola formato compatibile con Simple2DEnv
            self.people.append({
                "x": float(s[0]), "y": float(s[1]), 
                "vx": float(s[2]), "vy": float(s[3]), 
                "angle": float(s[4])
            })

    def _generate_static_obstacles(self):
        """Generazione ostacoli specifica per RL (sostituisce quella del padre se diversa)"""
        self.obstacles = []
        w, h = self.room_width, self.room_height
        cols, rows = 4, 3
        cell_w, cell_h = w/cols, h/rows
        cells = list(range(cols*rows)); random.shuffle(cells)
        
        for i in range(min(6, len(cells))):
            idx = cells[i]
            r, c = idx // cols, idx % cols
            cx = (c + 0.5) * cell_w + random.uniform(-0.5, 0.5)
            cy = (r + 0.5) * cell_h + random.uniform(-0.5, 0.5)
            
            # Evita spawn vicino a (2,2)
            if math.hypot(cx-2.0, cy-2.0) < 2.0: continue

            if i % 2 == 0:
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": random.uniform(0.5, 0.8)})
            else:
                sw, sh = random.uniform(0.8, 1.5), random.uniform(0.8, 1.5)
                self.obstacles.append({"type": "rect", "xmin": cx-sw/2, "xmax": cx+sw/2, "ymin": cy-sh/2, "ymax": cy+sh/2})

    def _setup_scenario(self):
        w, h = self.room_width, self.room_height
        states, goals = [], []
        
        rx, ry, rt = 2.0, 2.0, 0.0
        gx, gy = w-2.0, h-2.0

        if self.scenario_type == "parallel":
            rx, ry, rt = 1.5, h/2, 0.0; gx, gy = w-1.5, h/2
            for i in range(self.num_people):
                y_pos = (i + 1) * (h / (self.num_people + 1))
                if i % 2 == 0:
                    states.append([1.5, y_pos, 0,0, 0,0]); goals.append([w-1.5, y_pos])
                else:
                    states.append([w-1.5, y_pos, 0,0, math.pi,0]); goals.append([1.5, y_pos])
                    
        elif self.scenario_type == "perpendicular":
            rx, ry, rt = 1.5, h/2, 0.0; gx, gy = w-1.5, h/2
            cx = w/2
            for i in range(self.num_people):
                off = random.uniform(-2, 2)
                if i % 2 == 0:
                    states.append([cx+off, 1.5, 0,0, math.pi/2, 0]); goals.append([cx+off, h-1.5])
                else:
                    states.append([cx+off, h-1.5, 0,0, -math.pi/2, 0]); goals.append([cx+off, 1.5])
        
        elif self.scenario_type == "cross":
            rx, ry, rt = 2.0, 2.0, math.pi/4; gx, gy = w-2.0, h-2.0
            pts = [(2.0, h-2.0), (w-2.0, 2.0), (w-2.0, h-2.0)]
            trgs = [(w-2.0, 2.0), (2.0, h-2.0), (2.0, 2.0)]
            for i in range(self.num_people):
                states.append([pts[i%3][0], pts[i%3][1], 0,0,0,0])
                goals.append([trgs[i%3][0], trgs[i%3][1]])
        else:
            rx, ry = random.uniform(1,w-1), random.uniform(1,h-1)
            gx, gy = random.uniform(1,w-1), random.uniform(1,h-1)
            for _ in range(self.num_people):
                states.append([random.uniform(1,w-1), random.uniform(1,h-1), 0,0,0,0])
                goals.append([random.uniform(1,w-1), random.uniform(1,h-1)])

        self.x, self.y, self.theta = rx, ry, rt
        self.goal_x, self.goal_y = gx, gy
        
        self.humans_state_jax = jnp.array(states, dtype=jnp.float32)
        self.humans_goal_jax = jnp.array(goals, dtype=jnp.float32)
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

    def _compute_reward_custom(self, dist_before, dist_after, v, w, lidar):
        reward = 0.0
        reward += 20.0 * (dist_before - dist_after) # Progress
        reward -= 0.05 # Time
        to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        err = (to_goal - self.theta + math.pi) % (2*math.pi) - math.pi
        reward += 0.5 * v * math.cos(err) # Alignment
        if min(lidar) < 0.5: reward -= 0.2 / (min(lidar) + 0.01) # Safety
        return reward

    def _check_termination_custom(self):
        done = False
        info = {"termination_reason": "none"}
        
        if math.hypot(self.x - self.goal_x, self.y - self.goal_y) < 0.3:
            done = True; info["termination_reason"] = "goal_reached"
        elif self._is_collision_with_people(): # Usa metodo padre
            done = True; info["termination_reason"] = "people_collision"
        elif self.step_count >= self.max_steps:
            done = True; info["termination_reason"] = "timeout"
            
        return done, info