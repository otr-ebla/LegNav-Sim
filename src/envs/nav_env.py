import math
import matplotlib.pyplot as plt 
import random
import numpy as np

from collections import deque
import matplotlib
import pygame

MAX_LIN_VEL = 0.3  # m/s (TurtleBot4 max linear)
MAX_ANG_VEL = 0.7  # rad/s (TurtleBot4 max angular)
GAP_STATIC = 0.0  # meters di sicurezza extra tra robot e persone
GAP_PEOPLE = 0.0  # meters di sicurezza extra tra robot e persone

LEG_RADIUS = 0.07  # Raggio per rappresentare le gambe (se use_legs=True)

class Simple2DEnv:
    def __init__(
            self, 
            max_steps: int = 1000, 
            dt: float = 0.1,
            room_width: float = 12.0,
            room_height: float = 12.0,
            robot_radius: float = 0.2, 
            num_rays: int = 108,
            max_lidar_distance: float = 15.0,
            num_people: int = 10,
            people_radius: float = 0.2,
            people_speed: float = 0.0,
            reward_factor_progress: float = 5.0,
            num_obstacles: int = 0,
            render_skip: int = 1,
            use_legs: bool = False,
            human_distraction_prob: float = 0.0,
            lidar_noise_enable: bool = False,
            real_lidar_specs: bool = False,
            ):

        self.max_steps = max_steps
        self.step_count = 0
        self.dt = dt

        # Stato Robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = []
        
        # Variabili Metriche
        self.start_x = 0.0
        self.start_y = 0.0
        self.episode_path_length = 0.0
        self.episode_jerk_sum = 0.0

        # [MODIFICATO] Variabili Fisica (Limiti Reali TurtleBot4)
        self.last_v = 0.0
        self.last_w = 0.0
        self.wheel_separation = 0.233
        # Velocità massime imposte dall'hardware reale
        self.max_v = MAX_LIN_VEL  # m/s (TurtleBot4 max linear)
        self.max_w = MAX_ANG_VEL  # rad/s (TurtleBot4 max angular)
        self.v = 0.0
        self.w = 0.0

        self.reward_factor_progress = reward_factor_progress

        # Geometria Stanza
        self.room_width = room_width
        self.room_height = room_height
        self.robot_radius = robot_radius
        self.max_possible_dist = math.sqrt(room_width**2 + room_height**2)

        self.real_lidar_specs = real_lidar_specs
        
        if self.real_lidar_specs:
            # --- REAL TURTLEBOT4 CONFIGURATION ---
            self.num_rays = 1080
            
            # ORIENTATION FIX: 
            # User noted: "The first ray is to the right of the robot, not in front".
            # In ROS body frame: Front=0, Left=+pi/2, Right=-pi/2.
            # So we shift the starting angle by -90 degrees (-pi/2).
            self.lidar_start_angle_offset = -math.pi / 2.0
            
            # NOISE MODEL (From your 'lidar_static_test2' analysis):
            self.lidar_noise_std = 0.027  # Global Std Dev ~2.7cm
            
            # BLIND SPOTS (From your analysis):
            # You found rays 93-94 were hitting poles. We map them directly.
            self.simulated_blind_indices = [93, 94] 
            self.structural_obstacle_dist = 0.22 # Distance to the pole
            
            # BAD SECTORS (From your analysis):
            # You found instability around indices 580-608.
            self.noisy_sector_indices = list(range(580, 609)) 
            self.noisy_sector_multiplier = 3.0 # Increase noise in this sector
            
            self.lidar_dropout_prob = 0.01 # 1% chance of random max range (glitch)
        else:
            self.lidar_start_angle_offset = 0.0
            self.num_rays = num_rays


        self.max_lidar_distance = max_lidar_distance
        self.lidar_min_distance = 0.12  # Minima distanza rilevabile (blind spot)
        self.lidar_offset = -0.05       # 5cm indietro rispetto al centro

        # Agenti e Ostacoli
        self.num_people = num_people
        self.people_radius = people_radius
        self.people_speed = people_speed
        self.people = []
        self.obstacles = []
        self.num_obstacles = num_obstacles

        # Goal
        self.goal_x = None
        self.goal_y = None
        self.goal_radius = 0.3
        self.last_termination_reason = None

        # Stuck case
        # self.stuck_window = 300
        # self.stuck_threshold = 1.0
        # self.pose_history = deque(maxlen=self.stuck_window)

        # Rendering
        # self.fig = None
        # self.ax = None

        # --- RENDERING CONFIG (PYGAME) ---
        self.global_jerk_sum = 0.0
        self.global_step_count = 0
        self.persistent_outcome = "N/A" # Per mantenere la scritta tra episodi
        self.window_size = 800  # Dimensione finestra in pixel
        self.sidebar_width = 350 # <--- NUOVA COLONNA LATERALE
        self.scale = self.window_size / max(self.room_width, self.room_height) # Pixel per metro
        self.screen = None
        self.clock = None

        self.render_skip = render_skip
        self.render_counter = 0
        self.progress_reward = 0

        self.manual_skip_triggered = False
        self._listener_attached = False

        # Human distraction
        self.human_distraction_prob = human_distraction_prob

        # Use legs for rendering humans
        self.use_legs = use_legs  # <--- SALVA STATO
        self.humans_leg_phase = [] 
        self.smooth_v = []

        # Lidar noise config
        self.lidar_noise_enable = lidar_noise_enable
        self.lidar_noise_std = 0.03  # Standard deviation del rumore
        self.lidar_dropout_prob = 0.03  # Probabilità di dropout per ogni raggio

    # --- LOGICA MOVIMENTO PERSONE ---
    def _handle_person_obstacle_collisions(self, p):
        pr = self.people_radius
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy, radius = obs["cx"], obs["cy"], obs["radius"]
                dx = p["x"] - cx
                dy = p["y"] - cy
                dist_sq = dx*dx + dy*dy
                min_dist = pr + radius
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    if dist == 0: dist = 1e-6
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    p["x"] += nx * overlap
                    p["y"] += ny * overlap
                    v_dot_n = p["vx"] * nx + p["vy"] * ny
                    p["vx"] -= 2 * v_dot_n * nx
                    p["vy"] -= 2 * v_dot_n * ny
                    p["angle"] = math.atan2(p["vy"], p["vx"])

            elif obs["type"] == "rect":
                xmin, xmax = obs["xmin"], obs["xmax"]
                ymin, ymax = obs["ymin"], obs["ymax"]
                closest_x = max(xmin, min(p["x"], xmax))
                closest_y = max(ymin, min(p["y"], ymax))
                dx = p["x"] - closest_x
                dy = p["y"] - closest_y
                dist_sq = dx*dx + dy*dy
                if dist_sq < pr * pr:
                    dist = math.sqrt(dist_sq)
                    if dist == 0: dist = 1e-6
                    overlap = pr - dist
                    nx = dx / dist
                    ny = dy / dist
                    p["x"] += nx * overlap
                    p["y"] += ny * overlap
                    v_dot_n = p["vx"] * nx + p["vy"] * ny
                    p["vx"] -= 2 * v_dot_n * nx
                    p["vy"] -= 2 * v_dot_n * ny
                    p["angle"] = math.atan2(p["vy"], p["vx"])

    def _step_people(self):
        #if self.people_speed == 0: return
        for i, p in enumerate(self.people):

            self._apply_human_robot_repulsion(p)
            p["x"] += p["vx"] * self.dt
            p["y"] += p["vy"] * self.dt
            p["angle"] = math.atan2(p["vy"], p["vx"])

            if p["x"] - self.people_radius < 0:
                p["x"] = self.people_radius; p["vx"] = -p["vx"]
            elif p["x"] + self.people_radius > self.room_width:
                p["x"] = self.room_width - self.people_radius; p["vx"] = -p["vx"]
            
            if p["y"] - self.people_radius < 0:
                p["y"] = self.people_radius; p["vy"] = -p["vy"]
            elif p["y"] + self.people_radius > self.room_height:
                p["y"] = self.room_height - self.people_radius; p["vy"] = -p["vy"]
            
            p["angle"] = math.atan2(p["vy"], p["vx"])
            self._handle_person_obstacle_collisions(p)

            if self.use_legs:
                v_mag = math.hypot(p["vx"], p["vy"])
                smooth_v = self._get_smooth_speed(i, v_mag)
                
                # Aggiorna fase
                self.humans_leg_phase[i] += smooth_v * self.dt * 4.0
                
                # Calcola coordinate
                p["legs"] = self._calculate_leg_positions(
                    p["x"], p["y"], smooth_v, p["angle"], self.humans_leg_phase[i]
                )

    # --- [MODIFICATO] HELPER MATEMATICI RAY CASTING ---
    # Ora accettano x0, y0 come parametri per gestire l'offset del sensore
    def _ray_rect_intersection(self, angle, xmin, xmax, ymin, ymax, x0, y0):
        dx = math.cos(angle)
        dy = math.sin(angle)
        eps = 1e-8
        
        if abs(dx) < eps:
            if x0 < xmin or x0 > xmax: return None
            tmin_x, tmax_x = -math.inf, math.inf
        else:
            t1 = (xmin - x0) / dx
            t2 = (xmax - x0) / dx
            tmin_x, tmax_x = min(t1, t2), max(t1, t2)

        if abs(dy) < eps:
            if y0 < ymin or y0 > ymax: return None
            tmin_y, tmax_y = -math.inf, math.inf
        else:
            t1 = (ymin - y0) / dy
            t2 = (ymax - y0) / dy
            tmin_y, tmax_y = min(t1, t2), max(t1, t2)

        t_near = max(tmin_x, tmin_y)
        t_far = min(tmax_x, tmax_y)

        if t_near > t_far or t_far < 0: return None
        if t_near < 0: return t_far if t_far >= 0 else None
        return t_near

    def _ray_circle_intersection(self, angle, cx, cy, radius, x0, y0):
        dx = math.cos(angle)
        dy = math.sin(angle)
        fx = x0 - cx
        fy = y0 - cy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx*fx + fy*fy - radius*radius
        discriminant = b*b - 4.0*c
        if discriminant < 0: return None
        sqrt_disc = math.sqrt(discriminant)
        candidates = [t for t in [(-b - sqrt_disc)/2.0, (-b + sqrt_disc)/2.0] if t > 0]
        return min(candidates) if candidates else None

    # [MODIFICATO] _cast_ray ora richiede l'origine del raggio
    def _cast_ray(self, angle, x0, y0):
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []
        eps = 1e-6

        # Muri
        if abs(dx) > eps:
            t = (0 - x0) / dx; distances.append(t) if t >= 0 and 0 <= y0 + t * dy <= self.room_height else None
            t = (self.room_width - x0) / dx; distances.append(t) if t >= 0 and 0 <= y0 + t * dy <= self.room_height else None
        if abs(dy) > eps:
            t = (0 - y0) / dy; distances.append(t) if t >= 0 and 0 <= x0 + t * dx <= self.room_width else None
            t = (self.room_height - y0) / dy; distances.append(t) if t >= 0 and 0 <= x0 + t * dx <= self.room_width else None

        # Persone
        # Persone
        if self.use_legs:
            for p in self.people:
                if "legs" in p:
                    # Gamba Sinistra
                    t = self._ray_circle_intersection(angle, p["legs"][0][0], p["legs"][0][1], LEG_RADIUS, x0, y0)
                    if t is not None: distances.append(t)
                    # Gamba Destra
                    t = self._ray_circle_intersection(angle, p["legs"][1][0], p["legs"][1][1], LEG_RADIUS, x0, y0)
                    if t is not None: distances.append(t)
                else:
                    # Fallback
                    t = self._ray_circle_intersection(angle, p["x"], p["y"], self.people_radius, x0, y0)
                    if t is not None: distances.append(t)
        else:
            # Vecchio metodo
            for p in self.people:
                t = self._ray_circle_intersection(angle, p["x"], p["y"], self.people_radius, x0, y0)
                if t is not None: distances.append(t)

        # Ostacoli
        for obs in self.obstacles:
            if obs["type"] == "circle":
                t = self._ray_circle_intersection(angle, obs["cx"], obs["cy"], obs["radius"], x0, y0)
                if t is not None: distances.append(t)
            elif obs["type"] == "rect":
                t = self._ray_rect_intersection(angle, obs["xmin"], obs["xmax"], obs["ymin"], obs["ymax"], x0, y0)
                if t is not None: distances.append(t)

        if not distances: 
            return self.max_lidar_distance
        
        real_dist = min(distances)
        
        # [MODIFICATO] Implementazione limiti fisici del sensore
        # Se l'oggetto è più lontano del range massimo, clamp al massimo
        if real_dist > self.max_lidar_distance:
            return self.max_lidar_distance
        
        # Se l'oggetto è più vicino della distanza minima (blind spot), 
        # il sensore restituisce comunque il valore minimo (o noise, qui usiamo il minimo)
        if real_dist < self.lidar_min_distance:
            return self.lidar_min_distance
            
        return real_dist

    def _compute_lidar(self):
        # Calculate absolute sensor position
        lidar_x = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y = self.y + self.lidar_offset * math.sin(self.theta)
        
        # [MODIFIED] Apply Angular Offset
        # If real_lidar_specs is True, start_angle is (theta - 90 deg).
        # This makes index 0 point RIGHT, index ~270 point FRONT, index ~540 point LEFT.
        start_angle = self.theta + self.lidar_start_angle_offset
        
        # Generate angles (CCW rotation standard)
        angles = [start_angle + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]

        # 1. Geometry Pass (Perfect Ray Casting)
        raw_distances = [self._cast_ray(a, lidar_x, lidar_y) for a in angles]

        final_readings = []

        for i, d in enumerate(raw_distances):
            
            # --- A. STRUCTURAL OBSTACLES (The Poles) ---
            # If we are in "Real Mode", index 93 is physically blocked by a pole.
            # We override the ray cast result with the pole distance.
            if self.real_lidar_specs and i in self.simulated_blind_indices:
                # The sensor sees the pole (approx 0.22m) unless a wall is even closer
                d = min(d, self.structural_obstacle_dist)

            if not self.lidar_noise_enable:
                # Clamp and return
                final_readings.append(max(self.lidar_min_distance, min(d, self.max_lidar_distance)))
                continue

            # --- B. NOISE MODEL ---
            
            # B1. Dropout (Random Glitches)
            if random.random() < self.lidar_dropout_prob:
                final_readings.append(self.max_lidar_distance)
                continue

            # B2. Gaussian Noise (Sector-Dependent)
            sigma = self.lidar_noise_std
            
            # If this ray is in the "Bad Sector" (back of robot), increase noise
            if self.real_lidar_specs and i in self.noisy_sector_indices:
                sigma *= self.noisy_sector_multiplier
            
            noise = random.gauss(0.0, sigma)
            noisy_d = d + noise

            # --- C. SENSOR LIMITS ---
            noisy_d = max(self.lidar_min_distance, min(noisy_d, self.max_lidar_distance))
            
            final_readings.append(noisy_d)
            
        return final_readings

    # --- LOGICA FISICA E SAFETY (MODIFICATA PER TBOT4) ---
    def _apply_differential_drive_constraints(self, v, w):
        # [MODIFICATO] Sostituita logica ruote con clamping diretto ai limiti del robot
        # Questo assicura che il robot non superi MAI 0.3 m/s e 2.3 rad/s
        v = max(0.0, min(v, self.max_v))
        w = max(-self.max_w, min(w, self.max_w))
        return v, w

    def _check_path_existence(self, start_pos, goal_pos, resolution=0.4):
        sx, sy = start_pos; gx, gy = goal_pos; rr = self.robot_radius
        rows = int(math.ceil(self.room_height / resolution))
        cols = int(math.ceil(self.room_width / resolution))
        start_node = (int(sy / resolution), int(sx / resolution))
        goal_node = (int(gy / resolution), int(gx / resolution))
        
        if not (0 <= start_node[0] < rows and 0 <= start_node[1] < cols): return False
        if not (0 <= goal_node[0] < rows and 0 <= goal_node[1] < cols): return False

        queue = [start_node]; visited = {start_node}
        motions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        
        while queue:
            r, c = queue.pop(0)
            if (r, c) == goal_node: return True
            for dr, dc in motions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    wx = (nc * resolution) + (resolution/2); wy = (nr * resolution) + (resolution/2)
                    if (wx-rr<0 or wx+rr>self.room_width or wy-rr<0 or wy+rr>self.room_height):
                        visited.add((nr, nc)); continue
                    collision = False
                    for obs in self.obstacles:
                        if obs["type"] == "circle":
                            if (wx-obs["cx"])**2 + (wy-obs["cy"])**2 < (obs["radius"]+rr)**2: collision=True; break
                        elif obs["type"] == "rect":
                            cx = max(obs["xmin"], min(wx, obs["xmax"])); cy = max(obs["ymin"], min(wy, obs["ymax"]))
                            if (wx-cx)**2+(wy-cy)**2 < rr**2: collision=True; break
                    if not collision: visited.add((nr, nc)); queue.append((nr, nc))
                    else: visited.add((nr, nc))
        return False

    def reset(self):
        self.progress_reward = 0

        self.manual_skip_triggered = False
        self.step_count = 0
        self.last_termination_reason = None
        self.episode_path_length = 0.0
        self.episode_jerk_sum = 0.0

        # self.pose_history.clear()
        # self.pose_history.append((self.x, self.y))

        max_retries = 100
        for _ in range(max_retries):
            self._reset_obstacles()
            margin = self.robot_radius + 0.5
            
            placed_robot = False
            for _ in range(50):
                self.x = random.uniform(margin, self.room_width - margin)
                self.y = random.uniform(margin, self.room_height - margin)
                self.theta = random.uniform(0, 2 * math.pi)
                if not self._is_collision_with_obstacles() and not self._is_collision_with_walls():
                    placed_robot = True; break
            if not placed_robot: continue 
            
            self.trajectory = [(self.x, self.y)]
            self.start_x = self.x; self.start_y = self.y

            placed_goal = False
            for _ in range(50):
                gx = random.uniform(1.0, self.room_width - 1.0)
                gy = random.uniform(1.0, self.room_height - 1.0)
                if math.hypot(gx-self.x, gy-self.y) < 4.0 or self._point_inside_any_obstacle(gx, gy): continue
                goal_unsafe = False
                for obs in self.obstacles:
                    if obs["type"]=="circle" and (gx-obs["cx"])**2+(gy-obs["cy"])**2 < (obs["radius"])**2: goal_unsafe=True
                    elif obs["type"]=="rect":
                        cx = max(obs["xmin"], min(gx, obs["xmax"])); cy = max(obs["ymin"], min(gy, obs["ymax"]))
                        if (gx-cx)**2+(gy-cy)**2 < 0.4**2: goal_unsafe=True
                if not goal_unsafe: self.goal_x=gx; self.goal_y=gy; placed_goal=True; break
            
            if not placed_goal: continue
            if self._check_path_existence((self.x, self.y), (self.goal_x, self.goal_y)): break
        else:
            print("Warning: Could not find valid configuration after retries")

        self.last_v = 0.0; self.last_w = 0.0
        self._reset_people()
        
        self.humans_leg_phase = [random.uniform(0, 2*math.pi) for _ in range(self.num_people)]
        self.smooth_v = [0.0] * self.num_people
        
        return self._get_observation(self.last_v, self.last_w)
    
    # In src/envs/nav_env.py

    def _apply_human_robot_repulsion(self, p):
        # 1. Se l'umano è "distratto", ignora il robot (comportamento ostacolo mobile cieco)
        if p.get("distracted", False):
            return

        # Vettore dal Robot alla Persona
        dx = p["x"] - self.x
        dy = p["y"] - self.y
        dist = math.hypot(dx, dy)

        REACTION_DIST = 1.0 
        
        if dist < REACTION_DIST:
            if dist == 0: dist = 0.01

            urgency = (REACTION_DIST - dist) / REACTION_DIST
            
            rep_force_mag = 13.0 * urgency**2 

            nx = dx / dist
            ny = dy / dist
            

            tx = -ny
            ty = nx
            
            r_cos = math.cos(self.theta)
            r_sin = math.sin(self.theta)

            cross_prod = (r_cos * dy) - (r_sin * dx)
            
            if abs(self.last_v) < 0.05:

                dot_t = (p["vx"] * tx) + (p["vy"] * ty)
                side_sign = 1.0 if dot_t > 0 else -1.0
            else:
                # Schiva dalla parte dove sei già
                side_sign = 1.0 if cross_prod > 0 else -1.0
            
            dodge_mag = 2.0 * urgency 
            
            p["vx"] += (nx * rep_force_mag * 0.3) + (tx * side_sign * dodge_mag)
            p["vy"] += (ny * rep_force_mag * 0.3) + (ty * side_sign * dodge_mag)
            
            # --- C. CLAMP VELOCITÀ ---
            # Evitiamo che l'umano acceleri a velocità sovrumane per schivare
            current_speed = math.hypot(p["vx"], p["vy"])
            # Permettiamo un leggero scatto (1.5x) per emergenza, ma non oltre
            max_reaction_speed = max(self.people_speed * 1.5, 0.5) 
            
            if current_speed > max_reaction_speed:
                scale = max_reaction_speed / current_speed
                p["vx"] *= scale
                p["vy"] *= scale

    def _get_smooth_speed(self, human_index, current_v_mag):
        """Filtro per inerzia animazione gambe"""
        ALPHA = 0.15
        old_v = self.smooth_v[human_index]
        new_v = (ALPHA * current_v_mag) + ((1.0 - ALPHA) * old_v)
        self.smooth_v[human_index] = new_v
        return new_v

    def _calculate_leg_positions(self, x, y, v, theta, leg_phase):
        HIP_SPACING = 0.20
        K_PHASE = 6.0 
        target_amp = math.pi / (2 * K_PHASE)
        stride_amp = target_amp * min(1.0, v / 0.2) if v >= 0.05 else 0.0

        def get_offset(phi):
            p = phi % (2 * math.pi)
            if p < math.pi: return 1.0 - (2.0 * p / math.pi)
            else: return -math.cos(p - math.pi)

        off_l_val = get_offset(leg_phase)
        off_r_val = get_offset(leg_phase + math.pi)
        
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px, py = -sin_t, cos_t 

        # Gamba Sinistra
        lx = x - (px * HIP_SPACING / 2) + (cos_t * stride_amp * off_l_val)
        ly = y - (py * HIP_SPACING / 2) + (sin_t * stride_amp * off_l_val)
        # Gamba Destra
        rx = x + (px * HIP_SPACING / 2) + (cos_t * stride_amp * off_r_val)
        ry = y + (py * HIP_SPACING / 2) + (sin_t * stride_amp * off_r_val)
        
        return [(lx, ly), (rx, ry)]

    def _dist_point_to_segment(self, px, py, x1, y1, x2, y2):
        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0: return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        closest_x = x1 + t * dx; closest_y = y1 + t * dy
        return math.hypot(px - closest_x, py - closest_y)


    # def step(self, action):
    #     reward = 0.0
    #     done = False
    #     info = {}

    #     if self.manual_skip_triggered:
    #         obs = self._get_observation(0.0, 0.0)
    #         return obs, 0.0, True, {"termination_reason": "manual_skip"} 

    #     target_v, target_w = action 
    #     v, w = self._apply_differential_drive_constraints(target_v, target_w)

    #     dist_to_goal_prev = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
    #     #self.pose_history.append((self.x, self.y))

    #     # Update Physics
    #     self.theta += w * self.dt
    #     self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
    #     next_x = self.x + v * self.dt * math.cos(self.theta)
    #     next_y = self.y + v * self.dt * math.sin(self.theta)

    #     # Collision Check (Safety Layer)

    #     INFLATION = GAP_PEOPLE 
    #     eff_radius = self.robot_radius + INFLATION

    #     collision_static = False
    #     if (next_x - eff_radius < 0 or next_x + eff_radius > self.room_width or 
    #         next_y - eff_radius < 0 or next_y + eff_radius > self.room_height): 
    #         collision_static = True
        
    #     if not collision_static:
    #         # 2. Controllo Ostacoli (usando eff_radius)
    #         for obs in self.obstacles:
    #             if obs["type"] == "circle":
    #                 # Distanza Euclidea < (Raggio Robot aumentato + Raggio Ostacolo)
    #                 if (next_x - obs["cx"])**2 + (next_y - obs["cy"])**2 < (eff_radius + obs["radius"])**2: 
    #                     collision_static = True; break
                
    #             elif obs["type"] == "rect":
    #                 # Nearest Point tra cerchio (robot) e rettangolo
    #                 cx = max(obs["xmin"], min(next_x, obs["xmax"]))
    #                 cy = max(obs["ymin"], min(next_y, obs["ymax"]))
    #                 # Distanza dal punto più vicino < Raggio Robot aumentato
    #                 if (next_x - cx)**2 + (next_y - cy)**2 < eff_radius**2: 
    #                     collision_static = True; break
        
    #     # Stuck Detector
    #     # if len(self.pose_history) == self.stuck_window:
    #     #     past_x, past_y = self.pose_history[0]
    #     #     dist_moved = math.hypot(self.x - past_x, self.y - past_y)
            
    #     #     if dist_moved < 0.5 and abs(w) < 0.1: 
    #     #         reward = -50.0 
    #     #         done = True
    #     #         info["termination_reason"] = "stuck"
    #     #         obs = self._get_observation(v, w)
    #     #         return obs, reward, done, info
            
    #     #     if dist_moved < 0.5 and abs(w) >= 0.1:
    #     #         for _ in range(int(self.stuck_window/2)):
    #     #             self.pose_history.popleft()

    #     if collision_static:
    #         reward = -200.0
    #         done = True
    #         info["termination_reason"] = "collision_static"
    #         obs = self._get_observation(v, w)
    #         return obs, reward, done, info

    #     self.x = next_x
    #     self.y = next_y
    #     self.trajectory.append((self.x, self.y))
        
    #     self._step_people() 
    #     self.step_count += 1
        
    #     dist_to_goal_now = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
    #     self.episode_path_length += math.hypot(v * self.dt * math.cos(self.theta), v * self.dt * math.sin(self.theta))
    #     self.episode_jerk_sum += abs((w - self.last_w) / self.dt)




    #     # --------------------------
    #     # --- REWARD CALCULATION ---
    #     # --------------------------

    #     reward -= 0.05
    #     progress_reward = self.reward_factor_progress * (dist_to_goal_prev - dist_to_goal_now)
    #     reward += progress_reward
    #     self.progress_reward += progress_reward

    #     reward -= 0.05 * abs(w - self.last_w)
        
    #     self.last_w = w
    #     self.last_v = v

    #     # --- YIELDING & PRECEDENCE LOGIC (ORACLE BASED) ---
    #     # The environment knows WHO is a person, even if the robot just sees LIDAR.
    #     # This teaches the robot: "Moving patterns in LIDAR = DANGER", "Static patterns = SAFE".
    #     closest_human_dist = float('inf')
    #     closest_human_angle = 0.0
        
    #     for p in self.people:
    #         d = math.hypot(p["x"] - self.x, p["y"] - self.y)
    #         if d < closest_human_dist:
    #             closest_human_dist = d
    #             global_angle = math.atan2(p["y"] - self.y, p["x"] - self.x)
    #             rel_angle = global_angle - self.theta
    #             closest_human_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
    #     # YIELD PARAMETERS
    #     YIELD_DIST = 1.5       
    #     YIELD_FOV = math.radians(120) / 2  # +/- 60 degrees 
    #     human_in_yield_zone = (closest_human_dist < YIELD_DIST and 
    #                            abs(closest_human_angle) < YIELD_FOV)
    #     if human_in_yield_zone:
    #         urgency = (YIELD_DIST - closest_human_dist) / YIELD_DIST

    #         if v > 0.05:
    #             # PENALTY FOR MOVING NEAR HUMANS
    #             yield_violation_penalty = 20.0 * urgency * (v + 0.1)
    #             reward -= yield_violation_penalty
            
    #         elif v <= 0.05 and abs(w) < 0.2:
    #             # REWARD FOR WAITING FOR HUMANS
    #             reward += 0.5 + (0.5 * urgency)
        
    #     # NOTE: No penalty for being near static obstacles (walls) at high speed.
    #     # This teaches the robot to distinguish Static vs Dynamic via LIDAR history.


    #     # 5. Termination Check
    #     collision_people = self._is_collision_with_people()
    #     is_goal = self._is_goal_reached()
        
    #     if is_goal:
    #         done = True
    #         reward = 500.0
    #         info["termination_reason"] = "goal_reached"
            
    #     elif collision_people:
    #         done = True
            
    #         # --- LOGICA "ACTIVE VS PASSIVE COLLISION" ---
    #         # Se il robot era praticamente fermo, diamo per scontato che abbia fatto
    #         # il suo dovere (cedere il passo) e che la colpa sia dell'umano veloce.
    #         if v < 0.05 and abs(w) < 0.1:
    #             # Collisione Passiva: Il robot era fermo in sicurezza.
    #             # Penalità simbolica per l'evento sfortunato, ma non catastrofica.
    #             reward = -10.0 
    #             info["termination_reason"] = "people_collision"
    #         else:
    #             # Collisione Attiva: Il robot si muoveva verso il pericolo.
    #             # Punizione severa per scoraggiare l'impatto.
    #             reward = -200.0
    #             info["termination_reason"] = "people_collision"
                
    #     elif self.step_count >= self.max_steps:
    #         done = True
    #         reward = -100.0
    #         info["termination_reason"] = "max_steps_reached"

    #     if done:
    #         #print(f"Episode progress reward: {self.progress_reward:.2f} over {self.step_count} steps.")
    #         self.last_termination_reason = info.get("termination_reason", "unknown")
    #         info["path_length"] = self.episode_path_length
    #         info["total_time"] = self.step_count * self.dt
    #         info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0

    #     obs = self._get_observation(v, w)
        
    #     return obs, reward, done, info

    def step(self, action):
        # 0. Inizializzazione
        reward = 0.0
        done = False
        info = {}
        dt = self.dt

        # 1. Skip Manuale (Exit Rapido)
        if self.manual_skip_triggered:
            obs = self._get_observation(0.0, 0.0)
            return obs, 0.0, True, {"termination_reason": "manual_skip"}

        # 2. Cinematica & Fisica
        target_v, target_w = action
        self.v, self.w = self._apply_differential_drive_constraints(target_v, target_w)
        
        # Pre-calcolo distanze goal (per reward progress)
        dx_goal_prev = self.x - self.goal_x
        dy_goal_prev = self.y - self.goal_y
        dist_to_goal_prev = math.hypot(dx_goal_prev, dy_goal_prev)

        # Aggiornamento Posizione
        self.theta = (self.theta + self.w * dt + math.pi) % (2 * math.pi) - math.pi
        
        # Ottimizzazione: Calcolo sin/cos una volta sola
        sin_t, cos_t = math.sin(self.theta), math.cos(self.theta)
        step_dist = self.v * dt
        next_x = self.x + step_dist * cos_t
        next_y = self.y + step_dist * sin_t

        # 3. Collision Check Statici (Ottimizzato)
        eff_radius = self.robot_radius + GAP_STATIC
        eff_radius_sq = eff_radius**2
        collision_static = False

        # A. Check Muri (Bounds Check Rapido)
        if not (eff_radius < next_x < self.room_width - eff_radius and 
                eff_radius < next_y < self.room_height - eff_radius):
            collision_static = True
        else:
            # B. Check Ostacoli
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    # Distanza euclidea al quadrato
                    dx_obs = next_x - obs["cx"]
                    dy_obs = next_y - obs["cy"]
                    if (dx_obs*dx_obs + dy_obs*dy_obs) < (eff_radius + obs["radius"])**2:
                        collision_static = True; break
                elif obs["type"] == "rect":
                    # Nearest Point su rettangolo
                    cx = max(obs["xmin"], min(next_x, obs["xmax"]))
                    cy = max(obs["ymin"], min(next_y, obs["ymax"]))
                    if (next_x - cx)**2 + (next_y - cy)**2 < eff_radius_sq:
                        collision_static = True; break

        if collision_static:
            self.last_termination_reason = "collision_static"
            self.persistent_outcome = "collision_static"
            return self._get_observation(self.v, self.w), -200.0, True, {"termination_reason": "collision_static"}

        # 4. Commit dello Stato
        self.x, self.y = next_x, next_y
        self.trajectory.append((self.x, self.y))
        
        self._step_people()
        self.step_count += 1

        # 5. Metriche
        dist_to_goal_now = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        self.episode_path_length += step_dist
        
        current_jerk = abs((self.w - self.last_w) / dt)
        self.episode_jerk_sum += current_jerk
        self.global_jerk_sum += current_jerk # [FIX] Era duplicato nel tuo codice originale
        self.global_step_count += 1

        # 6. Reward Base
        reward -= 0.1 # Time Penalty
        
        progress = dist_to_goal_prev - dist_to_goal_now
        reward += 5.0 * progress
        self.progress_reward += 5.0 * progress
        
        reward -= 0.05 * abs(self.w - self.last_w) # Smoothness Penalty

        # 7. LOGICA UNIFICATA PERSONE (Yielding + Collision + Active/Passive)
        # Ottimizzazione: Iteriamo una sola volta per trovare l'umano più vicino.
        # Questo dato serve sia per il Yielding che per le Collisioni.
        
        closest_dist_sq = float('inf')
        closest_dx = 0.0
        closest_dy = 0.0
        
        for p in self.people:
            dx = p["x"] - self.x
            dy = p["y"] - self.y
            d_sq = dx*dx + dy*dy
            if d_sq < closest_dist_sq:
                closest_dist_sq = d_sq
                closest_dx = dx
                closest_dy = dy
        
        closest_dist = math.sqrt(closest_dist_sq)

        # Calcolo Angolo Relativo (Serve sia per Yielding che per Active Collision)
        global_angle = math.atan2(closest_dy, closest_dx)
        rel_angle = (global_angle - self.theta + math.pi) % (2 * math.pi) - math.pi
        
        # --- A. YIELDING LOGIC ---
        YIELD_DIST = 1.5
        # FOV +/- 60 gradi
        YIELD_FOV_LIMIT = math.radians(60) 
        
        if closest_dist < YIELD_DIST and abs(rel_angle) < YIELD_FOV_LIMIT:
            urgency = (YIELD_DIST - closest_dist) / YIELD_DIST
            
            if self.v > 0.1:
                # Penalità movimento veloce verso umano
                reward -= 15.0 * urgency * (self.v / self.max_v)
                if not hasattr(self, '_yield_violations'): self._yield_violations = 0
                self._yield_violations += 1
            elif self.v <= 0.1:
                # Reward attesa
                if not hasattr(self, '_time_stopped_in_zone'): self._time_stopped_in_zone = 0
                self._time_stopped_in_zone += 1
                decay = max(0.0, 1.0 - (self._time_stopped_in_zone / 50.0))
                reward += 0.2 * urgency * decay
        else:
            self._time_stopped_in_zone = 0

        # --- B. TERMINATION CHECKS ---
        
        # Goal Reached
        if dist_to_goal_now <= self.goal_radius:
            done = True
            time_eff = 1.0 - (self.step_count / self.max_steps)
            reward = 200.0 + (100.0 * max(0.0, time_eff))
            info["termination_reason"] = "goal_reached"

        # People Collision Check
        else:
            # Soglia unificata (Robot + Persona + Gap)
            unified_threshold = self.robot_radius + self.people_radius + GAP_PEOPLE
            
            if closest_dist_sq < unified_threshold**2:
                done = True
                
                # Active vs Passive Classification
                # Riutilizziamo rel_angle calcolato sopra!
                is_in_front = abs(rel_angle) < YIELD_FOV_LIMIT
                is_moving_fast = self.v >= 0.1
                
                if is_moving_fast and is_in_front:
                    # Active Collision
                    speed_factor = self.v / self.max_v
                    reward = -(150.0 * speed_factor)
                    info["termination_reason"] = "people_collision_active"
                    info["collision_type"] = "active"
                else:
                    # Passive Collision
                    reward = -20.0
                    info["termination_reason"] = "people_collision_passive"
                    info["collision_type"] = "passive"

            # Max Steps Timeout
            elif self.step_count >= self.max_steps:
                done = True
                dist_pen = -50.0 * (dist_to_goal_now / self.max_possible_dist)
                reward = -50.0 + dist_pen
                info["termination_reason"] = "max_steps_reached"

        # 8. Finalizzazione Info
        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")
            self.persistent_outcome = self.last_termination_reason
            info["path_length"] = self.episode_path_length
            info["total_time"] = self.step_count * dt
            info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0
            info["progress_reward_total"] = self.progress_reward

        self.last_w = self.w
        self.last_v = self.v

        obs = self._get_observation(self.v, self.w)
        return obs, reward, done, info
    



    def _get_observation(self, v, w):
        """
        Costruisce osservazione COMPLETAMENTE NORMALIZZATA.
        
        Returns:
            np.array([norm_dist, norm_heading, norm_v, norm_w, ...lidar...])
            - norm_dist: [0, 1]
            - norm_heading: [-1, 1]
            - norm_v: [0, 1]
            - norm_w: [-1, 1]
            - lidar: [0, 1] per ogni raggio
        """
        lidar = self._compute_lidar()
        
        # 1. GOAL (Normalizzato)
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        norm_dist = dist_to_goal / self.max_possible_dist
        
        angle_to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        heading_error = angle_to_goal - self.theta
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        norm_heading = heading_error / math.pi
        
        # 2. VELOCITÀ (Normalizzate)
        norm_v = v / self.max_v
        norm_w = w / self.max_w
        
        # 3. LIDAR (Normalizzazione lineare)
        SENSING_HORIZON = self.max_possible_dist
        denominator = SENSING_HORIZON - self.lidar_min_distance
        inverse_lidar = []
        
        for d in lidar:
            inv = (SENSING_HORIZON - d) / denominator
            inverse_lidar.append(max(0.0, min(1.0, inv)))
        
        obs_list = [norm_dist, norm_heading, norm_v, norm_w] + inverse_lidar
        return np.array(obs_list, dtype=np.float32)
    
    # [NUOVO] Callback per pressione tasti
    def _on_key_press(self, event):
        if event.key == 'right':
            print("\n>>> SKIP MANUALE RILEVATO <<<")
            self.manual_skip_triggered = True


    def _to_screen(self, x, y):
        """Converte coordinate Mondo (metri) -> Schermo (pixel)"""
        # PyGame ha l'origine (0,0) in alto a sinistra.
        # Il nostro mondo ha l'origine in basso a sinistra.
        # Quindi la Y va invertita: screen_y = height - (world_y * scale)
        sx = int(x * self.scale)
        sy = int(self.window_size - (y * self.scale))
        return sx, sy

    def render(self):
        self.render_counter += 1
        if self.render_counter % self.render_skip != 0:
            return

        # --- Inizializzazione PyGame (Dimensioni Aumentate) ---
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            
            # Larghezza totale = Stanza + Sidebar
            total_width = self.window_size + self.sidebar_width
            total_height = self.window_size
            
            self.screen = pygame.display.set_mode((total_width, total_height))
            pygame.display.set_caption("Turtlebot4 Simulation - Stats Monitor")
            self.clock = pygame.time.Clock()
            
            # La superficie LIDAR resta grande solo quanto la stanza (per efficienza)
            self.lidar_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            
            # Font
            self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_text = pygame.font.SysFont("Consolas", 18) # Consolas è monospaced, allinea meglio i numeri

        # Gestione eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    print("\n>>> SKIP MANUALE RILEVATO <<<")
                    self.manual_skip_triggered = True

        # 1. Pulisci lo schermo
        self.screen.fill((255, 255, 255)) # Bianco per la stanza
        
        # --- DISEGNO SIDEBAR ---
        # Rettangolo grigio sulla destra
        sidebar_rect = pygame.Rect(self.window_size, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.screen, (240, 240, 240), sidebar_rect) 
        # Linea divisoria nera
        pygame.draw.line(self.screen, (0, 0, 0), (self.window_size, 0), (self.window_size, self.window_size), 3)

        # 2. Pulisci la superficie LIDAR
        self.lidar_surface.fill((0, 0, 0, 0))

        # 3. Disegna Muri (Solo nell'area stanza)
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.window_size, self.window_size), 2)

        # 4. Disegna Ostacoli
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy = self._to_screen(obs["cx"], obs["cy"])
                r = int(obs["radius"] * self.scale)
                pygame.draw.circle(self.screen, (150, 150, 150), (cx, cy), r)
            elif obs["type"] == "rect":
                x_screen, y_screen = self._to_screen(obs["xmin"], obs["ymax"])
                w_screen = int((obs["xmax"] - obs["xmin"]) * self.scale)
                h_screen = int((obs["ymax"] - obs["ymin"]) * self.scale)
                pygame.draw.rect(self.screen, (150, 150, 150), (x_screen, y_screen, w_screen, h_screen))

        # 5. Calcolo LIDAR
        lidar_readings = self._compute_lidar()
        
        lidar_x_origin = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y_origin = self.y + self.lidar_offset * math.sin(self.theta)
        sx_origin, sy_origin = self._to_screen(lidar_x_origin, lidar_y_origin)

        start_angle = self.theta + self.lidar_start_angle_offset
        angles = [start_angle + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]

        # --- DISEGNO RAGGI (Fix Alpha + Visibilità) ---
        # --- DISEGNO RAGGI (Logica: Rosso Vicino / Invisibile Medio / Giallo Noise) ---
        # --- DISEGNO RAGGI (Logica: Rosso Vicino / Invisibile Medio / VIOLA Fondoscala) ---
        for i in range(self.num_rays):
            dist = lidar_readings[i]
            ang = angles[i]
            
            x_end = lidar_x_origin + dist * math.cos(ang)
            y_end = lidar_y_origin + dist * math.sin(ang)
            sx_end, sy_end = self._to_screen(x_end, y_end)

            if i == 0:
                color = (0, 0, 255, 255) # Blue opaco
                thickness = 2
                pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), thickness)
                continue # Passa al prossimo raggio

            # --- CASO 1: FONDOSCALA (Rumore o Fuori Range) ---
            if dist >= self.max_lidar_distance - 0.1:
                # VIOLA ACCESO (R=180, G=0, B=255)
                # Alpha=180 (Ben visibile)
                color = (180, 0, 255, 180) 
                thickness = 1 # <--- Raggi più spessi come richiesto
                
                pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), thickness)
            
            # --- CASO 2: MISURA VALIDA (Muro o Ostacolo) ---
            else:
                color_ray_dist = 5.0
                norm_d = min(dist, color_ray_dist) / color_ray_dist 
                proximity = (1.0 - norm_d)                          
                
                r = int(255 * (0.6 + 0.4 * proximity))
                g = int(255 * (0.6 - 0.6 * proximity))
                b = int(255 * (0.6 - 0.6 * proximity))
                a = int(255 * (0.9 * proximity))
                
                color = (r, g, b, a)

                if color[3] > 0:
                    # Raggi normali sottili (spessore 1)
                    pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), 1)

        # Sovrapponi il LIDAR (Solo sulla parte stanza)
        self.screen.blit(self.lidar_surface, (0, 0))

        # 6. Disegna Robot
        rx, ry = self._to_screen(self.x, self.y)
        rr = int(self.robot_radius * self.scale)
        pygame.draw.circle(self.screen, (0, 0, 255), (rx, ry), rr) 
        
        head_x = self.x + (self.robot_radius * 1.5) * math.cos(self.theta)
        head_y = self.y + (self.robot_radius * 1.5) * math.sin(self.theta)
        hx, hy = self._to_screen(head_x, head_y)
        pygame.draw.line(self.screen, (0, 0, 100), (rx, ry), (hx, hy), 3)

        # 7. Disegna Persone
        # 7. Draw People
        for p in self.people:
            # Legs / Shoes Rendering
            if self.use_legs and "legs" in p:
                foot_len = 0.30
                
                # Colors
                color_shoe_fill = (169, 169, 169) # Darker Grey
                color_shoe_edge = (50, 50, 50)    # Almost Black
                color_leg_circle = (0, 0, 0)      # Black (Ankle)

                foot_theta = p["angle"]
                cos_t = math.cos(foot_theta)
                sin_t = math.sin(foot_theta)

                for lx, ly in p["legs"]:
                    # [MODIFIED] Polygon definition for a "Shoe" shape
                    # Coordinates are local to the ankle (0,0)
                    # Format: (forward_offset, side_offset)
                    local_verts = [
                        (-LEG_RADIUS, -LEG_RADIUS),           # 1. Back Left (Heel)
                        (-LEG_RADIUS, LEG_RADIUS),            # 2. Back Right (Heel)
                        (foot_len * 0.65, LEG_RADIUS),        # 3. Ball of foot Right (Widest point)
                        (foot_len, LEG_RADIUS * 0.35),        # 4. Toe Tip Right (Tapered)
                        (foot_len, -LEG_RADIUS * 0.35),       # 5. Toe Tip Left (Tapered)
                        (foot_len * 0.65, -LEG_RADIUS)        # 6. Ball of foot Left (Widest point)
                    ]

                    screen_verts = []
                    for loc_x, loc_y in local_verts:
                        # Apply Rotation Matrix (Body Frame -> World Frame)
                        rot_x = loc_x * cos_t - loc_y * sin_t
                        rot_y = loc_x * sin_t + loc_y * cos_t
                        
                        # Convert to Screen Pixels
                        screen_verts.append(self._to_screen(lx + rot_x, ly + rot_y))

                    # Draw the shoe polygon
                    pygame.draw.polygon(self.screen, color_shoe_fill, screen_verts)
                    pygame.draw.polygon(self.screen, color_shoe_edge, screen_verts, 1)
                    
                    # Draw the ankle circle
                    slx, sly = self._to_screen(lx, ly)
                    pygame.draw.circle(self.screen, color_leg_circle, (slx, sly), int(LEG_RADIUS * self.scale))

            else:
                # [FIX] FALLBACK: Disegna cerchio verde per il corpo
                cx, cy = self._to_screen(p["x"], p["y"])
                r = int(self.people_radius * self.scale)
                pygame.draw.circle(self.screen, (0, 255, 0), (cx, cy), r)

        # 8. Disegna Goal
        if self.goal_x is not None:
            gx, gy = self._to_screen(self.goal_x, self.goal_y)
            num_points = 5
            radius_ext = int(0.15 * self.scale)
            radius_int = int(0.06 * self.scale)
            star_points = []
            for i in range(num_points * 2):
                radius = radius_ext if i % 2 == 0 else radius_int
                angle = i * math.pi / num_points - math.pi / 2
                px = gx + radius * math.cos(angle)
                py = gy + radius * math.sin(angle)
                star_points.append((px, py))
            pygame.draw.polygon(self.screen, (255, 165, 0), star_points)
            pygame.draw.polygon(self.screen, (200, 100, 0), star_points, 2)


        # --- 9. STATISTICHE NELLA COLONNA DESTRA (SIDEBAR) ---
        
        x_text = self.window_size + 20
        y_text = 20
        line_spacing = 25

        # Dati Calcolati
        dist_to_goal = math.hypot(self.x - self.goal_x, self.y - self.goal_y) if self.goal_x is not None else 0.0
        min_laser_dist = min(lidar_readings) if (lidar_readings and len(lidar_readings) > 0) else 0.0
        
        # Calcolo Total Average Jerk
        total_avg_jerk = 0.0
        if self.global_step_count > 0:
            total_avg_jerk = self.global_jerk_sum / self.global_step_count

        # --- TITOLO ---
        title = self.font_title.render("TELEMETRY", True, (0, 0, 0))
        self.screen.blit(title, (x_text, y_text))
        y_text += 40

        # Funzione helper per righe
        def draw_stat_line(label, value, color=(0,0,0)):
            label_surf = self.font_text.render(f"{label}:", True, (80, 80, 80))
            value_surf = self.font_text.render(str(value), True, color)
            self.screen.blit(label_surf, (x_text, y_text))
            val_w = value_surf.get_width()
            val_x = (self.window_size + self.sidebar_width) - val_w - 20
            self.screen.blit(value_surf, (val_x, y_text))
            return line_spacing

        # --- DATI EPISODIO ---
        y_text += draw_stat_line("Ep. Step", f"{self.step_count}")
        y_text += draw_stat_line("Goal Dist", f"{dist_to_goal:.2f} m")
        
        laser_color = (200, 0, 0) if min_laser_dist < 0.5 else (0, 100, 0)
        y_text += draw_stat_line("Lidar Min", f"{min_laser_dist:.2f} m", laser_color)
        
        y_text += 10 # Spacer
        
        # --- DATI GLOBALI ---
        y_text += draw_stat_line("Avg Jerk (Tot)", f"{total_avg_jerk:.2f}")

        # --- GRAFICO V/W (ACTION PLOT) ---
        y_text += 40
        
        # Configurazione Grafico
        plot_w = 200
        plot_h = 150
        plot_x = x_text + (self.sidebar_width - 20 - plot_w) // 2 # Centrato
        plot_y = y_text
        
        # Sfondo Grafico
        pygame.draw.rect(self.screen, (230, 230, 230), (plot_x, plot_y, plot_w, plot_h))
        pygame.draw.rect(self.screen, (0, 0, 0), (plot_x, plot_y, plot_w, plot_h), 2)
        
        # Assi
        # Asse W=0 (Orizzontale, a metà altezza)
        mid_y = plot_y + plot_h / 2
        pygame.draw.line(self.screen, (150, 150, 150), (plot_x, mid_y), (plot_x + plot_w, mid_y), 1)
        # Asse V=0 (Verticale, a sinistra)
        pygame.draw.line(self.screen, (150, 150, 150), (plot_x, plot_y), (plot_x, plot_y + plot_h), 1)

        # Etichette Assi
        font_small = pygame.font.SysFont("Arial", 12)
        lbl_w_max = font_small.render(f"+{self.max_w}", True, (0,0,0)); self.screen.blit(lbl_w_max, (plot_x - 25, plot_y))
        lbl_w_min = font_small.render(f"-{self.max_w}", True, (0,0,0)); self.screen.blit(lbl_w_min, (plot_x - 25, plot_y + plot_h - 10))
        lbl_v_max = font_small.render(f"{self.max_v} m/s", True, (0,0,0)); self.screen.blit(lbl_v_max, (plot_x + plot_w - 30, plot_y + plot_h + 5))
        lbl_chart = font_small.render("V / W Space", True, (0,0,0)); self.screen.blit(lbl_chart, (plot_x + 5, plot_y + 5))

        # Calcolo Posizione Pallino (Current Action)
        # X = V (0 a Max)
        dot_x_norm = self.v / self.max_v
        dot_screen_x = plot_x + int(dot_x_norm * plot_w)
        
        # Y = W (-Max a +Max). Nota: Y cresce verso il basso in Pygame
        # Normalize W to [-1, 1]
        dot_y_norm = self.w / self.max_w 
        # Invertiamo il segno perché Y in pygame va giù, ma noi vogliamo +W in alto
        dot_screen_y = mid_y - int(dot_y_norm * (plot_h / 2))
        
        # Disegna Pallino
        pygame.draw.circle(self.screen, (255, 0, 0), (dot_screen_x, dot_screen_y), 6)
        
        # Linee di proiezione (opzionali, per bellezza)
        pygame.draw.line(self.screen, (255, 100, 100), (plot_x, dot_screen_y), (dot_screen_x, dot_screen_y), 1)
        pygame.draw.line(self.screen, (255, 100, 100), (dot_screen_x, mid_y), (dot_screen_x, dot_screen_y), 1)

        y_text = plot_y + plot_h + 30

        # --- LAST EPISODE OUTCOME (PERSISTENTE) ---
        outcome_color = (0, 0, 0)
        txt = str(self.persistent_outcome).lower()
        
        # Logica Colori Aggiornata
        if "goal" in txt: 
            outcome_color = (0, 150, 0)   # Verde
        elif "passive" in txt: 
            outcome_color = (0, 150, 0)   # Verde (Collisione Passiva è OK)
        elif "collision" in txt: 
            outcome_color = (200, 0, 0)   # Rosso (Collisione Attiva/Statica)
        elif "timeout" in txt: 
            outcome_color = (150, 100, 0) # Arancione
        
        # Etichetta
        self.screen.blit(self.font_text.render("Last Outcome:", True, (0,0,0)), (x_text, y_text))
        y_text += 20 # Riduco leggermente lo spazio verticale
        
        # Scritta Outcome rimpicciolita (Arial 16 Bold invece di 24)
        font_outcome_small = pygame.font.SysFont("Arial", 16, bold=True)
        outcome_surf = font_outcome_small.render(self.persistent_outcome.upper(), True, outcome_color)
        self.screen.blit(outcome_surf, (x_text, y_text))

        self.clock.tick(30)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # --- METODI HELPER RESET ---
    def _reset_obstacles(self):
        # Configurazione Margini
        # Robot Radius = 0.2m -> Diametro 0.4m
        # Vogliamo un passaggio di almeno 0.6m (0.4 robot + 0.2 aria)
        # Quindi il centro del robot deve stare a (0.6 / 2) = 0.3m dagli ostacoli.
        REQUIRED_CLEARANCE = 0.3 # 0.2 radius + 0.1 margin
        
        max_attempts = 100
        for attempt in range(max_attempts):
            self.obstacles = []
            cols = 4; rows = 3
            cell_w = self.room_width / cols
            cell_h = self.room_height / rows
            
            # Generazione layout (come prima)
            cell_indices = [i for i in range(cols * rows)]
            while len(cell_indices) < self.num_obstacles: 
                cell_indices.extend([i for i in range(cols * rows)])
            random.shuffle(cell_indices)
            
            for i in range(self.num_obstacles):
                idx = cell_indices[i]
                r_idx = idx // cols; c_idx = idx % cols
                margin = 0.5 
                min_x = c_idx * cell_w + margin; max_x = (c_idx + 1) * cell_w - margin
                min_y = r_idx * cell_h + margin; max_y = (r_idx + 1) * cell_h - margin
                
                cx = random.uniform(min_x, max_x); cy = random.uniform(min_y, max_y)
                
                if i % 2 == 0:
                    r = random.uniform(0.5, 0.8) # Raggio ridotto per favorire passaggi
                    self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": r})
                else:
                    w = random.uniform(0.8, 1.2); h = random.uniform(0.8, 1.2)
                    self.obstacles.append({"type": "rect", "xmin": cx - w/2, "xmax": cx + w/2, "ymin": cy - h/2, "ymax": cy + h/2})

            # VALIDAZIONE DEL LAYOUT
            # Verifichiamo se esiste passaggio considerando il robot "ingrassato"
            if self._check_environment_connectivity(clearance=REQUIRED_CLEARANCE):
                return # Layout approvato!

        print("Warning: Could not generate valid obstacle layout with guaranteed passages.")

    def _check_environment_connectivity(self, clearance=0.3):
        """
        Verifica la connettività usando una griglia.
        Le celle sono segnate come OCCUPATE se sono entro 'clearance' metri da un ostacolo.
        """
        resolution = 0.2 # 20cm per cella (sufficiente per questa verifica)
        rows = int(self.room_height / resolution)
        cols = int(self.room_width / resolution)
        grid = np.zeros((rows, cols), dtype=int) # 0=Free, 1=Blocked
        
        # 1. Rasterizzazione Ostacoli "Gonfiati"
        # Se un punto è a meno di 'clearance' da un ostacolo, è impraticabile per il centro del robot
        
        for r in range(rows):
            for c in range(cols):
                wx = (c * resolution) + (resolution/2)
                wy = (r * resolution) + (resolution/2)
                
                # Check Muri (Anche i muri hanno spessore per il robot)
                if (wx - clearance < 0 or wx + clearance > self.room_width or 
                    wy - clearance < 0 or wy + clearance > self.room_height):
                    grid[r, c] = 1
                    continue
                
                # Check Ostacoli
                collision = False
                for obs in self.obstacles:
                    if obs["type"] == "circle":
                        # Distanza dal centro < (Raggio Ostacolo + Clearance)
                        if (wx - obs["cx"])**2 + (wy - obs["cy"])**2 < (obs["radius"] + clearance)**2:
                            collision = True; break
                    elif obs["type"] == "rect":
                        # Rettangolo espanso di 'clearance'
                        expanded_xmin = obs["xmin"] - clearance
                        expanded_xmax = obs["xmax"] + clearance
                        expanded_ymin = obs["ymin"] - clearance
                        expanded_ymax = obs["ymax"] + clearance
                        
                        if expanded_xmin <= wx <= expanded_xmax and expanded_ymin <= wy <= expanded_ymax:
                            collision = True; break
                
                if collision:
                    grid[r, c] = 1

        # 2. Conta celle libere totali
        free_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    free_cells.append((r, c))
        
        if not free_cells: return False # Stanza completamente bloccata

        # 3. Flood Fill dal primo punto libero
        start_node = free_cells[0]
        queue = [start_node]
        visited = {start_node}
        count = 0
        
        while queue:
            r, c = queue.pop(0)
            count += 1
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        # 4. Criterio di Accettazione
        # Se l'area connessa più grande copre almeno il 90% dello spazio libero teorico,
        # significa che non ci sono grandi sezioni isolate irraggiungibili.
        return count >= (len(free_cells) * 0.90)

    def _reset_people(self):
        self.people = []
        margin = 2.0
        for _ in range(self.num_people):
            px, py = self.x, self.y
            while math.hypot(px-self.x, py-self.y) < 2.0 or self._point_inside_any_obstacle(px, py):
                px = random.uniform(margin, self.room_width-margin)
                py = random.uniform(margin, self.room_height-margin)
            angle = random.uniform(0, 2*math.pi)
            vx = self.people_speed * math.cos(angle); vy = self.people_speed * math.sin(angle)
            
            is_distracted = random.random() < self.human_distraction_prob 
            
            self.people.append({
                "x": px, "y": py, 
                "vx": vx, "vy": vy, 
                "angle": angle,
                "distracted": is_distracted # Nuova proprietà
            })

    def _point_inside_any_obstacle(self, x, y):
        for obs in self.obstacles:
            if self._point_inside_obstacle(x, y, obs): return True
        return False
    
    def _point_inside_obstacle(self, x, y, obs):
        if obs["type"] == "circle":
            return (x - obs["cx"])**2 + (y - obs["cy"])**2 < obs["radius"]**2
        elif obs["type"] == "rect":
            return obs["xmin"] <= x <= obs["xmax"] and obs["ymin"] <= y <= obs["ymax"]
        return False
        
    def _is_collision_with_obstacles(self):
        rr = self.robot_radius
        reff = rr + GAP_STATIC
        for obs in self.obstacles:
            if obs["type"]=="circle":
                if (self.x-obs["cx"])**2+(self.y-obs["cy"])**2 < (reff+obs["radius"])**2: 
                    #print("Collisione con l'ostacolo circolare:", obs)
                    return True
            elif obs["type"]=="rect":
                cx = max(obs["xmin"], min(self.x, obs["xmax"])); cy = max(obs["ymin"], min(self.y, obs["ymax"]))
                if (self.x-cx)**2+(self.y-cy)**2 < reff**2: return True
        return False
    
    def _is_collision_with_walls(self):
        return (self.x-self.robot_radius<0 or self.x+self.robot_radius>self.room_width or 
                self.y-self.robot_radius<0 or self.y+self.robot_radius>self.room_height)
                
    def _is_collision_with_people(self):
        """
        Simplified collision check: 
        Treats every person as a cylinder with a fixed safety radius.
        Unified for both simple and leg rendering modes.
        """
        # UNIFIED THRESHOLD: Robot (0.2) + Person (0.2) + GAP (0.2) = 0.6m
        # This is the 'hard limit' that terminates the episode.
        rr = self.robot_radius  # 0.2
        pr = self.people_radius # 0.2
        unified_threshold = rr + pr + GAP_PEOPLE # 0.6m
        min_sq = unified_threshold**2

        for p in self.people:
            # Simple Euclidean distance check between centers
            dist_sq = (self.x - p["x"])**2 + (self.y - p["y"])**2
            if dist_sq < min_sq:
                return True
    
    def _is_goal_reached(self):
        if self.goal_x is None: return False
        return (self.x-self.goal_x)**2+(self.y-self.goal_y)**2 <= self.goal_radius**2
    
































    # --- RENDERING (MODIFICATO CON OFFSET) ---
    # def render(self):
    #     self.render_counter += 1
    
    #     if self.render_counter % self.render_skip != 0:
    #         return

    #     if self.fig is None:
    #         self.fig, self.ax = plt.subplots()
    #         plt.ion()

    #     if not self._listener_attached:
    #         self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    #         self._listener_attached = True

    #     self.ax.clear()

    #     # 1. Muri
    #     self.ax.plot([0, self.room_width, self.room_width, 0, 0], [0, 0, self.room_height, self.room_height, 0], 'k-')

    #     # 2. Ostacoli
    #     for obs in self.obstacles:
    #         if obs["type"] == "circle":
    #             circle = plt.Circle((obs["cx"], obs["cy"]), obs["radius"], color='gray', alpha=0.7, fill=True)
    #             self.ax.add_patch(circle)
    #         elif obs["type"] == "rect":
    #             rect = plt.Rectangle((obs["xmin"], obs["ymin"]), obs["xmax"] - obs["xmin"], obs["ymax"] - obs["ymin"], color='gray', alpha=0.7, fill=True)
    #             self.ax.add_patch(rect)

    #     # 3. Robot
    #     robot_circle = plt.Circle((self.x, self.y), self.robot_radius, color='blue', fill=True)
    #     collision_circle = plt.Circle((self.x, self.y), self.robot_radius + GAP_PEOPLE, color='black', fill=False, linestyle='--', linewidth=1, alpha=0.2)
    #     self.ax.add_patch(robot_circle)
    #     self.ax.add_patch(collision_circle)

    #     # Freccia direzione
    #     arrow_len = self.robot_radius * 1.5
    #     x_head = self.x + arrow_len * math.cos(self.theta)
    #     y_head = self.y + arrow_len * math.sin(self.theta)
    #     self.ax.plot([self.x, x_head], [self.y, y_head], 'b-', linewidth=2)

    #     # 4. Persone
    #     LEG_RADIUS = 0.09
    #     for p in self.people:
    #         if self.use_legs and "legs" in p:
    #             foot_len = 0.30
    #             foot_width = LEG_RADIUS * 2.0
    #             foot_theta = p["angle"]
    #             cos_t = math.cos(foot_theta)
    #             sin_t = math.sin(foot_theta)

    #             for lx, ly in p["legs"]:
    #                 local_x, local_y = -LEG_RADIUS, -LEG_RADIUS
    #                 anchor_x = lx + (local_x * cos_t - local_y * sin_t)
    #                 anchor_y = ly + (local_x * sin_t + local_y * cos_t)
    #                 rect = matplotlib.patches.Rectangle(
    #                     (anchor_x, anchor_y), width=foot_len, height=foot_width,
    #                     angle=math.degrees(foot_theta),
    #                     fill=True, facecolor='#D3D3D3', edgecolor='black',
    #                     linewidth=0.5, alpha=0.9
    #                 )
    #                 self.ax.add_patch(rect)
    #                 leg_circle = plt.Circle((lx, ly), LEG_RADIUS, color='black', fill=True)
    #                 self.ax.add_patch(leg_circle)
    #         else:
    #             person = plt.Circle((p["x"], p["y"]), self.people_radius, color='green', fill=True)
    #             self.ax.add_patch(person)

    #     # 5. LIDAR FIX (VISUALIZZAZIONE)
    #     lidar = self._compute_lidar()
        
    #     # [FIX 1] Applicare lo STESSO offset usato nel calcolo fisico
    #     start_angle = self.theta + self.lidar_start_angle_offset
    #     angles = [start_angle + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]
        
    #     lidar_x_origin = self.x + self.lidar_offset * math.cos(self.theta)
    #     lidar_y_origin = self.y + self.lidar_offset * math.sin(self.theta)

    #     # [FIX 2] OTTIMIZZAZIONE GRAFICA
    #     # Se usiamo il lidar reale (1080 raggi), disegniamone solo 1 ogni 10 (Stride)
    #     # Questo riduce il carico grafico da 1080 oggetti a 108, rendendo tutto fluido.
    #     VISUAL_STRIDE = 1 if self.real_lidar_specs else 1

    #     for i in range(0, self.num_rays, VISUAL_STRIDE):
    #         dist = lidar[i]
    #         ang = angles[i]
            
    #         x_end = lidar_x_origin + dist * math.cos(ang)
    #         y_end = lidar_y_origin + dist * math.sin(ang)

    #         # Colorazione
    #         color_ray_dist = 5.0
    #         norm_d = min(dist, color_ray_dist) / color_ray_dist
    #         proximity = (1.0 - norm_d) 
    #         rgba = (0.6 + 0.4*proximity, 0.6 - 0.6*proximity, 0.6 - 0.6*proximity, 0.9 * proximity)
            
    #         # Linea sottile
    #         self.ax.plot([lidar_x_origin, x_end], [lidar_y_origin, y_end], color=rgba, linewidth=0.5)
    #         # Pallino finale
    #         self.ax.plot(x_end, y_end, marker='o', markersize=1, color=rgba)

    #     # Disegna SEMPRE il raggio 0 per debug (per vedere dove "guarda" l'indice 0)
    #     dist0 = lidar[0]
    #     ang0 = angles[0]
    #     self.ax.plot(
    #         [lidar_x_origin, lidar_x_origin + dist0*math.cos(ang0)], 
    #         [lidar_y_origin, lidar_y_origin + dist0*math.sin(ang0)], 
    #         color='cyan', linewidth=2.0, label='Ray 0'
    #     )

    #     # 6. Goal
    #     if self.goal_x is not None:
    #         self.ax.plot(self.goal_x, self.goal_y, marker='*', markersize=7, color='orange')

    #     self.ax.set_xlim(-1, self.room_width + 1)
    #     self.ax.set_ylim(-1, self.room_height + 1)
    #     self.ax.set_aspect('equal', adjustable='box')
        
    #     plt.pause(0.0001)




    # def close(self):
    #     if self.fig: plt.close(self.fig); self.fig=None