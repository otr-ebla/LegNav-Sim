import math
import matplotlib.pyplot as plt 
import random
import numpy as np

from collections import deque
import matplotlib

MAX_LIN_VEL = 0.3  # m/s (TurtleBot4 max linear)
MAX_ANG_VEL = 0.7  # rad/s (TurtleBot4 max angular)
GAP_STATIC = 0.0  # meters di sicurezza extra tra robot e persone
GAP_PEOPLE = 0.2  # meters di sicurezza extra tra robot e persone

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

        self.reward_factor_progress = reward_factor_progress

        # Geometria Stanza
        self.room_width = room_width
        self.room_height = room_height
        self.robot_radius = robot_radius
        self.max_possible_dist = math.sqrt(room_width**2 + room_height**2)

        # [MODIFICATO] Sensori (Specifiche RPLIDAR / TBot4)
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
        self.fig = None
        self.ax = None
        self.render_skip = render_skip
        self.render_counter = 0
        self.progress_reward = 0

        self.manual_skip_triggered = False
        self._listener_attached = False

        # Use legs for rendering humans
        self.use_legs = use_legs  # <--- SALVA STATO
        self.humans_leg_phase = [] 
        self.smooth_v = []

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
            LEG_RADIUS = 0.09
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

    # [MODIFICATO] Calcolo posizione Lidar spostata
    def _compute_lidar(self):
        # Calcola la posizione assoluta del sensore lidar (traslato di lidar_offset lungo l'orientamento)
        lidar_x = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y = self.y + self.lidar_offset * math.sin(self.theta)
        
        angles = [self.theta + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]
        
        # Passiamo le coordinate del Lidar invece di quelle del centro robot
        return [self._cast_ray(a, lidar_x, lidar_y) for a in angles]

    # --- LOGICA FISICA E SAFETY (MODIFICATA PER TBOT4) ---
    def _apply_differential_drive_constraints(self, v, w):
        # [MODIFICATO] Sostituita logica ruote con clamping diretto ai limiti del robot
        # Questo assicura che il robot non superi MAI 0.3 m/s e 2.3 rad/s
        v = max(-self.max_v, min(v, self.max_v))
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
            
            rep_force_mag = 1.5 * urgency**2 

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
        reward = 0.0
        done = False
        info = {}

        if self.manual_skip_triggered:
            obs = self._get_observation(0.0, 0.0)
            return obs, 0.0, True, {"termination_reason": "manual_skip"}

        target_v, target_w = action
        v, w = self._apply_differential_drive_constraints(target_v, target_w)

        dist_to_goal_prev = math.hypot(self.x - self.goal_x, self.y - self.goal_y)

        # Update Physics
        self.theta += w * self.dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        next_x = self.x + v * self.dt * math.cos(self.theta)
        next_y = self.y + v * self.dt * math.sin(self.theta)

        # Collision Check (Safety Layer - Static Obstacles)
        eff_radius = self.robot_radius + GAP_STATIC
        collision_static = False

        if (next_x - eff_radius < 0 or next_x + eff_radius > self.room_width or 
            next_y - eff_radius < 0 or next_y + eff_radius > self.room_height):
            collision_static = True

        if not collision_static:
            rr = eff_radius
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    if (next_x - obs["cx"])**2 + (next_y - obs["cy"])**2 < (rr + obs["radius"])**2:
                        collision_static = True
                        break
                elif obs["type"] == "rect":
                    cx = max(obs["xmin"], min(next_x, obs["xmax"]))
                    cy = max(obs["ymin"], min(next_y, obs["ymax"]))
                    if (next_x - cx)**2 + (next_y - cy)**2 < rr**2:
                        collision_static = True
                        break

        if collision_static:
            reward = -200.0
            done = True
            info["termination_reason"] = "collision_static"
            obs = self._get_observation(v, w)
            return obs, reward, done, info

        self.x = next_x
        self.y = next_y
        self.trajectory.append((self.x, self.y))

        self._step_people()
        self.step_count += 1

        dist_to_goal_now = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        self.episode_path_length += math.hypot(v * self.dt * math.cos(self.theta), 
                                            v * self.dt * math.sin(self.theta))
        self.episode_jerk_sum += abs((w - self.last_w) / self.dt)

        # ==========================================
        # === REWARD SYSTEM ===
        # ==========================================

        # 1. TIME PENALTY
        reward -= 0.01

        # 2. PROGRESS REWARD
        progress = dist_to_goal_prev - dist_to_goal_now
        reward += 5.0 * progress
        self.progress_reward += 5.0 * progress

        # 3. SMOOTHNESS
        jerk_penalty = 0.05 * abs(w - self.last_w)
        reward -= jerk_penalty

        # === YIELDING LOGIC (Directional) ===
        # Calcoliamo l'angolo dell'umano PIÙ VICINO per capire se è davanti o dietro
        closest_human_dist = float('inf')
        closest_human_angle = 0.0
        
        for p in self.people:
            d = math.hypot(p["x"] - self.x, p["y"] - self.y)
            if d < closest_human_dist:
                closest_human_dist = d
                global_angle = math.atan2(p["y"] - self.y, p["x"] - self.x)
                rel_angle = global_angle - self.theta
                closest_human_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        # Yield Parameters
        YIELD_DIST = 1.5
        # FOV ristretto per Yielding: consideriamo solo il cono frontale (+/- 60 gradi)
        # Se l'umano è fuori da questo cono (es. dietro), il robot NON deve rallentare.
        YIELD_FOV = math.radians(120) / 2  
        
        human_in_yield_zone = (closest_human_dist < YIELD_DIST and 
                            abs(closest_human_angle) < YIELD_FOV)

        # 4. YIELD REWARD/PENALTY
        if human_in_yield_zone:
            urgency = (YIELD_DIST - closest_human_dist) / YIELD_DIST
            
            # Penalità se ci muoviamo veloce CONTRO qualcuno davanti
            if v > 0.1:
                yield_violation_penalty = 15.0 * urgency * (v / self.max_v)
                reward -= yield_violation_penalty
                if not hasattr(self, '_yield_violations'): self._yield_violations = 0
                self._yield_violations += 1
            
            # Reward se aspettiamo
            elif v <= 0.1:
                if not hasattr(self, '_time_stopped_in_zone'): self._time_stopped_in_zone = 0
                self._time_stopped_in_zone += 1
                decay_factor = max(0.0, 1.0 - (self._time_stopped_in_zone / 50.0))
                reward += 0.2 * urgency * decay_factor
        else:
            self._time_stopped_in_zone = 0

        # === TERMINATION CONDITIONS ===
        collision_people = self._is_collision_with_people()
        is_goal = self._is_goal_reached()

        if is_goal:
            done = True
            base_goal_reward = 200.0
            time_efficiency = 1.0 - (self.step_count / self.max_steps)
            time_bonus = 100.0 * max(0.0, time_efficiency)
            reward = base_goal_reward + time_bonus
            info["termination_reason"] = "goal_reached"

        elif collision_people:
            done = True
            
            # [MODIFICATO] Analisi Direzionale della Collisione
            # Capire se l'urto arriva da DAVANTI o da DIETRO
            is_front_collision = abs(closest_human_angle) < math.radians(90) # Emisfero frontale
            
            if is_front_collision:
                # Caso A: Urto frontale.
                if v < 0.05 and abs(w) < 0.1:
                     # Robot fermo: Passive Collision (Sfortuna)
                    reward = -20.0
                    info["collision_type"] = "passive_front"
                else:
                    # Robot in movimento: Active Collision (Grave)
                    speed_factor = v / self.max_v
                    reward = -100.0 - (100.0 * speed_factor)
                    info["collision_type"] = "active_front"
            else:
                # Caso B: Urto posteriore (Tamponamento subito)
                if v >= 0:
                    # Stavo andando avanti (o fermo) e mi hanno colpito da dietro.
                    # NON è colpa mia. Penalità minima simbolica.
                    reward = -10.0 
                    info["collision_type"] = "passive_rear"
                else:
                    # Stavo facendo retromarcia (v < 0) e ho investito qualcuno dietro.
                    # Colpa mia grave (blind reversing).
                    reward = -150.0 
                    info["collision_type"] = "active_reverse"
            
            info["termination_reason"] = "people_collision"

        elif self.step_count >= self.max_steps:
            done = True
            remaining_dist = dist_to_goal_now
            dist_penalty = -50.0 * (remaining_dist / self.max_possible_dist)
            reward = -50.0 + dist_penalty
            info["termination_reason"] = "max_steps_reached"

        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")
            info["path_length"] = self.episode_path_length
            info["total_time"] = self.step_count * self.dt
            info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0
            info["progress_reward_total"] = self.progress_reward

        self.last_w = w
        self.last_v = v

        obs = self._get_observation(v, w)
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

    # --- RENDERING (MODIFICATO CON OFFSET) ---
    def render(self):
        self.render_counter += 1
    
        # Renderizza solo ogni N frame per velocità
        if self.render_counter % self.render_skip != 0:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        if not self._listener_attached:
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            self._listener_attached = True

        self.ax.clear()

        # 1. Disegno Stanza
        self.ax.plot([0, self.room_width, self.room_width, 0, 0], [0, 0, self.room_height, self.room_height, 0], 'k-')

        # 2. Disegno Ostacoli Statici
        for obs in self.obstacles:
            if obs["type"] == "circle":
                circle = plt.Circle((obs["cx"], obs["cy"]), obs["radius"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(circle)
            elif obs["type"] == "rect":
                rect = plt.Rectangle((obs["xmin"], obs["ymin"]), obs["xmax"] - obs["xmin"], obs["ymax"] - obs["ymin"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(rect)

        # 3. Disegno Robot
        robot_circle = plt.Circle((self.x, self.y), self.robot_radius, color='blue', fill=True)
        # Cerchio tratteggiato per visualizzare il raggio di collisione esteso (Debug)
        collision_circle = plt.Circle((self.x, self.y), self.robot_radius + GAP_PEOPLE, color='black', fill=False, linestyle='--', linewidth=1, alpha=0.2)

        self.ax.add_patch(robot_circle)
        self.ax.add_patch(collision_circle)

        # Freccia direzione robot
        arrow_len = self.robot_radius * 1.5
        x_head = self.x + arrow_len * math.cos(self.theta)
        y_head = self.y + arrow_len * math.sin(self.theta)
        self.ax.plot([self.x, x_head], [self.y, y_head], 'b-', linewidth=2)

        # 4. DISEGNO PERSONE (Logica Gambe vs Standard)
        LEG_RADIUS = 0.09
        
        for p in self.people:
            # --- MODALITÀ GAMBE ---
            if self.use_legs and "legs" in p:
                # Disegna un "corpo fantasma" trasparente per capire dove si trova il baricentro
                ghost_body = plt.Circle((p["x"], p["y"]), self.people_radius, fill=False, color='green', linestyle=':', alpha=0.4)
                self.ax.add_patch(ghost_body)

                # Parametri Scarpa
                foot_len = 0.30
                foot_width = LEG_RADIUS * 2.0
                foot_theta = p["angle"]
                cos_t = math.cos(foot_theta)
                sin_t = math.sin(foot_theta)

                for lx, ly in p["legs"]:
                    # Calcolo vertice in basso a sinistra del rettangolo scarpa (ruotato)
                    # L'ancoraggio locale è (-R, -R) rispetto al centro della gamba
                    local_x = -LEG_RADIUS
                    local_y = -LEG_RADIUS
                    
                    # Rotazione 2D + Traslazione
                    anchor_x = lx + (local_x * cos_t - local_y * sin_t)
                    anchor_y = ly + (local_x * sin_t + local_y * cos_t)

                    # 1. Disegna Scarpa (Rettangolo)
                    rect = matplotlib.patches.Rectangle(
                        (anchor_x, anchor_y), width=foot_len, height=foot_width,
                        angle=math.degrees(foot_theta),
                        fill=True, facecolor='#D3D3D3', edgecolor='black', # Grigio chiaro con bordo nero
                        linewidth=0.5, alpha=0.9
                    )
                    self.ax.add_patch(rect)

                    # 2. Disegna Tibia (Cerchio Nero)
                    leg_circle = plt.Circle((lx, ly), LEG_RADIUS, color='black', fill=True)
                    self.ax.add_patch(leg_circle)

            # --- MODALITÀ STANDARD (Cerchio Verde) ---
            else:
                person = plt.Circle((p["x"], p["y"]), self.people_radius, color='green', fill=True)
                self.ax.add_patch(person)
            
            # Vettore direzione (comune)
            x_people_head = p["x"] + 0.3*math.cos(p["angle"])
            y_people_head = p["y"] + 0.3*math.sin(p["angle"])
            self.ax.plot([p["x"], x_people_head], [p["y"], y_people_head], 'g-', linewidth=2)

        # 5. Disegno Lidar
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
            
            alpha = 0.9 * proximity
            rgba = (r, g, b, alpha)
            
            if i == 0: 
                rgba = (0.5, 0.7, 1.0, 1.0)
                linewidth = 1.0             
            else:
                linewidth = 0.5

            self.ax.plot([lidar_x_origin, x_end], [lidar_y_origin, y_end], color=rgba, linewidth=linewidth)
            if i != 0:
                self.ax.plot(x_end, y_end, marker='o', markersize=2, color=rgba)

        # 6. Disegno Goal
        if self.goal_x is not None:
            self.ax.plot(self.goal_x, self.goal_y, marker='*', markersize=7, color='orange')

        self.ax.set_xlim(-1, self.room_width + 1)
        self.ax.set_ylim(-1, self.room_height + 1)
        self.ax.set_aspect('equal', adjustable='box')

        status_text = f"Step: {self.step_count} | Mode: {'LEGS' if self.use_legs else 'SIMPLE'}"
        self.ax.text(0.01, 0.99, status_text, verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        plt.pause(0.0001)

    def close(self):
        if self.fig: plt.close(self.fig); self.fig=None

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
            is_distracted = random.random() < 0.6 
            
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
        for obs in self.obstacles:
            if obs["type"]=="circle":
                if (self.x-obs["cx"])**2+(self.y-obs["cy"])**2 < (rr+obs["radius"]+0.2)**2: 
                    #print("Collisione con l'ostacolo circolare:", obs)
                    return True
            elif obs["type"]=="rect":
                cx = max(obs["xmin"], min(self.x, obs["xmax"])); cy = max(obs["ymin"], min(self.y, obs["ymax"]))
                if (self.x-cx)**2+(self.y-cy)**2 < rr**2: return True
        return False
    
    def _is_collision_with_walls(self):
        return (self.x-self.robot_radius<0 or self.x+self.robot_radius>self.room_width or 
                self.y-self.robot_radius<0 or self.y+self.robot_radius>self.room_height)
                
    def _is_collision_with_people(self):
        rr = self.robot_radius
        
        # CASO STANDARD (Cerchio)
        if not self.use_legs:
            # Usa GAP_PEOPLE definito globalmente
            min_sq = (rr + self.people_radius + GAP_PEOPLE)**2
            for p in self.people:
                if (self.x-p["x"])**2+(self.y-p["y"])**2 < min_sq: 
                    return True
            return False

        # CASO AVANZATO (Gambe + Piedi)
        LEG_R, FOOT_L = 0.09, 0.30
        for p in self.people:
            # 1. Check Ghost Body (Sicurezza per non passare ATTRAVERSO la pancia)
            if math.hypot(self.x-p["x"], self.y-p["y"]) < (rr + 0.15): return True
            
            # 2. Check Gambe
            if "legs" in p:
                ct, st = math.cos(p["angle"]), math.sin(p["angle"])
                for lx, ly in p["legs"]:
                    # Collisione Stinco
                    if math.hypot(self.x-lx, self.y-ly) < (rr + LEG_R): return True
                    # Collisione Piede (Segmento)
                    x_s, y_s = lx - LEG_R*ct, ly - LEG_R*st
                    x_e, y_e = lx + (FOOT_L-LEG_R)*ct, ly + (FOOT_L-LEG_R)*st
                    if self._dist_point_to_segment(self.x, self.y, x_s, y_s, x_e, y_e) < (rr + 0.05):
                        return True
        return False
    
    def _is_goal_reached(self):
        if self.goal_x is None: return False
        return (self.x-self.goal_x)**2+(self.y-self.goal_y)**2 <= self.goal_radius**2