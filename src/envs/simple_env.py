import math
import matplotlib.pyplot as plt 
import random
import numpy as np

class Simple2DEnv:
    def __init__(
            self, 
            max_steps: int = 1000, 
            dt: float = 0.1,
            room_width: float = 12.0,
            room_height: float = 12.0,
            robot_radius: float = 0.17, 
            num_rays: int = 108,
            max_lidar_distance: float = 12.0,
            num_people: int = 10,
            people_radius: float = 0.25,
            people_speed: float = 0.0,
            ):

        self.max_steps = max_steps
        self.step_count = 0
        self.dt = dt

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = []
        
        # ### NEW METRICS: Variabili di stato per metriche ###
        self.start_x = 0.0
        self.start_y = 0.0
        self.episode_path_length = 0.0
        self.episode_jerk_sum = 0.0
        # ###################################################

        self.last_v = 0.0
        self.last_w = 0.0
        
        self.wheel_separation = 0.233
        self.max_wheel_speed = 0.46

        self.room_width = room_width
        self.room_height = room_height
        self.robot_radius = robot_radius

        self.num_rays = num_rays
        self.max_lidar_distance = max_lidar_distance

        self.num_people = num_people
        self.people_radius = people_radius
        self.people_speed = people_speed
        self.people = []
        self.obstacles = []
        self.num_obstacles = 12

        self.goal_x = None
        self.goal_y = None
        self.goal_radius = 0.3
        self.last_termination_reason = None

        self.fig = None
        self.ax = None

    # ... [MANTIENI _apply_differential_drive_constraints e _check_path_existence UGUALI] ...
    # (Per brevità non li ricopio qui, sono identici al tuo file precedente)
    def _apply_differential_drive_constraints(self, v, w):
        half_base = self.wheel_separation / 2.0
        vr = v + (w * half_base)
        vl = v - (w * half_base)
        vr = max(-self.max_wheel_speed, min(vr, self.max_wheel_speed))
        vl = max(-self.max_wheel_speed, min(vl, self.max_wheel_speed))
        v_act = (vr + vl) / 2.0
        w_act = (vr - vl) / self.wheel_separation
        return v_act, w_act

    def _check_path_existence(self, start_pos, goal_pos, resolution=0.4):
        # ... (Copia incolla il metodo dal tuo file precedente o dalla mia risposta precedente) ...
        # È fondamentale che questo resti uguale per non rompere il reset
        sx, sy = start_pos
        gx, gy = goal_pos
        rr = self.robot_radius
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
                    wx = (nc * resolution) + (resolution / 2); wy = (nr * resolution) + (resolution / 2)
                    if (wx - rr < 0 or wx + rr > self.room_width or wy - rr < 0 or wy + rr > self.room_height):
                        visited.add((nr, nc)); continue
                    collision = False
                    for obs in self.obstacles:
                        if obs["type"] == "circle":
                            if (wx - obs["cx"])**2 + (wy - obs["cy"])**2 < (obs["radius"] + rr)**2: collision = True; break
                        elif obs["type"] == "rect":
                            cx = max(obs["xmin"], min(wx, obs["xmax"])); cy = max(obs["ymin"], min(wy, obs["ymax"]))
                            if (wx-cx)**2 + (wy-cy)**2 < rr**2: collision = True; break
                    if not collision: visited.add((nr, nc)); queue.append((nr, nc))
                    else: visited.add((nr, nc))
        return False

    def reset(self):
        self.step_count = 0
        self.last_termination_reason = None
        
        # ### NEW METRICS: Reset contatori ###
        self.episode_path_length = 0.0
        self.episode_jerk_sum = 0.0
        # ####################################

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
            
            # ### NEW METRICS: Salva Start Position ###
            self.start_x = self.x
            self.start_y = self.y
            # #########################################

            placed_goal = False
            for _ in range(50):
                gx = random.uniform(1.0, self.room_width - 1.0)
                gy = random.uniform(1.0, self.room_height - 1.0)
                dist = math.hypot(gx - self.x, gy - self.y)
                if dist < 4.0 or self._point_inside_any_obstacle(gx, gy): continue
                goal_collides = False
                for obs in self.obstacles:
                    if obs["type"] == "circle":
                        if (gx-obs["cx"])**2+(gy-obs["cy"])**2 < (obs["radius"]+0.4)**2: goal_collides=True
                    elif obs["type"] == "rect":
                        cx = max(obs["xmin"], min(gx, obs["xmax"])); cy = max(obs["ymin"], min(gy, obs["ymax"]))
                        if (gx-cx)**2+(gy-cy)**2 < 0.4**2: goal_collides=True
                if goal_collides: continue
                self.goal_x = gx; self.goal_y = gy
                placed_goal = True; break
            
            if not placed_goal: continue
            if self._check_path_existence((self.x, self.y), (self.goal_x, self.goal_y)): break
        else:
            print("Warning: Could not find valid configuration after retries")

        self.last_v = 0.0
        self.last_w = 0.0
        self._reset_people()

        lidar = self._compute_lidar()
        obs = (self.x, self.y, self.theta, self.last_v, self.last_w, lidar)
        return obs

    def step(self, action):
        target_v, target_w = action 
        v, w = self._apply_differential_drive_constraints(target_v, target_w)


        # --- 1. PREDIZIONE POSIZIONE FUTURA ---
        next_x = self.x + v * self.dt * math.cos(self.theta)
        next_y = self.y + v * self.dt * math.sin(self.theta)
        # (Nota: ignoriamo cambio theta per semplicità del check, o puoi calcolarlo)
        
        # --- 2. SAFETY LAYER (Il trucco per il 90% SR) ---
        # Controlliamo se la PROSSIMA posizione causerebbe collisione
        # Usiamo una funzione helper temporanea o logica inline
        will_collide = False
        
        # Check Muri
        if (next_x - self.robot_radius < 0 or next_x + self.robot_radius > self.room_width or 
            next_y - self.robot_radius < 0 or next_y + self.robot_radius > self.room_height):
            will_collide = True
            
        # Check Ostacoli Statici (Solo questi, le persone le lasciamo "morbide" o gestite dopo)
        if not will_collide:
            rr = self.robot_radius
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    if (next_x-obs["cx"])**2 + (next_y-obs["cy"])**2 < (rr+obs["radius"])**2:
                        will_collide = True; break
                elif obs["type"] == "rect":
                    cx = max(obs["xmin"], min(next_x, obs["xmax"]))
                    cy = max(obs["ymin"], min(next_y, obs["ymax"]))
                    if (next_x-cx)**2 + (next_y-cy)**2 < rr**2:
                        will_collide = True; break
                    

        # REWARD
        reward = 0.0
        reward -= 0.1 # Living

        if will_collide:
            # BLOCCA IL ROBOT: Non aggiornare self.x e self.y
            v = 0.0 
            # w = 0.0 # Opzionale: lasciamolo ruotare per liberarsi
            
            # Penalità severa ma NON terminale
            reward -= 5.0 
            collision_penalty = True
        else:
            # Movimento consentito
            self.x = next_x
            self.y = next_y
            self.theta += w * self.dt

        


        dist_before = math.hypot(self.x - self.goal_x, self.y - self.goal_y)

        # ### NEW METRICS: Accumula Jerk e Path ###
        # Jerk angolare: |delta_omega| / dt [rad/s^3]
        current_jerk = abs((w - self.last_w) / self.dt)
        self.episode_jerk_sum += current_jerk

        prev_x, prev_y = self.x, self.y
        # #########################################

        self.x += v * self.dt * math.cos(self.theta)
        self.y += v * self.dt * math.sin(self.theta)
        self.theta += w * self.dt
        
        # ### NEW METRICS: Aggiorna Path Length ###
        dist_stepped = math.hypot(self.x - prev_x, self.y - prev_y)
        self.episode_path_length += dist_stepped
        # #########################################

        self.trajectory.append((self.x, self.y))
        self._step_people()
        self.step_count += 1
        dist_after = math.hypot(self.x - self.goal_x, self.y - self.goal_y)

        # Collisioni
        collision_wall = self._is_collision_with_walls()    
        collision_people = self._is_collision_with_people()
        collision_obstacles = self._is_collision_with_obstacles()
        lidar = self._compute_lidar()
        obs = (self.x, self.y, self.theta, v, w, lidar)

        






        # CONTINUE WITH THE REWARD LOGIC

        if v < 0.05:
            reward -= 0.2  # "Muoviti!"

        progress = (dist_before - dist_after) 
        reward += 25.0 * progress 
        goal_angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        error_angle = (goal_angle - self.theta + math.pi) % (2 * math.pi) - math.pi
        reward += 1.0 * v * np.exp(-abs(error_angle))

        # Smoothness (abbassato come richiesto)
        diff_w = w - self.last_w
        diff_v = v - self.last_v
        reward -= 0.05 * (diff_w ** 2) 
        reward -= 0.01 * (diff_v ** 2)

        self.last_w = w
        self.last_v = v

        min_lidar = min(lidar)
        safe_dist = 0.35 
        if min_lidar < safe_dist:
            reward -= 10.0 * ((safe_dist - min_lidar) / safe_dist) ** 2

        done = False
        info = {}

        if self._is_goal_reached():
            done = True
            reward += 300.0
            info["termination_reason"] = "goal_reached"

        if collision_people or collision_wall or collision_obstacles:
            done = True
            reward -= 200.0
            if collision_people: info["termination_reason"] = "people_collision"
            elif collision_wall: info["termination_reason"] = "wall_collision"
            else: info["termination_reason"] = "obstacle_collision"

        if self.step_count >= self.max_steps:
            done = True
            reward -= 10.0 
            info["termination_reason"] = "max_steps_reached"

        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")
            
            # ### NEW METRICS: Esporta i dati per run_ppo.py ###
            info["path_length"] = self.episode_path_length
            info["total_time"] = self.step_count * self.dt
            info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0
            
            # Calcolo distanza ottima (Euclidea per ora, o Dijkstra se volessimo essere pignoli)
            optimal_dist = math.hypot(self.goal_x - self.start_x, self.goal_y - self.start_y)
            info["optimal_length"] = optimal_dist
            # ################################################

        return obs, reward, done, info

    # ... [MANTIENI METODI HELPER: _reset_people, _ray_cast, render, ecc.] ...
    # (Inserisci qui tutti gli altri metodi helper che erano nel file precedente, sono invariati)
    def _reset_obstacles(self):
        # ... (copia dal file precedente) ...
        self.obstacles = []
        margin = 1.0
        num_circles = self.num_obstacles // 2
        for _ in range(num_circles):
            r = random.uniform(0.6, 1.2)
            cx = random.uniform(margin+r, self.room_width-margin-r)
            cy = random.uniform(margin+r, self.room_height-margin-r)
            self.obstacles.append({"type":"circle", "cx":cx, "cy":cy, "radius":r})
        for _ in range(self.num_obstacles - num_circles):
            w, h = random.uniform(1.0, 2.5), random.uniform(1.0, 2.5)
            cx = random.uniform(margin+w, self.room_width-margin-w)
            cy = random.uniform(margin+h, self.room_height-margin-h)
            self.obstacles.append({"type":"rect", "xmin":cx-w/2, "xmax":cx+w/2, "ymin":cy-h/2, "ymax":cy+h/2})

    def _reset_people(self):
        # ... (copia dal file precedente) ...
        self.people = []
        margin = 2.0
        for _ in range(self.num_people):
            px, py = random.uniform(margin, self.room_width-margin), random.uniform(margin, self.room_height-margin)
            while self._point_inside_any_obstacle(px, py) or math.hypot(px-self.x, py-self.y) < 2.0:
                px, py = random.uniform(margin, self.room_width-margin), random.uniform(margin, self.room_height-margin)
            angle = random.uniform(0, 2*math.pi)
            vx = self.people_speed * math.cos(angle)
            vy = self.people_speed * math.sin(angle)
            self.people.append({"x":px, "y":py, "vx":vx, "vy":vy, "angle":angle})

    def _step_people(self):
        if self.people_speed == 0: return
        for p in self.people:
            p["x"] += p["vx"] * self.dt
            p["y"] += p["vy"] * self.dt
            if p["x"] < self.people_radius or p["x"] > self.room_width - self.people_radius: p["vx"] *= -1
            if p["y"] < self.people_radius or p["y"] > self.room_height - self.people_radius: p["vy"] *= -1

    def _is_collision_with_walls(self):
        return (self.x - self.robot_radius < 0 or self.x + self.robot_radius > self.room_width or 
                self.y - self.robot_radius < 0 or self.y + self.robot_radius > self.room_height)

    def _is_collision_with_obstacles(self):
        rr = self.robot_radius
        for obs in self.obstacles:
            if obs["type"] == "circle":
                if (self.x-obs["cx"])**2 + (self.y-obs["cy"])**2 < (rr+obs["radius"])**2: return True
            elif obs["type"] == "rect":
                cx = max(obs["xmin"], min(self.x, obs["xmax"]))
                cy = max(obs["ymin"], min(self.y, obs["ymax"]))
                if (self.x-cx)**2 + (self.y-cy)**2 < rr**2: return True
        return False
    
    def _point_inside_any_obstacle(self, x, y):
        for obs in self.obstacles:
            if obs["type"] == "circle":
                if (x-obs["cx"])**2 + (y-obs["cy"])**2 < obs["radius"]**2: return True
            elif obs["type"] == "rect":
                if obs["xmin"]<=x<=obs["xmax"] and obs["ymin"]<=y<=obs["ymax"]: return True
        return False
        
    def _is_collision_with_people(self):
        min_dist_sq = (self.robot_radius + self.people_radius)**2
        for p in self.people:
            if (self.x-p["x"])**2 + (self.y-p["y"])**2 < min_dist_sq: return True
        return False

    def _is_goal_reached(self):
        if self.goal_x is None: return False
        return (self.x-self.goal_x)**2 + (self.y-self.goal_y)**2 <= self.goal_radius**2

    def _cast_ray(self, angle):
        x0, y0 = self.x, self.y
        dx, dy = math.cos(angle), math.sin(angle)
        dists = [self.max_lidar_distance]
        if dx != 0:
            t = (0 - x0)/dx; dists.append(t) if t>0 and 0<=y0+t*dy<=self.room_height else None
            t = (self.room_width - x0)/dx; dists.append(t) if t>0 and 0<=y0+t*dy<=self.room_height else None
        if dy != 0:
            t = (0 - y0)/dy; dists.append(t) if t>0 and 0<=x0+t*dx<=self.room_width else None
            t = (self.room_height - y0)/dy; dists.append(t) if t>0 and 0<=x0+t*dx<=self.room_width else None
        for obs in self.obstacles:
            if obs["type"] == "circle":
                val = self._ray_circle_intersection(x0,y0,dx,dy, obs["cx"], obs["cy"], obs["radius"])
                if val: dists.append(val)
            elif obs["type"] == "rect":
                 val = self._ray_rect_intersection_logic(x0,y0,dx,dy, obs["xmin"], obs["xmax"], obs["ymin"], obs["ymax"])
                 if val: dists.append(val)
        for p in self.people:
            val = self._ray_circle_intersection(x0,y0,dx,dy, p["x"], p["y"], self.people_radius)
            if val: dists.append(val)
        return min(dists)

    def _ray_circle_intersection(self, x0, y0, dx, dy, cx, cy, r):
        fx, fy = x0 - cx, y0 - cy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        disc = b*b - 4*c
        if disc < 0: return None
        sqrt_disc = math.sqrt(disc)
        t1, t2 = (-b - sqrt_disc)/2, (-b + sqrt_disc)/2
        if t1 > 0: return t1
        if t2 > 0: return t2
        return None

    def _ray_rect_intersection_logic(self, x0, y0, dx, dy, xmin, xmax, ymin, ymax):
        t_min, t_max = -float('inf'), float('inf')
        if abs(dx) < 1e-9:
            if x0 < xmin or x0 > xmax: return None
        else:
            tx1, tx2 = (xmin - x0)/dx, (xmax - x0)/dx
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))
        if abs(dy) < 1e-9:
            if y0 < ymin or y0 > ymax: return None
        else:
            ty1, ty2 = (ymin - y0)/dy, (ymax - y0)/dy
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))
        return t_min if t_max >= t_min and t_max >= 0 else None

    def _compute_lidar(self):
        return [self._cast_ray(self.theta + i * 2*math.pi/self.num_rays) for i in range(self.num_rays)]

    def render(self):
        pass 
    def close(self):
        if self.fig: plt.close(self.fig)