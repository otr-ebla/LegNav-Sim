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
            robot_radius: float = 0.2, 
            num_rays: int = 108,
            max_lidar_distance: float = 15.0,
            num_people: int = 10,
            people_radius: float = 0.2,
            people_speed: float = 0.0,
            num_obstacles: int = 0,
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
        self.max_v = 0.3  # m/s (TurtleBot4 max linear)
        self.max_w = 2.3  # rad/s (TurtleBot4 max angular)

        # Geometria Stanza
        self.room_width = room_width
        self.room_height = room_height
        self.robot_radius = robot_radius

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

        # Rendering
        self.fig = None
        self.ax = None

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
        if self.people_speed == 0: return
        for p in self.people:
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
        self.step_count = 0
        self.last_termination_reason = None
        self.episode_path_length = 0.0
        self.episode_jerk_sum = 0.0

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
                    if obs["type"]=="circle" and (gx-obs["cx"])**2+(gy-obs["cy"])**2 < (obs["radius"]+0.4)**2: goal_unsafe=True
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
        
        return self._get_observation(self.last_v, self.last_w)

    def step(self, action):
        target_v, target_w = action 
        
        # 1. Applicazione limiti fisici (Clipping)
        v, w = self._apply_differential_drive_constraints(target_v, target_w)

        # Calcolo distanza precedente per il reward di progresso
        dist_to_goal_prev = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        
        # 2. Aggiornamento Fisica
        # Aggiorniamo prima theta per calcolare la nuova direzione
        self.theta += w * self.dt
        # Normalizzazione angolo tra -pi e pi (importante per la stabilità numerica)
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # Calcolo posizione candidata
        next_x = self.x + v * self.dt * math.cos(self.theta)
        next_y = self.y + v * self.dt * math.sin(self.theta)

        # 3. Controllo Collisioni (Safety Layer)
        collision_static = False
        
        # A. Check Muri
        if (next_x - self.robot_radius < 0 or next_x + self.robot_radius > self.room_width or 
            next_y - self.robot_radius < 0 or next_y + self.robot_radius > self.room_height): 
            collision_static = True
        
        # B. Check Ostacoli (Muri interni/Oggetti)
        if not collision_static:
            rr = self.robot_radius
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    if (next_x - obs["cx"])**2 + (next_y - obs["cy"])**2 < (rr + obs["radius"])**2: 
                        collision_static = True; break
                elif obs["type"] == "rect":
                    cx = max(obs["xmin"], min(next_x, obs["xmax"]))
                    cy = max(obs["ymin"], min(next_y, obs["ymax"]))
                    if (next_x - cx)**2 + (next_y - cy)**2 < rr**2: 
                        collision_static = True; break

        # Inizializzazione variabili di ritorno
        reward = 0.0
        done = False
        info = {}

        # GESTIONE COLLISIONE STATICA (Muri/Oggetti)
        # [MODIFICA CRITICA] Ora la collisione è TERMINALE.
        # Questo impedisce al robot di "strisciare" sui muri.
        if collision_static:
            reward = -100.0
            done = True
            info["termination_reason"] = "collision_static"
            
            # Calcolo l'osservazione finale prima di uscire (senza muovere il robot)
            # Nota: restituiamo lo stato corrente, non quello impossibile dentro il muro
            obs = self._get_observation(v, w)
            return obs, reward, done, info

        # Se non c'è collisione, applica il movimento
        self.x = next_x
        self.y = next_y
        self.trajectory.append((self.x, self.y))
        
        # Aggiornamento dinamica ambiente (persone)
        self._step_people() 
        self.step_count += 1
        
        # Calcolo metriche post-movimento
        dist_to_goal_now = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        self.episode_path_length += math.hypot(v * self.dt * math.cos(self.theta), v * self.dt * math.sin(self.theta))
        self.episode_jerk_sum += abs((w - self.last_w) / self.dt)

        # 4. Calcolo Reward
        # A. Penalità temporale (incentiva a fare presto)
        reward -= 0.05
        
        # B. Reward di progresso (molto forte per guidare l'apprendimento)
        # Premia l'avvicinamento netto al goal
        reward += 20.0 * (dist_to_goal_prev - dist_to_goal_now)

        obs = self._get_observation(v, w)
        inv_lidar = obs[4:]  # Lidar inversi normalizzati

        # reward negativo sul massimo del lidar inverso (cioè il più vicino)
        max_inv_lidar = max(inv_lidar)
        if max_inv_lidar > 0.6:
            reward -= np.exp(-4.0 * (0.6 - max_inv_lidar))  # Penalità più forte per ostacoli vicini

        # C. Reward di allineamento (incentiva a guardare il goal mentre si muove)
        # goal_angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        # heading_error = abs((goal_angle - self.theta + math.pi) % (2 * math.pi) - math.pi)
        # if v > 0.05: # Solo se si muove
        #     # Più l'errore è basso (allineato), più alto è il premio (max 1.0 * v)
        #     reward += 1.0 * v * np.exp(-heading_error)

        # D. Penalità per azioni brusche (Smoothness)
        reward -= 0.05 * abs(w - self.last_w)
        
        # Aggiorna variabili stato precedente
        self.last_w = w
        self.last_v = v

        # 5. Check Terminazione
        collision_people = self._is_collision_with_people()
        is_goal = self._is_goal_reached()
        
        if is_goal:
            done = True
            reward += 200.0 # Grande bonus finale
            info["termination_reason"] = "goal_reached"
        elif collision_people:
            done = True
            reward -= 200.0 # Grande penalità collisione dinamica
            info["termination_reason"] = "people_collision"
        elif self.step_count >= self.max_steps:
            done = True
            reward -= 10.0  # Penalità timeout
            info["termination_reason"] = "max_steps_reached"

        # Info finali per debugging/logging
        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")
            info["path_length"] = self.episode_path_length
            info["total_time"] = self.step_count * self.dt
            info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0

        # 6. Costruzione Osservazione
        
        
        return obs, reward, done, info

    # --- [NUOVO METODO HELPER] Costruzione Osservazione Normalizzata ---
    def _get_observation(self, v, w):
        # Calcolo Lidar
        lidar = self._compute_lidar()
        
        # Calcolo vettore Goal RELATIVO al robot
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        angle_to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        
        # Errore di prua: differenza tra dove devo andare e dove sto guardando
        heading_error = angle_to_goal - self.theta
        # Normalizza tra -pi e pi
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        # 2. Angolo Goal: Già tra -pi e pi, dividiamo per pi per avere [-1, 1]
        norm_heading = heading_error / math.pi
        
        # 3. Lidar: inversi i lidar che siano valori tra 0 e 1, 1 quando l'ostacolo è molto vicino cioè a min_lidar_distance
        inverse_lidar = []
        for d in lidar:
            if d >= self.max_lidar_distance:
                inverse_lidar.append(0.0)
            else:
                inv = (self.lidar_min_distance/d)**(1/3)
                inverse_lidar.append(min(inv, 1.0))
                
        # Costruzione vettore finale numpy (float32 è lo standard per Pytorch/TF)
        # Struttura: [dist_goal, angle_goal, vel_lin, vel_ang, ...lidar_beams...]
        obs_list = [dist_to_goal, norm_heading, v, w] + inverse_lidar
        return np.array(obs_list, dtype=np.float32)

    # --- RENDERING (MODIFICATO CON OFFSET) ---
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        self.ax.clear()

        # Draw room
        self.ax.plot([0, self.room_width, self.room_width, 0, 0], [0, 0, self.room_height, self.room_height, 0], 'k-')

        # Draw obstacles
        for obs in self.obstacles:
            if obs["type"] == "circle":
                circle = plt.Circle((obs["cx"], obs["cy"]), obs["radius"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(circle)
            elif obs["type"] == "rect":
                rect = plt.Rectangle((obs["xmin"], obs["ymin"]), obs["xmax"] - obs["xmin"], obs["ymax"] - obs["ymin"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(rect)

        # Draw robot
        robot_circle = plt.Circle((self.x, self.y), self.robot_radius, color='blue', fill=True)
        self.ax.add_patch(robot_circle)

        arrow_len = self.robot_radius * 1.5
        x_head = self.x + arrow_len * math.cos(self.theta)
        y_head = self.y + arrow_len * math.sin(self.theta)
        self.ax.plot([self.x, x_head], [self.y, y_head], 'b-', linewidth=2)

        # Draw people
        for p in self.people:
            person = plt.Circle((p["x"], p["y"]), self.people_radius, color='green', fill=True)
            x_people_head = p["x"] + 0.3*math.cos(p["angle"])
            y_people_head = p["y"] + 0.3*math.sin(p["angle"])
            self.ax.add_patch(person)
            self.ax.plot([p["x"], x_people_head], [p["y"], y_people_head], 'g-', linewidth=2)

        lidar = self._compute_lidar()
        angles = [self.theta + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]

        # [MODIFICATO] Calcola l'origine del lidar per il rendering
        lidar_x_origin = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y_origin = self.y + self.lidar_offset * math.sin(self.theta)

        if len(self.trajectory) >= 2:
            traj_xs, traj_ys = zip(*self.trajectory)
            self.ax.plot(traj_xs, traj_ys, 'b--', linewidth=1)

        base_grey = (0.6, 0.6, 0.6)
        base_red = (1.0, 0.0, 0.0)

        # Draw lidar rays
        for i, (dist, ang) in enumerate(zip(lidar, angles)): 
            # Il raggio parte da lidar_origin, non da self.x/y
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
            
            # --- MODIFICA VISIVA RICHIESTA (Raggio chiaro) ---
            if i == 0: 
                rgba = (0.5, 0.7, 1.0, 1.0) # Blu molto chiaro/brillante
                linewidth = 1.0             
            else:
                linewidth = 0.5
            # ---------------------------------------------

            self.ax.plot([lidar_x_origin, x_end], [lidar_y_origin, y_end], color=rgba, linewidth=linewidth)

            if i != 0:
                self.ax.plot(x_end, y_end, marker='o', markersize=2, color=rgba)

        # Draw goal
        if self.goal_x is not None:
            self.ax.plot(self.goal_x, self.goal_y, marker='*', markersize=7, color='orange')

        self.ax.set_xlim(-1, self.room_width + 1)
        self.ax.set_ylim(-1, self.room_height + 1)
        self.ax.set_aspect('equal', adjustable='box')

        status_text = f"step: {self.step_count}\nlast term: {self.last_termination_reason}"
        self.ax.text(0.01, 0.99, status_text, verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        plt.pause(0.0001)

    def close(self):
        if self.fig: plt.close(self.fig); self.fig=None

    # --- METODI HELPER RESET ---
    def _reset_obstacles(self):
        self.obstacles = []
        cols = 4; rows = 3
        cell_w = self.room_width / cols; cell_h = self.room_height / rows
        cell_indices = [i for i in range(cols * rows)]
        while len(cell_indices) < self.num_obstacles: cell_indices.extend([i for i in range(cols * rows)])
        random.shuffle(cell_indices)
        
        for i in range(self.num_obstacles):
            idx = cell_indices[i]
            r_idx = idx // cols; c_idx = idx % cols
            margin = 0.5 
            min_x = c_idx * cell_w + margin; max_x = (c_idx + 1) * cell_w - margin
            min_y = r_idx * cell_h + margin; max_y = (r_idx + 1) * cell_h - margin
            cx = random.uniform(min_x, max_x); cy = random.uniform(min_y, max_y)
            if i % 2 == 0:
                r = random.uniform(0.5, 1.0) 
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": r})
            else:
                w = random.uniform(0.8, 2.0); h = random.uniform(0.8, 2.0)
                self.obstacles.append({"type": "rect", "xmin": cx - w/2, "xmax": cx + w/2, "ymin": cy - h/2, "ymax": cy + h/2})

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
            self.people.append({"x":px, "y":py, "vx":vx, "vy":vy, "angle":angle})

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
                if (self.x-obs["cx"])**2+(self.y-obs["cy"])**2 < (rr+obs["radius"])**2: return True
            elif obs["type"]=="rect":
                cx = max(obs["xmin"], min(self.x, obs["xmax"])); cy = max(obs["ymin"], min(self.y, obs["ymax"]))
                if (self.x-cx)**2+(self.y-cy)**2 < rr**2: return True
        return False
    
    def _is_collision_with_walls(self):
        return (self.x-self.robot_radius<0 or self.x+self.robot_radius>self.room_width or 
                self.y-self.robot_radius<0 or self.y+self.robot_radius>self.room_height)
                
    def _is_collision_with_people(self):
        min_sq = (self.robot_radius + self.people_radius)**2
        for p in self.people:
            if (self.x-p["x"])**2+(self.y-p["y"])**2 < min_sq: return True
        return False
    
    def _is_goal_reached(self):
        if self.goal_x is None: return False
        return (self.x-self.goal_x)**2+(self.y-self.goal_y)**2 <= self.goal_radius**2