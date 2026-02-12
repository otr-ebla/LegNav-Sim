import math
import random
import numpy as np
import pygame
from numba import njit, float32, int32, boolean

# =============================================================================
# 🚀 KERNEL NUMBA (MOTORE FISICO VELOCE)
# =============================================================================

LEG_RADIUS = 0.07

@njit(fastmath=True)
def intersect_circle_robust(lx, ly, dx, dy, cx, cy, r):
    """Calcola l'intersezione raggio-cerchio. Ritorna la distanza o 9999.0 se nessuna."""
    fx = lx - cx
    fy = ly - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - (r * r)
    disc = b * b - 4.0 * c
    
    if disc < 0:
        return 9999.0
    
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    
    if t1 > 0: return t1
    if t2 > 0: return t2
    return 9999.0

@njit(fastmath=True)
def fast_ray_intersect_static(x0, y0, angle, obs_arr, num_obs, room_w, room_h, max_dist):
    dx = math.cos(angle)
    dy = math.sin(angle)
    min_dist = max_dist

    # 1. Muri
    if abs(dx) > 1e-6:
        t = (0 - x0) / dx
        if t > 0 and 0 <= y0 + t * dy <= room_h: min_dist = min(min_dist, t)
        t = (room_w - x0) / dx
        if t > 0 and 0 <= y0 + t * dy <= room_h: min_dist = min(min_dist, t)
    if abs(dy) > 1e-6:
        t = (0 - y0) / dy
        if t > 0 and 0 <= x0 + t * dx <= room_w: min_dist = min(min_dist, t)
        t = (room_h - y0) / dy
        if t > 0 and 0 <= x0 + t * dx <= room_w: min_dist = min(min_dist, t)

    # 2. Ostacoli Statici
    for i in range(num_obs):
        otype = obs_arr[i, 0]
        if otype == 0.0: # Circle
            d = intersect_circle_robust(x0, y0, dx, dy, obs_arr[i, 1], obs_arr[i, 2], obs_arr[i, 3])
            if d < min_dist: min_dist = d
        elif otype == 1.0: # Rect (Slab method)
            xmin, xmax, ymin, ymax = obs_arr[i, 1], obs_arr[i, 2], obs_arr[i, 3], obs_arr[i, 4]
            tmin = -1e9; tmax = 1e9
            if abs(dx) < 1e-9:
                if x0 < xmin or x0 > xmax: tmin = 1e9
            else:
                tx1 = (xmin - x0) / dx; tx2 = (xmax - x0) / dx
                tmin = max(tmin, min(tx1, tx2))
                tmax = min(tmax, max(tx1, tx2))
            if abs(dy) < 1e-9:
                if y0 < ymin or y0 > ymax: tmin = 1e9
            else:
                ty1 = (ymin - y0) / dy; ty2 = (ymax - y0) / dy
                tmin = max(tmin, min(ty1, ty2))
                tmax = min(tmax, max(ty1, ty2))
            if tmax >= tmin and tmin > 0:
                if tmin < min_dist: min_dist = tmin
    return min_dist

@njit(fastmath=True, parallel=False)
def fast_compute_lidar_core(x, y, theta, num_rays, start_angle, max_dist, 
                            obs_arr, num_obs, room_w, room_h, 
                            people_arr, num_people, 
                            use_legs, legs_coords): # <--- NUOVI PARAMETRI
    """
    Calcola i raggi Lidar. 
    Se use_legs=True, calcola collisione con le GAMBE (r=0.09).
    Altrimenti con il CORPO (r=0.2).
    """
    ranges = np.zeros(num_rays, dtype=np.float32)
    angle_inc = 2 * math.pi / num_rays
    lx = x + (-0.05) * math.cos(theta)
    ly = y + (-0.05) * math.sin(theta)
    
    BODY_RADIUS = 0.2

    for i in range(num_rays):
        ang = start_angle + i * angle_inc
        dx = math.cos(ang)
        dy = math.sin(ang)
        
        # 1. Statici (Muri + Ostacoli)
        d = fast_ray_intersect_static(lx, ly, ang, obs_arr, num_obs, room_w, room_h, max_dist)
        
        # 2. Dinamici (Persone)
        if use_legs:
            # Check intersezione con le coordinate delle GAMBE
            for p in range(num_people):
                # Gamba 1 (Left) - coordinate pre-calcolate
                l1x = legs_coords[p, 0, 0]
                l1y = legs_coords[p, 0, 1]
                t1 = intersect_circle_robust(lx, ly, dx, dy, l1x, l1y, LEG_RADIUS)
                if t1 < d: d = t1
                
                # Gamba 2 (Right)
                l2x = legs_coords[p, 1, 0]
                l2y = legs_coords[p, 1, 1]
                t2 = intersect_circle_robust(lx, ly, dx, dy, l2x, l2y, LEG_RADIUS)
                if t2 < d: d = t2
        else:
            # Check intersezione con CORPO (Cilindro classico)
            for p in range(num_people):
                px, py = people_arr[p, 0], people_arr[p, 1]
                t = intersect_circle_robust(lx, ly, dx, dy, px, py, BODY_RADIUS)
                if t < d: d = t
                
        ranges[i] = d
    return ranges

@njit(fastmath=True)
def fast_apply_lidar_noise(raw_ranges, num_rays, noise_std, max_dist):
    out = np.zeros_like(raw_ranges)
    min_dist = 0.12 
    for i in range(num_rays):
        val = raw_ranges[i]
        sp_roll = random.random()
        if sp_roll < 0.05: val = max_dist 
        elif sp_roll < 0.10: val = min_dist
        else:
            noise = random.gauss(0.0, noise_std)
            val += noise
        if val < min_dist: val = min_dist
        if val > max_dist: val = max_dist
        out[i] = val
    return out

@njit(fastmath=True)
def fast_check_static_collision(x, y, radius, obs_arr, num_obs, room_w, room_h):
    if x - radius < 0 or x + radius > room_w or y - radius < 0 or y + radius > room_h: return True
    for i in range(num_obs):
        otype = obs_arr[i, 0]
        if otype == 0.0: # Circle
            dx = x - obs_arr[i, 1]; dy = y - obs_arr[i, 2]; r_obs = obs_arr[i, 3]
            if (dx*dx + dy*dy) < (r_obs + radius)**2: return True
        elif otype == 1.0: # Rect
            xmin, xmax = obs_arr[i, 1], obs_arr[i, 2]
            ymin, ymax = obs_arr[i, 3], obs_arr[i, 4]
            cx = max(xmin, min(x, xmax)); cy = max(ymin, min(y, ymax))
            if (x - cx)**2 + (y - cy)**2 < radius**2: return True
    return False

@njit(fastmath=True)
def fast_grid_bfs(start_x, start_y, goal_x, goal_y, room_w, room_h, obs_arr, num_obs, res=0.4):
    cols = int(room_w / res) + 1; rows = int(room_h / res) + 1
    sx = int(start_x / res); sy = int(start_y / res)
    gx = int(goal_x / res); gy = int(goal_y / res)
    if sx == gx and sy == gy: return True
    visited = np.zeros((rows, cols), dtype=np.int8)
    q_x = np.zeros(rows * cols, dtype=np.int32)
    q_y = np.zeros(rows * cols, dtype=np.int32)
    head = 0; tail = 0
    q_x[tail] = sx; q_y[tail] = sy; tail += 1; visited[sy, sx] = 1
    
    dx = np.array([0, 0, -1, 1], dtype=np.int32)
    dy = np.array([-1, 1, 0, 0], dtype=np.int32)
    
    while head < tail:
        cx = q_x[head]; cy = q_y[head]; head += 1
        if cx == gx and cy == gy: return True
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if 0 <= nx < cols and 0 <= ny < rows:
                if visited[ny, nx] == 0:
                    wx = nx * res + res/2; wy = ny * res + res/2
                    is_coll = False
                    if wx < 0.2 or wx > room_w - 0.2 or wy < 0.2 or wy > room_h - 0.2: is_coll = True
                    else:
                        for k in range(num_obs):
                            otype = obs_arr[k, 0]
                            if otype == 0.0:
                                if (wx - obs_arr[k,1])**2 + (wy - obs_arr[k,2])**2 < (obs_arr[k,3] + 0.3)**2: is_coll = True; break
                            elif otype == 1.0:
                                rcx = max(obs_arr[k,1], min(wx, obs_arr[k,2])); rcy = max(obs_arr[k,3], min(wy, obs_arr[k,4]))
                                if (wx - rcx)**2 + (wy - rcy)**2 < 0.3**2: is_coll = True; break
                    if not is_coll:
                        visited[ny, nx] = 1; q_x[tail] = nx; q_y[tail] = ny; tail += 1
    return False

@njit(fastmath=True)
def fast_apply_repulsion(people_arr, num_people, robot_x, robot_y, robot_theta, robot_v, people_speed_base):
    REACTION_DIST = 1.0 
    for i in range(num_people):
        if people_arr[i, 5] > 0.5: continue 
        px = people_arr[i, 0]; py = people_arr[i, 1]
        vx = people_arr[i, 2]; vy = people_arr[i, 3]
        dx = px - robot_x; dy = py - robot_y; dist = math.sqrt(dx*dx + dy*dy)
        if dist < REACTION_DIST:
            if dist < 0.01: dist = 0.01
            urgency = (REACTION_DIST - dist) / REACTION_DIST
            rep_force_mag = 13.0 * urgency**2
            nx = dx / dist; ny = dy / dist
            tx = -ny; ty = nx
            r_cos = math.cos(robot_theta); r_sin = math.sin(robot_theta)
            cross_prod = (r_cos * dy) - (r_sin * dx)
            side_sign = 0.0
            if abs(robot_v) < 0.05:
                dot_t = (vx * tx) + (vy * ty)
                side_sign = 1.0 if dot_t > 0 else -1.0
            else:
                side_sign = 1.0 if cross_prod > 0 else -1.0
            dodge_mag = 2.0 * urgency
            vx += (nx * rep_force_mag * 0.3) + (tx * side_sign * dodge_mag)
            vy += (ny * rep_force_mag * 0.3) + (ty * side_sign * dodge_mag)
            current_speed = math.sqrt(vx*vx + vy*vy)
            max_reaction_speed = max(people_speed_base * 1.5, 0.5)
            if current_speed > max_reaction_speed:
                scale = max_reaction_speed / current_speed
                vx *= scale; vy *= scale
            people_arr[i, 2] = vx; people_arr[i, 3] = vy

@njit(fastmath=True)
def fast_move_people(people_arr, obs_arr, num_people, num_obs, room_w, room_h, dt, people_radius):
    for i in range(num_people):
        px = people_arr[i, 0]; py = people_arr[i, 1]
        vx = people_arr[i, 2]; vy = people_arr[i, 3]
        px += vx * dt; py += vy * dt
        if px - people_radius < 0: px = people_radius; vx = -vx
        elif px + people_radius > room_w: px = room_w - people_radius; vx = -vx
        if py - people_radius < 0: py = people_radius; vy = -vy
        elif py + people_radius > room_h: py = room_h - people_radius; vy = -vy
        for j in range(num_obs):
            otype = obs_arr[j, 0]
            if otype == 0.0: 
                cx, cy, cr = obs_arr[j, 1], obs_arr[j, 2], obs_arr[j, 3]
                dx = px - cx; dy = py - cy; d2 = dx*dx + dy*dy
                min_d = people_radius + cr
                if d2 < min_d*min_d:
                    dist = math.sqrt(d2); 
                    if dist < 1e-6: dist = 1e-6
                    nx = dx/dist; ny = dy/dist; overlap = min_d - dist
                    px += nx * overlap; py += ny * overlap
                    dot = vx*nx + vy*ny; vx -= 2*dot*nx; vy -= 2*dot*ny
            elif otype == 1.0: 
                xmin, xmax = obs_arr[j, 1], obs_arr[j, 2]; ymin, ymax = obs_arr[j, 3], obs_arr[j, 4]
                cx = min(max(px, xmin), xmax); cy = min(max(py, ymin), ymax)
                dx = px - cx; dy = py - cy; d2 = dx*dx + dy*dy
                if d2 < people_radius*people_radius:
                    dist = math.sqrt(d2)
                    if dist < 1e-6: dist = 1e-6
                    nx = dx/dist; ny = dy/dist; overlap = people_radius - dist
                    px += nx * overlap; py += ny * overlap
                    dot = vx*nx + vy*ny; vx -= 2*dot*nx; vy -= 2*dot*ny
        people_arr[i, 0] = px; people_arr[i, 1] = py
        people_arr[i, 2] = vx; people_arr[i, 3] = vy
        people_arr[i, 4] = math.atan2(vy, vx)

@njit(fastmath=True)
def fast_get_legs(x, y, v, theta, leg_phase):
    """Calcolo coordinate gambe (Left/Right) per una persona"""
    HIP_SPACING = 0.20
    K_PHASE = 6.0
    target_amp = math.pi / (2 * K_PHASE)
    stride_amp = 0.0
    if v >= 0.05:
        ratio = v / 0.2
        if ratio > 1.0: ratio = 1.0
        stride_amp = target_amp * ratio

    phi_l = leg_phase % (2 * math.pi)
    off_l = 1.0 - (2.0 * phi_l / math.pi) if phi_l < math.pi else -math.cos(phi_l - math.pi)

    phi_r = (leg_phase + math.pi) % (2 * math.pi)
    off_r = 1.0 - (2.0 * phi_r / math.pi) if phi_r < math.pi else -math.cos(phi_r - math.pi)

    cos_t = math.cos(theta); sin_t = math.sin(theta)
    px = -sin_t; py = cos_t
    
    lx = x - (px * HIP_SPACING * 0.5) + (cos_t * stride_amp * off_l)
    ly = y - (py * HIP_SPACING * 0.5) + (sin_t * stride_amp * off_l)
    
    rx = x + (px * HIP_SPACING * 0.5) + (cos_t * stride_amp * off_r)
    ry = y + (py * HIP_SPACING * 0.5) + (sin_t * stride_amp * off_r)
    
    return np.array([[lx, ly], [rx, ry]], dtype=np.float32)

# =============================================================================
# CLASSE AMBIENTE
# =============================================================================

class Simple2DEnv:
    def __init__(
            self, 
            max_steps: int = 1000, 
            dt: float = 0.1,
            room_width: float = 12.0,
            room_height: float = 12.0,
            robot_radius: float = 0.2, 
            num_rays: int = 1080, 
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

        self.dt = dt
        self.max_steps = max_steps
        self.room_width = room_width
        self.room_height = room_height
        self.num_rays = 1080 if real_lidar_specs else num_rays
        self.real_lidar_specs = real_lidar_specs
        self.lidar_start_angle_offset = -math.pi / 2.0 if real_lidar_specs else 0.0
        self.max_lidar_distance = max_lidar_distance
        self.lidar_noise_enable = lidar_noise_enable
        self.lidar_offset = -0.05
        
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.v = 0.0; self.w = 0.0
        self.max_v = 0.3; self.max_w = 0.7
        self.robot_radius = robot_radius
        self.trajectory = []

        # --- Dati Ibridi ---
        self.max_obstacles = 50 
        self.obstacles_arr = np.zeros((self.max_obstacles, 5), dtype=np.float32)
        self.num_active_obstacles = 0
        self.obstacles = [] 
        self.num_obstacles = num_obstacles

        self.num_people = num_people
        self.people_arr = np.zeros((self.num_people, 6), dtype=np.float32)
        self.people = [] 
        self.people_speed = people_speed
        self.people_radius = people_radius
        self.human_distraction_prob = human_distraction_prob
        
        # Buffer per le gambe (necessario per Numba)
        # [NumPeople, 2(Legs), 2(XY)]
        self.legs_coords = np.zeros((self.num_people, 2, 2), dtype=np.float32)
        
        # Rendering
        self.goal_x = None; self.goal_y = None; self.goal_radius = 0.3
        self.screen = None
        self.render_skip = render_skip
        self.render_counter = 0
        self.window_size = 800
        self.scale = self.window_size / max(room_width, room_height)
        self.sidebar_width = 350
        
        # Inizializza lidar
        self.lidar_readings = np.zeros(self.num_rays, dtype=np.float32)

        # Rewards & Stats
        self.progress_reward = 0
        self.step_count = 0
        self.episode_jerk_sum = 0
        self.global_jerk_sum = 0
        self.global_step_count = 0
        self.last_v = 0; self.last_w = 0
        self.episode_path_length = 0
        self.max_possible_dist = math.sqrt(room_width**2 + room_height**2)
        self.last_termination_reason = "N/A"
        self.persistent_outcome = "N/A"
        self.manual_skip_triggered = False

        # Legs Animation
        self.use_legs = use_legs
        self.humans_leg_phase = [0.0] * num_people
        self.smooth_v = [0.0] * num_people

        # NOISE CONFIG
        if self.real_lidar_specs:
            self.lidar_noise_std = 0.027
        else:
            self.lidar_noise_std = 0.03

    def reset(self):
        self.step_count = 0
        self.trajectory = []
        self.episode_path_length = 0
        self.episode_jerk_sum = 0
        self.last_v = 0; self.last_w = 0
        self.manual_skip_triggered = False

        while True:
            self._reset_obstacles()

            # Posiziona Robot
            valid_robot = False
            margin = self.robot_radius + 0.2
            for _ in range(50):
                rx = random.uniform(margin, self.room_width - margin)
                ry = random.uniform(margin, self.room_height - margin)
                if not fast_check_static_collision(rx, ry, self.robot_radius + 0.05, self.obstacles_arr, self.num_active_obstacles, self.room_width, self.room_height):
                    self.x = rx; self.y = ry; valid_robot = True; break
            
            if not valid_robot: continue

            self.theta = random.uniform(0, 2*math.pi)

            # Posiziona Goal
            valid_goal = False
            for _ in range(50):
                gx = random.uniform(1, self.room_width - 1)
                gy = random.uniform(1, self.room_height - 1)
                if math.hypot(gx - self.x, gy - self.y) > 4.0:
                     if not fast_check_static_collision(gx, gy, 0.3, self.obstacles_arr, self.num_active_obstacles, self.room_width, self.room_height):
                         if fast_grid_bfs(self.x, self.y, gx, gy, self.room_width, self.room_height, self.obstacles_arr, self.num_active_obstacles):
                             self.goal_x = gx; self.goal_y = gy; valid_goal = True; break
            
            if valid_goal: break

        # Inizializza Persone
        self.people = []
        for i in range(self.num_people):
            while True:
                px = random.uniform(1, self.room_width - 1)
                py = random.uniform(1, self.room_height - 1)
                if math.hypot(px-self.x, py-self.y) > 2.0:
                    if not fast_check_static_collision(px, py, self.people_radius, self.obstacles_arr, self.num_active_obstacles, self.room_width, self.room_height): break
            
            angle = random.uniform(0, 2*math.pi)
            speed = self.people_speed
            vx = speed * math.cos(angle); vy = speed * math.sin(angle)
            is_distracted = 1.0 if random.random() < self.human_distraction_prob else 0.0
            
            self.people_arr[i] = [px, py, vx, vy, angle, is_distracted]
            self.people.append({"x": px, "y": py, "vx": vx, "vy": vy, "angle": angle, "distracted": bool(is_distracted)})

        return self._get_observation(0, 0)

    def step(self, action):
        if self.manual_skip_triggered:
            return self._get_observation(0,0), 0.0, True, {"termination_reason": "manual_skip"}

        target_v, target_w = action
        self.v = max(0.0, min(target_v, self.max_v))
        self.w = max(-self.max_w, min(target_w, self.max_w))

        dt = self.dt
        self.theta = (self.theta + self.w * dt + math.pi) % (2 * math.pi) - math.pi
        next_x = self.x + self.v * dt * math.cos(self.theta)
        next_y = self.y + self.v * dt * math.sin(self.theta)

        # Stats
        current_jerk = abs((self.w - self.last_w) / dt)
        self.episode_jerk_sum += current_jerk
        self.global_jerk_sum += current_jerk
        self.global_step_count += 1
        self.episode_path_length += (self.v * dt)

        # Collisioni Statiche
        if fast_check_static_collision(next_x, next_y, self.robot_radius - 0.02, self.obstacles_arr, self.num_active_obstacles, self.room_width, self.room_height):
            self.persistent_outcome = "collision_static"
            return self._get_observation(self.v, self.w), -200.0, True, {"termination_reason": "collision_static"}

        self.x = next_x
        self.y = next_y
        self.step_count += 1
        
        # 1. Repulsione Umani
        fast_apply_repulsion(self.people_arr, self.num_people, self.x, self.y, self.theta, self.last_v, self.people_speed)

        # 2. Fisica Umani
        fast_move_people(self.people_arr, self.obstacles_arr, self.num_people, self.num_active_obstacles, self.room_width, self.room_height, dt, self.people_radius)

        # 3. Sync Python + Animazione Gambe
        for i in range(self.num_people):
            px, py = self.people_arr[i, 0], self.people_arr[i, 1]
            vx, vy = self.people_arr[i, 2], self.people_arr[i, 3]
            angle = self.people_arr[i, 4]
            p = self.people[i]
            p["x"] = px; p["y"] = py; p["vx"] = vx; p["vy"] = vy; p["angle"] = angle
            
            # Calcolo gambe SEMPRE, perché servono al Lidar
            if self.use_legs:
                v_mag = math.hypot(vx, vy)
                self.humans_leg_phase[i] += v_mag * dt * 4.0
                legs_arr = fast_get_legs(px, py, v_mag, angle, self.humans_leg_phase[i])
                
                # Salva per Lidar Numba
                self.legs_coords[i, 0, 0] = legs_arr[0,0]
                self.legs_coords[i, 0, 1] = legs_arr[0,1]
                self.legs_coords[i, 1, 0] = legs_arr[1,0]
                self.legs_coords[i, 1, 1] = legs_arr[1,1]
                
                # Salva per Rendering Pygame
                p["legs"] = [(legs_arr[0,0], legs_arr[0,1]), (legs_arr[1,0], legs_arr[1,1])]

        # 4. Lidar (Core + Noise)
        self.lidar_readings = self._compute_lidar_fast()

        # Rewards & Logic
        dx_goal = self.x - self.goal_x; dy_goal = self.y - self.goal_y
        dist_goal = math.hypot(dx_goal, dy_goal)
        done = False
        reward = -0.1 

        prev_dist = math.hypot(self.x - self.v*dt*math.cos(self.theta) - self.goal_x, self.y - self.v*dt*math.sin(self.theta) - self.goal_y)
        reward += 5.0 * (prev_dist - dist_goal)
        self.progress_reward += 5.0 * (prev_dist - dist_goal)
        reward -= 0.05 * abs(self.w - self.last_w)

        # Yielding Logic
        closest_dist = float('inf'); closest_rel_angle = 0.0
        for p in self.people:
            d = math.hypot(p["x"] - self.x, p["y"] - self.y)
            if d < closest_dist:
                closest_dist = d
                glob_angle = math.atan2(p["y"] - self.y, p["x"] - self.x)
                closest_rel_angle = (glob_angle - self.theta + math.pi) % (2 * math.pi) - math.pi
        
        YIELD_DIST = 1.5; YIELD_FOV = math.radians(60)
        if closest_dist < YIELD_DIST and abs(closest_rel_angle) < YIELD_FOV:
            urgency = (YIELD_DIST - closest_dist) / YIELD_DIST
            if self.v > 0.1: reward -= 15.0 * urgency * (self.v / self.max_v)
            elif self.v <= 0.1: reward += 0.2 * urgency
        
        if dist_goal <= self.goal_radius:
            done = True; reward = 200.0; info = {"termination_reason": "goal_reached"}
            self.persistent_outcome = "goal"
        else:
            coll_thresh = self.robot_radius + self.people_radius + 0.0 
            if closest_dist < coll_thresh:
                done = True
                is_front = abs(closest_rel_angle) < YIELD_FOV
                if self.v >= 0.1 and is_front:
                    reward = -150.0 * (self.v/self.max_v)
                    info = {"termination_reason": "people_collision_active", "collision_type": "active"}
                    self.persistent_outcome = "collision_people"
                else:
                    reward = -20.0
                    info = {"termination_reason": "people_collision_passive", "collision_type": "passive"}
                    self.persistent_outcome = "collision_passive"
            elif self.step_count >= self.max_steps:
                done = True; reward = -50.0; info = {"termination_reason": "max_steps_reached"}
                self.persistent_outcome = "timeout"
            else:
                info = {}

        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")
            info["path_length"] = self.episode_path_length
            info["total_time"] = self.step_count * dt
            info["mean_jerk"] = self.episode_jerk_sum / self.step_count if self.step_count > 0 else 0.0

        self.last_v = self.v; self.last_w = self.w
        return self._get_observation(self.v, self.w), reward, done, info

    def _compute_lidar_fast(self):
        start_angle = self.theta + self.lidar_start_angle_offset
        
        # Chiamata aggiornata con supporto gambe
        raw = fast_compute_lidar_core(
            self.x, self.y, self.theta, 
            self.num_rays, start_angle, self.max_lidar_distance,
            self.obstacles_arr, self.num_active_obstacles,
            self.room_width, self.room_height,
            self.people_arr, self.num_people,
            self.use_legs, self.legs_coords # <--- PASSAGGIO COORDINATE GAMBE
        )
        
        if not self.lidar_noise_enable:
            return raw

        final = fast_apply_lidar_noise(
            raw, self.num_rays, 
            self.lidar_noise_std, 
            self.max_lidar_distance
        )
        
        return final.astype(np.float32)

    def _get_observation(self, v, w):
        if not hasattr(self, 'lidar_readings') or self.lidar_readings is None:
             self.lidar_readings = self._compute_lidar_fast()
        
        norm_dist = math.hypot(self.goal_x - self.x, self.goal_y - self.y) / self.max_possible_dist
        inv_lidar = (self.max_lidar_distance - self.lidar_readings) / (self.max_lidar_distance - 0.12)
        inv_lidar = np.clip(inv_lidar, 0.0, 1.0)
        gt = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        he = (gt - self.theta + math.pi) % (2*math.pi) - math.pi
        obs_scalars = np.array([norm_dist, he/math.pi, v/self.max_v, w/self.max_w], dtype=np.float32)
        return np.concatenate([obs_scalars, inv_lidar])

    def _reset_obstacles(self):
        self.obstacles = [] 
        self.num_active_obstacles = 0
        if self.num_obstacles > 0:
            num_circles = self.num_obstacles // 2
            num_rects = self.num_obstacles - num_circles
        else:
            num_circles = 0; num_rects = 0

        for _ in range(num_circles):
            cx = random.uniform(1, self.room_width-1); cy = random.uniform(1, self.room_height-1); r = random.uniform(0.4, 0.7)
            if self.num_active_obstacles < self.max_obstacles:
                self.obstacles_arr[self.num_active_obstacles] = [0.0, cx, cy, r, 0.0]; self.num_active_obstacles += 1
                self.obstacles.append({"type": "circle", "cx": cx, "cy": cy, "radius": r})

        for _ in range(num_rects):
            cx = random.uniform(1, self.room_width-1); cy = random.uniform(1, self.room_height-1)
            w = random.uniform(0.5, 1.0); h = random.uniform(0.5, 1.0)
            if self.num_active_obstacles < self.max_obstacles:
                self.obstacles_arr[self.num_active_obstacles] = [1.0, cx-w/2, cx+w/2, cy-h/2, cy+h/2]; self.num_active_obstacles += 1
                self.obstacles.append({"type": "rect", "xmin": cx-w/2, "xmax": cx+w/2, "ymin": cy-h/2, "ymax": cy+h/2})

    # =========================================================================
    # 🎨 RENDERING
    # =========================================================================
    
    def _to_screen(self, x, y):
        sx = int(x * self.scale)
        sy = int(self.window_size - (y * self.scale))
        return sx, sy

    def render(self):
        self.render_counter += 1
        if self.render_counter % self.render_skip != 0:
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            total_width = self.window_size + self.sidebar_width
            total_height = self.window_size
            self.screen = pygame.display.set_mode((total_width, total_height))
            pygame.display.set_caption("Turtlebot4 Simulation - Stats Monitor")
            self.clock = pygame.time.Clock()
            self.lidar_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_text = pygame.font.SysFont("Consolas", 18)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    print("\n>>> SKIP MANUALE RILEVATO <<<")
                    self.manual_skip_triggered = True

        self.screen.fill((255, 255, 255))
        
        sidebar_rect = pygame.Rect(self.window_size, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.screen, (240, 240, 240), sidebar_rect) 
        pygame.draw.line(self.screen, (0, 0, 0), (self.window_size, 0), (self.window_size, self.window_size), 3)

        self.lidar_surface.fill((0, 0, 0, 0))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.window_size, self.window_size), 2)

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

        lidar_readings = self.lidar_readings
        lidar_x_origin = self.x + self.lidar_offset * math.cos(self.theta)
        lidar_y_origin = self.y + self.lidar_offset * math.sin(self.theta)
        sx_origin, sy_origin = self._to_screen(lidar_x_origin, lidar_y_origin)
        start_angle = self.theta + self.lidar_start_angle_offset
        angles = [start_angle + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]

        for i in range(self.num_rays):
            dist = lidar_readings[i]
            ang = angles[i]
            x_end = lidar_x_origin + dist * math.cos(ang)
            y_end = lidar_y_origin + dist * math.sin(ang)
            sx_end, sy_end = self._to_screen(x_end, y_end)

            if i == 0:
                color = (0, 0, 255, 255)
                thickness = 2
                pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), thickness)
                continue

            if dist >= self.max_lidar_distance - 0.1:
                color = (180, 0, 255, 180) 
                thickness = 1
                pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), thickness)
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
                    pygame.draw.line(self.lidar_surface, color, (sx_origin, sy_origin), (sx_end, sy_end), 1)

        self.screen.blit(self.lidar_surface, (0, 0))

        rx, ry = self._to_screen(self.x, self.y)
        rr = int(self.robot_radius * self.scale)
        pygame.draw.circle(self.screen, (0, 0, 255), (rx, ry), rr) 
        head_x = self.x + (self.robot_radius * 1.5) * math.cos(self.theta)
        head_y = self.y + (self.robot_radius * 1.5) * math.sin(self.theta)
        hx, hy = self._to_screen(head_x, head_y)
        pygame.draw.line(self.screen, (0, 0, 100), (rx, ry), (hx, hy), 3)

        for p in self.people:
            if self.use_legs and "legs" in p:
                foot_len = 0.30
                color_shoe_fill = (169, 169, 169)
                color_shoe_edge = (50, 50, 50)
                color_leg_circle = (0, 0, 0)
                foot_theta = p["angle"]
                cos_t = math.cos(foot_theta)
                sin_t = math.sin(foot_theta)

                for lx, ly in p["legs"]:
                    local_verts = [
                        (-LEG_RADIUS, -LEG_RADIUS),
                        (-LEG_RADIUS, LEG_RADIUS),
                        (foot_len * 0.65, LEG_RADIUS),
                        (foot_len, LEG_RADIUS * 0.35),
                        (foot_len, -LEG_RADIUS * 0.35),
                        (foot_len * 0.65, -LEG_RADIUS)
                    ]
                    screen_verts = []
                    for loc_x, loc_y in local_verts:
                        rot_x = loc_x * cos_t - loc_y * sin_t
                        rot_y = loc_x * sin_t + loc_y * cos_t
                        screen_verts.append(self._to_screen(lx + rot_x, ly + rot_y))
                    pygame.draw.polygon(self.screen, color_shoe_fill, screen_verts)
                    pygame.draw.polygon(self.screen, color_shoe_edge, screen_verts, 1)
                    slx, sly = self._to_screen(lx, ly)
                    pygame.draw.circle(self.screen, color_leg_circle, (slx, sly), int(LEG_RADIUS * self.scale))
            else:
                # [FIX] FALLBACK: Disegna cerchio verde per il corpo se use_legs=False
                cx, cy = self._to_screen(p["x"], p["y"])
                r = int(self.people_radius * self.scale)
                pygame.draw.circle(self.screen, (0, 200, 0), (cx, cy), r)

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

        x_text = self.window_size + 20
        y_text = 20
        line_spacing = 25
        dist_to_goal = math.hypot(self.x - self.goal_x, self.y - self.goal_y) if self.goal_x is not None else 0.0
        
        if lidar_readings is not None and len(lidar_readings) > 0:
            min_laser_dist = np.min(lidar_readings)
        else:
            min_laser_dist = 0.0

        total_avg_jerk = 0.0
        if self.global_step_count > 0:
            total_avg_jerk = self.global_jerk_sum / self.global_step_count

        title = self.font_title.render("TELEMETRY", True, (0, 0, 0))
        self.screen.blit(title, (x_text, y_text))
        y_text += 40

        def draw_stat_line(label, value, color=(0,0,0)):
            label_surf = self.font_text.render(f"{label}:", True, (80, 80, 80))
            value_surf = self.font_text.render(str(value), True, color)
            self.screen.blit(label_surf, (x_text, y_text))
            val_w = value_surf.get_width()
            val_x = (self.window_size + self.sidebar_width) - val_w - 20
            self.screen.blit(value_surf, (val_x, y_text))
            return line_spacing

        y_text += draw_stat_line("Ep. Step", f"{self.step_count}")
        y_text += draw_stat_line("Goal Dist", f"{dist_to_goal:.2f} m")
        laser_color = (200, 0, 0) if min_laser_dist < 0.5 else (0, 100, 0)
        y_text += draw_stat_line("Lidar Min", f"{min_laser_dist:.2f} m", laser_color)
        y_text += 10
        y_text += draw_stat_line("Avg Jerk (Tot)", f"{total_avg_jerk:.2f}")

        y_text += 40
        plot_w = 200; plot_h = 150
        plot_x = x_text + (self.sidebar_width - 20 - plot_w) // 2 
        plot_y = y_text
        pygame.draw.rect(self.screen, (230, 230, 230), (plot_x, plot_y, plot_w, plot_h))
        pygame.draw.rect(self.screen, (0, 0, 0), (plot_x, plot_y, plot_w, plot_h), 2)
        mid_y = plot_y + plot_h / 2
        pygame.draw.line(self.screen, (150, 150, 150), (plot_x, mid_y), (plot_x + plot_w, mid_y), 1)
        pygame.draw.line(self.screen, (150, 150, 150), (plot_x, plot_y), (plot_x, plot_y + plot_h), 1)

        font_small = pygame.font.SysFont("Arial", 12)
        lbl_w_max = font_small.render(f"+{self.max_w}", True, (0,0,0)); self.screen.blit(lbl_w_max, (plot_x - 25, plot_y))
        lbl_w_min = font_small.render(f"-{self.max_w}", True, (0,0,0)); self.screen.blit(lbl_w_min, (plot_x - 25, plot_y + plot_h - 10))
        lbl_v_max = font_small.render(f"{self.max_v} m/s", True, (0,0,0)); self.screen.blit(lbl_v_max, (plot_x + plot_w - 30, plot_y + plot_h + 5))
        lbl_chart = font_small.render("V / W Space", True, (0,0,0)); self.screen.blit(lbl_chart, (plot_x + 5, plot_y + 5))

        dot_x_norm = self.v / self.max_v
        dot_screen_x = plot_x + int(dot_x_norm * plot_w)
        dot_y_norm = self.w / self.max_w 
        dot_screen_y = mid_y - int(dot_y_norm * (plot_h / 2))
        
        pygame.draw.circle(self.screen, (255, 0, 0), (dot_screen_x, dot_screen_y), 6)
        pygame.draw.line(self.screen, (255, 100, 100), (plot_x, dot_screen_y), (dot_screen_x, dot_screen_y), 1)
        pygame.draw.line(self.screen, (255, 100, 100), (dot_screen_x, mid_y), (dot_screen_x, dot_screen_y), 1)

        y_text = plot_y + plot_h + 30
        outcome_color = (0, 0, 0)
        txt = str(self.persistent_outcome).lower()
        if "goal" in txt: outcome_color = (0, 150, 0)
        elif "passive" in txt: outcome_color = (0, 150, 0)
        elif "collision" in txt: outcome_color = (200, 0, 0)
        elif "timeout" in txt: outcome_color = (150, 100, 0)
        
        self.screen.blit(self.font_text.render("Last Outcome:", True, (0,0,0)), (x_text, y_text))
        y_text += 20
        font_outcome_small = pygame.font.SysFont("Arial", 16, bold=True)
        outcome_surf = font_outcome_small.render(self.persistent_outcome.upper(), True, outcome_color)
        self.screen.blit(outcome_surf, (x_text, y_text))

        self.clock.tick(30)
        pygame.display.flip()

    def close(self):
        if self.screen: pygame.quit()