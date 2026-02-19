import numpy as np
import math
from src.config import LidarConfig, SimConfig, RobotConfig

class DWAPlanner:
    def __init__(self, config):
        # --- Robot Physics ---
        self.max_speed = config.MAX_LINEAR_VEL
        self.min_speed = 0.0 
        self.max_yaw_rate = config.MAX_W
        
        # Accelerazioni (ora usate per finestra ampia)
        self.max_accel = 2.5 
        self.max_yaw_accel = 20.0 
        
        # --- DWA Parameters MIGLIORATI ---
        self.dt = 0.1             # Passo più ampio per stabilità
        self.predict_time = 1.0   # Orizzonte temporale esteso
        self.sim_steps = 20       # Più granularità
        
        # Risoluzione campionamento più fine
        self.v_res = 0.08       
        self.y_res = 0.08       
        
        # --- Weights OTTIMIZZATI ---
        self.to_goal_cost_gain = 1.5     # Ridotto per favorire deviazioni
        self.speed_cost_gain = 25      # Leggermente aggressivo
        self.obstacle_cost_gain = 1.5    # Alto per sicurezza
        
        self.robot_radius = config.RADIUS
        
        # Parametri Lidar
        self.max_lidar_dist = 15.0 
        self.min_lidar_dist = 0.12
        self.fov = LidarConfig.FOV

    def plan(self, obs):
        state_vec = obs['state']
        current_v = state_vec[0]
        current_w = state_vec[1]
        dist_to_goal = state_vec[3]
        angle_to_goal = state_vec[4]
        
        # Goal locali
        local_goal_x = dist_to_goal * np.cos(angle_to_goal)
        local_goal_y = dist_to_goal * np.sin(angle_to_goal)

        # 2. Dynamic Window ESPANSA (finestra su tutto l'orizzonte)
        dw_v_min = max(self.min_speed, current_v - self.max_accel * self.predict_time)
        dw_v_max = min(self.max_speed, current_v + self.max_accel * self.predict_time)
        dw_w_min = -self.max_yaw_rate  # Full range yaw per curve!
        dw_w_max = self.max_yaw_rate

        v_samples = np.arange(dw_v_min, dw_v_max + 1e-4, self.v_res)
        w_samples = np.arange(dw_w_min, dw_w_max + 1e-4, self.y_res)
        
        # Recovery options
        v_samples = np.unique(np.concatenate((v_samples, [0.0, 0.2])))
        w_samples = np.unique(np.concatenate((w_samples, [0.0, -0.8, 0.8])))

        v_grid, w_grid = np.meshgrid(v_samples, w_samples)
        v_flat = v_grid.flatten()
        w_flat = w_grid.flatten()
        n_samples = v_flat.shape[0]

        # 3. Lidar points
        lidar_norm = obs['lidar'][-108:] 
        lidar_dists = self.max_lidar_dist - (lidar_norm * (self.max_lidar_dist - self.min_lidar_dist))
        lidar_angles = np.linspace(-self.fov/2, self.fov/2, len(lidar_dists))
        
        valid_mask = lidar_dists < (self.max_lidar_dist - 0.5)
        obs_x = lidar_dists[valid_mask] * np.cos(lidar_angles[valid_mask])
        obs_y = lidar_dists[valid_mask] * np.sin(lidar_angles[valid_mask])
        obs_points = np.stack((obs_x, obs_y), axis=1)

        # 4. Simulazione traiettorie
        dt_step = self.predict_time / self.sim_steps
        
        traj_x = np.zeros((n_samples, self.sim_steps))
        traj_y = np.zeros((n_samples, self.sim_steps))
        traj_theta = np.zeros((n_samples, self.sim_steps))
        
        curr_x = np.zeros(n_samples)
        curr_y = np.zeros(n_samples)
        curr_theta = np.zeros(n_samples)

        for i in range(self.sim_steps):
            curr_x += v_flat * np.cos(curr_theta) * dt_step
            curr_y += v_flat * np.sin(curr_theta) * dt_step
            curr_theta += w_flat * dt_step
            
            traj_x[:, i] = curr_x
            traj_y[:, i] = curr_y
            traj_theta[:, i] = curr_theta

        # 5. Costo Ostacoli MIGLIORATO (TUTTI gli step)
        if len(obs_points) > 0:
            # Appiattisci traiettorie per check completo: (n_samples * sim_steps, 2)
            traj_flat_x = traj_x.flatten()
            traj_flat_y = traj_y.flatten()
            
            dx = traj_flat_x[:, np.newaxis] - obs_x[np.newaxis, :]
            dy = traj_flat_y[:, np.newaxis] - obs_y[np.newaxis, :]
            d2 = dx**2 + dy**2
            min_d2_per_point = np.min(d2, axis=1)
            min_dist_per_point = np.sqrt(min_d2_per_point)
            
            # Min per traiettoria
            min_dist_per_traj = np.min(min_dist_per_point.reshape(n_samples, self.sim_steps), axis=1)
        else:
            min_dist_per_traj = np.full(n_samples, 100.0)

        # Margine dinamico (più conservativo vicino al goal)
        safety_margin = self.robot_radius + 0.15 + 0.1 * (1 - dist_to_goal / 5.0)
        collision_mask = min_dist_per_traj < safety_margin
        
        # 6. Altri costi
        final_x = traj_x[:, -1]
        final_y = traj_y[:, -1]
        final_theta = traj_theta[:, -1]
        
        dx_goal = local_goal_x - final_x
        dy_goal = local_goal_y - final_y
        error_angle = np.arctan2(dy_goal, dx_goal) - final_theta
        error_angle = (error_angle + np.pi) % (2 * np.pi) - np.pi
        cost_goal = np.abs(error_angle)

        cost_speed = self.max_speed - v_flat

        # Costi normalizzati
        cost_goal_norm = cost_goal / np.pi
        cost_speed_norm = cost_speed / self.max_speed
        dist_clamped = np.maximum(min_dist_per_traj, 0.01)
        cost_obs = 1.0 / dist_clamped

        total_cost = (self.to_goal_cost_gain * cost_goal_norm + 
                      self.speed_cost_gain * cost_speed_norm + 
                      self.obstacle_cost_gain * cost_obs)

        total_cost[collision_mask] = float('inf')

        # 7. Selezione + Recovery Intelligente
        best_idx = np.argmin(total_cost)
        
        if total_cost[best_idx] == float('inf') or np.all(total_cost == float('inf')):
            # Recovery: ruota verso direzione libera nel lidar
            if len(lidar_dists) > 0:
                free_dir_idx = np.argmax(lidar_dists)
                free_angle = lidar_angles[free_dir_idx]
                return np.array([0.0, np.clip(free_angle * 2.0, -self.max_yaw_rate/2, self.max_yaw_rate/2)])
            return np.array([0.0, 0.0])

        return np.array([v_flat[best_idx], w_flat[best_idx]])
