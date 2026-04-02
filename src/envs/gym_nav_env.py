import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from .fast_nav_env import Simple2DEnv
from src.config import RobotConfig, LidarConfig, SimConfig 

NUM_RAYS = LidarConfig.NUM_RAYS  # 216 rays, 360° full-circle LiDAR
STACK_DIM = RobotConfig.LIDAR_STACK_DIM
MAX_STEPS = SimConfig.MAX_STEPS
NUM_OBSTACLES = 12
PEOPLE_SPEED = 1.0

class GymNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays: int = NUM_RAYS,
        num_people: int = 0,
        num_obstacles: int = NUM_OBSTACLES,
        max_steps: int = MAX_STEPS,
        stack_dim: int = STACK_DIM,
        reward_factor_progress: float = 5.0,
        use_legs: bool = False,
        distraction_prob: float = 0.0,  
        render_skip: int = 1,
        lidar_noise_enable: bool = False,
        real_lidar_specs: bool = False,
    ):  
        super().__init__()
        
        self.render_mode = render_mode
        self.real_lidar_specs = real_lidar_specs
        
        # Lock ray count to config value (216)
        self.num_rays = num_rays 

        self.stack_dim = stack_dim
        self.max_steps = max_steps

        # Initialize physical environment
        self.env = Simple2DEnv(
            num_rays=self.num_rays,
            max_steps=max_steps,
            num_people=num_people,
            robot_radius=RobotConfig.RADIUS,
            room_width=SimConfig.ROOM_SIDE_LENGTH,
            room_height=SimConfig.ROOM_SIDE_LENGTH,
            num_obstacles=num_obstacles,
            reward_factor_progress=reward_factor_progress,
            people_speed=PEOPLE_SPEED,
            use_legs=use_legs,
            human_distraction_prob=distraction_prob,
            render_skip=render_skip,
            lidar_noise_enable=lidar_noise_enable,
            real_lidar_specs=real_lidar_specs,
        )
        
        # Observation Space (216 rays, 360° FOV)
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32),
            "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32), 
            # State vector is strictly 5 elements: [v_t, omega_t, v_max, dist, alignment]
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

        self.action_space = spaces.Box(
            low=np.array([0.0, -RobotConfig.MAX_W]), 
            high=np.array([RobotConfig.MAX_LINEAR_VEL, RobotConfig.MAX_W]), 
            dtype=np.float32,
        )
        
        self.last_v = 0.0
        self.last_w = 0.0

    def _process_obs(self, env_obs, v, w):
        # 1. Extract raw lidar safely from the end of the array
        lidar_proc = env_obs[-self.num_rays:]

        # 2. Normalized Pose
        s_x = self.env.x / SimConfig.ROOM_SIDE_LENGTH
        s_y = self.env.y / SimConfig.ROOM_SIDE_LENGTH
        s_theta = self.env.theta / math.pi
        pose_vec = np.array([s_x, s_y, s_theta], dtype=np.float32)

        # 3. Exact 5-element State Vector from flowchart
        dx = self.env.goal_x - self.env.x
        dy = self.env.goal_y - self.env.y
        goal_distance = math.hypot(dx, dy)

        # Calculate relative alignment error in radians [-pi, pi]
        angle_to_goal = math.atan2(dy, dx)
        goal_error_alignment = (angle_to_goal - self.env.theta + math.pi) % (2 * math.pi) - math.pi

        # Create the state vector exactly as requested
        state_vec = np.array([
            v,                    # v_t
            w,                    # omega_t
            self.env.max_v,       # v_max
            goal_distance,        # goal_distance
            goal_error_alignment  # goal_error_alignment
        ], dtype=np.float32)

        return {
            "lidar": lidar_proc.astype(np.float32),
            "pose": pose_vec,
            "state": state_vec
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.last_v = 0.0
        self.last_w = 0.0
        
        env_obs = self.env.reset()
        return self._process_obs(env_obs, 0.0, 0.0), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], 0.0, RobotConfig.MAX_LINEAR_VEL))
        w = float(np.clip(action[1], -RobotConfig.MAX_W, RobotConfig.MAX_W))

        env_obs, reward, done, info = self.env.step((v, w))
        
        terminated = False
        truncated = False
        reason = info.get("termination_reason", "unknown")
        if done:
            if reason == "max_steps_reached": truncated = True
            else: terminated = True

        if self.render_mode == "human": self.env.render()
        
        processed_obs = self._process_obs(env_obs, v, w)
        self.last_v = v
        self.last_w = w

        return processed_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human": self.env.render()

    def close(self):
        self.env.close()