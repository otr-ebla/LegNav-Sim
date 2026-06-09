import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math
from collections import deque
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from gymnasium import spaces

# --- IMPORTAZIONI SB3 ---
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize, VecEnv
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    raise ImportError("Installa le librerie necessarie: pip install sb3-contrib stable-baselines3 shimmy gymnasium")

# --- CONFIGURAZIONE ---
MODEL_PATH = "./25MSAC_jack"
NORMALIZATION_FILE = "./25MSAC_jack_vecnormalize.pkl" 

# Parametri Ambiente
MAX_LIN_VEL = 0.46
MAX_ANG_VEL = 2.0
NUM_RAYS_MODEL = 216
STACK_SIZE = 3
MAX_LIDAR_DIST = 12.0  
ROOM_DIAGONAL = math.sqrt(12.0**2 + 12.0**2) 

# --- PARAMETRI PATTUGLIAMENTO (Patrol) ---
WAYPOINT_A = (6.0, -0.2)  # Punto A
WAYPOINT_B = (0.3, -0.2)  # Punto B (Origine)

TRAINING_DT = 0.25       # 4 Hz
GOAL_THRESHOLD = 0.1 # Distanza per considerare il punto raggiunto

# Parametri Logging
PRINT_EVERY_N_STEPS = 4  # Stampa ogni ~1 secondo

class DummyVecEnv(VecEnv):
    def __init__(self, observation_space, action_space):
        super().__init__(1, observation_space, action_space)
    def step_async(self, actions): pass
    def step_wait(self): return np.zeros((1,) + self.observation_space.shape), 0.0, False, False, {}
    def reset(self): return np.zeros((1,) + self.observation_space.shape), {}
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None]
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def seed(self, seed=None): return [set_random_seed(seed)]
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

class RLNavNode(Node):
    def __init__(self):
        super().__init__('rl_nav_node')

        # Stato Interno
        self.scan_stack = deque(maxlen=STACK_SIZE)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_v = 0.0
        self.last_w = 0.0
        
        # --- GESTIONE GOAL DINAMICO ---
        self.current_target_name = "A"
        self.goal_x = WAYPOINT_A[0]
        self.goal_y = WAYPOINT_A[1]
        
        self.step_counter = 0
        
        # Flag
        self.odom_ready = False
        self.scan_ready = False
        self.odom_first_received = False
        self.scan_first_received = False

        # Load AI
        self.get_logger().info(f"Caricamento modello da {MODEL_PATH}...")
        self.model = SAC.load(MODEL_PATH, device="cpu")
        
        obs_dim = 4 + (NUM_RAYS_MODEL * STACK_SIZE) 
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=np.array([0.0, -MAX_ANG_VEL]), high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), dtype=np.float32)

        dummy_env = DummyVecEnv(observation_space, action_space)
        self.vec_normalize = VecNormalize.load(NORMALIZATION_FILE, dummy_env)
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False
        
        self.get_logger().info(f"Sistema RL pronto. Start verso Punto {self.current_target_name}: {self.goal_x}, {self.goal_y}")

        # ROS2
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(TRAINING_DT, self.control_loop)

    def odom_callback(self, msg):
        if not self.odom_first_received:
            self.get_logger().info("✅ Odom ricevuta.")
            self.odom_first_received = True
            
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)
        self.odom_ready = True

    def scan_callback(self, msg):
        if not self.scan_first_received:
            self.get_logger().info(f"✅ Scan ricevuta.")
            self.scan_first_received = True
            
        raw_ranges = np.array(msg.ranges)
        
        # 1. Cleaning
        cleaned_ranges = np.nan_to_num(raw_ranges, nan=MAX_LIDAR_DIST, posinf=MAX_LIDAR_DIST, neginf=MAX_LIDAR_DIST)
        cleaned_ranges[cleaned_ranges == 0.0] = MAX_LIDAR_DIST 
        cleaned_ranges = np.clip(cleaned_ranges, 0.0, MAX_LIDAR_DIST)

        # 2. Rotazione (Fix Cecità)
        shift_amount = -(len(cleaned_ranges) // 4)
        aligned_ranges = np.roll(cleaned_ranges, shift_amount)

        # 3. Downsampling
        if len(aligned_ranges) >= NUM_RAYS_MODEL:
            indices = np.linspace(0, len(aligned_ranges) - 1, NUM_RAYS_MODEL, dtype=int)
            downsampled = aligned_ranges[indices]
        else:
            downsampled = np.pad(aligned_ranges, (0, NUM_RAYS_MODEL - len(aligned_ranges)), 'edge')

        # 4. Normalizzazione
        normalized_scan = downsampled / MAX_LIDAR_DIST
        
        if len(self.scan_stack) == 0:
            for _ in range(STACK_SIZE): self.scan_stack.append(normalized_scan)
        else:
            self.scan_stack.append(normalized_scan)
            
        self.scan_ready = True

    def get_obs(self):
        # Usa goal_x/y dinamici della classe invece di costanti globali
        goal_dx = self.goal_x - self.x
        goal_dy = self.goal_y - self.y
        
        dist_to_goal = math.sqrt(goal_dx**2 + goal_dy**2)
        rho_norm = np.clip(dist_to_goal / ROOM_DIAGONAL, 0.0, 1.0)

        theta_goal = math.atan2(goal_dy, goal_dx)
        angle_diff = theta_goal - self.theta
        wrapped_angle = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        alpha_norm = wrapped_angle / math.pi

        last_v_norm = np.clip(self.last_v / MAX_LIN_VEL, 0.0, 1.0)
        last_w_norm = np.clip(self.last_w / MAX_ANG_VEL, -1.0, 1.0)

        lidar_stack_flat = np.concatenate(list(self.scan_stack), axis=0).astype(np.float32)

        obs = np.concatenate([
            [rho_norm, alpha_norm, last_v_norm, last_w_norm],
            lidar_stack_flat
        ]).astype(np.float32)
        
        return obs.reshape(1, -1)

    def control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        # 1. Calcoli Distanza dal Target CORRENTE
        dist_dx = self.goal_x - self.x
        dist_dy = self.goal_y - self.y
        dist_to_goal = math.sqrt(dist_dx**2 + dist_dy**2)
        
        goal_angle = math.atan2(dist_dy, dist_dx)
        angle_err = goal_angle - self.theta
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi
        
        current_scan_normalized = self.scan_stack[-1]
        min_dist_norm = np.min(current_scan_normalized)
        min_dist = min_dist_norm * MAX_LIDAR_DIST # Rinormalizza

        # 2. LOGGING PERIODICO
        self.step_counter += 1
        if self.step_counter % PRINT_EVERY_N_STEPS == 0:
            self.get_logger().info(
                f"🚀 To {self.current_target_name} | Dist: {dist_to_goal:.2f} m | Min LiDAR: {min_dist:.2f} m | Err: {math.degrees(angle_err):.1f}°"
            )

        # 3. CHECK CAMBIO TARGET (LOGICA PATTUGLIAMENTO)
        if dist_to_goal < GOAL_THRESHOLD:
            self.get_logger().info(f"🏆 Raggiunto Punto {self.current_target_name}! Cambio direzione...")
            
            # Scambio Waypoint
            if self.current_target_name == "A":
                self.goal_x = WAYPOINT_B[0]
                self.goal_y = WAYPOINT_B[1]
                self.current_target_name = "B"
            else:
                self.goal_x = WAYPOINT_A[0]
                self.goal_y = WAYPOINT_A[1]
                self.current_target_name = "A"
            
            # Nota: Non fermiamo il robot (v=0), lasciamo che l'RL gestisca l'inversione fluida
            # Aggiorniamo subito la distanza per il prossimo calcolo obs
            return # Saltiamo un ciclo di controllo per permettere l'aggiornamento stato pulito

        # 4. Inferenza
        raw_obs = self.get_obs()
        normalized_obs = self.vec_normalize.normalize_obs(raw_obs) 
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        v = float(action[0][0])
        w = float(action[0][1])

        v = np.clip(v, 0.0, MAX_LIN_VEL)
        w = np.clip(w, -MAX_ANG_VEL, MAX_ANG_VEL)

        self.last_v = v
        self.last_w = w

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = RLNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.cmd_vel_pub.publish(stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
