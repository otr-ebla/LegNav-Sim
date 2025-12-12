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
MODEL_PATH = "./checkpoints/2_Mno_obstacles_part2"
NORMALIZATION_FILE = "./2_Mno_obstacles_part2.pkl" 

# [AGGIORNATO] Parametri allineati con nav_env.py e gym_nav_env.py
MAX_LIN_VEL = 0.3   # TurtleBot4 Limit
MAX_ANG_VEL = 2.3   # TurtleBot4 Limit
NUM_RAYS_MODEL = 108
STACK_SIZE = 3

# [AGGIORNATO] Parametri Lidar/Ambiente (NavEnv)
MAX_LIDAR_DIST = 15.0   # Training env usa 15.0 per normalizzare
LIDAR_MIN_DIST = 0.12   # Blind spot
ROOM_WIDTH = 12.0
ROOM_HEIGHT = 12.0
# La diagonale usata per normalizzare la distanza nell'env
ROOM_DIAGONAL = math.hypot(ROOM_WIDTH, ROOM_HEIGHT) 

# Parametri Navigazione
GOAL_X = 7.0
GOAL_Y = 0.0
TRAINING_DT = 0.1 # Importante per il timer del loop
GOAL_THRESHOLD = 0.3

# Parametri Logging
PRINT_EVERY_N_STEPS = 5 

class DummyVecEnv(VecEnv):
    """Necessario per caricare VecNormalize senza creare l'env completo."""
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
        
        self.step_counter = 0
        
        # Flags
        self.odom_ready = False
        self.scan_ready = False
        self.odom_first_received = False
        self.scan_first_received = False

        # 1. Caricamento Modello e Normalizzazione
        self.get_logger().info(f"Caricamento modello da {MODEL_PATH}...")
        
        # Calcolo dimensione osservazione: 4 scalari + (108 * 3) lidar
        obs_dim = 4 + (NUM_RAYS_MODEL * STACK_SIZE) 
        
        # Definizione spazi per VecNormalize
        observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, shape=(obs_dim,), 
            dtype=np.float32)
        
        action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANG_VEL]), 
            high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), 
            dtype=np.float32)

        # Caricamento VecNormalize
        dummy_env = DummyVecEnv(observation_space, action_space)
        self.vec_normalize = VecNormalize.load(NORMALIZATION_FILE, dummy_env)
        self.vec_normalize.training = False # IMPORTANTE: Non aggiornare le statistiche in inferenza
        self.vec_normalize.norm_reward = False
        
        self.model = SAC.load(MODEL_PATH, device="cpu")
        self.get_logger().info("Sistema RL pronto.")

        # ROS2 Communication
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer Loop (esegue alla frequenza di training)
        self.create_timer(TRAINING_DT, self.control_loop)

    def odom_callback(self, msg):
        if not self.odom_first_received:
            self.get_logger().info("✅ Odom ricevuta.")
            self.odom_first_received = True
            
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Conversione Quaternione -> Eulero (Theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)
        self.odom_ready = True

    def scan_callback(self, msg):
        """
        Gestione Lidar: deve replicare esattamente nav_env.py
        """
        if not self.scan_first_received:
            self.get_logger().info(f"✅ Scan ricevuta.")
            self.scan_first_received = True
            
        raw_ranges = np.array(msg.ranges)
        
        # 1. Handling Inf/Nan
        cleaned_ranges = np.nan_to_num(raw_ranges, nan=MAX_LIDAR_DIST, posinf=MAX_LIDAR_DIST, neginf=MAX_LIDAR_DIST)
        cleaned_ranges[cleaned_ranges == 0.0] = MAX_LIDAR_DIST 

        # 2. Rotazione (Adatta in base al tuo robot fisico se necessario)
        # Assumiamo che Index 0 sia il FRONTE per il modello RL.
        # Se il lidar fisico ha il fronte a index 0, shift_amount = 0.
        # Se ha il fronte a index 270 (es. rplidar montato girato), shiftare come prima.
        # [NOTA] Mantengo la rotazione precedente se funzionava per il tuo robot:
        shift_amount = -(len(cleaned_ranges) // 4) 
        aligned_ranges = np.roll(cleaned_ranges, shift_amount)

        # 3. Downsampling a 108 raggi
        if len(aligned_ranges) >= NUM_RAYS_MODEL:
            indices = np.linspace(0, len(aligned_ranges) - 1, NUM_RAYS_MODEL, dtype=int)
            downsampled = aligned_ranges[indices]
        else:
            downsampled = np.pad(aligned_ranges, (0, NUM_RAYS_MODEL - len(aligned_ranges)), 'edge')

        # 4. Clipping ai limiti fisici (come in nav_env._cast_ray)
        # Se > 15.0 -> 15.0. Se < 0.12 -> 0.12
        downsampled = np.clip(downsampled, LIDAR_MIN_DIST, MAX_LIDAR_DIST)

        # 5. Inversione del valore degli scans (come in nav_env)
        for d in range(len(downsampled)):
            downsampled[d] = (LIDAR_MIN_DIST/downsampled[d])**(1/3)
        
        # Stacking
        if len(self.scan_stack) == 0:
            for _ in range(STACK_SIZE): self.scan_stack.append(downsampled)
        else:
            self.scan_stack.append(downsampled)

        self.scan_ready = True

    def get_obs(self):
        """
        Costruisce l'osservazione esattamente come _compose_obs in gym_nav_env 
        e _get_observation in nav_env.
        """
        # 1. Calcoli relativi al Goal
        goal_dx = GOAL_X - self.x
        goal_dy = GOAL_Y - self.y
        dist_to_goal = math.hypot(goal_dx, goal_dy)
        
        theta_goal = math.atan2(goal_dy, goal_dx)
        heading_error = theta_goal - self.theta
        # Normalizzazione angolo tra -pi e pi
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

    
        
        # Angolo: Normalizzato su PI -> range [-1, 1]
        norm_heading = heading_error / math.pi
        
       

        # 3. Preparazione Lidar appiattito
        lidar_stack_flat = np.concatenate(list(self.scan_stack), axis=0).astype(np.float32)

        # 4. Concatenazione finale
        # Struttura: [dist, angle, v, w, ...lidar...]
        obs = np.concatenate([
            [dist_to_goal, norm_heading, self.last_v, self.last_w],
            lidar_stack_flat
        ]).astype(np.float32)
        
        return obs.reshape(1, -1)

    def control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        # Calcoli per logging
        dist_to_goal = math.hypot(GOAL_X - self.x, GOAL_Y - self.y)
        
        # Logging Periodico
        self.step_counter += 1
        if self.step_counter % PRINT_EVERY_N_STEPS == 0:
            self.get_logger().info(
                f"📉 STATUS -> Dist: {dist_to_goal:.2f} m | V: {self.last_v:.2f} | W: {self.last_w:.2f}"
            )

        # 1. Ottieni osservazione "Raw" (ma strutturata come l'env)
        raw_obs = self.get_obs()
        
        # 2. Applica VecNormalize (scaling statistiche training)
        # Nota: vec_normalize si aspetta l'array piatto
        normalized_obs = self.vec_normalize.normalize_obs(raw_obs) 
        
        # 3. Predizione AI
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        
        # 4. Estrazione e Clipping (Safety)
        v = float(action[0][0])
        w = float(action[0][1])

        # Assicuriamoci che l'uscita rispetti i limiti hardware
        v = np.clip(v, 0.0, MAX_LIN_VEL)
        w = np.clip(w, -MAX_ANG_VEL, MAX_ANG_VEL)

        # 5. Check Goal
        if dist_to_goal < GOAL_THRESHOLD:
            v = 0.0; w = 0.0
            if self.step_counter % 20 == 0:
                self.get_logger().info(f"🏆 GOAL RAGGIUNTO! (Dist: {dist_to_goal:.2f}m)")

        self.last_v = v
        self.last_w = w

        # 6. Invio comando
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
