#!/usr/bin/env python3
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
import sys
import os

# --- IMPORTAZIONI SB3 & CUSTOM NN ---
try:
    # IMPORTANTE: Usiamo TQC perché il modello è stato allenato con TQC
    from sb3_contrib import TQC 
    from stable_baselines3.common.vec_env import VecNormalize, VecEnv
    from stable_baselines3.common.utils import set_random_seed
    
    # Importiamo la definizione della rete neurale custom
    # Assicurati che models/hybrid_cnn_mlp.py esista
    from models.hybrid_cnn_mlp import HybridCnnMlp
except ImportError as e:
    raise ImportError(f"Librerie mancanti: {e}. Installa: pip install sb3-contrib stable-baselines3 shimmy gymnasium")

# --- CONFIGURAZIONE ---
# Nome del modello caricato (senza .zip)
MODEL_NAME = "Training_Noise_1080_7Humans"
MODEL_PATH = f"./checkpoints/{MODEL_NAME}"
NORMALIZATION_FILE = f"./checkpoints/{MODEL_NAME}_vecnormalize.pkl"

# Parametri Ambiente (DEVONO COMBACIARE COL TRAINING)
MAX_LIN_VEL = 0.3      # Config.py: 0.3
MAX_ANG_VEL = 0.8      # Config.py: 0.8
NUM_RAYS_MODEL = 1080  # Rete allenata con 1080 raggi
STACK_SIZE = 5         # Rete allenata con 5 frame
MAX_LIDAR_DIST = 12.0  # Config.py: 12.0
ROOM_DIAGONAL = math.sqrt(12.0**2 + 12.0**2) 

# --- PARAMETRI PATTUGLIAMENTO (Coordinate Real World) ---
# Modifica questi punti in base alla tua stanza reale
WAYPOINT_A = (4.0, 0.0)   # Esempio: 4 metri avanti
WAYPOINT_B = (0.0, 0.0)   # Torna all'origine

TRAINING_DT = 0.1        # 10 Hz (Matcha nav_env dt=0.1)
GOAL_THRESHOLD = 0.3     # Distanza per considerare il punto raggiunto

# --- CLASSE DUMMY PER CARICARE VECNORMALIZE ---
class DummyVecEnv(VecEnv):
    """Classe fittizia necessaria solo per caricare le statistiche di normalizzazione"""
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

class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node_real')

        # Stato Interno
        self.scan_stack = deque(maxlen=STACK_SIZE)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_v = 0.0
        self.last_w = 0.0
        
        # Gestione Goal
        self.current_target_name = "A"
        self.goal_x = WAYPOINT_A[0]
        self.goal_y = WAYPOINT_A[1]
        
        self.step_counter = 0
        self.first_scan_received = False
        self.first_odom_received = False

        # --- 1. CARICAMENTO MODELLO ---
        self.get_logger().info(f"🧠 Caricamento modello TQC da: {MODEL_PATH}")
        
        # Definiamo gli oggetti custom necessari per deserializzare il modello
        custom_objects = {
            "HybridCnnMlp": HybridCnnMlp,
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0
        }

        try:
            # Carichiamo il modello TQC
            self.model = TQC.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
            self.get_logger().info("✅ Modello caricato con successo.")
        except Exception as e:
            self.get_logger().error(f"❌ Errore caricamento modello: {e}")
            sys.exit(1)
        
        # --- 2. CARICAMENTO NORMALIZZAZIONE ---
        # Creiamo spazi fittizi per inizializzare VecNormalize
        # Dimensione Osservazione: 4 scalari + (1080 raggi * 5 stack) = 5404
        obs_dim = 4 + (NUM_RAYS_MODEL * STACK_SIZE) 
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=np.array([0.0, -MAX_ANG_VEL]), high=np.array([MAX_LIN_VEL, MAX_ANG_VEL]), dtype=np.float32)

        dummy_env = DummyVecEnv(observation_space, action_space)
        
        try:
            self.vec_normalize = VecNormalize.load(NORMALIZATION_FILE, dummy_env)
            self.vec_normalize.training = False # IMPORTANTE: Non aggiornare le statistiche in inferenza
            self.vec_normalize.norm_reward = False
            self.get_logger().info("✅ VecNormalize caricato.")
        except Exception as e:
            self.get_logger().error(f"❌ Errore caricamento normalizzazione: {e}")
            sys.exit(1)

        # --- 3. ROS 2 SETUP ---
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        
        # Sottoscrizioni
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer di Controllo (10Hz)
        self.create_timer(TRAINING_DT, self.control_loop)
        
        self.get_logger().info(f"🚀 PATTUGLIA ATTIVA. Target iniziale: {self.current_target_name} ({self.goal_x}, {self.goal_y})")

    def odom_callback(self, msg):
        self.first_odom_received = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Conversione Quaternione -> Eulero (Yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        # NOTA: Il TurtleBot4 reale potrebbe restituire un numero variabile di raggi (es. 720, 1081, 1440).
        # La rete richiede ESATTAMENTE 1080 raggi.
        # Qui facciamo un resampling lineare per adattare qualsiasi input a 1080.
        
        raw_ranges = np.array(msg.ranges)
        
        # 1. Pulizia Inf/Nan
        # Sostituiamo infiniti e nan con MAX_DIST
        cleaned = np.nan_to_num(raw_ranges, nan=MAX_LIDAR_DIST, posinf=MAX_LIDAR_DIST, neginf=MAX_LIDAR_DIST)
        # Sostituiamo gli zeri (errore lidar) con MAX_DIST
        cleaned[cleaned < 0.05] = MAX_LIDAR_DIST 
        # Clamping
        cleaned = np.clip(cleaned, 0.0, MAX_LIDAR_DIST)

        # 2. Resampling a 1080 (Senza rotazioni/riallineamenti)
        current_len = len(cleaned)
        if current_len != NUM_RAYS_MODEL:
            # Interpolazione lineare per ottenere esattamente 1080 punti
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, NUM_RAYS_MODEL)
            resampled = np.interp(x_new, x_old, cleaned)
        else:
            resampled = cleaned

        # 3. Normalizzazione [0, 1]
        normalized_scan = resampled / MAX_LIDAR_DIST
        
        # 4. Aggiornamento Stack
        # Se è la prima volta, riempiamo lo stack duplicando la scansione
        if not self.first_scan_received:
            for _ in range(STACK_SIZE):
                self.scan_stack.append(normalized_scan)
            self.first_scan_received = True
            self.get_logger().info("✅ Primo scan ricevuto e processato.")
        else:
            self.scan_stack.append(normalized_scan)

    def get_obs(self):
        """Costruisce il vettore di osservazione per la rete neurale"""
        # 1. Calcolo Scalari Relativi al Goal
        goal_dx = self.goal_x - self.x
        goal_dy = self.goal_y - self.y
        dist_to_goal = math.hypot(goal_dx, goal_dy)
        
        # Normalizzazione Distanza
        rho_norm = np.clip(dist_to_goal / ROOM_DIAGONAL, 0.0, 1.0)

        # Calcolo Errore Angolare
        theta_goal = math.atan2(goal_dy, goal_dx)
        angle_diff = theta_goal - self.theta
        # Wrap angle in [-pi, pi]
        wrapped_angle = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        # Normalizzazione Angolo [-1, 1]
        alpha_norm = wrapped_angle / math.pi

        # Normalizzazione Velocità Precedenti
        last_v_norm = np.clip(self.last_v / MAX_LIN_VEL, 0.0, 1.0)
        last_w_norm = np.clip(self.last_w / MAX_ANG_VEL, -1.0, 1.0)

        # 2. Appiattimento Stack Lidar
        # [oldest, ..., newest] -> Concatenazione
        lidar_stack_flat = np.concatenate(list(self.scan_stack), axis=0).astype(np.float32)

        # 3. Concatenazione Finale
        obs = np.concatenate([
            [rho_norm, alpha_norm, last_v_norm, last_w_norm],
            lidar_stack_flat
        ]).astype(np.float32)
        
        return obs.reshape(1, -1) # Shape (1, 5404)

    def control_loop(self):
        # Attendi inizializzazione sensori
        if not self.first_scan_received or not self.first_odom_received:
            return

        # --- LOGICA PATTUGLIAMENTO ---
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        
        if dist_to_goal < GOAL_THRESHOLD:
            self.get_logger().info(f"🏆 Checkpoint {self.current_target_name} Raggiunto! Inversione rotta.")
            
            if self.current_target_name == "A":
                self.goal_x, self.goal_y = WAYPOINT_B
                self.current_target_name = "B"
            else:
                self.goal_x, self.goal_y = WAYPOINT_A
                self.current_target_name = "A"
            return # Pausa di 1 tick per aggiornare lo stato

        # --- INFERENZA ---
        try:
            # 1. Ottieni osservazione grezza
            raw_obs = self.get_obs()
            
            # 2. Normalizza usando le statistiche del training
            normalized_obs = self.vec_normalize.normalize_obs(raw_obs) 
            
            # 3. Predizione Rete (Deterministic=True per robot reale)
            action, _ = self.model.predict(normalized_obs, deterministic=True)
            
            # 4. Denormalizzazione Azione (Anche se la rete outputta già valori scalati, clippiamo per sicurezza)
            v_raw = float(action[0][0])
            w_raw = float(action[0][1])

            # Applicazione limiti hardware
            v = np.clip(v_raw, 0.0, MAX_LIN_VEL)
            w = np.clip(w_raw, -MAX_ANG_VEL, MAX_ANG_VEL)

            # --- PUBLISH COMANDO ---
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)
            
            # Aggiorna stato per prossimo step
            self.last_v = v
            self.last_w = w
            
            # Log sporadico
            self.step_counter += 1
            if self.step_counter % 20 == 0: # Ogni 2 secondi
                self.get_logger().info(f"To {self.current_target_name} | Dist: {dist_to_goal:.2f}m | V: {v:.2f} | W: {w:.2f}")

        except Exception as e:
            self.get_logger().error(f"Errore nel control loop: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Stop di emergenza
        stop = Twist()
        node.cmd_vel_pub.publish(stop)
        node.get_logger().info("🛑 Arresto manuale.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()