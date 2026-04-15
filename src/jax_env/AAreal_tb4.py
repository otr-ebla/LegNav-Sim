'''
AAreal_tb4.py
ROS2 Node for Patrolling the real Turtlebot4 between two waypoints using a JAX-trained PPO model.

'''


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
import sys
import os

import jax
import jax.numpy as jnp
import flax.serialization

# --- IMPORT JAX TRAINING MODULES ---
try:
    from jax_env import EnvState, get_obs, ROOM_W, ROOM_H, ROBOT_RADIUS, NUM_RAYS, MAX_LIDAR_DIST, FOV
    from jax_network import EndToEndActorCritic, scale_action_to_env
except ImportError as e:
    raise ImportError(f"Missing JAX environment modules: {e}. Make sure this script is in the same folder as jax_env.py and jax_network.py")

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/ppo_model_best.msgpack"

# Environment parameters (MUST MATCH TRAINING)
MAX_LIN_VEL = 0.46     # Max linear speed requested
STACK_DIM = 3          # Network trained with 3 frames
POSE_SIZE = 3
STATE_VEC_SIZE = 5

# --- PATROL PARAMETERS (Real World Coordinates) ---
WAYPOINT_A = (4.0, 0.0)   # Example: 4 meters straight ahead
WAYPOINT_B = (0.0, 0.0)   # Return to origin

TRAINING_DT = 0.1         # 10 Hz control loop
GOAL_THRESHOLD = 0.3      # Distance to consider waypoint reached


class PatrolNodeJAX(Node):
    def __init__(self):
        super().__init__('patrol_node_jax')

        # Internal State
        self.lidar_stack = deque(maxlen=STACK_DIM)
        self.pose_stack = deque(maxlen=STACK_DIM)
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_v = 0.0
        self.last_w = 0.0
        
        # Goal Management
        self.current_target_name = "A"
        self.goal_x = WAYPOINT_A[0]
        self.goal_y = WAYPOINT_A[1]
        
        self.step_counter = 0
        self.first_scan_received = False
        self.first_odom_received = False
        self.latest_scan_normalized = np.zeros(NUM_RAYS, dtype=np.float32)

        # PRNG Key for JAX operations
        self.rng = jax.random.PRNGKey(0)

        # --- 1. LOAD JAX MODEL ---
        self.get_logger().info(f"🧠 Loading JAX PPO model from: {MODEL_PATH}")
        
        # Initialize network architecture
        # OBS_SIZE is 662: (3 * 3) + 5 + (216 * 3)
        self.obs_size = (POSE_SIZE * STACK_DIM) + STATE_VEC_SIZE + (NUM_RAYS * STACK_DIM)
        self.network = EndToEndActorCritic(action_dim=2, stack_dim=STACK_DIM, num_rays=NUM_RAYS)
        
        # Initialize dummy parameters to get the exact structure expected by flax
        dummy_obs = jnp.zeros((1, self.obs_size))
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.network.init(init_rng, dummy_obs)["params"]

        try:
            with open(MODEL_PATH, "rb") as f:
                raw_bytes = f.read()
            # Msgpack restore fills the initialized params structure
            bundle = flax.serialization.msgpack_restore(raw_bytes)
            self.params = bundle["params"]
            self.get_logger().info("✅ JAX Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # JIT compile the inference step for maximum speed
        @jax.jit
        def _fast_inference(p, o):
            mean, _, _ = self.network.apply({"params": p}, o)
            return scale_action_to_env(jnp.squeeze(mean, axis=0), MAX_LIN_VEL)
        self.fast_inference = _fast_inference

        # --- 2. ROS 2 SETUP ---
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        
        # Subscriptions
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control Loop Timer (10Hz)
        self.create_timer(TRAINING_DT, self.control_loop)
        
        self.get_logger().info(f"🚀 PATROL ACTIVE. Initial target: {self.current_target_name} ({self.goal_x}, {self.goal_y})")

    def odom_callback(self, msg):
        self.first_odom_received = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Quaternion to Euler (Yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        """
        Interpolates the real LiDAR scan down to the 216 rays expected by the network,
        covering the specific FOV used during training.
        """
        raw_ranges = np.array(msg.ranges)
        real_angles = np.linspace(msg.angle_min, msg.angle_max, len(raw_ranges))
        
        # Target angles based on the training environment's FOV
        target_angles = np.linspace(-FOV / 2.0, FOV / 2.0, NUM_RAYS)
        
        # 1. Clean Inf/Nan/Zeros
        cleaned = np.nan_to_num(raw_ranges, nan=MAX_LIDAR_DIST, posinf=MAX_LIDAR_DIST, neginf=MAX_LIDAR_DIST)
        cleaned[cleaned < 0.05] = MAX_LIDAR_DIST 
        cleaned = np.clip(cleaned, 0.0, MAX_LIDAR_DIST)

        # 2. Resample to 216 rays over the trained FOV
        resampled = np.interp(target_angles, real_angles, cleaned, left=MAX_LIDAR_DIST, right=MAX_LIDAR_DIST)

        # 3. Normalize matching jax_eval_multi.py logic
        # network receives: (MAX_LIDAR_DIST - raw_lidar) / (MAX_LIDAR_DIST - ROBOT_RADIUS)
        normalized_scan = (MAX_LIDAR_DIST - resampled) / (MAX_LIDAR_DIST - ROBOT_RADIUS)
        self.latest_scan_normalized = np.clip(normalized_scan, 0.0, 1.0).astype(np.float32)
        
        self.first_scan_received = True

    def _get_base_obs(self):
        """
        Constructs a dummy EnvState with real robot variables, calls the training
        get_obs function to guarantee exact math parity for pose and state_vec.
        """
        # Create a dummy array for 12 humans (all set to -1.0 sentinel to mask them out)
        dummy_people = jnp.zeros((12, 11))
        dummy_people = dummy_people.at[:, 10].set(-1.0) 

        dummy_state = EnvState(
            x=self.x, 
            y=self.y, 
            theta=self.theta, 
            v=self.last_v, 
            w=self.last_w,
            goal_x=self.goal_x, 
            goal_y=self.goal_y, 
            max_v=MAX_LIN_VEL,
            people=dummy_people,
            obs_circles=jnp.zeros((6, 3)),
            obs_boxes=jnp.zeros((6, 4)),
            time_step=0,
            foot_state=jnp.zeros((12, 10)),
            time_stopped=0,
            sp_mask=jnp.zeros(NUM_RAYS, dtype=jnp.bool_)
        )

        self.rng, obs_key = jax.random.split(self.rng)
        
        # We only care about base_obs; we ignore the returned mask
        base_obs, _ = get_obs(dummy_state, obs_key)
        return np.array(base_obs)

    def get_stacked_obs(self):
        """Builds the flattened observation vector with temporal stacks."""
        base_obs = self._get_base_obs()
        
        # Extract features (ignoring the simulated LiDAR at the end of base_obs)
        current_pose = base_obs[0:POSE_SIZE]
        current_state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        
        # First step initialization
        if len(self.pose_stack) == 0:
            for _ in range(STACK_DIM):
                self.pose_stack.append(current_pose)
                self.lidar_stack.append(self.latest_scan_normalized)
        else:
            self.pose_stack.append(current_pose)
            self.lidar_stack.append(self.latest_scan_normalized)

        # Flatten stacks: oldest to newest
        pose_stack_flat = np.concatenate(list(self.pose_stack), axis=0)
        lidar_stack_flat = np.concatenate(list(self.lidar_stack), axis=0)

        # Final concatenation: [pose_stack(9) | state_vec(9) | lidar_stack(324)]
        obs_flat = np.concatenate([
            pose_stack_flat, 
            current_state_vec, 
            lidar_stack_flat
        ]).astype(np.float32)
        
        return jnp.array(obs_flat[None, :]) # Shape (1, 662)

    def control_loop(self):
        # Wait for sensors
        if not self.first_scan_received or not self.first_odom_received:
            return

        # --- PATROL LOGIC ---
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        
        if dist_to_goal < GOAL_THRESHOLD:
            self.get_logger().info(f"🏆 Checkpoint {self.current_target_name} reached! Reversing route.")
            
            if self.current_target_name == "A":
                self.goal_x, self.goal_y = WAYPOINT_B
                self.current_target_name = "B"
            else:
                self.goal_x, self.goal_y = WAYPOINT_A
                self.current_target_name = "A"
            return # Pause for 1 tick to update state cleanly

        # --- INFERENCE ---
        try:
            # 1. Build observation vector
            stacked_obs = self.get_stacked_obs()
            
            # 2. JIT compiled forward pass + action scaling
            env_action = self.fast_inference(self.params, stacked_obs)
            
            # 3. Extract actions
            v = float(env_action[0])
            w = float(env_action[1])

            # --- PUBLISH COMMAND ---
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)
            
            # Update state for next step
            self.last_v = v
            self.last_w = w
            
            # Sporadic logging
            self.step_counter += 1
            if self.step_counter % 20 == 0: # Every 2 seconds
                self.get_logger().info(f"To {self.current_target_name} | Dist: {dist_to_goal:.2f}m | V: {v:.2f} | W: {w:.2f}")

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PatrolNodeJAX()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Emergency stop
        stop = Twist()
        node.cmd_vel_pub.publish(stop)
        node.get_logger().info("🛑 Manual stop triggered.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()