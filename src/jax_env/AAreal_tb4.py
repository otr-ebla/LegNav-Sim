import math
import sys

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from collections import deque
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

try:
    from jax_network import EndToEndActorCritic, scale_action_to_env
except ImportError as e:
    raise ImportError(f"Missing JAX modules: {e}. Ensure jax_network.py is available.")


# ── Constants (must match jax_env.py / config.py) ─────────────────────────────
ROBOT_RADIUS   = 0.17
NUM_RAYS       = 216
MAX_LIDAR_DIST = 12.0
FOV            = 2.0 * math.pi          # full 360° — same as jax_env.py

MODEL_PATH     = "checkpoints/ppo_attn_final.msgpack"

# FIX 2: use a single DEPLOY_MAX_V that encodes the curriculum stage correctly.
# Set this to the max_v value you want the robot to target at deploy time.
# 0.46 m/s = TB4 safe indoor speed; change to e.g. 1.2 if the checkpoint was
# trained at higher speeds and you want more aggressive navigation.
DEPLOY_MAX_V   = 0.46

STACK_DIM      = 3
POSE_SIZE      = 3
STATE_VEC_SIZE = 5
MAX_GOAL_DIST  = math.hypot(12.0, 12.0)   # diagonal of 12×12 m room

# Patrol waypoints in odom frame (metres). Origin = robot start position.
WAYPOINT_A = (6.0, 0.0)
WAYPOINT_B = (0.0, 0.0)

TRAINING_DT    = 0.1    # control loop period — must match RobotConfig.DT
GOAL_THRESHOLD = 0.3    # metres — same as GOAL_RADIUS in jax_env.py

# ── Sim LiDAR angle grid (matches compute_lidar in jax_physics.py) ────────────
# angles = theta - FOV/2 + arange(NUM_RAYS) * (FOV / (NUM_RAYS - 1))
# At theta=0: ray 0 → -π (LEFT), ray 108 → 0 (FORWARD), ray 215 → +π (LEFT again)
# We store the offset from theta=0, i.e. the per-ray angular offsets.
_SIM_RAY_OFFSETS = -FOV * 0.5 + np.arange(NUM_RAYS) * (FOV / (NUM_RAYS - 1))


class PatrolNodeJAX(Node):
    def __init__(self):
        super().__init__("patrol_node_jax")

        # ── Robot state ───────────────────────────────────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0
        self.last_v = 0.0
        self.last_w = 0.0

        # ── Observation stacks ────────────────────────────────────────────────
        self.lidar_stack = deque(maxlen=STACK_DIM)
        self.pose_stack  = deque(maxlen=STACK_DIM)
        self.latest_scan_normalized = np.zeros(NUM_RAYS, dtype=np.float32)

        # ── Patrol state ──────────────────────────────────────────────────────
        self.current_target_name = "A"
        self.goal_x = WAYPOINT_A[0]
        self.goal_y = WAYPOINT_A[1]
        self.step_counter = 0

        # ── Sensor ready flags ────────────────────────────────────────────────
        self.first_scan_received = False
        self.first_odom_received = False

        # ── JAX network setup ─────────────────────────────────────────────────
        self.get_logger().info(f"Loading JAX model from {MODEL_PATH}")
        self.rng = jax.random.PRNGKey(0)

        obs_size = POSE_SIZE * STACK_DIM + STATE_VEC_SIZE + NUM_RAYS * STACK_DIM  # 662
        self.network = EndToEndActorCritic(action_dim=2, stack_dim=STACK_DIM, num_rays=NUM_RAYS)

        dummy_obs = jnp.zeros((1, obs_size))
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.network.init(init_rng, dummy_obs)["params"]

        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = flax.serialization.msgpack_restore(f.read())
            self.params = bundle["params"]
            self.get_logger().info("JAX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

        # JIT-compile inference (runs once on first call, then cached)
        @jax.jit
        def _fast_inference(p, o):
            mean, _, _ = self.network.apply({"params": p}, o)
            return scale_action_to_env(jnp.squeeze(mean, axis=0), DEPLOY_MAX_V)
        self.fast_inference = _fast_inference

        # ── ROS2 subscriptions & publishers ───────────────────────────────────
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(LaserScan, "/turtlebot1/scan",  self.scan_callback, qos)
        self.create_subscription(Odometry,  "/turtlebot1/odom",  self.odom_callback, qos)

        self.cmd_vel_pub = self.create_publisher(Twist, "/turtlebot1/cmd_vel", 1)
        self.create_timer(TRAINING_DT, self.control_loop)

        self.get_logger().info(
            f"Patrol ACTIVE | target: {self.current_target_name} | "
            f"max_v: {DEPLOY_MAX_V} m/s"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.first_odom_received = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp  = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        raw = np.array(msg.ranges, dtype=np.float32)

        # ── Clean invalid readings ─────────────────────────────────────────
        cleaned = np.where(np.isfinite(raw), raw, MAX_LIDAR_DIST)
        #cleaned = np.where(cleaned < 0.12,   MAX_LIDAR_DIST, cleaned)
        cleaned = np.where(cleaned < 0.16, MAX_LIDAR_DIST, cleaned)
        cleaned = np.clip(cleaned, 0.0, MAX_LIDAR_DIST)

        # ── FIX 1: angle-aware resampling to match sim ray convention ─────
        # Real sensor angles (world-frame offsets from robot heading = 0).
        # ROS REP-103: angle_min = right side (negative), increases CCW.
        n_real = len(cleaned)
        real_angles = np.linspace(msg.angle_min, msg.angle_max, n_real)

        # Simulation target angles: _SIM_RAY_OFFSETS are robot-relative offsets.
        # At theta=0 these are absolute world angles → we sample the real scan
        # at these same robot-relative angles.
        # np.interp with period=2π handles the circular wraparound correctly
        # regardless of where angle_min/angle_max fall.
        downsampled = np.interp(
            _SIM_RAY_OFFSETS,
            real_angles,
            cleaned,
            period=2.0 * math.pi,
        )

        inv_lidar = (MAX_LIDAR_DIST - downsampled) / (MAX_LIDAR_DIST - ROBOT_RADIUS)
        self.latest_scan_normalized = np.clip(inv_lidar, 0.0, 1.0).astype(np.float32)
        self.first_scan_received = True

    def get_stacked_obs(self) -> jnp.ndarray:
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        cos_t   = math.cos(-self.theta)
        sin_t   = math.sin(-self.theta)
        gdx_ego = dx * cos_t - dy * sin_t
        gdy_ego = dx * sin_t + dy * cos_t

        goal_dist  = math.hypot(dx, dy)
        goal_angle = math.atan2(dy, dx)
        goal_align = (goal_angle - self.theta + math.pi) % (2.0 * math.pi) - math.pi

        current_pose = np.array([
            gdx_ego / MAX_GOAL_DIST,
            gdy_ego / MAX_GOAL_DIST,
            self.theta / math.pi,
        ], dtype=np.float32)

        # FIX 2: state_vec uses DEPLOY_MAX_V consistently for indices 0 and 2
        # state_vec: [v/max_v, w, (max_v-0.2)/1.8, goal_dist/D, goal_align/π]
        current_state_vec = np.array([
            self.last_v / max(DEPLOY_MAX_V, 1e-3),
            self.last_w,
            (DEPLOY_MAX_V - 0.2) / 1.8,
            goal_dist / MAX_GOAL_DIST,
            goal_align / math.pi,
        ], dtype=np.float32)

        # Cold-start: tile current frame (identical to reset_stacked tile)
        if len(self.pose_stack) == 0:
            for _ in range(STACK_DIM):
                self.pose_stack.append(current_pose.copy())
                self.lidar_stack.append(self.latest_scan_normalized.copy())
        else:
            self.pose_stack.append(current_pose)
            self.lidar_stack.append(self.latest_scan_normalized)

        pose_stack_flat  = np.concatenate(list(self.pose_stack))   # (9,)
        lidar_stack_flat = np.concatenate(list(self.lidar_stack))  # (648,)

        obs_flat = np.concatenate([
            pose_stack_flat,    # 9
            current_state_vec,  # 5
            lidar_stack_flat,   # 648
        ]).astype(np.float32)   # total: 662

        return jnp.array(obs_flat[None, :])  # (1, 662)

    # ── Control loop ─────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.first_scan_received or not self.first_odom_received:
            return

        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)

        # ── Waypoint switch ───────────────────────────────────────────────
        if dist_to_goal < GOAL_THRESHOLD:
            self.get_logger().info(
                f"Waypoint {self.current_target_name} reached! Switching target."
            )
            if self.current_target_name == "A":
                self.goal_x, self.goal_y = WAYPOINT_B
                self.current_target_name = "B"
            else:
                self.goal_x, self.goal_y = WAYPOINT_A
                self.current_target_name = "A"
            self.last_v = 0.0
            self.last_w = 0.0
            # Reset stacks so the new episode starts clean (mirrors env reset)
            self.pose_stack.clear()
            self.lidar_stack.clear()
            return

        # ── NN Inference ──────────────────────────────────────────────────
        try:
            stacked_obs = self.get_stacked_obs()
            env_action  = self.fast_inference(self.params, stacked_obs)

            v = float(np.array(env_action[0]))
            w = float(np.array(env_action[1]))

            cmd = Twist()
            cmd.linear.x  = v
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)

            self.last_v = v
            self.last_w = w
            self.step_counter += 1

            if self.step_counter % 20 == 0:
                self.get_logger().info(
                    f"→ {self.current_target_name} | dist: {dist_to_goal:.2f} m | "
                    f"v: {v:.2f} m/s | w: {w:.2f} rad/s"
                )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self._publish_stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    def destroy_node(self):
        """FIX 3: always stop the robot before ROS2 tears down the node."""
        self.get_logger().info("Stopping robot (destroy_node).")
        self._publish_stop()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PatrolNodeJAX()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # FIX 3: reuse existing publisher (guaranteed to flush before shutdown)
        node.get_logger().info("KeyboardInterrupt — stopping robot.")
        node._publish_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()