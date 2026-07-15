import numpy as np


class RobotConfig:
    RADIUS = 0.2  # robot radius meters
    MAX_LINEAR_VEL = 2.0 #0.3  # m/s
    MAX_W = 1.0
    DT = 0.15 # 0.25
    LIDAR_OFFSET = -0.05  # meters
    LIDAR_STACK_DIM = 3  #3
    # Temporal stride between stacked frames. With DT small, consecutive scans
    # are nearly identical, so we stack every STRIDE-th frame: the exposed stack
    # holds o_t, o_{t-STRIDE}, o_{t-2*STRIDE}, ... (STRIDE=1 => classic o_t,o_{t-1},o_{t-2}).
    LIDAR_STACK_STRIDE = 2

class SimConfig:
    HSFM_DT = 0.01
    ROOM_SIDE_LENGTH = 10.0  # meters
    ROOM_SIZE = (ROOM_SIDE_LENGTH, ROOM_SIDE_LENGTH)  # meters
    MAX_STEPS = 500
    HUMANS_RADIUS = 0.4  # meters
    PEOPLE_RADIUS = 0.4
    HUMANS_VELOCITY = 1.0 # m/s
    NUM_HUMANS = 5
    JAX_NUM_PEOPLE = 24
    JAX_NUM_OBS_CIR = 12
    JAX_NUM_OBS_BOX = 12
    RADIUS_EXTENDED = 0.3
    LEG_RADIUS = 0.08
    SHOE_WIDTH = 0.12
    SHOE_LENGTH = 0.3
    HIP_WIDTH = 0.32


class LidarConfig:
    NUM_RAYS = 216          # 360° full-circle LiDAR (108 front + 108 rear)
    MAX_DISTANCE = SimConfig.ROOM_SIDE_LENGTH * np.sqrt(2)  # meters
    MIN_DIST = 0.12  # meters
    FOV = 2 * np.pi          # 360° full-circle FOV
