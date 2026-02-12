import numpy as np


class RobotConfig:
    RADIUS = 0.2  # robot radius meters
    MAX_LINEAR_VEL = 0.3  # m/s
    MAX_W = 0.8
    DT = 0.1 # 0.25
    LIDAR_OFFSET = -0.05  # meters
    LIDAR_STACK_DIM = 3 #3

class SimConfig:
    HSFM_DT = 0.01
    ROOM_SIDE_LENGTH = 7.0  # meters
    ROOM_SIZE = (ROOM_SIDE_LENGTH, ROOM_SIDE_LENGTH)  # meters
    MAX_STEPS = 1000
    HUMANS_RADIUS = 0.2  # meters
    HUMANS_VELOCITY = 1.0 # m/s
    NUM_OBSTACLES = 25
    NUM_HUMANS = 5
    RADIUS_EXTENDED = 0.3


class LidarConfig:
    NUM_RAYS = 108
    MAX_DISTANCE = SimConfig.ROOM_SIDE_LENGTH * np.sqrt(2)  # meters
    MIN_DIST = 0.12  # meters
    FOV = 2*np.pi
