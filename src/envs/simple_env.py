# envs/simple_env.py

import math
import matplotlib.pyplot as plt 
import random
import numpy as np

class Simple2DEnv:
    """
    Very simple 2D environment skeleton.
    For now it only keeps track of step counter.
    Later we'll add:
    -   rooms, walls, doors, people
    -   robot state and lidar
    -   reset() randomized worlds
    """

    def __init__(
            self, 
            max_steps: int = 400, 
            dt: float = 0.25,
            room_width: float = 20.0,
            room_height: float = 20.0,
            robot_radius: float = 0.2,
            num_rays: int = 8,
            max_lidar_distance: float = 20.0,
            num_people: int = 3,
            people_radius: float = 0.2,
            people_speed: float = 0.5,
            ):

        self.max_steps = max_steps
        self.step_count = 0
        self.dt = dt

        # Robot state variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # orientation
        self.trajectory = []  # list of (x,y) positions

        # Room geometry
        self.room_width = room_width
        self.room_height = room_height
        self.robot_radius = robot_radius

        self.num_rays = num_rays
        self.max_lidar_distance = max_lidar_distance

        # People parameters
        self.num_people = num_people
        self.people_radius = people_radius
        self.people_speed = people_speed

        self.last_termination_reason = None

        self.people = [] # list of dicts with x,y,vx,vy

        self.obstacles = []
        self.num_obstacles = 10

        # GOAL position
        self.goal_x = None
        self.goal_y = None
        self.goal_radius = 0.3

        self.fig = None
        self.ax = None

    def _ray_rect_intersection(self, angle, xmin, xmax, ymin, ymax):
        """
        Compute intersection of a ray from (self.x, self.y) at angle `angle`
        with an axis-aligned rectangle defined by (xmin, xmax, ymin, ymax).
        Return the distance to the intersection point, or None if no intersection.
        """
        x0, y0 = self.x, self.y
        dx = math.cos(angle)
        dy = math.sin(angle)

        eps = 1e-8

        # X slab
        if abs(dx) < eps:
            if x0 < xmin or x0 > xmax:
                return None
            tmin_x = -math.inf
            tmax_x = math.inf
        else:
            t1 = (xmin - x0) / dx
            t2 = (xmax - x0) / dx
            tmin_x = min(t1, t2)
            tmax_x = max(t1, t2)

        # Y slab
        if abs(dy) < eps:
            if y0 < ymin or y0 > ymax:
                return None
            tmin_y = -math.inf
            tmax_y = math.inf
        else:
            t1 = (ymin - y0) / dy
            t2 = (ymax - y0) / dy
            tmin_y = min(t1, t2)
            tmax_y = max(t1, t2)

        t_near = max(tmin_x, tmin_y)
        t_far = min(tmax_x, tmax_y)

        # No intersection or behind the origin
        if t_near > t_far or t_far < 0:
            return None
        
        if t_near < 0:
            if t_far >= 0:
                return t_far
        return t_near

    def _ray_circle_intersection(self, angle, cx, cy, radius):
        """
        Compute intersection of a ray from (self.x, self.y) at angle `angle`
        with a circle centered at (cx, cy) with given radius.
        Return the distance to the intersection point, or None if no intersection.
        """
        x0, y0 = self.x, self.y
        dx = math.cos(angle)
        dy = math.sin(angle)

        # f = o - c
        fx = x0 - cx
        fy = y0 - cy

        # Quadratic coefficients: t^2 + bt + c = 0 (a=1)
        b = 2.0 * (fx * dx + fy * dy)
        c = fx*fx + fy*fy -radius*radius

        # Discriminant
        discriminant = b*b-4.0*c

        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc)/2.0
        t2 = (-b + sqrt_disc)/2.0

        candidates = []
        if t1 > 0:
            candidates.append(t1)
        if t2 > 0:
            candidates.append(t2)

        if not candidates:
            return None
        
        return min(candidates)
    
    def _reset_obstacles(self):
        """
        Initialize random static obstacles:
        - circles
        - axis-aligned rectangles
        For now they are just visual + LiDAR obstacles (no collisions/physics yet).
        """
        self.obstacles = []
        margin = 1.0

        num_circles = self.num_obstacles // 2
        num_rects = self.num_obstacles - num_circles

        # ---- Circles ----
        for _ in range(num_circles):
            radius = random.uniform(0.6, 1.5)
            cx = random.uniform(margin + radius, self.room_width - margin - radius)
            cy = random.uniform(margin + radius, self.room_height - margin - radius)

            self.obstacles.append({"type": "circle", 
                                   "cx": cx, 
                                   "cy": cy, 
                                   "radius": radius})
            
        # ---- Rectangles ----    
        for _ in range(num_rects):
            w = random.uniform(1.0, 3.0)
            h = random.uniform(1.0, 3.0)
            cx = random.uniform(margin + w/2.0, self.room_width - margin - w/2.0)
            cy = random.uniform(margin + h/2.0, self.room_height - margin - h/2.0)

            xmin = cx - w/2.0
            xmax = cx + w/2.0
            ymin = cy - h/2.0
            ymax = cy + h/2.0

            self.obstacles.append({"type": "rect",
                                   "xmin": xmin,
                                   "xmax": xmax,
                                   "ymin": ymin,
                                   "ymax": ymax})

    def _reset_people(self):
        """Initialize people at random positions with random velocities"""
        self.people = []

        margin = self.people_radius + 0.5

        for _ in range(self.num_people):
            px, py = self.x, self.y
            while np.linalg.norm(np.array([px, py]) - np.array([self.x, self.y])) < 1.0:
                px = random.uniform(margin, self.room_width - margin)
                py = random.uniform(margin, self.room_height - margin)
            while self._point_inside_any_obstacle(px, py):
                px = random.uniform(margin, self.room_width - margin)
                py = random.uniform(margin, self.room_height - margin)

            # random direction and speed
            angle = random.uniform(0, 2 * math.pi)
            vx = self.people_speed * math.cos(angle) * random.uniform(0.5, 1.0)
            vy = self.people_speed * math.sin(angle) * random.uniform(0.5, 1.0)

            self.people.append({"x": px, "y": py, "angle": angle, "vx": vx, "vy": vy})

    def _is_goal_reached(self) -> bool:
        if self.goal_x is None or self.goal_y is None:
            return False
        dx = self.x - self.goal_x
        dy = self.y - self.goal_y
        dist_sq = dx * dx + dy * dy
        return dist_sq <= self.goal_radius * self.goal_radius

    def _handle_person_obstacle_collisions(self, p):
        """
        Resolve collisions between one person (disk) and all static obstacles.
        Simple resolution: push person out and reflect velocity.
        """
        pr = self.people_radius
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy, radius = obs["cx"], obs["cy"], obs["radius"]
                dx = p["x"] - cx
                dy = p["y"] - cy
                dist_sq = dx*dx + dy*dy
                min_dist = pr + radius
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    if dist == 0:
                        dist = 1e-6
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    p["x"] += nx * overlap
                    p["y"] += ny * overlap

                    # Reflect velocity
                    v_dot_n = p["vx"] * nx + p["vy"] * ny
                    p["vx"] -= 2 * v_dot_n * nx
                    p["vy"] -= 2 * v_dot_n * ny
                    p["angle"] = math.atan2(p["vy"], p["vx"])
            elif obs["type"] == "rect":
                xmin, xmax = obs["xmin"], obs["xmax"]
                ymin, ymax = obs["ymin"], obs["ymax"]

                closest_x = max(xmin, min(p["x"], xmax))
                closest_y = max(ymin, min(p["y"], ymax))

                dx = p["x"] - closest_x
                dy = p["y"] - closest_y
                dist_sq = dx*dx + dy*dy
                if dist_sq < pr * pr:
                    dist = math.sqrt(dist_sq)
                    if dist == 0:
                        dist = 1e-6
                    overlap = pr - dist
                    nx = dx / dist
                    ny = dy / dist
                    p["x"] += nx * overlap
                    p["y"] += ny * overlap

                    # Reflect velocity
                    v_dot_n = p["vx"] * nx + p["vy"] * ny
                    p["vx"] -= 2 * v_dot_n * nx
                    p["vy"] -= 2 * v_dot_n * ny
                    p["angle"] = math.atan2(p["vy"], p["vx"])

    def _step_people(self):
        """move people and make them bounce on walls"""

        for p in self.people:
            p["x"] += p["vx"] * self.dt
            p["y"] += p["vy"] * self.dt
            p["angle"] = math.atan2(p["vy"], p["vx"])

            # bounce on vertical walls
            if p["x"] - self.people_radius < 0:
                p["x"] = self.people_radius
                p["vx"] = -p["vx"]
                p["angle"] = math.atan2(p["vy"], p["vx"])
            elif p["x"] + self.people_radius > self.room_width:
                p["x"] = self.room_width - self.people_radius
                p["vx"] = -p["vx"]
                p["angle"] = math.atan2(p["vy"], p["vx"])
            # bounce on horizontal walls
            if p["y"] - self.people_radius < 0:
                p["y"] = self.people_radius
                p["vy"] = -p["vy"]
                p["angle"] = math.atan2(p["vy"], p["vx"])
            elif p["y"] + self.people_radius > self.room_height:
                p["y"] = self.room_height - self.people_radius
                p["vy"] = -p["vy"]
                p["angle"] = math.atan2(p["vy"], p["vx"])

            self._handle_person_obstacle_collisions(p)

    def _is_collision_with_people(self) -> bool:
        """
        Check if the robot (a disk with radius robot_radius) 
        is intersecting with any of the people.
        """
        rr = self.robot_radius
        pr = self.people_radius
        min_dist_sq = (rr + pr) ** 2

        for p in self.people:
            dx = self.x - p["x"]
            dy = self.y - p["y"]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                return True
        return False
            
    def _is_collision_with_obstacles(self) -> bool:
        """
        Check if the robot (a disk with radius robot_radius) 
        is intersecting with any of the static obstacles.
        """
        rr = self.robot_radius

        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy, radius = obs["cx"], obs["cy"], obs["radius"]
                dx = self.x - cx
                dy = self.y - cy
                dist_sq = dx*dx + dy*dy
                min_dist = rr + radius
                if dist_sq < min_dist * min_dist:
                    return True
            elif obs["type"] == "rect":
                xmin, xmax = obs["xmin"], obs["xmax"]
                ymin, ymax = obs["ymin"], obs["ymax"]

                closest_x = max(xmin, min(self.x, xmax))
                closest_y = max(ymin, min(self.y, ymax))

                dx = self.x - closest_x
                dy = self.y - closest_y
                dist_sq = dx*dx + dy*dy
                if dist_sq < rr * rr:
                    return True
        return False

    def render(self):
        """render the room, robot, and lidar rays using matplotlib"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion() # interactive mode ON

        self.ax.clear()

        # Draw room rectangle
        self.ax.plot(
            [0, self.room_width, self.room_width, 0, 0],
            [0, 0, self.room_height, self.room_height, 0],
            'k-'
        )

        # Draw static obstacles
        for obs in self.obstacles:
            if obs["type"] == "circle":
                circle = plt.Circle((obs["cx"], obs["cy"]), obs["radius"], color='gray', alpha=0.7, fill=True)
                self.ax.add_patch(circle)
            elif obs["type"] == "rect":
                rect = plt.Rectangle(
                    (obs["xmin"], obs["ymin"]),
                    obs["xmax"] - obs["xmin"],
                    obs["ymax"] - obs["ymin"],
                    color='gray',
                    alpha=0.7,
                    fill=True
                )
                self.ax.add_patch(rect)

        # Draw robot as a circle
        robot_circle = plt.Circle((self.x, self.y), self.robot_radius, color='blue', fill=True)
        self.ax.add_patch(robot_circle)

        arrow_len = self.robot_radius * 1.5
        x_head = self.x + arrow_len * math.cos(self.theta)
        y_head = self.y + arrow_len * math.sin(self.theta)

        self.ax.plot(
            [self.x, x_head],
            [self.y, y_head],
            'b-', 
            linewidth=2
        )

        # Draw people
        for p in self.people:
            person = plt.Circle((p["x"], p["y"]), self.people_radius, color='green', fill=True)
            x_people_head = p["x"] + 0.3*math.cos(p["angle"])
            y_people_head = p["y"] + 0.3*math.sin(p["angle"])
            self.ax.add_patch(person)
            self.ax.plot(
                [p["x"], x_people_head],
                [p["y"], y_people_head],
                'g-',
                linewidth=2
            )

        lidar = self._compute_lidar()
        angles = [
            self.theta + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)
        ]

        if len(self.trajectory) >= 2:
            traj_xs, traj_ys = zip(*self.trajectory)
            self.ax.plot(traj_xs, traj_ys, 'b--', linewidth=1)

        base_grey = (0.6, 0.6, 0.6)
        base_red = (1.0, 0.0, 0.0)

        # Draw lidar rays
        for dist, ang in zip(lidar, angles):
            x_end = self.x + dist * math.cos(ang)
            y_end = self.y + dist * math.sin(ang)

            # max_range_for_color = max(self.room_width, self.room_height)
            # norm_d = min(dist, max_range_for_color) / max_range_for_color
            # proximity = (1.0 - norm_d) ** 2
            color_ray_dist = 5.0
            norm_d = min(dist, color_ray_dist) / color_ray_dist
            proximity = (1.0 - norm_d) 

            r = base_grey[0] + proximity * (base_red[0] - base_grey[0])
            g = base_grey[1] + proximity * (base_red[1] - base_grey[1])
            b = base_grey[2] + proximity * (base_red[2] - base_grey[2])

            alpha = 0.9 * proximity
            rgba = (r, g, b, alpha)

            self.ax.plot([self.x, x_end], [self.y, y_end], color=rgba, linewidth=0.5)

            self.ax.plot(
                x_end, y_end,
                marker='o',
                markersize=2,
                color=rgba,
            )

        # Draw goal
        if self.goal_x is not None and self.goal_y is not None:
            self.ax.plot(
                self.goal_x, self.goal_y,
                marker='*',
                markersize=7,
                color='orange',
            )

        # limits and aspect
        self.ax.set_xlim(-1, self.room_width + 1)
        self.ax.set_ylim(-1, self.room_height + 1)
        self.ax.set_aspect('equal', adjustable='box')

        status_lines = [f"step: {self.step_count}"]

        if self.last_termination_reason is not None:
            status_lines.append(f"last termination: {self.last_termination_reason}")

        status_text = "\n".join(status_lines)

        self.ax.text(
            0.01, 0.99,
            status_text,
            verticalalignment='top',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
        )

        plt.pause(0.0001)

    def _point_inside_obstacle(self, x, y, obs):
        """
        Return True if point (x,y) is inside the given obstacle
        """
        if obs["type"] == "circle":
            cx, cy, radius = obs["cx"], obs["cy"], obs["radius"]
            dx = x - cx
            dy = y - cy
            dist_sq = dx*dx + dy*dy
            return dist_sq < radius * radius
        elif obs["type"] == "rect":
            xmin, xmax = obs["xmin"], obs["xmax"]
            ymin, ymax = obs["ymin"], obs["ymax"]
            return xmin <= x <= xmax and ymin <= y <= ymax
        return False

    def _point_inside_any_obstacle(self, x, y):
        """
        Return True if point (x,y) is inside any obstacle
        """
        for obs in self.obstacles:
            if self._point_inside_obstacle(x, y, obs):
                return True
        return False

    def reset(self):
        """Reset the environment to an initial state"""
        self.step_count = 0
        self.last_termination_reason = None

        self.x = self.room_width / 2.0
        self.y = self.room_height / 2.0
        self.theta = np.random.uniform(0, 2 * math.pi)
        


        
        self._reset_obstacles()
        margin = self.robot_radius + 1.3
        if self._is_collision_with_obstacles() or self._is_collision_with_walls():
            while True:
                self.x = random.uniform(margin, self.room_width - margin)
                self.y = random.uniform(margin, self.room_height - margin)
                if not self._is_collision_with_obstacles() and not self._is_collision_with_walls():
                    break
        self.trajectory = [(self.x, self.y)]    

        while True:
            gx = random.uniform(1.0, self.room_width - 1.0)
            gy = random.uniform(1.0, self.room_height - 1.0)
            dx = gx - self.x
            dy = gy - self.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 2.0:
                continue
            if self._point_inside_any_obstacle(gx, gy):
                continue
            break

        self.goal_x = gx
        self.goal_y = gy

        self._reset_people()

        lidar = self._compute_lidar()
        obs = (self.x, self.y, self.theta, lidar)
        return obs
    
    def _is_collision_with_walls(self) -> bool:
        """
        Check if the robot (a disk with radius robot_radius) 
        is intersecting with the walls of the room.
        """
        if self.x - self.robot_radius < 0 or self.x + self.robot_radius > self.room_width:
            return True
        if self.y - self.robot_radius < 0 or self.y + self.robot_radius > self.room_height:
            return True
        return False

    def _cast_ray(self, angle):
        """
        Cast 1 ray from (x,y in direction = angle)
        Return the distance until it hits a wall.
        """
        x0, y0 = self.x, self.y
        dx = math.cos(angle)
        dy = math.sin(angle)

        distances = [] # all possible intersections that 1 ray can have, only the minimum will be the important

        eps = 1e-6
        # Intersection with x = 0
        if abs(dx) > eps:
            t = (0 - x0) / dx
            if t >= 0:
                y_hit = y0 + t * dy
                if 0 <= y_hit <= self.room_height:
                    distances.append(t)

        # Intersection with x = room_width
        if abs(dx) > eps:
            t = (self.room_width - x0) / dx
            if t >= 0:
                y_hit = y0 + t * dy
                if 0 <= y_hit <= self.room_height:
                    distances.append(t)

        # Intersection with y = 0
        if abs(dy) > eps:
            t = (0 - y0) / dy
            if t >= 0:
                x_hit = x0 + t * dx
                if 0 <= x_hit <= self.room_width:
                    distances.append(t)

        # Intersection with y = room_height
        if abs(dy) > eps:
            t = (self.room_height - y0) / dy
            if t >= 0:
                x_hit = x0 + t * dx
                if 0 <= x_hit <= self.room_width:
                    distances.append(t)

        # ------- Intersection with people -------
        for p in self.people:
            cx, cy = p["x"], p["y"]
            t_circle = self._ray_circle_intersection(angle, cx, cy, self.people_radius)
            if t_circle is not None:
                distances.append(t_circle)

        # ------- Intersection with obstacles -------
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy, radius = obs["cx"], obs["cy"], obs["radius"]
                t_circle = self._ray_circle_intersection(angle, cx, cy, radius)
                if t_circle is not None:
                    distances.append(t_circle)
            elif obs["type"] == "rect":
                xmin, xmax = obs["xmin"], obs["xmax"]
                ymin, ymax = obs["ymin"], obs["ymax"]
                t_rect = self._ray_rect_intersection(angle, xmin, xmax, ymin, ymax)
                if t_rect is not None:
                    distances.append(t_rect)

        if len(distances) == 0:
            return self.max_lidar_distance
        
        return min(distances) # for every ray-angle return min computed value, that is the intersection
    
    def _compute_lidar(self):
        """
        Compute lidar distance for all rays"""

    
    def step(self, action):
        """
        Advance the simulation by one step.
        For now:
            - ignore the action
            - just increase step counter
        """
        v, w = action # unpack tuple

        dist_before = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)

        # Differential drive update
        self.x += v * self.dt * math.cos(self.theta)
        self.y += v * self.dt * math.sin(self.theta)
        self.theta += w * self.dt

        self.trajectory.append((self.x, self.y))
        self._step_people()
        self.step_count += 1

        dist_after = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)

        # Compute observations and Collisions
        lidar = self._compute_lidar()
        obs = (self.x, self.y, self.theta, lidar)

        collision_wall = self._is_collision_with_walls()    
        collision_people = self._is_collision_with_people()
        collision_obstacles = self._is_collision_with_obstacles()


        # --- REWARD ENGINNERING ---

        reward = 0.0

        # A. Progress reward: Positive if getting closer, negative if moving away
        progress = (dist_before - dist_after) # positive if getting closer
        reward += 5*progress

        # B. Negative penalty for difference in heading angle to goal in two consecutive steps
        # Negative penalty for change in heading-to-goal between consecutive steps
        if self.goal_x is not None and self.goal_y is not None and len(self.trajectory) >= 2:
            x_prev, y_prev = self.trajectory[-2]
            ang_prev = math.atan2(self.goal_y - y_prev, self.goal_x - x_prev)
            ang_curr = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
            # normalize to [-pi, pi]
            d_ang = ang_curr - ang_prev
            d_ang = (d_ang + math.pi) % (2 * math.pi) - math.pi
            reward -= 5 * abs(d_ang)

        

        # D. Safety/LiDAR penalty
        min_lidar = min(lidar)
        if min_lidar < 1.0:
            reward -= math.exp(1.0 - min_lidar)


        done = False
        info = {}

        if self._is_goal_reached():
            done = True
            reward += 200.0
            info["termination_reason"] = "goal_reached"

        if collision_people:
            done = True
            reward -= 50.0
            info["termination_reason"] = "people_collision"

        if collision_wall:
            done = True
            reward -= 50.0
            info["termination_reason"] = "wall_collision"

        if collision_obstacles:
            done = True
            reward -= 50.0
            info["termination_reason"] = "obstacle_collision"

        if self.step_count >= self.max_steps:
            done = True
            reward = -1.0
            info["termination_reason"] = "max_steps_reached"

        if done:
            self.last_termination_reason = info.get("termination_reason", "unknown")

        return obs, reward, done, info
    
    def _compute_lidar(self):
        """
        Compute lidar distance for all rays
        """
        angles = [self.theta + i * (2 * math.pi / self.num_rays) for i in range(self.num_rays)]
        distances = [self._cast_ray(angle) for angle in angles]

        # clip values to max_lidar_distance
        distances = [min(d, self.max_lidar_distance) for d in distances]
        return distances
    
    def close(self):
        """Clean up resources (matplotlib figure)"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None