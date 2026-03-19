"""
jax_scenarios.py — Modular Scenario Generator (JHSFM Version)
==============================================================
Outputs an 11-element array per human to support JHSFM waypoints:
[x, y, vx, vy, theta, omega, g1x, g1y, g2x, g2y, goal_idx]
"""

import jax
import jax.numpy as jnp
import math

ROOM_W = 12.0
ROOM_H = 12.0
ROBOT_RADIUS = 0.2
PEOPLE_RADIUS = 0.2
GOAL_RADIUS = 0.3
NUM_PEOPLE = 12
NUM_OBS_CIR = 6
NUM_OBS_BOX = 6
MAX_RESAMPLE_ITERS = 200

def _min_dist_to_circles(x, y, circles):
    dx = circles[:, 0] - x
    dy = circles[:, 1] - y
    return jnp.min(jnp.sqrt(dx**2 + dy**2) - circles[:, 2])

def _min_dist_to_boxes(x, y, boxes):
    def _box_dist(box):
        cx, cy, hw, hh = box
        ddx = jnp.maximum(jnp.abs(x - cx) - hw, 0.0)
        ddy = jnp.maximum(jnp.abs(y - cy) - hh, 0.0)
        return jnp.sqrt(ddx**2 + ddy**2)
    return jnp.min(jax.vmap(_box_dist)(boxes))

def _is_safe(x, y, clearance, obs_circles, obs_boxes):
    wall_ok = (x > clearance) & (x < ROOM_W - clearance) & \
              (y > clearance) & (y < ROOM_H - clearance)
    cir_ok  = _min_dist_to_circles(x, y, obs_circles) > clearance
    box_ok  = _min_dist_to_boxes(x, y, obs_boxes) > clearance
    return wall_ok & cir_ok & box_ok

def generate_scenario(key: jnp.ndarray, min_goal_dist: float, scenario_idx: int = -1):
    k_scen, k_branch = jax.random.split(key)
    idx = jnp.where(scenario_idx < 0, jax.random.randint(k_scen, (), 0, 7), jnp.int32(scenario_idx))

    def pack_human(px, py, th, g1x, g1y, g2x, g2y):
        return jnp.stack([px, py, jnp.zeros_like(px), jnp.zeros_like(px), th, jnp.zeros_like(px), 
                          g1x, g1y, g2x, g2y, jnp.zeros_like(px)], axis=-1)

    # --- 0: RANDOM STATIC ---
    def _random_scen(k):
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(k, 9)
        max_v = jax.random.uniform(k1, minval=0.2, maxval=2.0)
        margin = ROBOT_RADIUS + 0.5

        def init_circle(ck):
            c1, c2, c3 = jax.random.split(ck, 3)
            return jnp.array([jax.random.uniform(c1, minval=1.5, maxval=ROOM_W-1.5),
                              jax.random.uniform(c2, minval=1.5, maxval=ROOM_H-1.5),
                              jax.random.uniform(c3, minval=0.15, maxval=0.45)])
        obs_circles = jax.vmap(init_circle)(jax.random.split(k8, NUM_OBS_CIR))

        def init_box(bk):
            b1, b2, b3, b4 = jax.random.split(bk, 4)
            return jnp.array([jax.random.uniform(b1, minval=1.5, maxval=ROOM_W-1.5),
                              jax.random.uniform(b2, minval=1.5, maxval=ROOM_H-1.5),
                              jax.random.uniform(b3, minval=0.2, maxval=0.7),
                              jax.random.uniform(b4, minval=0.2, maxval=0.7)])
        obs_boxes = jax.vmap(init_box)(jax.random.split(k9, NUM_OBS_BOX))

        rx = jax.random.uniform(k2, minval=margin, maxval=ROOM_W-margin)
        ry = jax.random.uniform(k3, minval=margin, maxval=ROOM_H-margin)
        rtheta = jax.random.uniform(k4, minval=-jnp.pi, maxval=jnp.pi)

        GOAL_CLEARANCE = GOAL_RADIUS + 0.3
        def _goal_cond(carry):
            gx, gy, k, i = carry
            too_close = jnp.sqrt((gx - rx)**2 + (gy - ry)**2) < min_goal_dist
            return (too_close | ~_is_safe(gx, gy, GOAL_CLEARANCE, obs_circles, obs_boxes)) & (i < MAX_RESAMPLE_ITERS)
        def _goal_body(carry):
            _, _, k, i = carry
            k, ka, kb = jax.random.split(k, 3)
            return jax.random.uniform(ka, minval=margin, maxval=ROOM_W-margin), jax.random.uniform(kb, minval=margin, maxval=ROOM_H-margin), k, i+1
        k5a, k5b = jax.random.split(k5)
        gx, gy, _, _ = jax.lax.while_loop(_goal_cond, _goal_body, (jax.random.uniform(k5a, minval=margin, maxval=ROOM_W-margin), jax.random.uniform(k5b, minval=margin, maxval=ROOM_H-margin), k6, 0))

        PERSON_CLEARANCE = PEOPLE_RADIUS + 0.15
        PERSON_ROBOT_CLEAR = ROBOT_RADIUS + PEOPLE_RADIUS + 0.3
        PERSON_GOAL_CLEAR = GOAL_RADIUS + PEOPLE_RADIUS + 0.1

        def init_person(pkey):
            pk1, pk2, pk3, pk4, pk5 = jax.random.split(pkey, 5)
            def _p_cond(carry):
                px, py, k, i = carry
                return (~_is_safe(px, py, PERSON_CLEARANCE, obs_circles, obs_boxes) | (jnp.sqrt((px-rx)**2+(py-ry)**2)<PERSON_ROBOT_CLEAR) | (jnp.sqrt((px-gx)**2+(py-gy)**2)<PERSON_GOAL_CLEAR)) & (i < MAX_RESAMPLE_ITERS)
            def _p_body(carry):
                _, _, k, i = carry
                k, ka, kb = jax.random.split(k, 3)
                return jax.random.uniform(ka, minval=1.0, maxval=ROOM_W-1.0), jax.random.uniform(kb, minval=1.0, maxval=ROOM_H-1.0), k, i+1
            px, py, _, _ = jax.lax.while_loop(_p_cond, _p_body, (jax.random.uniform(pk1, minval=1.0, maxval=ROOM_W-1.0), jax.random.uniform(pk2, minval=1.0, maxval=ROOM_H-1.0), pk5, 0))
            g1x, g1y = jax.random.uniform(pk3, minval=1.0, maxval=ROOM_W-1.0), jax.random.uniform(pk4, minval=1.0, maxval=ROOM_H-1.0)
            return jnp.array([px, py, 0.0, 0.0, 0.0, 0.0, g1x, g1y, g1x, g1y, 0.0])
        people = jax.vmap(init_person)(jax.random.split(k7, NUM_PEOPLE))
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 1: PARALLEL TRAFFIC (CORRIDOR) ---
    # --- 1: PARALLEL TRAFFIC (CORRIDOR) ---
    def _parallel_scen(k):
        N_PRL = 5
        k1, k2, k3, k4, k5, k_gx, k_rx = jax.random.split(k, 7)
        
        # Corridor geometry (Room is 12x12. 4m wide corridor in the center)
        corridor_width = 4.0
        wall_width = (ROOM_W - corridor_width) / 2.0
        
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))
        
        # Left wall
        obs_boxes = obs_boxes.at[0].set([wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0])
        # Right wall
        obs_boxes = obs_boxes.at[1].set([ROOM_W - wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0])

        rx, ry, rtheta = jax.random.uniform(k_rx, minval=4.8, maxval=7.2), 0.4, jnp.pi / 2.0
        
        # Robot Goal
        gx = jax.random.uniform(k_gx, minval=4.3, maxval=7.8)
        gy = ROOM_H - 4.0
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        
        # 2 humans spawn directly adjacent to the walls (4.5 and 7.5)
        px_walls = jnp.array([4.5, 7.5])
        
        # Loop condition: Keep resampling if any two humans are closer than 1.0m
        def _cond(carry):
            px_rand, py, key, i = carry
            px_all = jnp.concatenate([px_walls, px_rand])
            dx = px_all[:, None] - px_all[None, :]
            dy = py[:, None] - py[None, :]
            # Add 100 to the diagonal so we don't count self-distance as a collision
            dist = jnp.sqrt(dx**2 + dy**2) + jnp.eye(N_PRL) * 100.0
            return (jnp.min(dist) < 1.0) & (i < MAX_RESAMPLE_ITERS)

        # Loop body: Resample internal X positions and ALL Y positions
        # Loop body: Resample internal X positions and ALL Y positions
        def _body(carry):
            px_rand, py, key, i = carry
            key, k_x, k_y = jax.random.split(key, 3)
            new_px_rand = jax.random.uniform(k_x, (N_PRL - 2,), minval=4.8, maxval=7.2)
            
            # --- MODIFIED: Spawn at least 2m ahead of the robot (ry + 2.0) ---
            new_py = jax.random.uniform(k_y, (N_PRL,), minval=ry + 3.0, maxval=ROOM_H - 0.2)
            
            return new_px_rand, new_py, key, i + 1

        # Initial random guess
        px_rand_init = jax.random.uniform(k2, (N_PRL - 2,), minval=4.8, maxval=7.2)
        
        # --- MODIFIED: Initial guess also constrained to ry + 2.0 ---
        py_init = jax.random.uniform(k3, (N_PRL,), minval=ry + 2.0, maxval=ROOM_H - 1.0)

        # Execute the vectorized rejection sampling
        px_random, py, _, _ = jax.lax.while_loop(_cond, _body, (px_rand_init, py_init, k4, 0))
        
        px = jnp.concatenate([px_walls, px_random])
        
        # Human Goal targets: wall walkers go straight down, inner flow crosses lanes randomly
        g1x_walls = px_walls  
        g1x_random = jax.random.uniform(k5, (N_PRL - 2,), minval=4.8, maxval=7.2) 
        g1x = jnp.concatenate([g1x_walls, g1x_random])
        
        # Fixed goal height to trigger teleportation in jax_env_multi.py
        g1y = jnp.full((N_PRL,), 1.0)
        
        # They walk DOWN (-pi/2)
        people_prl = pack_human(px, py, jnp.full((N_PRL,), -jnp.pi/2), g1x, g1y, g1x, g1y)
        
        # Dummy padding
        n_pad = NUM_PEOPLE - N_PRL
        dummy_x  = jnp.full((n_pad,), -999.0)
        dummy_y  = jnp.full((n_pad,), -999.0)
        dummy_rows = jnp.stack([
            dummy_x, dummy_y,
            jnp.zeros(n_pad), jnp.zeros(n_pad),   # vx, vy
            jnp.zeros(n_pad), jnp.zeros(n_pad),   # theta, omega
            dummy_x, dummy_y,                     # g1 == position
            dummy_x, dummy_y,                     # g2 == position
            jnp.full((n_pad,), -1.0),             # goal_idx = -1 sentinel
        ], axis=-1)
        
        people = jnp.concatenate([people_prl, dummy_rows], axis=0)
        
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 2: PERPENDICULAR CROSSING ---
    # People walk left<->right, bouncing between the two opposite walls.
    # g1 = target wall ahead, g2 = the wall they came from (for the return leg).
    # The existing goal_idx toggle (0→1→0→...) in jax_env_multi handles the
    # back-and-forth automatically once g1 ≠ g2.
    def _perpendicular_scen(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        rx, ry, rtheta = jax.random.uniform(k1, minval=1.5, maxval=ROOM_W-1.5), 1.0, jnp.pi / 2.0
        gx, gy = jax.random.uniform(k2, minval=1.5, maxval=ROOM_W-1.5), ROOM_H - 1.0
        max_v = jax.random.uniform(k3, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))

        # Spread people at random y positions across the room
        py = jax.random.uniform(k4, (NUM_PEOPLE,), minval=1.5, maxval=ROOM_H - 1.5)

        # Alternate: even-indexed start on the left, odd-indexed on the right
        left_mask = jnp.arange(NUM_PEOPLE) % 2 == 0

        # Spawn position: near the wall they start from
        px = jnp.where(left_mask, 0.6, ROOM_W - 0.6)

        # g1 = far wall (first target), g2 = near wall (return target)
        # Left starters → g1 is right wall, g2 is left wall
        # Right starters → g1 is left wall, g2 is right wall
        wall_near = jnp.where(left_mask, 0.6,          ROOM_W - 0.6)
        wall_far  = jnp.where(left_mask, ROOM_W - 0.6, 0.6)

        g1x = wall_far;  g1y = py   # head toward far wall first
        g2x = wall_near; g2y = py   # then bounce back to near wall

        # Initial heading: left starters face right (0), right starters face left (π)
        angles = jnp.where(left_mask, 0.0, jnp.pi)

        people = pack_human(px, py, angles, g1x, g1y, g2x, g2y)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 3: CIRCULAR CROSSING ---
    def _circular_scen(k):
        k1, k2 = jax.random.split(k, 2)
        rx, ry, rtheta = ROOM_W / 2.0, 1.5, jnp.pi / 2.0
        gx, gy = ROOM_W / 2.0, ROOM_H - 1.5
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))
        
        cx, cy, radius = ROOM_W / 2.0, ROOM_H / 2.0, jnp.minimum(ROOM_W, ROOM_H) / 2.0 - 1.5
        spawn_angles = jnp.linspace(0, 2*jnp.pi, NUM_PEOPLE, endpoint=False) + jax.random.uniform(k2, (NUM_PEOPLE,), minval=-0.2, maxval=0.2)
        px = cx + radius * jnp.cos(spawn_angles)
        py = cy + radius * jnp.sin(spawn_angles)
        
        # Goal 1: The diametrically opposite side of the circle
        g1x = cx + radius * jnp.cos(spawn_angles + jnp.pi)
        g1y = cy + radius * jnp.sin(spawn_angles + jnp.pi)
        
        # Goal 2: Their own spawn position so they repeat the path indefinitely
        g2x, g2y = px, py
        
        people = pack_human(px, py, spawn_angles + jnp.pi, g1x, g1y, g2x, g2y)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 4: BOTTLENECK ---
    # --- 4: BOTTLENECK ---
    def _bottleneck_scen(k):
        N_BTL = 5
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k, 7)
        gap_center_x = jax.random.uniform(k1, minval=2.5, maxval=ROOM_W-2.5)
        
        # Sample an initial random X coordinate
        rx_raw = jax.random.uniform(k7, minval=1.0, maxval=ROOM_W-1.0)
        
        # Evaluate distance to the gap center
        dist_to_gap = rx_raw - gap_center_x
        in_zone = jnp.abs(dist_to_gap) < 1.5
        
        # If inside the exclusion zone, push it strictly outside to the nearest side
        push_dir = jnp.where(dist_to_gap >= 0.0, 1.0, -1.0)
        rx = jnp.where(in_zone, gap_center_x + push_dir * 1.5, rx_raw)
        rx = jnp.clip(rx, 1.0, ROOM_W - 1.0)
        
        ry, rtheta = 1.5, jnp.pi / 2.0
        gx, gy = gap_center_x, ROOM_H - 1.5
        max_v = jax.random.uniform(k2, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        
        gap_size, wall_y = 2.8, ROOM_H / 2.0
        left_w, right_w = gap_center_x - gap_size / 2.0, ROOM_W - (gap_center_x + gap_size / 2.0)
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4)).at[0].set([left_w / 2.0, wall_y, left_w / 2.0, 0.2]).at[1].set([ROOM_W - right_w / 2.0, wall_y, right_w / 2.0, 0.2])
        
        px = jax.random.uniform(k3, (N_BTL,), minval=1.0, maxval=ROOM_W-1.0)
        py = jax.random.uniform(k4, (N_BTL,), minval=ROOM_H-2.5, maxval=ROOM_H-1.5)
        
        g1x = gap_center_x + jax.random.uniform(k5, (N_BTL,), minval=-0.5, maxval=0.5)
        g1y = jnp.full((N_BTL,), wall_y - 0.15)
        g2x = jax.random.uniform(k6, (N_BTL,), minval=1.0, maxval=ROOM_W-1.0)
        g2y = jnp.full((N_BTL,), 0.0)
        
        people_btl = pack_human(px, py, jnp.full((N_BTL,), -1.57), g1x, g1y, g2x, g2y)
        
        # Dummy rows use goal_idx = -1 as a sentinel
        n_pad = NUM_PEOPLE - N_BTL
        dummy_x  = jnp.full((n_pad,), -999.0)
        dummy_y  = jnp.full((n_pad,), -999.0)
        dummy_rows = jnp.stack([
            dummy_x, dummy_y,
            jnp.zeros(n_pad), jnp.zeros(n_pad),   # vx, vy
            jnp.zeros(n_pad), jnp.zeros(n_pad),   # theta, omega
            dummy_x, dummy_y,                      # g1 == position
            dummy_x, dummy_y,                      # g2 == position
            jnp.full((n_pad,), -1.0),              # goal_idx = -1 sentinel
        ], axis=-1)
        people = jnp.concatenate([people_btl, dummy_rows], axis=0)
        
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 5: INTERSECTION ---
    def _intersection_scen(k):
        k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 4))
        
        cw, ch, gap = ROOM_W / 2.0, ROOM_H / 2.0, 4.0
        hw, hh = (ROOM_W - gap) / 4.0, (ROOM_H - gap) / 4.0  
        obs_boxes = obs_boxes.at[0].set([hw, hh, hw, hh]).at[1].set([ROOM_W - hw, hh, hw, hh])
        obs_boxes = obs_boxes.at[2].set([hw, ROOM_H - hh, hw, hh]).at[3].set([ROOM_W - hw, ROOM_H - hh, hw, hh])
        
        ends_x  = jnp.array([cw, cw, 1.5, ROOM_W - 1.5])
        ends_y  = jnp.array([1.5, ROOM_H - 1.5, ch, ch])
        ends_th = jnp.array([jnp.pi/2, -jnp.pi/2, 0.0, jnp.pi])
        
        start_idx = jax.random.randint(k2, (), 0, 4)
        goal_idx  = (start_idx + jax.random.randint(k3, (), 1, 4)) % 4
        rx, ry, rtheta = ends_x[start_idx], ends_y[start_idx], ends_th[start_idx]
        gx, gy = ends_x[goal_idx],  ends_y[goal_idx]
        
        sides = jax.random.randint(k4, (NUM_PEOPLE,), 0, 2)
        vx = jax.random.uniform(k5, (NUM_PEOPLE,), minval=cw - gap/2 + 0.5, maxval=cw + gap/2 - 0.5)
        vy = jax.random.uniform(k6, (NUM_PEOPLE,), minval=1.0, maxval=ROOM_H - 1.0)
        hx = jax.random.uniform(k5, (NUM_PEOPLE,), minval=1.0, maxval=ROOM_W - 1.0)
        hy = jax.random.uniform(k6, (NUM_PEOPLE,), minval=ch - gap/2 + 0.5, maxval=ch + gap/2 - 0.5)
        
        px = jnp.where(sides == 0, vx, hx)
        py = jnp.where(sides == 0, vy, hy)
        
        g1x = jnp.where(sides == 0, px, 1.0)
        g1y = jnp.where(sides == 0, ROOM_H - 1.0, py)
        g2x = jnp.where(sides == 0, px, ROOM_W - 1.0)
        g2y = jnp.where(sides == 0, 1.0, py)
        
        angles = jnp.where(sides == 0, jnp.pi/2, 0.0)
        people = pack_human(px, py, angles, g1x, g1y, g2x, g2y)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 6: STATIC GROUPS ---
    def _static_groups_scen(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        rx, ry, rtheta = ROOM_W / 2.0, 1.0, jnp.pi / 2.0
        gx, gy = ROOM_W / 2.0, ROOM_H - 1.0
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))
        
        group_centers_x = jax.random.uniform(k2, (NUM_PEOPLE,), minval=2.0, maxval=ROOM_W-2.0)
        group_centers_y = jax.random.uniform(k3, (NUM_PEOPLE,), minval=2.0, maxval=ROOM_H-2.0)
        angles = jax.random.uniform(k4, (NUM_PEOPLE,), minval=0, maxval=2*jnp.pi)
        
        px = group_centers_x + 0.6 * jnp.cos(angles)
        py = group_centers_y + 0.6 * jnp.sin(angles)
        people = pack_human(px, py, angles, px, py, px, py)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people
    
    branches = [
        _random_scen, 
        _parallel_scen, 
        _perpendicular_scen, 
        _circular_scen, 
        _bottleneck_scen, 
        _intersection_scen, 
        _static_groups_scen
    ]

    # Compile all scenarios into the XLA switch statement
    raw_rx, raw_ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people = jax.lax.switch(idx, branches, k_branch)

    # ── Universal Robot Safety Check ──
    # Post-process: Guarantees the robot NEVER spawns directly on top of a human OR inside a wall in ANY scenario
    k_safe = jax.random.split(key)[0]
    
    def _safe_cond(carry):
        x, y, k, i = carry
        dist = jnp.sqrt((people[:, 0] - x)**2 + (people[:, 1] - y)**2)
        human_overlap = jnp.min(dist) < 1.0
        wall_overlap = ~_is_safe(x, y, ROBOT_RADIUS + 0.3, obs_circles, obs_boxes)
        return (human_overlap | wall_overlap) & (i < MAX_RESAMPLE_ITERS)
        
    def _safe_body(carry):
        x, y, k, i = carry
        k, k1, k2 = jax.random.split(k, 3)
        # Apply a micro random-walk to gently push the robot into a safe coordinate
        nx = jnp.clip(x + jax.random.uniform(k1, minval=-1.0, maxval=1.0), 1.0, ROOM_W-1.0)
        ny = jnp.clip(y + jax.random.uniform(k2, minval=-1.0, maxval=1.0), 1.0, ROOM_H-1.0)
        return nx, ny, k, i+1
    
    rx_safe, ry_safe, _, _ = jax.lax.while_loop(_safe_cond, _safe_body, (raw_rx, raw_ry, k_safe, 0))

    return rx_safe, ry_safe, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people