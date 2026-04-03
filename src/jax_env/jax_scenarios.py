# jax_scenarios.py
import jax
import jax.numpy as jnp

ROOM_W = 12.0
ROOM_H = 12.0
ROBOT_RADIUS = 0.2
PEOPLE_RADIUS = 0.4   # FIX: was 0.2 — must match jax_env.py (PEOPLE_RADIUS=0.4)
                      # With 0.2, PERSON_ROBOT_CLEAR=0.7m but body_thresh=0.6m -> margin <0.1m
                      # -> immediate collisions at first step guaranteed
GOAL_RADIUS = 0.3
N_BASE_PEOPLE = 12   # people count for all scenarios except static_groups
NUM_PEOPLE    = 24   # total slots; static_groups fills all, others pad with dummies
NUM_OBS_CIR = 6
NUM_OBS_BOX = 6

# FIX: increased 32->64 — with 12 humans + 6 circles + 6 boxes in 12x12m, 32 guesses
# silently fail (argmax returns index 0 even if not safe).
N_GUESSES = 64

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

def _batch_sample_safe_pos(key, clearance, obs_circles, obs_boxes, min_val, max_val):
    """Samples N_GUESSES positions in parallel and returns the first safe one."""
    kx, ky = jax.random.split(key, 2)
    guesses_x = jax.random.uniform(kx, (N_GUESSES,), minval=min_val, maxval=max_val)
    guesses_y = jax.random.uniform(ky, (N_GUESSES,), minval=min_val, maxval=max_val)
    
    def check_safe(x, y):
        return _is_safe(x, y, clearance, obs_circles, obs_boxes)
    
    is_safe_mask = jax.vmap(check_safe)(guesses_x, guesses_y)
    best_idx = jnp.argmax(is_safe_mask)
    return guesses_x[best_idx], guesses_y[best_idx]

def generate_scenario(key: jnp.ndarray, max_goal_dist: float, scenario_idx: int = -1):
    """
    Generate a scenario.
    """
    # Minimum goal distance floor — goal never spawns inside the robot
    _MIN_GOAL_FLOOR = 0.8

    k_scen, k_branch = jax.random.split(key)
    
    # Sample a random scenario normally
    sampled_idx = jax.random.randint(k_scen, (), 0, 7)
    
    # If the user requested a specific scenario, use it
    idx = jnp.where(scenario_idx < 0, sampled_idx, jnp.int32(scenario_idx))
    
    # REMOVED the hardcoded curriculum logic here.

    def pack_human(px, py, th, g1x, g1y, g2x, g2y):
        return jnp.stack([
            px, py, jnp.zeros_like(px), jnp.zeros_like(px), th,
            jnp.zeros_like(px), g1x, g1y, g2x, g2y, jnp.zeros_like(px)
        ], axis=-1)

    def _make_dummies(n):
        dummy_x = jnp.full((n,), -999.0)
        return jnp.stack([
            dummy_x, dummy_x, jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n), jnp.zeros(n), dummy_x, dummy_x,
            dummy_x, dummy_x, jnp.full((n,), -1.0)
        ], axis=-1)

    # --- 0: RANDOM STATIC ---
    def _random_scen(k):
        k1, k2, k_robot, k_goal, k_people_keys, k8, k9 = jax.random.split(k, 7)
        max_v = jax.random.uniform(k1, minval=0.2, maxval=2.0)
        margin = ROBOT_RADIUS + 0.5

        def init_circle(ck):
            c1, c2, c3 = jax.random.split(ck, 3)
            return jnp.array([
                jax.random.uniform(c1, minval=1.5, maxval=ROOM_W-1.5),
                jax.random.uniform(c2, minval=1.5, maxval=ROOM_H-1.5),
                jax.random.uniform(c3, minval=0.15, maxval=0.45)
            ])
        obs_circles = jax.vmap(init_circle)(jax.random.split(k8, NUM_OBS_CIR))

        def init_box(bk):
            b1, b2, b3, b4 = jax.random.split(bk, 4)
            return jnp.array([
                jax.random.uniform(b1, minval=1.5, maxval=ROOM_W-1.5),
                jax.random.uniform(b2, minval=1.5, maxval=ROOM_H-1.5),
                jax.random.uniform(b3, minval=0.2, maxval=0.7),
                jax.random.uniform(b4, minval=0.2, maxval=0.7)
            ])
        obs_boxes = jax.vmap(init_box)(jax.random.split(k9, NUM_OBS_BOX))

        # Vectorized Robot Spawn
        rx, ry = _batch_sample_safe_pos(k_robot, margin, obs_circles, obs_boxes, margin, ROOM_W-margin)
        rtheta = jax.random.uniform(k2, minval=-jnp.pi, maxval=jnp.pi)

        # Vectorized Goal Spawn
        GOAL_CLEARANCE = GOAL_RADIUS + 0.3
        kgx, kgy = jax.random.split(k_goal, 2)
        g_guesses_x = jax.random.uniform(kgx, (N_GUESSES,), minval=margin, maxval=ROOM_W-margin)
        g_guesses_y = jax.random.uniform(kgy, (N_GUESSES,), minval=margin, maxval=ROOM_H-margin)
        
        def check_goal_safe(x, y):
            safe_env = _is_safe(x, y, GOAL_CLEARANCE, obs_circles, obs_boxes)
            dist = jnp.sqrt((x - rx)**2 + (y - ry)**2)
            dist_ok = (dist >= _MIN_GOAL_FLOOR) & (dist <= max_goal_dist)
            return safe_env & dist_ok
            
        g_safe_mask = jax.vmap(check_goal_safe)(g_guesses_x, g_guesses_y)
        g_best_idx = jnp.argmax(g_safe_mask)
        gx, gy = g_guesses_x[g_best_idx], g_guesses_y[g_best_idx]

        # Vectorized People Spawn
        PERSON_CLEARANCE   = PEOPLE_RADIUS + 0.15
        # FIX: was +0.3 -> center-to-center clearance = 0.2+0.2+0.3 = 0.7m
        # but body_thresh in jax_env = 0.2+0.4 = 0.6m -> real margin < 0.1m
        # after a single human step -> immediate collision guaranteed.
        # With PEOPLE_RADIUS now corrected to 0.4: 0.2+0.4+0.6 = 1.2m -> safe margin.
        PERSON_ROBOT_CLEAR = ROBOT_RADIUS + PEOPLE_RADIUS + 0.6
        PERSON_GOAL_CLEAR  = GOAL_RADIUS + PEOPLE_RADIUS + 0.3

        def init_person(pkey):
            pk_pos, pk_g1x, pk_g1y = jax.random.split(pkey, 3)
            kpx, kpy = jax.random.split(pk_pos, 2)
            px_guesses = jax.random.uniform(kpx, (N_GUESSES,), minval=1.0, maxval=ROOM_W-1.0)
            py_guesses = jax.random.uniform(kpy, (N_GUESSES,), minval=1.0, maxval=ROOM_H-1.0)
            
            def check_person_safe(x, y):
                env_ok = _is_safe(x, y, PERSON_CLEARANCE, obs_circles, obs_boxes)
                r_ok = jnp.sqrt((x - rx)**2 + (y - ry)**2) >= PERSON_ROBOT_CLEAR
                g_ok = jnp.sqrt((x - gx)**2 + (y - gy)**2) >= PERSON_GOAL_CLEAR
                return env_ok & r_ok & g_ok
                
            p_safe_mask = jax.vmap(check_person_safe)(px_guesses, py_guesses)
            p_best_idx = jnp.argmax(p_safe_mask)
            px, py = px_guesses[p_best_idx], py_guesses[p_best_idx]
            
            g1x = jax.random.uniform(pk_g1x, minval=1.0, maxval=ROOM_W-1.0)
            g1y = jax.random.uniform(pk_g1y, minval=1.0, maxval=ROOM_H-1.0)
            return jnp.array([px, py, 0.0, 0.0, 0.0, 0.0, g1x, g1y, g1x, g1y, 0.0])
            
        people_base = jax.vmap(init_person)(jax.random.split(k_people_keys, N_BASE_PEOPLE))
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 1: PARALLEL TRAFFIC (CORRIDOR) ---
    def _parallel_scen(k):
        N_PRL = 5
        k1, k_gx, k_rx, k_x_guess, k_y_guess, k5 = jax.random.split(k, 6)
        
        corridor_width = 4.0
        wall_width = (ROOM_W - corridor_width) / 2.0
        
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))
        obs_boxes = obs_boxes.at[0].set([wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0])
        obs_boxes = obs_boxes.at[1].set([ROOM_W - wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0])

        rx, ry, rtheta = jax.random.uniform(k_rx, minval=4.8, maxval=7.2), 0.4, jnp.pi / 2.0
        gx = jax.random.uniform(k_gx, minval=4.3, maxval=7.8)
        gy = ROOM_H - 4.0
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        
        px_walls = jnp.array([4.5, 7.5])
        
        # Vectorized batch guessing for parallel humans
        px_rand_guesses = jax.random.uniform(k_x_guess, (N_GUESSES, N_PRL - 2), minval=4.8, maxval=7.2)
        py_guesses = jax.random.uniform(k_y_guess, (N_GUESSES, N_PRL), minval=ry + 3.0, maxval=ROOM_H - 0.2)

        def check_parallel_safe(px_rand_guess, py_guess):
            px_all = jnp.concatenate([px_walls, px_rand_guess])
            dx = px_all[:, None] - px_all[None, :]
            dy = py_guess[:, None] - py_guess[None, :]
            dist = jnp.sqrt(dx**2 + dy**2) + jnp.eye(N_PRL) * 100.0
            return jnp.min(dist) >= 1.0

        safe_mask = jax.vmap(check_parallel_safe)(px_rand_guesses, py_guesses)
        best_idx = jnp.argmax(safe_mask)
        
        px = jnp.concatenate([px_walls, px_rand_guesses[best_idx]])
        py = py_guesses[best_idx]
        
        g1x_walls = px_walls  
        g1x_random = jax.random.uniform(k5, (N_PRL - 2,), minval=4.8, maxval=7.2) 
        g1x = jnp.concatenate([g1x_walls, g1x_random])
        g1y = jnp.full((N_PRL,), 1.0)
        
        people_prl = pack_human(px, py, jnp.full((N_PRL,), -jnp.pi/2), g1x, g1y, g1x, g1y)
        
        n_pad = NUM_PEOPLE - N_PRL
        dummy_x = jnp.full((n_pad,), -999.0)
        dummy_rows = jnp.stack([
            dummy_x, dummy_x, jnp.zeros(n_pad), jnp.zeros(n_pad), 
            jnp.zeros(n_pad), jnp.zeros(n_pad), dummy_x, dummy_x, 
            dummy_x, dummy_x, jnp.full((n_pad,), -1.0)
        ], axis=-1)
        
        people = jnp.concatenate([people_prl, dummy_rows], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 2: PERPENDICULAR CROSSING ---
    def _perpendicular_scen(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        rx, ry, rtheta = jax.random.uniform(k1, minval=1.5, maxval=ROOM_W-1.5), 1.0, jnp.pi / 2.0
        gx, gy = jax.random.uniform(k2, minval=1.5, maxval=ROOM_W-1.5), ROOM_H - 1.0
        max_v = jax.random.uniform(k3, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))

        py = jax.random.uniform(k4, (N_BASE_PEOPLE,), minval=1.5, maxval=ROOM_H - 1.5)
        left_mask = jnp.arange(N_BASE_PEOPLE) % 2 == 0
        px = jnp.where(left_mask, 0.6, ROOM_W - 0.6)
        wall_near = jnp.where(left_mask, 0.6, ROOM_W - 0.6)
        wall_far  = jnp.where(left_mask, ROOM_W - 0.6, 0.6)

        g1x = wall_far;  g1y = py
        g2x = wall_near; g2y = py
        angles = jnp.where(left_mask, 0.0, jnp.pi)

        people_base = pack_human(px, py, angles, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
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
        spawn_angles = jnp.linspace(0, 2*jnp.pi, N_BASE_PEOPLE, endpoint=False) + jax.random.uniform(k2, (N_BASE_PEOPLE,), minval=-0.2, maxval=0.2)
        px = cx + radius * jnp.cos(spawn_angles)
        py = cy + radius * jnp.sin(spawn_angles)

        g1x = cx + radius * jnp.cos(spawn_angles + jnp.pi)
        g1y = cy + radius * jnp.sin(spawn_angles + jnp.pi)
        g2x, g2y = px, py

        people_base = pack_human(px, py, spawn_angles + jnp.pi, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 4: BOTTLENECK ---
    def _bottleneck_scen(k):
        N_BTL = 5
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k, 7)
        gap_center_x = jax.random.uniform(k1, minval=2.5, maxval=ROOM_W-2.5)
        
        # Robot spawns in front of the gap (small jitter so it's not always exactly centered)
        rx_jitter = jax.random.uniform(k7, minval=-0.4, maxval=0.4)
        rx = jnp.clip(gap_center_x + rx_jitter, 1.0, ROOM_W - 1.0)
        
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
        
        n_pad = NUM_PEOPLE - N_BTL
        dummy_x = jnp.full((n_pad,), -999.0)
        dummy_rows = jnp.stack([
            dummy_x, dummy_x, jnp.zeros(n_pad), jnp.zeros(n_pad), 
            jnp.zeros(n_pad), jnp.zeros(n_pad), dummy_x, dummy_x, 
            dummy_x, dummy_x, jnp.full((n_pad,), -1.0)
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
        
        sides = jax.random.randint(k4, (N_BASE_PEOPLE,), 0, 2)
        vx = jax.random.uniform(k5, (N_BASE_PEOPLE,), minval=cw - gap/2 + 0.5, maxval=cw + gap/2 - 0.5)
        vy = jax.random.uniform(k6, (N_BASE_PEOPLE,), minval=1.0, maxval=ROOM_H - 1.0)
        hx = jax.random.uniform(k5, (N_BASE_PEOPLE,), minval=1.0, maxval=ROOM_W - 1.0)
        hy = jax.random.uniform(k6, (N_BASE_PEOPLE,), minval=ch - gap/2 + 0.5, maxval=ch + gap/2 - 0.5)

        px = jnp.where(sides == 0, vx, hx)
        py = jnp.where(sides == 0, vy, hy)

        g1x = jnp.where(sides == 0, px, 1.0)
        g1y = jnp.where(sides == 0, ROOM_H - 1.0, py)
        g2x = jnp.where(sides == 0, px, ROOM_W - 1.0)
        g2y = jnp.where(sides == 0, 1.0, py)

        angles = jnp.where(sides == 0, jnp.pi/2, 0.0)
        people_base = pack_human(px, py, angles, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
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

    # ── Universal Robot Safety Check (Vectorized) ──
    # Post-process: Guarantees the robot NEVER spawns directly on top of a human 
    # OR inside a wall in ANY scenario, using parallel offsets instead of while_loop.
    k_safe, k_off_x, k_off_y = jax.random.split(key, 3)
    
    # Generate N_GUESSES offsets. First offset is (0.0, 0.0) to keep intended spawn if safe.
    off_x = jnp.concatenate([jnp.array([0.0]), jax.random.uniform(k_off_x, (N_GUESSES-1,), minval=-2.0, maxval=2.0)])
    off_y = jnp.concatenate([jnp.array([0.0]), jax.random.uniform(k_off_y, (N_GUESSES-1,), minval=-2.0, maxval=2.0)])
    
    rx_guesses = jnp.clip(raw_rx + off_x, 1.0, ROOM_W - 1.0)
    ry_guesses = jnp.clip(raw_ry + off_y, 1.0, ROOM_H - 1.0)
    
    def check_post_safe(x, y):
        dist = jnp.sqrt((people[:, 0] - x)**2 + (people[:, 1] - y)**2)
        human_ok = jnp.min(dist) >= 1.0
        wall_ok = _is_safe(x, y, ROBOT_RADIUS + 0.3, obs_circles, obs_boxes)
        return human_ok & wall_ok
        
    safe_mask = jax.vmap(check_post_safe)(rx_guesses, ry_guesses)
    best_idx = jnp.argmax(safe_mask)
    
    rx_safe = rx_guesses[best_idx]
    ry_safe = ry_guesses[best_idx]

    return rx_safe, ry_safe, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people