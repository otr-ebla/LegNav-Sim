# jax_scenarios.py
import math
import pathlib
import numpy as np
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
NUM_OBS_CIR = 16
NUM_OBS_BOX = 300

N_GUESSES = 64

# ── NPZ map (scenario 16) ─────────────────────────────────────────────────────
_MAP_NPZ = pathlib.Path(__file__).parent.parent / "map" / "map_obbs.npz"

def _load_map_data():
    """Load pre-fitted OBBs from map_obbs.npz. Returns a dict of numpy arrays."""
    if not _MAP_NPZ.exists():
        raise FileNotFoundError(
            f"Map OBBs not found: {_MAP_NPZ}\n"
            "Run: python -m legnav.map.preprocess_map"
        )
    return np.load(_MAP_NPZ)

_map_data = _load_map_data()
MAP_ROOM_W  = float(_map_data["room_w"])
MAP_ROOM_H  = float(_map_data["room_h"])
_MAP_BOXES  = jnp.array(_map_data["obs_boxes"], dtype=jnp.float32)   # (N_KEEP, 8)
_MAP_ROBOT_X = float(_map_data["robot_x"])
_MAP_ROBOT_Y = float(_map_data["robot_y"])
_MAP_GOAL_X  = float(_map_data["goal_x"])
_MAP_GOAL_Y  = float(_map_data["goal_y"])

# Pad to NUM_OBS_BOX columns so jax.lax.switch sees uniform shape
if _MAP_BOXES.shape[0] < NUM_OBS_BOX:
    _MAP_BOXES = jnp.concatenate(
        [_MAP_BOXES, jnp.zeros((NUM_OBS_BOX - _MAP_BOXES.shape[0], 8), dtype=jnp.float32)],
        axis=0,
    )

# ── CCW tour waypoints for floor training scenario (scenario 17) ─────────────
# WP 0 = robot spawn; WP 1-10 = CCW tour targets.  10 training legs total.
_FLOOR_CCW_WPS = jnp.array([
    [_MAP_ROBOT_X, _MAP_ROBOT_Y],   # 0: robot spawn ≈ (13.26, 26.12)
    [25.0, 26.0],                   # 1: bottom corridor mid
    [37.0, 26.0],                   # 2: bottom corridor right
    [37.5, 36.0],                   # 3: right corridor lower
    [37.5, 47.0],                   # 4: right corridor upper
    [37.5, 55.0],                   # 5: top corridor right
    [25.0, 55.0],                   # 6: top corridor mid
    [13.0, 55.0],                   # 7: top corridor left
    [13.0, 49.0],                   # 8: left corridor upper
    [13.0, 36.0],                   # 9: left corridor lower
    [13.0, 29.0],                   # 10: back near spawn
], dtype=jnp.float32)   # shape (11, 2)

def _min_dist_to_circles(x, y, circles):
    dx = circles[:, 0] - x
    dy = circles[:, 1] - y
    return jnp.min(jnp.sqrt(dx**2 + dy**2) - circles[:, 2])


def _aabb_box(cx, cy, hw, hh):
    """Axis-aligned rectangle → 8-float vertex array (CCW from bottom-left)."""
    cx = jnp.asarray(cx, jnp.float32); cy = jnp.asarray(cy, jnp.float32)
    hw = jnp.asarray(hw, jnp.float32); hh = jnp.asarray(hh, jnp.float32)
    return jnp.array([cx - hw, cy - hh,
                      cx + hw, cy - hh,
                      cx + hw, cy + hh,
                      cx - hw, cy + hh], dtype=jnp.float32)


def _rot_box(cx, cy, hw, hh, theta):
    """Rotated rectangle → 8-float vertex array (CCW)."""
    cos_t = jnp.cos(theta); sin_t = jnp.sin(theta)
    lx = jnp.array([-hw,  hw,  hw, -hw])
    ly = jnp.array([-hh, -hh,  hh,  hh])
    wx = cx + lx * cos_t - ly * sin_t
    wy = cy + lx * sin_t + ly * cos_t
    return jnp.stack([wx[0], wy[0], wx[1], wy[1],
                      wx[2], wy[2], wx[3], wy[3]]).astype(jnp.float32)


def _min_dist_to_boxes(x, y, boxes):
    """Signed-ish point-to-quad distance for (K, 8) convex-quad boxes.
    Returns 0 inside the quad, positive distance outside. Used only for
    safe-spawn checks here, so the >clearance comparison treats inside as
    unsafe (distance is 0 ≤ clearance)."""
    def _quad_dist(b):
        def nearest_on_seg(ax, ay, bx, by):
            ex, ey = bx - ax, by - ay
            t = jnp.clip(((x - ax)*ex + (y - ay)*ey) /
                         jnp.maximum(ex*ex + ey*ey, 1e-12), 0.0, 1.0)
            npx, npy = ax + t*ex, ay + t*ey
            return jnp.sqrt((x - npx)**2 + (y - npy)**2)
        d0 = nearest_on_seg(b[0], b[1], b[2], b[3])
        d1 = nearest_on_seg(b[2], b[3], b[4], b[5])
        d2 = nearest_on_seg(b[4], b[5], b[6], b[7])
        d3 = nearest_on_seg(b[6], b[7], b[0], b[1])
        edge_dist = jnp.minimum(jnp.minimum(d0, d1), jnp.minimum(d2, d3))
        # Point-in-convex-quad: boxes are CCW (see _aabb_box / _rot_box), so
        # the point is inside iff every edge's left-side cross-product is ≥ 0.
        def cross(ax, ay, bx, by):
            return (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        c0 = cross(b[0], b[1], b[2], b[3])
        c1 = cross(b[2], b[3], b[4], b[5])
        c2 = cross(b[4], b[5], b[6], b[7])
        c3 = cross(b[6], b[7], b[0], b[1])
        inside = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        dist = jnp.where(inside, jnp.float32(0.0), edge_dist)
        # Degenerate (all vertices coincident) → ignore the box.
        span = jnp.maximum(jnp.abs(b[0] - b[4]), jnp.abs(b[1] - b[5]))
        return jnp.where(span < 1e-6, 1e6, dist)
    return jnp.min(jax.vmap(_quad_dist)(boxes))

def _is_safe(x, y, clearance, obs_circles, obs_boxes, room_w=ROOM_W, room_h=ROOM_H):
    wall_ok = (x > clearance) & (x < room_w - clearance) & \
              (y > clearance) & (y < room_h - clearance)
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
    any_safe = jnp.any(is_safe_mask)
    best_safe_idx = jnp.argmax(is_safe_mask)

    def _choose_best_of_all():
        # Compute per-guess minimum clearance to obstacles and walls, then
        # pick the guess with the largest minimum clearance (least-bad).
        vcirc = jax.vmap(lambda x, y: _min_dist_to_circles(x, y, obs_circles))(guesses_x, guesses_y)
        vbox  = jax.vmap(lambda x, y: _min_dist_to_boxes(x, y, obs_boxes))(guesses_x, guesses_y)
        wall_x = jnp.minimum(guesses_x - min_val, max_val - guesses_x)
        wall_y = jnp.minimum(guesses_y - min_val, max_val - guesses_y)
        wall_clear = jnp.minimum(wall_x, wall_y)
        min_clear = jnp.minimum(jnp.minimum(vcirc, vbox), wall_clear)
        best_idx = jnp.argmax(min_clear)
        return guesses_x[best_idx], guesses_y[best_idx]

    return jax.lax.cond(any_safe,
                        lambda: (guesses_x[best_safe_idx], guesses_y[best_safe_idx]),
                        _choose_best_of_all)

def generate_scenario(key: jnp.ndarray, max_goal_dist: float,
                      scenario_idx: int = -1, max_scenario: int = 6,
                      min_goal_dist: float = 0.8):
    """
    Generate a scenario.

    scenario_idx < 0  → random draw from [0, max_scenario] (uniform).
    scenario_idx >= 0 → pin to that exact index (used for eval / test scenarios).
    max_scenario      → upper bound for random training draws; set to cur_max_scen
                        from the curriculum so the robot always sees *all* unlocked
                        scenarios, not just the hardest one (fixes Scenario Amnesia).
    """
    # Minimum goal distance floor — goal never spawns inside the robot
    _MIN_GOAL_FLOOR = 0.8

    k_scen, k_branch = jax.random.split(key)

    # Sample uniformly from the full range of currently-unlocked scenarios.
    # max_scenario is a JAX integer so randint can handle it dynamically.
    sampled_idx = jax.random.randint(k_scen, (), 0, jnp.int32(max_scenario) + 1)

    # If the caller pinned a specific scenario (>= 0), use it; else use the draw.
    idx = jnp.where(scenario_idx < 0, sampled_idx, jnp.int32(scenario_idx))

    # Redirect visual-only scenarios (13-15, max_v=0) and the static NPZ map
    # scenario (16) to scenario 0 when sampled randomly so they never appear
    # as training episodes.  Pinned evals are unaffected.
    non_train_scen = ((idx == 13) | (idx == 14) | (idx == 15) | (idx == 16)) & (jnp.int32(scenario_idx) < 0)
    idx = jnp.where(non_train_scen, jnp.int32(0), idx)

    # Scenarios where enforcing a large minimum initial goal distance is
    # impractical: intersections and multi-waypoint test scenarios.
    # Include `bottleneck` (4) alongside intersections and multi-waypoint tests
    _EXCEPTION_SCENS = jnp.array([4, 5, 7, 9, 11, 12, 13, 14, 15], dtype=jnp.int32)
    is_exception_scen = jnp.any(jnp.int32(idx) == _EXCEPTION_SCENS)

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

    # --- 0: RANDOM (obstacles + moving people) ---
    def _random_scen(k):
        k1, k2, k_robot, k_goal, k_people_keys, k_perm, k_circ, k_box = jax.random.split(k, 8)
        max_v = jax.random.uniform(k1, minval=0.2, maxval=2.0)
        margin = ROBOT_RADIUS + 0.5

        # Sparse grid: ~10 obstacles in a 12×12 m room leaves ample space for
        # 12 people and the robot.  4 circles + 6 boxes = 10 total.
        N_RAND_CIR = 4
        N_RAND_BOX = 6
        N_RAND     = N_RAND_CIR + N_RAND_BOX   # 10
        GRID_NX    = 4
        GRID_NY    = 3
        N_CELLS    = GRID_NX * GRID_NY          # 12 cells for 10 obstacles

        INNER_LO, INNER_HI = 0.8, ROOM_W - 0.8
        cell_w = (INNER_HI - INNER_LO) / GRID_NX   # ~2.6 m
        cell_h = (INNER_HI - INNER_LO) / GRID_NY   # ~3.5 m
        cell_id = jnp.arange(N_CELLS)
        cell_cx = INNER_LO + (cell_id %  GRID_NX + 0.5) * cell_w
        cell_cy = INNER_LO + (cell_id // GRID_NX + 0.5) * cell_h
        perm = jax.random.permutation(k_perm, N_CELLS)
        cell_cx = cell_cx[perm]
        cell_cy = cell_cy[perm]

        EDGE_MARGIN = PEOPLE_RADIUS + 0.15
        half_min = min(cell_w, cell_h) / 2.0
        MAX_R   = max(0.15, min(0.45, half_min - EDGE_MARGIN - 0.05))
        MAX_BHW = max(0.2,  min(0.5,  (half_min - EDGE_MARGIN - 0.05) / math.sqrt(2.0)))
        MIN_R   = min(0.15, MAX_R)
        MIN_BHW = min(0.2,  MAX_BHW)
        jit_x_c = max(0.0, cell_w / 2.0 - MAX_R   - EDGE_MARGIN)
        jit_y_c = max(0.0, cell_h / 2.0 - MAX_R   - EDGE_MARGIN)
        jit_x_b = max(0.0, cell_w / 2.0 - MAX_BHW * math.sqrt(2.0) - EDGE_MARGIN)
        jit_y_b = max(0.0, cell_h / 2.0 - MAX_BHW * math.sqrt(2.0) - EDGE_MARGIN)

        def init_circle(ck, cx0, cy0):
            c1, c2, c3 = jax.random.split(ck, 3)
            cx = cx0 + jax.random.uniform(c1, minval=-jit_x_c, maxval=jit_x_c)
            cy = cy0 + jax.random.uniform(c2, minval=-jit_y_c, maxval=jit_y_c)
            r  = jax.random.uniform(c3, minval=MIN_R, maxval=MAX_R)
            return jnp.array([cx, cy, r])
        rand_circles = jax.vmap(init_circle)(
            jax.random.split(k_circ, N_RAND_CIR),
            cell_cx[:N_RAND_CIR], cell_cy[:N_RAND_CIR],
        )
        # Pad to NUM_OBS_CIR
        obs_circles = jnp.concatenate([
            rand_circles,
            jnp.zeros((NUM_OBS_CIR - N_RAND_CIR, 3), dtype=jnp.float32),
        ], axis=0)

        def init_box(bk, cx0, cy0):
            b1, b2, b3, b4 = jax.random.split(bk, 4)
            cx = cx0 + jax.random.uniform(b1, minval=-jit_x_b, maxval=jit_x_b)
            cy = cy0 + jax.random.uniform(b2, minval=-jit_y_b, maxval=jit_y_b)
            hw = jax.random.uniform(b3, minval=MIN_BHW, maxval=MAX_BHW)
            hh = jax.random.uniform(b4, minval=MIN_BHW, maxval=MAX_BHW)
            return _aabb_box(cx, cy, hw, hh)   # axis-aligned, no rotation
        rand_boxes = jax.vmap(init_box)(
            jax.random.split(k_box, N_RAND_BOX),
            cell_cx[N_RAND_CIR:N_RAND_CIR + N_RAND_BOX],
            cell_cy[N_RAND_CIR:N_RAND_CIR + N_RAND_BOX],
        )
        # Pad to NUM_OBS_BOX
        obs_boxes = jnp.concatenate([
            rand_boxes,
            jnp.zeros((NUM_OBS_BOX - N_RAND_BOX, 8), dtype=jnp.float32),
        ], axis=0)

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
            # If this is an exception scenario, keep original loose floor.
            min_req = jnp.where(is_exception_scen, _MIN_GOAL_FLOOR, jnp.float32(min_goal_dist))
            dist_ok = (dist >= min_req) & (dist <= max_goal_dist)
            return safe_env & dist_ok

        g_safe_mask = jax.vmap(check_goal_safe)(g_guesses_x, g_guesses_y)
        # If no candidate satisfies all constraints, fall back to the one with
        # the largest obstacle clearance so the goal never lands inside a wall.
        g_env_clear = jnp.minimum(
            jax.vmap(lambda x, y: _min_dist_to_circles(x, y, obs_circles))(g_guesses_x, g_guesses_y),
            jax.vmap(lambda x, y: _min_dist_to_boxes(x, y, obs_boxes))(g_guesses_x, g_guesses_y),
        )
        g_best_idx = jnp.where(jnp.any(g_safe_mask),
                               jnp.argmax(g_safe_mask),
                               jnp.argmax(g_env_clear))
        gx, gy = g_guesses_x[g_best_idx], g_guesses_y[g_best_idx]


        PERSON_CLEARANCE   = PEOPLE_RADIUS + 0.15
        PERSON_ROBOT_CLEAR = ROBOT_RADIUS + PEOPLE_RADIUS + 0.6
        PERSON_GOAL_CLEAR  = GOAL_RADIUS + PEOPLE_RADIUS + 0.3


        if GRID_NX > 1:
            lane_x = INNER_LO + jnp.arange(1, GRID_NX) * cell_w
            n_lanes_x = GRID_NX - 1
        else:
            lane_x = jnp.array([ROOM_W / 2.0])
            n_lanes_x = 1
        if GRID_NY > 1:
            lane_y = INNER_LO + jnp.arange(1, GRID_NY) * cell_h
            n_lanes_y = GRID_NY - 1
        else:
            lane_y = jnp.array([ROOM_H / 2.0])
            n_lanes_y = 1

        def init_person(pkey):
            pk_dir, pk_lane_y, pk_lane_x, pk_x, pk_y = jax.random.split(pkey, 5)
            is_horizontal = jax.random.bernoulli(pk_dir)
            iy = jax.random.randint(pk_lane_y, (), 0, n_lanes_y)
            ix = jax.random.randint(pk_lane_x, (), 0, n_lanes_x)
            ly = lane_y[iy]
            lx = lane_x[ix]

            x_h = jax.random.uniform(pk_x, (N_GUESSES,), minval=0.5, maxval=ROOM_W - 0.5)
            y_h = jnp.full((N_GUESSES,), ly)
            x_v = jnp.full((N_GUESSES,), lx)
            y_v = jax.random.uniform(pk_y, (N_GUESSES,), minval=0.5, maxval=ROOM_H - 0.5)
            xc = jnp.where(is_horizontal, x_h, x_v)
            yc = jnp.where(is_horizontal, y_h, y_v)

            def check_person_safe(x, y):
                env_ok = _is_safe(x, y, PERSON_CLEARANCE, obs_circles, obs_boxes)
                r_ok = jnp.sqrt((x - rx)**2 + (y - ry)**2) >= PERSON_ROBOT_CLEAR
                g_ok = jnp.sqrt((x - gx)**2 + (y - gy)**2) >= PERSON_GOAL_CLEAR
                return env_ok & r_ok & g_ok

            mask = jax.vmap(check_person_safe)(xc, yc)
            bi = jnp.argmax(mask)
            px, py = xc[bi], yc[bi]

            # End-to-end waypoints: g1 at one wall, g2 at the opposite wall.
            # Person patrols back and forth along the full lane length.
            g1x = jnp.where(is_horizontal, jnp.float32(0.5),         lx)
            g1y = jnp.where(is_horizontal, ly,                       jnp.float32(0.5))
            g2x = jnp.where(is_horizontal, jnp.float32(ROOM_W - 0.5), lx)
            g2y = jnp.where(is_horizontal, ly,                       jnp.float32(ROOM_H - 0.5))
            angle = jnp.where(is_horizontal, jnp.float32(jnp.pi),
                                              jnp.float32(-jnp.pi / 2.0))
            return jnp.array([px, py, 0.0, 0.0, angle, 0.0, g1x, g1y, g2x, g2y, 0.0])

        people_base = jax.vmap(init_person)(jax.random.split(k_people_keys, N_BASE_PEOPLE))
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 1: PARALLEL TRAFFIC (CORRIDOR) ---
    def _parallel_scen(k):
        N_PRL = 5
        (k1, k_gx, k_rx, k_x_guess, k_y_guess, k5,
         k_mode, k_obs_x, k_obs_y, k_w) = jax.random.split(k, 10)

        # Corridor gap randomized per episode: 1.2 m (barely passable) → 4.0 m.
        corridor_width = jax.random.uniform(k_w, minval=1.2, maxval=4.0)
        wall_width = (ROOM_W - corridor_width) / 2.0
        cx0  = ROOM_W / 2.0            # corridor centre line
        half = corridor_width / 2.0

        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        obs_boxes = obs_boxes.at[0].set(_aabb_box(wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0))
        obs_boxes = obs_boxes.at[1].set(_aabb_box(ROOM_W - wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0))
        # Spawn spans shrink with the gap (collapse to the centre line when narrow).
        r_span = jnp.maximum(half - 0.8, 0.0)
        g_span = jnp.maximum(half - 0.7, 0.0)
        rx, ry, rtheta = cx0 + jax.random.uniform(k_rx, minval=-1.0, maxval=1.0) * r_span, 0.4, jnp.pi / 2.0
        gx = cx0 + jax.random.uniform(k_gx, minval=-1.0, maxval=1.0) * g_span
        gy = ROOM_H - 0.6
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)

        # ── Humans variant: 5 pedestrians walking down the corridor ───────
        def _humans_branch(_):
            # Wall-huggers stay 0.5 m off the walls; random walkers keep a 0.8 m
            # margin. Both collapse toward the centre line as the gap narrows,
            # so every human is always inside the corridor gap.
            wall_off = jnp.maximum(half - 0.5, 0.0)
            x_span   = jnp.maximum(half - 0.8, 0.0)
            px_walls = jnp.array([cx0 - wall_off, cx0 + wall_off])
            px_rand_guesses = cx0 + jax.random.uniform(k_x_guess, (N_GUESSES, N_PRL - 2), minval=-1.0, maxval=1.0) * x_span
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
            g1x_random = cx0 + jax.random.uniform(k5, (N_PRL - 2,), minval=-1.0, maxval=1.0) * x_span
            g1x = jnp.concatenate([g1x_walls, g1x_random])
            g1y = jnp.full((N_PRL,), 1.0)

            g2x = g1x
            g2y = jnp.full((N_PRL,), ROOM_H - 0.2)
            people_prl = pack_human(px, py, jnp.full((N_PRL,), -jnp.pi/2), g1x, g1y, g2x, g2y)
            people = jnp.concatenate([people_prl, _make_dummies(NUM_PEOPLE - N_PRL)], axis=0)
            obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
            return obs_circles, obs_boxes, people

        # ── Static-obstacle variant: 6 obstacles (3 circles + 3 boxes) ────
        def _static_branch(_):
            N_OBS  = 6                       # 3 circles + 3 rectangles, no humans
            CIR_R  = jnp.float32(0.35)
            BOX_HW = jnp.float32(0.40)
            BOX_HH = jnp.float32(0.35)

            # One obstacle per band along the corridor between robot and goal.
            # x alternates left/right of the corridor centre and type alternates
            # circle/box, giving a slalom the robot must weave through.
            y_lo = ry + 1.5
            y_hi = gy - 1.0
            band = (y_hi - y_lo) / N_OBS
            i  = jnp.arange(N_OBS, dtype=jnp.float32)
            cy = y_lo + (i + 0.5) * band + jax.random.uniform(k_obs_y, (N_OBS,),
                                                              minval=-0.3, maxval=0.3)
            # Slalom amplitude scales with the gap (±0.7 at 4 m width, centred
            # when narrow); jitter included, obstacles stay ≥1 m off the walls.
            lateral = jnp.maximum(half - 1.3, 0.0)
            x_base = jnp.where(i % 2 == 0, cx0 - lateral, cx0 + lateral)
            cx = x_base + jax.random.uniform(k_obs_x, (N_OBS,), minval=-0.3, maxval=0.3)

            # Even indices → circles, odd indices → rectangles.
            cir_cx, cir_cy = cx[0::2], cy[0::2]        # 3 circles
            box_cx, box_cy = cx[1::2], cy[1::2]        # 3 rectangles

            circle_rows = jnp.stack([cir_cx, cir_cy, jnp.full((3,), CIR_R)], axis=-1)
            obs_circles = jnp.zeros((NUM_OBS_CIR, 3)).at[:3].set(circle_rows)

            box_rows = jax.vmap(lambda bx, by: _aabb_box(bx, by, BOX_HW, BOX_HH))(box_cx, box_cy)
            obs_boxes_out = obs_boxes.at[2:5].set(box_rows)   # walls at 0,1; boxes at 2,3,4

            people = _make_dummies(NUM_PEOPLE)
            return obs_circles, obs_boxes_out, people

        # Static slalom only when the gap is wide enough to pass a centred
        # obstacle (r≤0.4 + robot diameter); narrow corridors use humans only.
        use_static = jax.random.bernoulli(k_mode, 0.5) & (corridor_width >= 2.5)
        obs_circles, obs_boxes, people = jax.lax.cond(
            use_static, _static_branch, _humans_branch, operand=None)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 2: PERPENDICULAR CROSSING ---
    def _perpendicular_scen(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        rx, ry, rtheta = jax.random.uniform(k1, minval=1.5, maxval=ROOM_W-1.5), 1.0, jnp.pi / 2.0
        gx, gy = jax.random.uniform(k2, minval=1.5, maxval=ROOM_W-1.5), ROOM_H - 1.0
        max_v = jax.random.uniform(k3, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))

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
        # People spawn on a ring around the room centre; no central obstacle.
        cx, cy, radius = ROOM_W / 2.0, ROOM_H / 2.0, jnp.minimum(ROOM_W, ROOM_H) / 2.0 - 1.5
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        
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
        obs_boxes = (jnp.zeros((NUM_OBS_BOX, 8))
                     .at[0].set(_aabb_box(left_w / 2.0, wall_y, left_w / 2.0, 0.2))
                     .at[1].set(_aabb_box(ROOM_W - right_w / 2.0, wall_y, right_w / 2.0, 0.2)))
        
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
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 8))
        
        cw, ch, gap = ROOM_W / 2.0, ROOM_H / 2.0, 4.0
        hw, hh = (ROOM_W - gap) / 4.0, (ROOM_H - gap) / 4.0
        obs_boxes = (obs_boxes
                     .at[0].set(_aabb_box(hw,            hh,            hw, hh))
                     .at[1].set(_aabb_box(ROOM_W - hw,   hh,            hw, hh))
                     .at[2].set(_aabb_box(hw,            ROOM_H - hh,   hw, hh))
                     .at[3].set(_aabb_box(ROOM_W - hw,   ROOM_H - hh,   hw, hh)))
        
        # End indices: 0=bottom, 1=top, 2=left, 3=right (pairs 0↔1, 2↔3 are opposite).
        ends_x  = jnp.array([cw, cw, 1.5, ROOM_W - 1.5])
        ends_y  = jnp.array([4.5, ROOM_H - 4.5, ch, ch])
        ends_th = jnp.array([jnp.pi/2, -jnp.pi/2, 0.0, jnp.pi])

        start_idx = jax.random.randint(k2, (), 0, 4)
        # Goal: any of the other three ends (straight, left or right turn).
        # Random offset in {1,2,3} mod 4 always lands on a different end.
        goal_idx  = (start_idx + jax.random.randint(k3, (), 1, 4)) % 4
        rx, ry, rtheta = ends_x[start_idx], ends_y[start_idx], ends_th[start_idx]
        gx, gy = ends_x[goal_idx],  ends_y[goal_idx]
        
        # People spawn at the THREE ends the robot does NOT use, then walk
        # across the intersection. Round-robin assignment over the 3 non-start
        # ends → N_BASE_PEOPLE/3 people per end. End→corridor: 0/1 (bottom/top)
        # are vertical, 2/3 (left/right) are horizontal.
        other_ends = (start_idx + jnp.array([1, 2, 3])) % 4
        pe    = other_ends[jnp.arange(N_BASE_PEOPLE) % 3]      # per-person end (0..3)
        sides = jnp.where(pe < 2, 0, 1)                        # 0 vertical, 1 horizontal

        # lane = across-corridor offset [4.5, 7.5]; depth = distance into the arm.
        lane  = jax.random.uniform(k5, (N_BASE_PEOPLE,),
                                   minval=cw - gap/2 + 0.5, maxval=cw + gap/2 - 0.5)
        depth = jax.random.uniform(k6, (N_BASE_PEOPLE,), minval=1.0, maxval=4.0)

        # Vertical ends: x = lane, y = depth (bottom) / ROOM_H-depth (top).
        # Horizontal ends: y = lane, x = depth (left) / ROOM_W-depth (right).
        px = jnp.where(sides == 0, lane,
                       jnp.where(pe == 2, depth, ROOM_W - depth))
        py = jnp.where(sides == 0,
                       jnp.where(pe == 0, depth, ROOM_H - depth), lane)

        # Waypoints: g1 = opposite arm exit (cross the intersection first since
        # the toggle index starts at 0), g2 = own arm. People ping-pong across.
        g1x = jnp.where(sides == 0, lane, jnp.where(pe == 2, ROOM_W - 1.0, 1.0))
        g1y = jnp.where(sides == 0, jnp.where(pe == 0, ROOM_H - 1.0, 1.0), lane)
        g2x = jnp.where(sides == 0, lane, jnp.where(pe == 2, 1.0, ROOM_W - 1.0))
        g2y = jnp.where(sides == 0, jnp.where(pe == 0, 1.0, ROOM_H - 1.0), lane)

        # Face the intersection at spawn.
        angles = jnp.where(pe == 0, jnp.pi/2,
                 jnp.where(pe == 1, -jnp.pi/2,
                 jnp.where(pe == 2, 0.0, jnp.pi)))

        # Keep the goal road end clear so the goal post-processing never
        # resamples it off the end. People within 1.3 m of the goal mouth are
        # nudged along their arm: deeper (away from centre) for a vertical goal
        # where the arm has room, toward the centre for a wall-side horizontal
        # goal. The robot's own end has no people by design, so it never gets
        # nudged by the universal safety check.
        def _clear_goal_end(px, py, gx, gy):
            too_close = jnp.hypot(px - gx, py - gy) < 1.3
            goal_vert = jnp.abs(gx - cw) < 0.5
            ydir = jnp.where(gy < ch, -1.0, 1.0)
            xdir = jnp.where(gx < cw, 1.0, -1.0)
            npy = jnp.where(too_close & goal_vert,    gy + ydir * 1.6, py)
            npx = jnp.where(too_close & ~goal_vert,   gx + xdir * 1.6, px)
            return (jnp.clip(npx, 1.0, ROOM_W - 1.0),
                    jnp.clip(npy, 1.0, ROOM_H - 1.0))

        px, py = _clear_goal_end(px, py, gx, gy)

        people_base = pack_human(px, py, angles, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_BASE_PEOPLE)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 6: STATIC GROUPS ---
    def _static_groups_scen(k):
        # 4 groups × 6 people = 24 = NUM_PEOPLE.
        # Group centers are placed in the 4 room quadrants + random jitter so
        # they are always well-separated regardless of seed.
        N_GROUPS      = 4
        PPL_PER_GROUP = 6
        (k1, k_jx, k_jy, k_offr, k_offa, k_ang,
         k_rob_x, k_rob_y, k_rob_a, k_goal) = jax.random.split(k, 10)

        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 8))

        # ── Group centers: one per quadrant + jitter ──────────────────────────
        # Base centers at quadrant mid-points (3,3), (9,3), (3,9), (9,9).
        # Jitter ±1.5 m keeps them inside [1.8, 10.2] after clipping.
        base_qx = jnp.array([3.0, 9.0, 3.0, 9.0])
        base_qy = jnp.array([3.0, 3.0, 9.0, 9.0])
        jx = jax.random.uniform(k_jx, (N_GROUPS,), minval=-1.5, maxval=1.5)
        jy = jax.random.uniform(k_jy, (N_GROUPS,), minval=-1.5, maxval=1.5)
        gc_x = jnp.clip(base_qx + jx, 1.8, ROOM_W - 1.8)
        gc_y = jnp.clip(base_qy + jy, 1.8, ROOM_H - 1.8)

        # ── Per-person positions: ring around group center ────────────────────
        # Computed before robot/goal so they can be used as safety constraints.
        person_cx = jnp.repeat(gc_x, PPL_PER_GROUP)
        person_cy = jnp.repeat(gc_y, PPL_PER_GROUP)

        group_rot = jax.random.uniform(k_offa, (N_GROUPS,), minval=0.0, maxval=2 * jnp.pi)
        base_angles          = jnp.tile(
            jnp.arange(PPL_PER_GROUP) * (2 * jnp.pi / PPL_PER_GROUP), N_GROUPS
        )
        group_rot_per_person = jnp.repeat(group_rot, PPL_PER_GROUP)
        off_a = base_angles + group_rot_per_person

        off_r = jax.random.uniform(k_offr, (NUM_PEOPLE,), minval=0.8, maxval=1.1)
        px = jnp.clip(person_cx + off_r * jnp.cos(off_a), 1.0, ROOM_W - 1.0)
        py = jnp.clip(person_cy + off_r * jnp.sin(off_a), 1.0, ROOM_H - 1.0)

        # Group exclusion circles — used to keep robot AND goal out of groups.
        # Radius covers furthest person + people/goal radii + a small margin.
        _GRP_OBS_R = 1.1 + PEOPLE_RADIUS + GOAL_RADIUS + 0.3   # 2.1 m
        group_circles = jnp.concatenate([
            jnp.stack([gc_x, gc_y, jnp.full(N_GROUPS, _GRP_OBS_R)], axis=-1),
            jnp.zeros((NUM_OBS_CIR - N_GROUPS, 3)),
        ], axis=0)   # (6, 3)

        # ── Robot: safe-sampled in bottom strip, > 1 m from any person and
        # outside every group circle. By landing on an already-safe spot the
        # generate_scenario robot post-processing keeps it unchanged, which is
        # critical for the goal-vs-robot distance check below.
        rx_guesses = jax.random.uniform(k_rob_x, (N_GUESSES,), minval=1.0, maxval=ROOM_W - 1.0)
        ry_guesses = jax.random.uniform(k_rob_y, (N_GUESSES,), minval=1.0, maxval=2.5)

        def _rob_safe(x, y):
            dp = jnp.hypot(px - x, py - y)
            person_ok = jnp.min(dp) > 1.0
            circ_ok   = _min_dist_to_circles(x, y, group_circles) > ROBOT_RADIUS + 0.3
            return person_ok & circ_ok

        rob_safe_mask = jax.vmap(_rob_safe)(rx_guesses, ry_guesses)
        rob_best = jnp.argmax(rob_safe_mask)
        rx = rx_guesses[rob_best]
        ry = ry_guesses[rob_best]
        rtheta = jax.random.uniform(k_rob_a, minval=0.0, maxval=2 * jnp.pi)

        # ── Goal: outside every group AND > 1 m from the robot. The robot is
        # included as an extra exclusion circle so the generate_scenario push
        # step (triggered when dist < min_goal_dist=0.8 m) never fires and
        # therefore cannot shove the goal into a group on the other side.
        full_excl_circles = jnp.concatenate([
            jnp.stack([gc_x, gc_y, jnp.full(N_GROUPS, _GRP_OBS_R)], axis=-1),
            jnp.array([[rx, ry, 1.0]]),
            jnp.zeros((NUM_OBS_CIR - N_GROUPS - 1, 3)),
        ], axis=0)
        gx, gy = _batch_sample_safe_pos(
            k_goal, clearance=GOAL_RADIUS,
            obs_circles=full_excl_circles, obs_boxes=obs_boxes,
            min_val=1.0, max_val=ROOM_W - 1.0,
        )

        angles = jax.random.uniform(k_ang, (NUM_PEOPLE,), minval=0.0, maxval=2 * jnp.pi)

        # Static: goal == current position so nobody moves
        people = pack_human(px, py, angles, px, py, px, py)
        # group_circles were only used as virtual exclusion zones during
        # robot/goal sampling — the env itself has no static obstacles here.
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people
    
    # ── 7: S-CORRIDOR (test) ─────────────────────────────────────────────────
    def _s_corridor_scen(k):
        """S-shaped corridor. Robot: bottom-left → right bottleneck (y=4) →
           left bottleneck (y=8) → top-right. People spread across all 3 sections."""
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k, 7)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        # Wall 1: center (4,4), hw=4 → spans x=0..8; gap on RIGHT (x=8..12)
        obs_boxes = obs_boxes.at[0].set(_aabb_box(4.0, 4.0, 4.0, 0.2))
        # Wall 2: center (8,8), hw=4 → spans x=4..12; gap on LEFT (x=0..4)
        obs_boxes = obs_boxes.at[1].set(_aabb_box(8.0, 8.0, 4.0, 0.2))

        rx, ry, rtheta = 1.5, 1.5, jnp.pi / 2.0
        # First robot waypoint = right-side gap of wall 1
        gx, gy = 10.0, 4.0

        # --- 3 people in bottom section (y < 4), patrolling left ↔ right ---
        N_BOT = 3
        px_b = jax.random.uniform(k2, (N_BOT,), minval=1.0, maxval=7.0)
        py_b = jax.random.uniform(k3, (N_BOT,), minval=1.0, maxval=3.5)
        ang_b = jax.random.uniform(k4, (N_BOT,), minval=-jnp.pi, maxval=jnp.pi)
        ppl_bot = pack_human(px_b, py_b, ang_b,
                             jnp.full((N_BOT,), 1.0), py_b,
                             jnp.full((N_BOT,), 7.0), py_b)

        # --- 3 people in middle section (4 < y < 8), patrolling left ↔ right ---
        N_MID = 3
        px_m = jax.random.uniform(k5, (N_MID,), minval=5.0, maxval=11.0)
        py_m = jax.random.uniform(k6, (N_MID,), minval=4.5, maxval=7.5)
        ang_m = jax.random.uniform(k7, (N_MID,), minval=-jnp.pi, maxval=jnp.pi)
        ppl_mid = pack_human(px_m, py_m, ang_m,
                             jnp.full((N_MID,), 5.0), py_m,
                             jnp.full((N_MID,), 11.0), py_m)

        # --- 2 people in top section (y > 8), patrolling left ↔ right ---
        N_TOP = 2
        k8, k9, k10 = jax.random.split(k2, 3)   # reuse k2 branch — distinct enough
        px_t = jax.random.uniform(k8, (N_TOP,), minval=1.0, maxval=11.0)
        py_t = jax.random.uniform(k9, (N_TOP,), minval=8.5, maxval=11.0)
        ang_t = jax.random.uniform(k10, (N_TOP,), minval=-jnp.pi, maxval=jnp.pi)
        ppl_top = pack_human(px_t, py_t, ang_t,
                             jnp.full((N_TOP,), 1.0), py_t,
                             jnp.full((N_TOP,), 11.0), py_t)

        N_PPL = N_BOT + N_MID + N_TOP
        people = jnp.concatenate([ppl_bot, ppl_mid, ppl_top,
                                   _make_dummies(NUM_PEOPLE - N_PPL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 8: CONVERGING CROWDS (test) ──────────────────────────────────────────
    def _converging_crowds_scen(k):
        """4 groups of 3 from each corner, patrolling between corner and center."""
        k1, k2 = jax.random.split(k, 2)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        rx, ry, rtheta = ROOM_W / 2.0, 1.0, jnp.pi / 2.0
        gx, gy = ROOM_W / 2.0, ROOM_H - 1.0

        N_PPL = 12
        cx, cy = ROOM_W / 2.0, ROOM_H / 2.0
        corners_x = jnp.array([2.0, ROOM_W - 2.0, 2.0, ROOM_W - 2.0])
        corners_y = jnp.array([2.0, 2.0, ROOM_H - 2.0, ROOM_H - 2.0])
        offsets = jax.random.uniform(k2, (N_PPL, 2), minval=-0.8, maxval=0.8)
        group_idx = jnp.arange(N_PPL) // 3
        px = jnp.clip(corners_x[group_idx] + offsets[:, 0], 1.0, ROOM_W - 1.0)
        py = jnp.clip(corners_y[group_idx] + offsets[:, 1], 1.0, ROOM_H - 1.0)
        angles = jnp.arctan2(cy - py, cx - px)
        # Patrol: corner <-> center
        people_base = pack_human(px, py, angles,
                                 jnp.full((N_PPL,), cx), jnp.full((N_PPL,), cy),
                                 px, py)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_PPL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 9: SEQUENTIAL ROOMS — robot multi-waypoint (test) ────────────────────
    def _sequential_rooms_scen(k):
        """3 rooms with doorways. Robot's first goal is in room 2 (eval script advances)."""
        k1, k2, k3, k4, k5, k_rx, k_ry_jitter = jax.random.split(k, 7)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        # Two vertical walls at x=4 and x=8 with doorway gaps
        door1_y = jax.random.uniform(k2, minval=3.0, maxval=ROOM_H - 3.0)
        door2_y = jax.random.uniform(k3, minval=3.0, maxval=ROOM_H - 3.0)
        gap = 2.0
        w1_bot_hh = (door1_y - gap / 2.0) / 2.0
        w1_top_hh = (ROOM_H - door1_y - gap / 2.0) / 2.0
        obs_boxes = obs_boxes.at[0].set(_aabb_box(4.0, w1_bot_hh, 0.15, w1_bot_hh))
        obs_boxes = obs_boxes.at[1].set(_aabb_box(4.0, door1_y + gap / 2.0 + w1_top_hh, 0.15, w1_top_hh))
        w2_bot_hh = (door2_y - gap / 2.0) / 2.0
        w2_top_hh = (ROOM_H - door2_y - gap / 2.0) / 2.0
        obs_boxes = obs_boxes.at[2].set(_aabb_box(8.0, w2_bot_hh, 0.15, w2_bot_hh))
        obs_boxes = obs_boxes.at[3].set(_aabb_box(8.0, door2_y + gap / 2.0 + w2_top_hh, 0.15, w2_top_hh))

        # Robot spawns randomly in room 1 but approximately in front of door1_y.
        # x random in [0.8, 3.2], y = door1_y + small jitter (±1.2), facing right.
        rx     = jax.random.uniform(k_rx, minval=0.8, maxval=3.2)
        ry_off = jax.random.uniform(k_ry_jitter, minval=-1.2, maxval=1.2)
        ry     = jnp.clip(door1_y + ry_off, 1.0, ROOM_H - 1.0)
        rtheta = 0.0   # facing right toward door
        
        # Goal iniziale: Stanza 2, subito dopo la porta 1 (x=5.0, y=door1_y).
        gx, gy = 5.0, door1_y

        N_PPL = 6
        px = jnp.array([1.5, 2.5, 5.0, 7.0, 9.5, 10.5])
        py = jax.random.uniform(k4, (N_PPL,), minval=2.0, maxval=ROOM_H - 2.0)
        angles = jax.random.uniform(k5, (N_PPL,), minval=-jnp.pi, maxval=jnp.pi)
        people_base = pack_human(px, py, angles,
                                 px, jnp.full((N_PPL,), 2.0),
                                 px, jnp.full((N_PPL,), ROOM_H - 2.0))
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_PPL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 10: ZIGZAG COUNTER-FLOW (test) ───────────────────────────────────────
    def _zigzag_counterflow_scen(k):
        """Two groups walking in opposite directions with lateral shifts."""
        k1, k2, k3, k4 = jax.random.split(k, 4)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        rx, ry, rtheta = ROOM_W / 2.0, 1.0, jnp.pi / 2.0
        gx, gy = ROOM_W / 2.0, ROOM_H - 1.0

        N_PPL = 12
        is_group_a = jnp.arange(N_PPL) % 2 == 0
        px_a = jax.random.uniform(k2, (N_PPL,), minval=2.0, maxval=5.0)
        px_b = jax.random.uniform(k3, (N_PPL,), minval=7.0, maxval=10.0)
        px = jnp.where(is_group_a, px_a, px_b)
        py_a = jax.random.uniform(k4, (N_PPL,), minval=1.0, maxval=4.0)
        py_b = jax.random.uniform(k4, (N_PPL,), minval=8.0, maxval=11.0)
        py = jnp.where(is_group_a, py_a, py_b)
        angles = jnp.where(is_group_a, jnp.pi / 2.0, -jnp.pi / 2.0)
        # Shifted goals for zigzag effect
        g1x = jnp.where(is_group_a, px + 2.0, px - 2.0)
        g1y = jnp.where(is_group_a, ROOM_H - 1.0, 1.0)
        people_base = pack_human(px, py, angles, g1x, g1y, px, py)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_PPL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 11: LONG PARALLEL CORRIDOR (test) ────────────────────────────────────
    def _parallel_long_scen(k):
        """Double-length parallel-traffic corridor.
        The room height for this scenario is 24 m. Robot starts at the bottom 
        and must reach halfway (12.0) then 4/5 of the way (19.2) through a flow of
        24 pedestrians teleporting to the top when reaching the bottom."""
        H = 24.0   # physical room height for this scenario
        k1, k_rx, k_px, k_py = jax.random.split(k, 4)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)

        corridor_width = 4.0
        wall_width = (ROOM_W - corridor_width) / 2.0   # 4.0

        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        # Left wall: fills [0, wall_width] × [0, H]
        obs_boxes = obs_boxes.at[0].set(_aabb_box(wall_width / 2.0, H / 2.0,
                                                  wall_width / 2.0, H / 2.0))
        # Right wall: fills [ROOM_W-wall_width, ROOM_W] × [0, H]
        obs_boxes = obs_boxes.at[1].set(_aabb_box(ROOM_W - wall_width / 2.0, H / 2.0,
                                                  wall_width / 2.0, H / 2.0))

        # Robot: random x within corridor, starts at the very bottom, faces up
        rx = jax.random.uniform(k_rx, minval=4.8, maxval=7.2)
        ry, rtheta = 0.5, jnp.pi / 2.0
        # First goal: half height of the 24 m corridor
        gx, gy = 6.0, H / 2.0   # 12.0

        # 18 pedestrians — spawning randomly from 1/3 height to the top, tracking downwards
        N_PRL = 13
        px = jax.random.uniform(k_px, (N_PRL,), minval=4.8, maxval=7.2)
        py = jax.random.uniform(k_py, (N_PRL,), minval=H / 3.0, maxval=H - 0.2)
        
        # Bottom destination is 1.0 so that is_parallel logic in jax_env_multi.py triggers
        g1x = px;  g1y = jnp.full((N_PRL,), 1.0)
        g2x = px;  g2y = jnp.full((N_PRL,), H - 0.4)

        people_prl = pack_human(px, py, jnp.full((N_PRL,), -jnp.pi / 2.0),
                                g1x, g1y, g2x, g2y)
        n_pad = NUM_PEOPLE - N_PRL
        dummy_x = jnp.full((n_pad,), -999.0)
        dummy_rows = jnp.stack([
            dummy_x, dummy_x, jnp.zeros(n_pad), jnp.zeros(n_pad),
            jnp.zeros(n_pad), jnp.zeros(n_pad), dummy_x, dummy_x,
            dummy_x, dummy_x, jnp.full((n_pad,), -1.0)
        ], axis=-1)
        people = jnp.concatenate([people_prl, dummy_rows], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 12: U-TURN CORRIDOR — robot multi-waypoint (test) ────────────────────
    def _uturn_corridor_scen(k):
        """U-shaped corridor. 4 people per corridor.
        Left/bottom corridors: random bounce. Right corridor: continuous top→bottom
        flow (staggered start, all heading down initially — same mechanism as the
        parallel-traffic training scenario)."""
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)
        max_v = jax.random.uniform(k1, minval=0.5, maxval=1.5)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes = jnp.zeros((NUM_OBS_BOX, 8))
        corridor_w = 3.0
        block_hw = (ROOM_W - 2 * corridor_w) / 2.0
        block_hh = (ROOM_H - corridor_w) / 2.0
        obs_boxes = obs_boxes.at[0].set(_aabb_box(ROOM_W / 2.0, corridor_w + block_hh, block_hw, block_hh))

        rx, ry, rtheta = corridor_w / 2.0, ROOM_H - 1.5, -jnp.pi / 2.0
        gx, gy = corridor_w / 2.0, corridor_w / 2.0   # first waypoint: bottom-left

        N_PER  = 4                            # people per corridor
        cL     = corridor_w / 2.0            # 1.5  — left corridor centre x
        cR     = ROOM_W - corridor_w / 2.0   # 10.5 — right corridor centre x
        cB     = corridor_w / 2.0            # 1.5  — bottom corridor centre y
        y_top  = ROOM_H - 1.5               # 10.5
        y_bot  = 1.5
        x_left  = 1.5
        x_right = ROOM_W - 1.5              # 10.5

        # ── Left corridor: 4 people, random y positions, bounce up↔down ──────
        left_px  = jnp.full((N_PER,), cL)
        left_py  = jax.random.uniform(k2, (N_PER,), minval=y_bot + 1.0, maxval=y_top)
        left_g1x = jnp.full((N_PER,), cL);  left_g1y = jnp.full((N_PER,), y_bot)
        left_g2x = jnp.full((N_PER,), cL);  left_g2y = jnp.full((N_PER,), y_top)
        left_ang = jax.random.uniform(k3, (N_PER,), minval=-jnp.pi, maxval=jnp.pi)

        # ── Bottom corridor: 4 people, spread along x, bounce left↔right ─────
        bot_px  = jax.random.uniform(k4, (N_PER,), minval=x_left + 0.5, maxval=x_right - 0.5)
        bot_py  = jnp.full((N_PER,), cB)
        bot_g1x = jnp.full((N_PER,), x_left);  bot_g1y = jnp.full((N_PER,), cB)
        bot_g2x = jnp.full((N_PER,), x_right); bot_g2y = jnp.full((N_PER,), cB)
        bot_ang = jax.random.uniform(k5, (N_PER,), minval=-jnp.pi, maxval=jnp.pi)

        # ── Right corridor: continuous top→bottom flow ────────────────────────
        # 4 people evenly staggered from top to bottom (all waypoint_idx=0 → heading
        # to g1=bottom). When each reaches the bottom it bounces back to g2=top,
        # restarting the cycle. Staggering ensures someone is always going down.
        right_px  = jnp.full((N_PER,), cR)
        right_py  = jnp.linspace(y_top, y_bot + 2.0, N_PER)   # [10.5, 7.5, 5.0, 3.5]
        right_g1x = jnp.full((N_PER,), cR);  right_g1y = jnp.full((N_PER,), y_bot)
        right_g2x = jnp.full((N_PER,), cR);  right_g2y = jnp.full((N_PER,), y_top)
        right_ang = jnp.full((N_PER,), -jnp.pi / 2.0)          # all pointing DOWN

        # Concatenate all three corridors (12 people total)
        N_PPL = 3 * N_PER
        px  = jnp.concatenate([left_px,  bot_px,  right_px])
        py  = jnp.concatenate([left_py,  bot_py,  right_py])
        g1x = jnp.concatenate([left_g1x, bot_g1x, right_g1x])
        g1y = jnp.concatenate([left_g1y, bot_g1y, right_g1y])
        g2x = jnp.concatenate([left_g2x, bot_g2x, right_g2x])
        g2y = jnp.concatenate([left_g2y, bot_g2y, right_g2y])
        ang = jnp.concatenate([left_ang,  bot_ang,  right_ang])

        people_base = pack_human(px, py, ang, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - N_PPL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 13: GAIT TREADMILL — one person walking top→bottom, teleports back ──
    # Visual-only scenario for inspecting the jax_legs gait model. The robot
    # is parked in the bottom-right corner (max_v = 0 so it cannot move).
    def _gait_treadmill_scen(k):
        # Robot: bottom-right corner, max_v = 0 so policy actions don't move it.
        rx = jnp.float32(ROOM_W - 0.6)
        ry = jnp.float32(0.6)
        rtheta = jnp.float32(3.0 * jnp.pi / 4.0)   # facing up-left, into the room
        gx = jnp.float32(ROOM_W * 0.5)
        gy = jnp.float32(ROOM_H * 0.5)
        max_v = jnp.float32(0.0)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 8))

        # Single person spawned a little below the top of the room (so the
        # initial stationary pose is fully visible inside the viewport), lane
        # in the centre-left so they stay clear of the parked robot. g1x == g2x
        # and g1y ≈ 1.0 activates the vertical-teleport mechanism in
        # jax_env_multi.py (same trick used by the long parallel corridor):
        # when y < 1.5, respawn near the top.
        person_x = jnp.float32(ROOM_W * 0.4)
        person_y = jnp.float32(ROOM_H - 1.5)
        g1x = person_x;  g1y = jnp.float32(1.0)
        g2x = person_x;  g2y = jnp.float32(ROOM_H - 0.4)
        ang = jnp.float32(-jnp.pi / 2.0)   # heading straight down
        person = pack_human(person_x[None], person_y[None], ang[None],
                            g1x[None], g1y[None], g2x[None], g2y[None])
        people = jnp.concatenate([person, _make_dummies(NUM_PEOPLE - 1)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 14: GAIT CROSSING — two people walking against each other ───────────
    # They patrol left↔right via the standard waypoint-toggle mechanism so the
    # motion is endless. Used to inspect the gait model from both directions.
    def _gait_crossing_scen(k):
        rx = jnp.float32(ROOM_W - 0.6)
        ry = jnp.float32(0.6)
        rtheta = jnp.float32(3.0 * jnp.pi / 4.0)
        gx = jnp.float32(ROOM_W * 0.5)
        gy = jnp.float32(ROOM_H * 0.5)
        max_v = jnp.float32(0.0)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 8))

        # Both people share the same lane y so they walk head-on. HSFM will
        # resolve the near-miss naturally.
        y_lane = jnp.float32(ROOM_H * 0.5)
        # Person 0: starts on the left, heads right.
        # Person 1: mirror image — starts on the right, heads left.
        # Waypoints flip ends so they bounce back-and-forth endlessly.
        px  = jnp.array([0.6,           ROOM_W - 0.6], dtype=jnp.float32)
        py  = jnp.array([y_lane,        y_lane],       dtype=jnp.float32)
        ang = jnp.array([0.0,           jnp.pi],       dtype=jnp.float32)
        g1x = jnp.array([ROOM_W - 0.6,  0.6],          dtype=jnp.float32)
        g1y = py
        g2x = jnp.array([0.6,           ROOM_W - 0.6], dtype=jnp.float32)
        g2y = py
        people_base = pack_human(px, py, ang, g1x, g1y, g2x, g2y)
        people = jnp.concatenate([people_base, _make_dummies(NUM_PEOPLE - 2)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 15: GAIT CIRCLE — one person walking on a constant-radius circle ────
    # Visual-only scenario. HSFM motion is overridden by view_leg_scenarios.py
    # which integrates unicycle kinematics (constant v, omega) every frame.
    # The spawn position here only seeds the integrator; waypoints sit on the
    # spawn so HSFM would have no goal force even if it ran un-overridden.
    def _gait_circle_scen(k):
        rx = jnp.float32(ROOM_W - 0.6)
        ry = jnp.float32(0.6)
        rtheta = jnp.float32(3.0 * jnp.pi / 4.0)
        gx = jnp.float32(ROOM_W * 0.5)
        gy = jnp.float32(ROOM_H * 0.5)
        max_v = jnp.float32(0.0)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3))
        obs_boxes   = jnp.zeros((NUM_OBS_BOX, 8))

        # Spawn on the +x side of the room centre, heading +y → CCW circle.
        # The viewer recomputes the actual radius from CIRCLE_V / CIRCLE_OMEGA;
        # this seed just needs to be a valid starting pose somewhere reasonable.
        person_x = jnp.float32(ROOM_W * 0.5 + 2.0)
        person_y = jnp.float32(ROOM_H * 0.5)
        ang = jnp.float32(jnp.pi / 2.0)
        g1x = person_x;  g1y = person_y
        g2x = person_x;  g2y = person_y
        person = pack_human(person_x[None], person_y[None], ang[None],
                            g1x[None], g1y[None], g2x[None], g2y[None])
        people = jnp.concatenate([person, _make_dummies(NUM_PEOPLE - 1)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # ── 17: FLOOR TOUR CCW — random leg + patrol people (training) ───────────
    def _floor_tour_scen(k):
        """Random leg of the CCW floor tour on the real map, with 8 corridor patrollers.
        Robot spawns at CCW waypoint i (+ small jitter), goal at CCW waypoint i+1.
        Exercises long straight-corridor navigation with people present.
        """
        k1, k2, k3, k4, k5, k_theta, k_people = jax.random.split(k, 7)
        max_v = jnp.float32(1.2)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3), dtype=jnp.float32)
        obs_boxes   = _MAP_BOXES

        # 11 waypoints → 10 legs; pick one at random
        N_LEGS = 10
        leg_idx = jax.random.randint(k1, (), 0, N_LEGS)

        # Robot: CCW[leg_idx] + small position jitter for spawn diversity
        jitter_rx = jax.random.uniform(k2, minval=-0.5, maxval=0.5)
        jitter_ry = jax.random.uniform(k3, minval=-0.5, maxval=0.5)
        rx = jnp.clip(_FLOOR_CCW_WPS[leg_idx, 0] + jitter_rx,
                      ROBOT_RADIUS + 0.1, MAP_ROOM_W - ROBOT_RADIUS - 0.1)
        ry = jnp.clip(_FLOOR_CCW_WPS[leg_idx, 1] + jitter_ry,
                      ROBOT_RADIUS + 0.1, MAP_ROOM_H - ROBOT_RADIUS - 0.1)

        # Goal: CCW[leg_idx + 1] + small jitter
        jitter_gx = jax.random.uniform(k4, minval=-0.5, maxval=0.5)
        jitter_gy = jax.random.uniform(k5, minval=-0.5, maxval=0.5)
        gx = jnp.clip(_FLOOR_CCW_WPS[leg_idx + 1, 0] + jitter_gx,
                      GOAL_RADIUS + 0.1, MAP_ROOM_W - GOAL_RADIUS - 0.1)
        gy = jnp.clip(_FLOOR_CCW_WPS[leg_idx + 1, 1] + jitter_gy,
                      GOAL_RADIUS + 0.1, MAP_ROOM_H - GOAL_RADIUS - 0.1)

        # Heading: face the goal with ±0.8 rad noise
        rtheta = jnp.arctan2(gy - ry, gx - rx) + jax.random.uniform(k_theta, minval=-0.8, maxval=0.8)
        rtheta = (rtheta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        # 8 patrol people: 2 per corridor (bottom/right/top/left)
        # Format: [g1x, g1y, g2x, g2y, is_horizontal(1=h, 0=v)]
        corridor_data = jnp.array([
            [13.0, 26.0, 37.0, 26.0, 1.0],   # bottom, person 1
            [13.0, 26.0, 37.0, 26.0, 1.0],   # bottom, person 2
            [37.5, 26.0, 37.5, 55.0, 0.0],   # right,  person 3
            [37.5, 26.0, 37.5, 55.0, 0.0],   # right,  person 4
            [13.0, 55.0, 37.5, 55.0, 1.0],   # top,    person 5
            [13.0, 55.0, 37.5, 55.0, 1.0],   # top,    person 6
            [13.0, 26.0, 13.0, 55.0, 0.0],   # left,   person 7
            [13.0, 26.0, 13.0, 55.0, 0.0],   # left,   person 8
        ], dtype=jnp.float32)   # (8, 5)
        N_PATROL = 8

        def _init_patroller(pk, cd):
            g1x, g1y, g2x, g2y, is_h = cd
            k_t, k_wp = jax.random.split(pk)
            t    = jax.random.uniform(k_t)
            wp_i = jax.random.randint(k_wp, (), 0, 2).astype(jnp.float32)
            # Position along the corridor
            px = jnp.where(is_h > 0.5, g1x + t * (g2x - g1x), g1x)
            py = jnp.where(is_h > 0.5, g1y, g1y + t * (g2y - g1y))
            # Heading toward active waypoint
            cur_gx = jnp.where(wp_i == 0, g1x, g2x)
            cur_gy = jnp.where(wp_i == 0, g1y, g2y)
            theta  = jnp.arctan2(cur_gy - py, cur_gx - px)
            return jnp.array([px, py, 0.0, 0.0, theta, 0.0,
                               g1x, g1y, g2x, g2y, wp_i], dtype=jnp.float32)

        patrol_people = jax.vmap(_init_patroller)(
            jax.random.split(k_people, N_PATROL), corridor_data
        )
        people = jnp.concatenate([patrol_people, _make_dummies(NUM_PEOPLE - N_PATROL)], axis=0)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    # --- 16: NPZ MAP (static real-world floor-plan) ---
    def _npz_map_scen(_k):
        rx       = jnp.float32(_MAP_ROBOT_X)
        ry       = jnp.float32(_MAP_ROBOT_Y)
        rtheta   = jnp.float32(0.0)
        gx       = jnp.float32(_MAP_GOAL_X)
        gy       = jnp.float32(_MAP_GOAL_Y)
        max_v    = jnp.float32(1.0)
        obs_circles = jnp.zeros((NUM_OBS_CIR, 3), dtype=jnp.float32)
        obs_boxes   = _MAP_BOXES                                       # (NUM_OBS_BOX, 8)
        people      = _make_dummies(NUM_PEOPLE)
        return rx, ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people

    branches = [
        _random_scen,
        _parallel_scen,
        _perpendicular_scen,
        _circular_scen,
        _bottleneck_scen,
        _intersection_scen,
        _static_groups_scen,
        _s_corridor_scen,          # 7
        _converging_crowds_scen,   # 8
        _sequential_rooms_scen,    # 9
        _zigzag_counterflow_scen,  # 10
        _parallel_long_scen,       # 11
        _uturn_corridor_scen,      # 12
        _gait_treadmill_scen,      # 13 — visual gait check (single walker)
        _gait_crossing_scen,       # 14 — visual gait check (two walkers crossing)
        _gait_circle_scen,         # 15 — visual gait check (single walker, circle)
        _npz_map_scen,             # 16 — real-world floor-plan from NPZ map
        _floor_tour_scen,          # 17 — CCW floor tour with corridor patrollers (training)
    ]

    # Compile all scenarios into the XLA switch statement
    raw_rx, raw_ry, rtheta, gx, gy, max_v, obs_circles, obs_boxes, people = jax.lax.switch(idx, branches, k_branch)

    # Room dimensions: scenario 11 uses 24 m height; scenarios 16-17 use the NPZ map size.
    _uses_map = (jnp.int32(idx) == 16) | (jnp.int32(idx) == 17)
    room_h = jnp.where(jnp.int32(idx) == 11, jnp.float32(24.0),
             jnp.where(_uses_map, jnp.float32(MAP_ROOM_H), jnp.float32(ROOM_H)))
    room_w = jnp.where(_uses_map, jnp.float32(MAP_ROOM_W), jnp.float32(ROOM_W))

    # ── Universal Robot Safety Check (Vectorized) ──
    # Post-process: Guarantees the robot NEVER spawns directly on top of a human
    # OR inside a wall in ANY scenario, using parallel offsets instead of while_loop.
    k_safe, k_off_x, k_off_y = jax.random.split(key, 3)

    # Generate N_GUESSES offsets. First offset is (0.0, 0.0) to keep intended spawn if safe.
    off_x = jnp.concatenate([jnp.array([0.0]), jax.random.uniform(k_off_x, (N_GUESSES-1,), minval=-2.0, maxval=2.0)])
    off_y = jnp.concatenate([jnp.array([0.0]), jax.random.uniform(k_off_y, (N_GUESSES-1,), minval=-2.0, maxval=2.0)])

    rx_guesses = jnp.clip(raw_rx + off_x, 1.0, room_w - 1.0)
    ry_guesses = jnp.clip(raw_ry + off_y, 1.0, room_h - 1.0)

    def check_post_safe(x, y):
        dist = jnp.sqrt((people[:, 0] - x)**2 + (people[:, 1] - y)**2)
        human_ok = jnp.min(dist) >= 1.0
        wall_ok = _is_safe(x, y, ROBOT_RADIUS + 0.3, obs_circles, obs_boxes, room_w, room_h)
        return human_ok & wall_ok

    safe_mask = jax.vmap(check_post_safe)(rx_guesses, ry_guesses)
    best_idx = jnp.argmax(safe_mask)

    rx_safe = rx_guesses[best_idx]
    ry_safe = ry_guesses[best_idx]

    # Enforce a minimum initial goal distance for non-exception scenarios by
    # post-processing the chosen goal: if the goal is too close to the robot,
    # push it along the same direction until it sits at `min_goal_dist`.
    min_req_global = jnp.where(is_exception_scen, _MIN_GOAL_FLOOR, jnp.float32(min_goal_dist))
    dx = gx - rx_safe
    dy = gy - ry_safe
    dist = jnp.hypot(dx, dy)

    def _push_goal():
        # If dist is tiny, pick +x direction as fallback
        safe_dir_x = jnp.where(dist > 1e-6, dx / dist, jnp.array(1.0, dtype=dx.dtype))
        safe_dir_y = jnp.where(dist > 1e-6, dy / dist, jnp.array(0.0, dtype=dy.dtype))
        ngx = rx_safe + safe_dir_x * min_req_global
        ngy = ry_safe + safe_dir_y * min_req_global
        ngx = jnp.clip(ngx, 1.0, room_w - 1.0)
        ngy = jnp.clip(ngy, 1.0, room_h - 1.0)
        return ngx, ngy

    need_push = dist < min_req_global
    new_gx, new_gy = jax.lax.cond(need_push, _push_goal, lambda: (gx, gy))

    # Ensure the selected goal is not inside an obstacle AND not on top of any
    # active person. We attempt a vectorized re-sampling of candidate goals
    # (N_GUESSES) and pick the first candidate satisfying all constraints.
    # If none are valid, fall back to the currently computed `new_gx,new_gy`.
    k_post = jax.random.split(k_branch)[0]
    GOAL_CLEARANCE = GOAL_RADIUS + 0.3
    gkx, gky = jax.random.split(k_post, 2)
    rand_x = jax.random.uniform(gkx, (N_GUESSES,), minval=GOAL_CLEARANCE, maxval=room_w-GOAL_CLEARANCE)
    # For bottleneck scenario (idx==4) sample x randomly within the gap
    half_gap = 1.4  # matches gap_size/2 used in _bottleneck_scen (gap_size=2.8)
    minx = jnp.clip(gx - half_gap + GOAL_CLEARANCE, GOAL_CLEARANCE, room_w - GOAL_CLEARANCE)
    maxx = jnp.clip(gx + half_gap - GOAL_CLEARANCE, GOAL_CLEARANCE, room_w - GOAL_CLEARANCE)
    width = jnp.maximum(maxx - minx, 0.0)
    around_x = minx + jax.random.uniform(gkx, (N_GUESSES,)) * width
    # For parallel corridor scenarios prefer sampling x inside the corridor
    # bounds. The gap is randomized per episode, so derive it from the wall
    # boxes (box 0 = left wall, box 1 = right wall) instead of hardcoding it.
    # Clamp both bounds to the centre line so a gap narrower than
    # 2*GOAL_CLEARANCE degenerates to centred candidates instead of inverting.
    corridor_left  = jnp.max(obs_boxes[0, 0::2])
    corridor_right = jnp.min(obs_boxes[1, 0::2])
    corridor_mid   = 0.5 * (corridor_left + corridor_right)
    corridor_min = jnp.minimum(corridor_left + GOAL_CLEARANCE, corridor_mid)
    corridor_max = jnp.maximum(corridor_right - GOAL_CLEARANCE, corridor_mid)
    corridor_x = jax.random.uniform(gkx, (N_GUESSES,), minval=corridor_min, maxval=corridor_max)
    g_guesses_x = jnp.where(jnp.int32(idx) == 4, around_x,
                            jnp.where(jnp.int32(idx) == 1, corridor_x, rand_x))
    g_guesses_y = jax.random.uniform(gky, (N_GUESSES,), minval=GOAL_CLEARANCE, maxval=room_h-GOAL_CLEARANCE)
    # Bottleneck: keep candidate goal strictly above the wall so it never lands
    # back on the robot's side. Wall midline at ROOM_H/2, half-thickness 0.2;
    # require at least 1 m above the wall's top edge.
    btl_y_min = jnp.float32(ROOM_H / 2.0) + jnp.float32(0.2) + jnp.float32(1.0)
    btl_y = jax.random.uniform(gky, (N_GUESSES,), minval=btl_y_min, maxval=room_h-GOAL_CLEARANCE)
    g_guesses_y = jnp.where(jnp.int32(idx) == 4, btl_y, g_guesses_y)

    def _check_candidate(x, y):
        safe_env = _is_safe(x, y, GOAL_CLEARANCE, obs_circles, obs_boxes, room_w, room_h)
        d_to_robot = jnp.hypot(x - rx_safe, y - ry_safe)
        dist_ok = (d_to_robot >= min_req_global) & (d_to_robot <= max_goal_dist)
        # Minimum clearance from any active person
        p_mask = people[:, 10] >= 0.0
        # Compute distances to people, ignore dummies by large value
        pdists = jnp.hypot(people[:, 0] - x, people[:, 1] - y)
        pdists = jnp.where(p_mask, pdists, 1e6)
        min_pd = jnp.min(pdists)
        person_ok = min_pd >= (PEOPLE_RADIUS + GOAL_RADIUS + 0.05)
        return safe_env & dist_ok & person_ok

    cand_mask = jax.vmap(_check_candidate)(g_guesses_x, g_guesses_y)
    any_cand_ok = jnp.any(cand_mask)
    # If no candidate satisfies every constraint, fall back to the candidate
    # with the largest obstacle clearance so the goal is never inside a wall.
    cand_env_clear = jnp.minimum(
        jax.vmap(lambda x, y: _min_dist_to_circles(x, y, obs_circles))(g_guesses_x, g_guesses_y),
        jax.vmap(lambda x, y: _min_dist_to_boxes(x, y, obs_boxes))(g_guesses_x, g_guesses_y),
    )
    best_idx = jnp.where(any_cand_ok, jnp.argmax(cand_mask), jnp.argmax(cand_env_clear))
    cand_gx = g_guesses_x[best_idx]
    cand_gy = g_guesses_y[best_idx]

    # Keep the scenario's intended goal whenever it sits outside every static
    # obstacle. People are *not* part of this check: a moving person that
    # happens to spawn on the goal must be nudged away instead of displacing
    # the goal, so the position specified in the scenario is preserved.
    wall_ok = (new_gx > GOAL_CLEARANCE) & (new_gx < room_w - GOAL_CLEARANCE) & \
              (new_gy > GOAL_CLEARANCE) & (new_gy < room_h - GOAL_CLEARANCE)
    cir_ok  = _min_dist_to_circles(new_gx, new_gy, obs_circles) > GOAL_CLEARANCE
    box_ok  = _min_dist_to_boxes(new_gx, new_gy, obs_boxes)   > GOAL_CLEARANCE
    orig_safe = wall_ok & cir_ok & box_ok

    # When the scenario's goal is unsafe, always use the candidate — cand_gx/gy
    # is either the first fully-passing candidate, or (if none passed) the one
    # with the largest obstacle clearance. Falling back to `new_gx,new_gy` here
    # would re-introduce in-obstacle goals.
    final_gx = jnp.where(orig_safe, new_gx, cand_gx)
    final_gy = jnp.where(orig_safe, new_gy, cand_gy)

    # Displace any active person that would overlap the finalised goal at
    # spawn — push them radially outward to a safe clearance. Dummies (flag<0)
    # are untouched. Static-groups (scenario 6) already places its goal
    # outside every person, so `needs_push` is uniformly False there.
    PERSON_GOAL_SAFE = PEOPLE_RADIUS + GOAL_RADIUS + 0.05
    p_active = people[:, 10] >= 0.0
    ddx = people[:, 0] - final_gx
    ddy = people[:, 1] - final_gy
    ddist = jnp.hypot(ddx, ddy)
    needs_push = p_active & (ddist < PERSON_GOAL_SAFE)
    ux = jnp.where(ddist > 1e-6, ddx / ddist, jnp.float32(1.0))
    uy = jnp.where(ddist > 1e-6, ddy / ddist, jnp.float32(0.0))
    push_to = PERSON_GOAL_SAFE + 0.05
    new_px = jnp.where(needs_push, final_gx + ux * push_to, people[:, 0])
    new_py = jnp.where(needs_push, final_gy + uy * push_to, people[:, 1])
    new_px = jnp.clip(new_px, 0.4, room_w - 0.4)
    new_py = jnp.clip(new_py, 0.4, room_h - 0.4)
    people = people.at[:, 0].set(new_px).at[:, 1].set(new_py)

    return rx_safe, ry_safe, rtheta, final_gx, final_gy, max_v, obs_circles, obs_boxes, people, room_h, room_w


# ── Robot waypoint sequences for test scenarios (used by eval scripts) ────────
# The env always has a single goal. The eval loop advances through these
# waypoints: when goal_reached fires, it updates state.goal_x/goal_y to the
# next waypoint via state.replace(). Metrics reset between waypoints.
TEST_SCENARIO_NAMES = {
    7: "S-Corridor",
    8: "Converging Crowds",
    9: "Sequential Rooms",
    10: "Zigzag Counter-flow",
    11: "Long Parallel Corridor",
    12: "U-Turn Corridor",
}

# Robot waypoint lists per test scenario.
# Single-goal scenarios have one entry; the eval loop completes after it.
# Multi-goal scenarios list waypoints in order; goal_x/goal_y starts at [0].
TEST_ROBOT_WAYPOINTS = {
    7:  [(10.0, 4.0), (2.0, 8.0), (10.5, 10.5)],            
    8:  [(6.0, 11.0)],                                       
    
    # -1.0 = flag per door1_y, -2.0 = flag per door2_y (calcolati dinamicamente in eval).
    # WP1: stanza 2 subito dopo porta 1 | WP2: stanza 3 subito dopo porta 2.
    9:  [(5.0, -1.0), (9.0, -2.0)],
    
    10: [(6.0, 11.0)],                                       
    11: [(6.0, 12.0), (6.0, 19.2)],   
    12: [(1.5, 1.5), (10.5, 1.5), (10.5, 10.5)],            
}