"""
Fit axis-aligned bounding boxes (AABBs) to each polygon obstacle in the NPZ map,
then save the results as map_obbs.npz for use by the JAX simulator.

All obstacles in this floor-plan are axis-aligned, so AABBs (min/max of vertices)
give exact tight fits.  Complex hollow polygons (low area-ratio, e.g. door frames
and room-wall outlines) are decomposed into thin wall segments instead of a single
large AABB that would block open space.

Output: map_obbs.npz with keys
  obs_boxes  – (N_KEEP, 8) float32 AABBs in map-local coordinates (origin at 0)
  room_w     – scalar: map width  (m)
  room_h     – scalar: map height (m)
  origin_x   – scalar: x-offset subtracted from all coordinates
  origin_y   – scalar: y-offset subtracted from all coordinates
  robot_x    – scalar: recommended robot start x (map-local)
  robot_y    – scalar: recommended robot start y (map-local)
  goal_x     – scalar: recommended goal x (map-local)
  goal_y     – scalar: recommended goal y (map-local)
"""

import pathlib
import numpy as np

NPZ_IN  = pathlib.Path(__file__).parent / "data_sim_lidar_1.npz"
NPZ_OUT = pathlib.Path(__file__).parent / "map_obbs.npz"

N_KEEP        = 300    # keep this many AABBs (≤ NUM_OBS_BOX=300 in jax_scenarios)
AREA_MIN      = 0.01   # m² — skip near-degenerate slivers
ROOM_PADDING  = 1.5    # extra margin added around all AABB vertices

WALL_MAX      = 0.6    # max thickness of a single wall segment (m)
RATIO_THRESH  = 0.40   # decompose polygons whose area/bbox_area is below this
MIN_SEG       = 0.10   # minimum dimension of a kept wall segment (m)
COORD_MERGE   = 0.05   # merge vertex coordinates closer than this (m); must be
                       # smaller than thinnest wall (~0.12 m for door frames)


# ── Helpers ───────────────────────────────────────────────────────────────────

def polygon_area(verts: np.ndarray) -> float:
    """Shoelace formula."""
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_polygon(px: float, py: float, verts: np.ndarray) -> bool:
    """Ray-casting PIP test."""
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _merge_close(coords: list, eps: float) -> list:
    """Merge consecutive coordinate values that are within eps of each other."""
    if len(coords) < 2:
        return coords
    merged = [coords[0]]
    for c in coords[1:]:
        if c - merged[-1] < eps:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)
    return merged


def _consecutive_groups(indices: list) -> list:
    """Split a sorted list of integers into groups of consecutive values."""
    if not indices:
        return []
    groups, s, p = [], indices[0], indices[0]
    for k in indices[1:]:
        if k == p + 1:
            p = k
        else:
            groups.append((s, p))
            s = p = k
    groups.append((s, p))
    return groups


def fit_aabb(verts: np.ndarray) -> np.ndarray:
    """
    Axis-aligned bounding box from polygon vertices.
    Returns 8-float CCW vertex array [x0,y0, x1,y1, x2,y2, x3,y3].
    """
    min_x, max_x = float(verts[:, 0].min()), float(verts[:, 0].max())
    min_y, max_y = float(verts[:, 1].min()), float(verts[:, 1].max())
    return np.array([min_x, min_y,
                     max_x, min_y,
                     max_x, max_y,
                     min_x, max_y], dtype=np.float32)


def decompose_wall_polygon(verts: np.ndarray) -> list:
    """
    Decompose a hollow/complex axis-aligned wall polygon into thin wall AABBs.

    Strategy:
      1. Snap+merge near-duplicate vertex coordinates (LIDAR noise artefacts).
      2. Sweep thin vertical strips (width ≤ WALL_MAX); collect filled y-bands
         using PIP tests on the mid-point of each cell.
      3. Sweep remaining cells as thin horizontal strips.
      4. Filter out degenerate results (either dim < MIN_SEG).

    Falls back to a single AABB if no valid strips are found.
    """
    raw_xs = sorted(set(np.round(verts[:, 0], 3).tolist()))
    raw_ys = sorted(set(np.round(verts[:, 1], 3).tolist()))
    xs = _merge_close(raw_xs, COORD_MERGE)
    ys = _merge_close(raw_ys, COORD_MERGE)

    aabbs = []
    used: set = set()

    # --- Thin vertical strips ---
    for i in range(len(xs) - 1):
        w = xs[i + 1] - xs[i]
        if w < MIN_SEG or w > WALL_MAX:
            continue
        mx = (xs[i] + xs[i + 1]) / 2.0
        active = [
            j for j in range(len(ys) - 1)
            if _point_in_polygon(mx, (ys[j] + ys[j + 1]) / 2.0, verts)
        ]
        for j0, j1 in _consecutive_groups(active):
            y0, y1 = ys[j0], ys[j1 + 1]
            if y1 - y0 >= MIN_SEG:
                aabbs.append(np.array([xs[i], y0, xs[i+1], y0,
                                       xs[i+1], y1, xs[i], y1], dtype=np.float32))
                for j in range(j0, j1 + 1):
                    used.add((i, j))

    # --- Thin horizontal strips (cells not already covered by verticals) ---
    for j in range(len(ys) - 1):
        h = ys[j + 1] - ys[j]
        if h < MIN_SEG or h > WALL_MAX:
            continue
        my = (ys[j] + ys[j + 1]) / 2.0
        active = [
            i for i in range(len(xs) - 1)
            if (i, j) not in used and
               _point_in_polygon((xs[i] + xs[i + 1]) / 2.0, my, verts)
        ]
        for i0, i1 in _consecutive_groups(active):
            x0, x1 = xs[i0], xs[i1 + 1]
            if x1 - x0 >= MIN_SEG:
                aabbs.append(np.array([x0, ys[j], x1, ys[j],
                                       x1, ys[j+1], x0, ys[j+1]], dtype=np.float32))

    if not aabbs:
        aabbs = [fit_aabb(verts)]

    return aabbs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = np.load(NPZ_IN, allow_pickle=True)
    obstacles = data["Obstacles"]   # object array of (K, 2) vertex arrays
    true_map  = data["trueMap"]     # (N, 2) — used for coordinate bounds
    pose      = data["Pose"]        # (T, 3) robot trajectory

    # ── Step 1: Preliminary origin from ALL obstacle vertices + trueMap ───────
    obs_verts = np.concatenate([np.array(p, dtype=np.float64) for p in obstacles], axis=0)
    all_pts   = np.concatenate([true_map, obs_verts], axis=0)
    origin_x  = float(all_pts[:, 0].min())
    origin_y  = float(all_pts[:, 1].min())

    # ── Step 2: Filter polygons and fit AABBs (in preliminary local frame) ────
    # Complex hollow polygons (door frames, room wall outlines) are decomposed
    # into thin wall segments so their large bounding box does not block open space.
    obbs_list:   list = []
    areas_list:  list = []   # parent polygon area — used for sorting/priority

    for poly in obstacles:
        verts = np.array(poly, dtype=np.float64)
        area  = polygon_area(verts)
        if area < AREA_MIN:
            continue
        verts_local = verts - [origin_x, origin_y]

        min_x, max_x = verts_local[:, 0].min(), verts_local[:, 0].max()
        min_y, max_y = verts_local[:, 1].min(), verts_local[:, 1].max()
        bbox_area  = (max_x - min_x) * (max_y - min_y)
        area_ratio = area / max(bbox_area, 1e-9)

        # Decompose hollow polygons whose AABB over-fills open space.
        # Only attempt decomposition when BOTH bbox dims exceed WALL_MAX
        # (truly thin walls are already correct as a single AABB).
        both_thick = (max_x - min_x) > WALL_MAX and (max_y - min_y) > WALL_MAX
        if area_ratio < RATIO_THRESH and both_thick:
            for seg in decompose_wall_polygon(verts_local):
                obbs_list.append(seg)
                areas_list.append(area)
        else:
            obbs_list.append(fit_aabb(verts_local))
            areas_list.append(area)

    # Sort by parent-polygon area descending, keep top N_KEEP
    order      = np.argsort(areas_list)[::-1]
    obbs_list  = [obbs_list[i]  for i in order[:N_KEEP]]
    areas_list = [areas_list[i] for i in order[:N_KEEP]]
    n_actual   = len(obbs_list)

    # ── Step 3: Compute room bounds from AABB vertices ───────────────────────
    obb_arr = np.array(obbs_list, dtype=np.float64)   # (N, 8)
    obb_x   = obb_arr[:, [0, 2, 4, 6]].ravel()
    obb_y   = obb_arr[:, [1, 3, 5, 7]].ravel()

    tm_local_x = true_map[:, 0] - origin_x
    tm_local_y = true_map[:, 1] - origin_y
    all_local_x = np.concatenate([obb_x, tm_local_x])
    all_local_y = np.concatenate([obb_y, tm_local_y])

    shift_x = min(0.0, float(all_local_x.min()))
    shift_y = min(0.0, float(all_local_y.min()))
    if shift_x < 0 or shift_y < 0:
        origin_x += shift_x
        origin_y += shift_y
        obb_arr[:, [0, 2, 4, 6]] -= shift_x
        obb_arr[:, [1, 3, 5, 7]] -= shift_y
        obb_x = obb_arr[:, [0, 2, 4, 6]].ravel()
        obb_y = obb_arr[:, [1, 3, 5, 7]].ravel()
        tm_local_x -= shift_x
        tm_local_y -= shift_y

    # ── Manual adjustment: widen the right corridor ───────────────────────────
    # The large right-wall block (3 stacked AABBs at x≈[38.0, 39.6], y≈[24, 57])
    # leaves only ~1.5 m between itself and the shelf-row ends (x≈36.5).
    # Pull its left edge right so the corridor is ~2.3 m wide.
    RW_OLD_LEFT, RW_NEW_LEFT = 38.04, 38.80
    bx = obb_arr[:, [0, 2, 4, 6]]
    by = obb_arr[:, [1, 3, 5, 7]]
    sel = (np.abs(bx.min(1) - RW_OLD_LEFT) < 0.1) & (by.min(1) > 20.0) & (by.max(1) < 57.0)
    for r in np.where(sel)[0]:
        left = obb_arr[r, [0, 2, 4, 6]].min()
        for k in (0, 2, 4, 6):
            if abs(obb_arr[r, k] - left) < 1e-6:
                obb_arr[r, k] = RW_NEW_LEFT

    room_w = float(np.ceil(max(obb_x.max(), tm_local_x.max()) + ROOM_PADDING))
    room_h = float(np.ceil(max(obb_y.max(), tm_local_y.max()) + ROOM_PADDING))

    # Pad to exactly N_KEEP with zero rows (degenerate → ignored by simulator)
    obs_boxes = np.zeros((N_KEEP, 8), dtype=np.float32)
    for i, obb in enumerate(obb_arr):
        obs_boxes[i] = obb.astype(np.float32)

    # ── Robot and goal positions (map-local) ──────────────────────────────────
    robot_x = float(pose[0, 0] - origin_x)
    robot_y = float(pose[0, 1] - origin_y)
    goal_x  = float(pose[-1, 0] - origin_x)
    goal_y  = float(pose[-1, 1] - origin_y)

    # ── trueMap in local frame — stored as float16 for compactness ───────────
    true_map_local = (true_map - [origin_x, origin_y]).astype(np.float16)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(NPZ_OUT,
             obs_boxes=obs_boxes,
             room_w=np.float32(room_w),
             room_h=np.float32(room_h),
             origin_x=np.float32(origin_x),
             origin_y=np.float32(origin_y),
             robot_x=np.float32(robot_x),
             robot_y=np.float32(robot_y),
             goal_x=np.float32(goal_x),
             goal_y=np.float32(goal_y),
             true_map_local=true_map_local)

    print(f"Saved {NPZ_OUT}")
    print(f"  Obstacles kept : {n_actual} (area ≥ {AREA_MIN} m²)")
    print(f"  Room           : {room_w:.1f} × {room_h:.1f} m")
    print(f"  Origin offset  : ({origin_x:.2f}, {origin_y:.2f})")
    print(f"  Robot start    : ({robot_x:.2f}, {robot_y:.2f})")
    print(f"  Goal           : ({goal_x:.2f}, {goal_y:.2f})")
    if areas_list:
        print(f"  Area range     : {min(areas_list):.2f} – {max(areas_list):.2f} m²")


if __name__ == "__main__":
    main()
