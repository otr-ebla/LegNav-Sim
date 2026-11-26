"""
sim_static_robot_lidar.py — Static robot + two pacing humans, LiDAR visualisation.

Scene
-----
  * Empty rectangular room, white background.
  * Robot parked at the bottom-centre of the room, facing up, never moves.
  * Two humans walk horizontally back and forth: one starts on the left
    (heading right), the other on the right (heading left). When a human
    reaches a wall it bounces back (velocity reversed).
  * Legs/feet are driven by the same planted-foot gait model used in training
    (legnav.core.jax_legs), and the LiDAR is the same ray caster
    (legnav.core.jax_physics.compute_lidar) fed with the leg circles, so the
    rays hit the legs exactly as the policy sees them.

Left panel : top-down scene. Rays that hit a leg are red, rays that hit a
             wall are light grey.
Right panel: range per beam (polar) — the current scan versus the empty-room
             scan, so you can see where the beams are cut short by the legs.

The full trajectory is precomputed in one jitted lax.scan; matplotlib only
plays it back.

Run:
  python -m legnav.plotting.sim_static_robot_lidar
  python -m legnav.plotting.sim_static_robot_lidar --save figures/static_lidar.mp4
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import math

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, Rectangle

from legnav.config import RobotConfig, LidarConfig
from legnav.core.jax_legs import (init_foot_state, advance_feet, get_leg_circles,
                                  get_shoe_boxes, LEG_RADIUS, PEOPLE_RADIUS)
from legnav.core.jax_physics import compute_lidar

ROBOT_RADIUS = RobotConfig.RADIUS
NUM_RAYS     = LidarConfig.NUM_RAYS
FOV          = 2.0 * math.pi
MAX_DIST     = 12.0
# Dummy degenerate box far outside the room (compute_lidar needs >= 1 box).
_NO_BOXES    = jnp.full((1, 8), -100.0)

C_HIT   = "#d62728"
C_FREE  = "#c7c7c7"
C_ROBOT = "#1f77b4"
C_HUMAN = ("#2ca02c", "#9467bd")


# ── Simulation (fully jitted) ─────────────────────────────────────────────────

def build_sim(room_w, room_h, robot_xy, lanes_y, speeds, wall_margin, dt, steps):
    rx, ry = robot_xy
    rtheta = math.pi / 2.0          # facing up, into the room

    # people (N, 5): px, py, vx, vy, theta
    starts_x = jnp.array([wall_margin, room_w - wall_margin])
    vx0      = jnp.array([speeds[0], -speeds[1]])
    people0  = jnp.stack([starts_x, jnp.asarray(lanes_y), vx0,
                          jnp.zeros(2), jnp.where(vx0 > 0, 0.0, jnp.pi)], axis=-1)
    feet0    = init_foot_state(people0, jax.random.PRNGKey(0))

    x_lo, x_hi = wall_margin, room_w - wall_margin

    def scan_lidar(circles):
        return compute_lidar(rx, ry, rtheta, circles, _NO_BOXES,
                             NUM_RAYS, FOV, MAX_DIST, room_w, room_h)

    def step(carry, _):
        people, feet = carry
        px, vx = people[:, 0], people[:, 2]
        nx = px + vx * dt
        # Bounce off the side walls: reflect position and reverse velocity.
        bounce_hi = nx > x_hi
        bounce_lo = nx < x_lo
        nx = jnp.where(bounce_hi, 2 * x_hi - nx, jnp.where(bounce_lo, 2 * x_lo - nx, nx))
        vx = jnp.where(bounce_hi | bounce_lo, -vx, vx)
        theta = jnp.where(vx > 0, 0.0, jnp.pi)
        people = people.at[:, 0].set(nx).at[:, 2].set(vx).at[:, 4].set(theta)
        feet = advance_feet(feet, people, dt)

        legs = get_leg_circles(people, feet, use_legs=True)
        out = (people, feet[:, 0:4], get_shoe_boxes(people, feet), scan_lidar(legs))
        return (people, feet), out

    @jax.jit
    def run():
        empty = scan_lidar(jnp.array([[-100.0, -100.0, 0.01]]))
        _, traj = jax.lax.scan(step, (people0, feet0), None, length=steps)
        return traj, empty

    return run, (rx, ry, rtheta)


def shoe_polys(shoes):
    """(K, 5) [cx, cy, hL, hW, th] → (K, 4, 2) corner polygons (vectorised)."""
    cx, cy, hl, hw, th = (shoes[:, i] for i in range(5))
    c, s = np.cos(th), np.sin(th)
    sl = np.array([1, 1, -1, -1])[None]
    sw = np.array([1, -1, -1, 1])[None]
    lx, ly = hl[:, None] * sl, hw[:, None] * sw
    return np.stack([cx[:, None] + lx * c[:, None] - ly * s[:, None],
                     cy[:, None] + lx * s[:, None] + ly * c[:, None]], axis=-1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room-w", type=float, default=8.0)
    ap.add_argument("--room-h", type=float, default=6.0)
    ap.add_argument("--lanes", type=float, nargs=2, default=[2.4, 3.8],
                    help="y of the two humans' lanes (m)")
    ap.add_argument("--speeds", type=float, nargs=2, default=[1.0, 1.25],
                    help="walking speed of left/right human (m/s)")
    ap.add_argument("--wall-margin", type=float, default=0.45,
                    help="body-centre distance from the wall at which a human bounces")
    ap.add_argument("--dt", type=float, default=RobotConfig.DT)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--save", type=str, default=None,
                    help="output file (.mp4 or .gif); omit for an interactive window")
    ap.add_argument("--fps", type=int, default=None, help="default: real time (1/dt)")
    ap.add_argument("--no-radar", action="store_true", help="hide the polar range panel")
    args = ap.parse_args()

    W, H = args.room_w, args.room_h
    run, (rx, ry, rth) = build_sim(W, H, (W / 2.0, 0.6), args.lanes, args.speeds,
                                   args.wall_margin, args.dt, args.steps)
    (people, feet, shoes, lidar), empty = jax.device_get(run())
    people, feet, shoes, lidar, empty = map(np.asarray, (people, feet, shoes, lidar, empty))

    angles = rth - FOV * 0.5 + np.arange(NUM_RAYS) * (FOV / (NUM_RAYS - 1))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    hit = lidar < empty - 1e-3                                  # (T, R) beams cut by legs
    hx, hy = rx + lidar * cos_a, ry + lidar * sin_a             # (T, R) hit points

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    if args.no_radar:
        fig, ax = plt.subplots(figsize=(8, 8 * H / W))
        axr = None
    else:
        fig = plt.figure(figsize=(14, 6))
        ax  = fig.add_subplot(1, 2, 1)
        axr = fig.add_subplot(1, 2, 2, projection="polar")

    ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(-0.2, H + 0.2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, lw=3, ec="black", zorder=5))

    rays = LineCollection([], linewidths=0.8, zorder=1)
    ax.add_collection(rays)
    hit_pts = ax.scatter([], [], s=10, c=C_HIT, zorder=4)

    ax.add_patch(Circle((rx, ry), ROBOT_RADIUS, fc=C_ROBOT, ec="black", zorder=6))
    ax.plot([rx, rx + ROBOT_RADIUS * math.cos(rth)], [ry, ry + ROBOT_RADIUS * math.sin(rth)],
            c="white", lw=2, zorder=7)

    shoe_coll = PolyCollection([], edgecolors="black", linewidths=0.6, alpha=0.55, zorder=2)
    # get_shoe_boxes returns [left shoes (N), right shoes (N)].
    shoe_coll.set_facecolors([C_HUMAN[0], C_HUMAN[1], C_HUMAN[0], C_HUMAN[1]])
    ax.add_collection(shoe_coll)
    bodies, legs = [], []
    for i in range(2):
        bodies.append(ax.add_patch(Circle((0, 0), PEOPLE_RADIUS, fill=False, ls="--",
                                          ec=C_HUMAN[i], lw=1.0, zorder=2)))
        for _ in range(2):
            legs.append(ax.add_patch(Circle((0, 0), LEG_RADIUS, fc=C_HUMAN[i],
                                            ec="black", lw=0.8, zorder=3)))
    title = ax.set_title("", fontsize=11)

    if axr is not None:
        axr.set_theta_zero_location("E")
        axr.set_rlim(0, float(np.max(empty)) * 1.05)
        axr.plot(angles, empty, c="#7f7f7f", lw=1.0, ls="--", label="empty room")
        scan_line, = axr.plot([], [], c="black", lw=1.0, label="current scan")
        scan_hits = axr.scatter([], [], s=10, c=C_HIT, zorder=3, label="leg hits")
        axr.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, frameon=False)
        axr.set_title("LiDAR range per beam [m]", fontsize=11, pad=18)

    def update(t):
        seg = np.stack([np.broadcast_to([rx, ry], (NUM_RAYS, 2)),
                        np.stack([hx[t], hy[t]], -1)], axis=1)
        rays.set_segments(seg)
        rays.set_colors(np.where(hit[t][:, None], matplotlib.colors.to_rgba_array(C_HIT),
                                 matplotlib.colors.to_rgba_array(C_FREE)))
        hit_pts.set_offsets(np.stack([hx[t][hit[t]], hy[t][hit[t]]], -1).reshape(-1, 2))
        shoe_coll.set_verts(shoe_polys(shoes[t]))
        for i in range(2):
            bodies[i].center = people[t, i, :2]
            legs[2 * i].center     = feet[t, i, 0:2]
            legs[2 * i + 1].center = feet[t, i, 2:4]
        title.set_text(f"t = {t * args.dt:5.2f} s   |   beams on legs: {int(hit[t].sum())}/{NUM_RAYS}")
        artists = [rays, hit_pts, shoe_coll, title, *bodies, *legs]
        if axr is not None:
            scan_line.set_data(np.append(angles, angles[0]), np.append(lidar[t], lidar[t, 0]))
            scan_hits.set_offsets(np.stack([angles[hit[t]], lidar[t][hit[t]]], -1).reshape(-1, 2))
            artists += [scan_line, scan_hits]
        return artists

    fps = args.fps or max(1, round(1.0 / args.dt))
    anim = FuncAnimation(fig, update, frames=args.steps, interval=1000.0 / fps,
                         blit=args.save is None)
    fig.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        writer = "pillow" if args.save.lower().endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=fps, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
