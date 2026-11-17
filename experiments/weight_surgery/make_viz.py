"""Representative-episode visualisations: original vs candidate on IDENTICAL episodes (same seed/keys).
usage: make_viz.py CAND_NAME [seed]"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection
from matplotlib import animation
import ws_lib as L, ws_trace as T
cand = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 4242
p0 = L.load_params(); p1 = L.load_params(os.path.join(L.CAND_DIR, cand + ".msgpack"))
T_STEPS = 900
r0, st = T.trace(p0, seed=seed, n_per_scen=28, T=T_STEPS)
r1, _ = T.trace(p1, seed=seed, n_per_scen=28, T=T_STEPS)
C_ORIG, C_CAND = "#c44e52", "#2a7f62"

def outcome(r, i):
    if r["succ"][:, i][r["active"][:, i]].any(): return "success"
    if r["col"][:, i][r["active"][:, i]].any(): return "collision"
    return "timeout"
out0 = np.array([outcome(r0, i) for i in range(len(st["scen"]))]); out1 = np.array([outcome(r1, i) for i in range(len(st["scen"]))])
print("trace batch outcomes  original:", {k: int((out0 == k).sum()) for k in ("success", "collision", "timeout")},
      " candidate:", {k: int((out1 == k).sum()) for k in ("success", "collision", "timeout")})

def draw_static(ax, i):
    rw, rh = float(st["room_w"][i]), float(st["room_h"][i])
    ax.add_patch(Polygon([[0, 0], [rw, 0], [rw, rh], [0, rh]], closed=True, fill=False, ec="#333", lw=1.2))
    for b in st["boxes"][i]:
        v = b.reshape(4, 2)
        if np.abs(v[0] - v[2]).max() > 1e-6: ax.add_patch(Polygon(v, closed=True, fc="#8a8a8a", ec="none"))
    for c in st["circ"][i]:
        if c[2] > 0: ax.add_patch(Circle(c[:2], c[2], fc="#8a8a8a", ec="none"))
    ax.set_xlim(-0.2, rw + 0.2); ax.set_ylim(-0.2, rh + 0.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

def draw_episode(ax, r, i, color, label, out):
    n = int(r["active"][:, i].sum()); x, y = r["x"][:n, i], r["y"][:n, i]
    ppl, pidx = r["ppl"][:n, i], r["pidx"][0, i]
    for h in np.where(pidx >= 0)[0]:
        hx, hy = ppl[:, h, 0], ppl[:, h, 1]
        brk = np.where(np.hypot(np.diff(hx), np.diff(hy)) > 1.0)[0] + 1       # respawn teleports
        for seg in np.split(np.arange(n), brk):
            ax.plot(hx[seg], hy[seg], color="#9db7d5", lw=0.6, alpha=0.7, zorder=1)
        ax.plot(hx[-1], hy[-1], "o", color="#5b8fc7", ms=3, zorder=2)
    ax.plot(x, y, color=color, lw=1.8, zorder=4)
    ax.plot(x[0], y[0], "s", color="k", ms=5, zorder=5)
    gs = np.unique(np.stack([r["gx"][:n, i], r["gy"][:n, i]], 1), axis=0)
    ax.plot(gs[:, 0], gs[:, 1], "*", color="#e0a100", ms=11, mec="k", mew=0.5, zorder=5)
    mk = {"success": ("o", color), "collision": ("X", "k"), "timeout": ("P", "k")}[out]
    ax.plot(x[-1], y[-1], mk[0], color=mk[1], ms=8, zorder=6)
    ax.set_title(f"{label}: {out}  ({n*0.15:.0f}s)", fontsize=8, color=color)

def pick(scen):
    idx = np.where(st["scen"] == scen)[0]
    fixed = [i for i in idx if out0[i] != "success" and out1[i] == "success"]
    both = [i for i in idx if out0[i] == "success" and out1[i] == "success"]
    broke = [i for i in idx if out0[i] == "success" and out1[i] != "success"]
    return fixed, both, broke

rows = []
for s in (12, 11, 9, 7, 5, 1):
    fixed, both, broke = pick(s)
    if fixed: rows.append((s, fixed[0], "original fails → candidate succeeds"))
    elif both: rows.append((s, both[0], "both succeed"))
for s in (12, 9, 1, 0, 5):
    f, b, br = pick(s)
    if br: rows.append((s, br[0], "REGRESSION: original succeeds → candidate fails")); break
rows.append((10, pick(10)[1][0], "both succeed")) if pick(10)[1] else None
n = len(rows); fig, axs = plt.subplots(2, n, figsize=(3.0 * n, 6.6))
for c, (s, i, why) in enumerate(rows):
    for rr, (r, col, lab, o) in enumerate([(r0, C_ORIG, "original", out0[i]), (r1, C_CAND, "candidate", out1[i])]):
        ax = axs[rr, c]; draw_static(ax, i); draw_episode(ax, r, i, col, lab, o)
    axs[0, c].text(0.5, 1.16, f"{L.SCEN_NAMES[s]} · v_max={st['max_v'][i]:.2f} m/s\n{why}", transform=axs[0, c].transAxes, ha="center", fontsize=7.5)
fig.suptitle(f"Same episodes (identical seed & noise): original CALF PPO Legs vs {cand}   ■ start  ★ goal/waypoints  ✕ collision  blue = pedestrians", fontsize=9, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"figures/episodes_{cand}.png", dpi=150); print("saved", f"figures/episodes_{cand}.png")

# ── animated side-by-side GIF of one "fixed" episode ─────────────────────────
s, i, why = rows[0]
n0, n1 = int(r0["active"][:, i].sum()), int(r1["active"][:, i].sum()); N = max(n0, n1)
fig, axs = plt.subplots(1, 2, figsize=(8, 4.4))
arts = []
for ax, r, col, lab, nn, o in [(axs[0], r0, C_ORIG, "original", n0, out0[i]), (axs[1], r1, C_CAND, cand, n1, out1[i])]:
    draw_static(ax, i)
    gs = np.unique(np.stack([r["gx"][:nn, i], r["gy"][:nn, i]], 1), axis=0); ax.plot(gs[:, 0], gs[:, 1], "*", color="#e0a100", ms=11, mec="k", mew=0.5)
    hum = ax.scatter([], [], s=28, c="#5b8fc7"); rob = Circle((0, 0), 0.2, fc=col, ec="k", zorder=5); ax.add_patch(rob)
    trail, = ax.plot([], [], color=col, lw=1.5); hd, = ax.plot([], [], color="k", lw=1.2, zorder=6); tt = ax.set_title(lab, fontsize=9, color=col)
    arts.append((r, nn, hum, rob, trail, hd, tt, lab, o))
def upd(f):
    t = f * 2
    for r, nn, hum, rob, trail, hd, tt, lab, o in arts:
        k = min(t, nn - 1); x, y, th = r["x"][k, i], r["y"][k, i], r["th"][k, i]
        m = r["pidx"][0, i] >= 0; hum.set_offsets(r["ppl"][k, i][m]); rob.center = (x, y)
        trail.set_data(r["x"][:k + 1, i], r["y"][:k + 1, i]); hd.set_data([x, x + 0.3 * np.cos(th)], [y, y + 0.3 * np.sin(th)])
        tt.set_text(f"{lab}: {o.upper() if t >= nn - 1 else 'running'}  t={k*0.15:.1f}s")
    return []
ani = animation.FuncAnimation(fig, upd, frames=N // 2 + 12, interval=60)
fig.suptitle(f"{L.SCEN_NAMES[s]} — identical episode, v_max={st['max_v'][i]:.2f} m/s", fontsize=9)
ani.save(f"figures/episode_{cand}_{L.SCEN_NAMES[s]}.gif", writer=animation.PillowWriter(fps=15), dpi=80); print("saved gif")
