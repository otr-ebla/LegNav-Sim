"""Build ranking tables + charts from cached per-episode results.
usage: make_report.py SPLIT name1 name2 ...   (original implicit)"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import ws_lib as L, evalcand as E
split, names = sys.argv[1], sys.argv[2:]
ld = lambda n: dict(np.load(os.path.join(L.RES_DIR, f"{split}__{n}.npz")))
base = ld("original"); R = {"original": base, **{n: ld(n) for n in names}}
rows = []
for n, a in R.items():
    S = L.summarize(a, by_scen=True); r = dict(policy=n, **S["ALL"]); r["sr_train"] = S["TRAIN(0-6)"]["success"]; r["sr_test"] = S["TEST(7-12)"]["success"]
    r["per_scen_sr"] = {k: v["success"] for k, v in S.items() if ":" in k}
    r["paired"] = E.paired(base, a) if n != "original" else None; rows.append(r)
rows.sort(key=lambda r: (-r["success"], r["collision"]))
json.dump(rows, open(f"results/RANKING_{split}.json", "w"), indent=1)
md = [f"### Ranking on `{split}` split — {rows[0]['n']} paired episodes per policy (13 scenarios)\n",
      "| # | policy | success % | collision % (act/pas/static) | timeout % | reward | nav time s | path m | final dist m | SPL | Δw/step | ang. jerk | w-flip | SR train 0-6 | SR test 7-12 | Δsuccess vs orig (95% CI) | Δcollision (95% CI) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows):
    p = r["paired"]; f = lambda k: "—" if p is None else f"{100*p[k][0]:+.2f} [{100*p[k][1]:+.2f}, {100*p[k][2]:+.2f}]" + (" ✔" if (p[k][1] > 0 or p[k][2] < 0) else "")
    md.append(f"| {i+1} | {r['policy']} | {100*r['success']:.2f} | {100*r['collision']:.2f} ({100*r['active_col']:.1f}/{100*r['passive_col']:.1f}/{100*r['static_col']:.1f}) | {100*r['timeout']:.2f} | {r['reward']:.2f} | {r['nav_time']:.1f} | {r['path_len']:.2f} | {r['final_dist']:.2f} | {r['spl']:.3f} | {r['dw']:.3f} | {r['ang_jerk']:.2f} | {r['w_flip_rate']:.3f} | {100*r['sr_train']:.1f} | {100*r['sr_test']:.1f} | {f('success')} | {f('collision')} |")
md.append("\n✔ = 95% paired-bootstrap CI excludes 0.\n\n#### Success % per scenario\n")
sc = list(rows[0]["per_scen_sr"]); md.append("| policy | " + " | ".join(s.split(":")[1] for s in sc) + " |"); md.append("|---|" + "---|" * len(sc))
for r in rows: md.append(f"| {r['policy']} | " + " | ".join(f"{100*r['per_scen_sr'][s]:.1f}" for s in sc) + " |")
open(f"results/RANKING_{split}.md", "w").write("\n".join(md)); print("\n".join(md))

# ── charts ───────────────────────────────────────────────────────────────────
INK, INK2, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})
best = next(r for r in rows if r["policy"] != "original"); orig = next(r for r in rows if r["policy"] == "original")
# 1) failure rate per scenario, original vs best (lower is better)
fig, ax = plt.subplots(figsize=(10, 3.6)); x = np.arange(len(sc)); w = 0.38
fo = [100 * (1 - orig["per_scen_sr"][s]) for s in sc]; fb = [100 * (1 - best["per_scen_sr"][s]) for s in sc]
ax.bar(x - w / 2 - 0.01, fo, w, color=ORANGE, label="original"); ax.bar(x + w / 2 + 0.01, fb, w, color=BLUE, label=best["policy"])
for xi, (a, b) in enumerate(zip(fo, fb)):
    ax.text(xi - w / 2, a + 0.2, f"{a:.1f}", ha="center", fontsize=7, color=INK2); ax.text(xi + w / 2, b + 0.2, f"{b:.1f}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([s.split(":")[1] for s in sc], rotation=30, ha="right"); ax.set_ylabel("failure rate (%)")
ax.yaxis.grid(True, color="#e6e5df", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False)
ax.set_title(f"Failure rate per scenario on held-out `{split}` episodes (lower is better)", color=INK, fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"figures/failure_by_scenario_{split}.png", dpi=150)
# 2) paired differences vs original with CI, one panel per metric (separate axes — no dual scale)
mets = [("success", 100, "Δ success (pp)"), ("collision", 100, "Δ collision (pp)"), ("reward", 1, "Δ episode reward"), ("spl", 1, "Δ SPL"), ("ang_jerk", 1, "Δ angular jerk")]
cr = [r for r in rows if r["policy"] != "original"][::-1]
fig, axs = plt.subplots(1, len(mets), figsize=(13, 0.42 * len(cr) + 1.6), sharey=True)
for ax, (k, s, lab) in zip(axs, mets):
    for yi, r in enumerate(cr):
        m, lo, hi = [v * s for v in r["paired"][k]]
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round"); ax.plot(m, yi, "o", color=BLUE, ms=6, mec=SURF, mew=1.5)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel(lab); ax.xaxis.grid(True, color="#e6e5df", lw=0.6); ax.set_axisbelow(True)
axs[0].set_yticks(range(len(cr))); axs[0].set_yticklabels([r["policy"] for r in cr])
fig.suptitle(f"Paired difference vs original on `{split}` (dot = mean, bar = 95% bootstrap CI; 0 = no change)", color=INK, fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"figures/paired_diffs_{split}.png", dpi=150); print("charts saved")
