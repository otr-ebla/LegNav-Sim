"""EXP-02  Behaviour-level probes (no checkpoint produced): does symmetrisation / a LiDAR
front-gap speed shield around the ORIGINAL network help?  Guides which targets to distil into weights."""
import sys, jax, numpy as np, ws_lib as L, evalcand as E, ws_teachers as T
split = sys.argv[1]
p0 = L.load_params(); base = E.run(split, "original", p0)
print(L.fmt_row("original", L.summarize(base)["ALL"]))
probes = {"PROBE_sym": T.mean_sym,
          "PROBE_shield_tau0.5": T.make_shield(0.5), "PROBE_shield_tau1.0": T.make_shield(1.0),
          "PROBE_shield_tau0.5_sym": T.make_shield(0.5, sym=True)}
def _spec(a):          # "tau,half_width,floor,sym"
    t, hw, fl, sy = a.split(","); return f"PROBE_shield_t{t}_hw{hw}_f{fl}" + ("_sym" if sy == "1" else ""), T.make_shield(float(t), float(hw), sy == "1", float(fl))
for a in sys.argv[2:]:
    if "," in a:
        n, f = _spec(a); probes[n] = f
names = [(_spec(a)[0] if "," in a else a) for a in sys.argv[2:]]
for name in names or probes:
    ck = jax.jit(L._make_chunk(T.act_from_mean(probes[name])))
    a = E.run(split, name, p0, chunk_fn=ck)
    S = L.summarize(a, by_scen=True)
    print(L.fmt_row(name, S["ALL"])); print("      ", E.fmt_paired(E.paired(base, a)))
    print("       per-scen SR:", " ".join(f"{k.split(':')[0]}:{100*v['success']:.0f}" for k, v in S.items() if ":" in k), flush=True)
