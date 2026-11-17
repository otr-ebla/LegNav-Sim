"""EXP-01  Actor-head calibration (exact, analytic edits of Dense_2 only).
raw = h·K + b ;  v = (tanh(raw_v)·.5+.5)·max_v ; w = tanh(raw_w)
Edit:  K[:,0]*=gv, b[0]=b[0]*gv+bv ;  K[:,1]*=gw, b[1]=b[1]*gw+bw
"""
import sys, numpy as np, ws_lib as L, evalcand as E
split = sys.argv[1] if len(sys.argv) > 1 else "s1"
p0 = L.load_params(); base = E.run(split, "original", p0)
print(L.fmt_row("original", L.summarize(base)["ALL"]))
grid = [dict(gv=1, bv=0, gw=1, bw=-0.035),      # remove measured left-turn bias (mean w=+0.033)
        dict(gv=0.6, bv=0, gw=1, bw=0), dict(gv=1.5, bv=0, gw=1, bw=0),
        dict(gv=1, bv=-0.7, gw=1, bw=0), dict(gv=1, bv=+0.7, gw=1, bw=0),
        dict(gv=1, bv=0, gw=0.75, bw=0), dict(gv=1, bv=0, gw=1.3, bw=0), dict(gv=1, bv=0, gw=1.7, bw=0)]
grid += [eval(a) for a in sys.argv[2:]]
for g in grid:
    name = "A_headcal_gv{gv}_bv{bv}_gw{gw}_bw{bw}".format(**g)
    p = L.copy_params(p0); K, b = p["Dense_2"]["kernel"], p["Dense_2"]["bias"]
    K[:, 0] *= g["gv"]; b[0] = b[0] * g["gv"] + g["bv"]; K[:, 1] *= g["gw"]; b[1] = b[1] * g["gw"] + g["bw"]
    L.save_candidate(p, name, dict(family="A: actor-head calibration", parent="original", edit=g,
                                   layers_modified=["Dense_2/kernel", "Dense_2/bias"]))
    a = E.run(split, name, p)
    print(L.fmt_row(name, L.summarize(a)["ALL"])); print("      ", E.fmt_paired(E.paired(base, a)), flush=True)
L.assert_original_untouched()
