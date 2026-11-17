"""EXP-05  LLM-in-the-loop composite edits (head calibration ⊕ skip implants).
usage: exp05_combo.py SPLIT TAG "dict(...)" ...   → candidates C_<TAG><i>"""
import sys, json, numpy as np, ws_lib as L, evalcand as E, ws_surgery as S
split, tag = sys.argv[1], sys.argv[2]
p0 = L.load_params(); base = E.run(split, "original", p0)
d0, d1 = S.dead_units(p0, np.load("results/onpolicy_obs_seed100.npy"))
print(L.fmt_row("original", L.summarize(base)["ALL"]))
for i, a in enumerate(sys.argv[3:]):
    spec = eval(a); name = spec.pop("name", f"C_{tag}{i}")
    p = S.build(p0, d0, d1, **spec)
    L.save_candidate(p, name, dict(family="C: composite analytic edit (head calibration + dead-neuron skip implant)", parent="original", edit=spec,
                                   layers_modified=["Dense_0", "Dense_1", "Dense_2", "Dense_3(row zeroed)"]))
    a = E.run(split, name, p); Sm = L.summarize(a, by_scen=True)
    print(f"{name}: {json.dumps(spec)}")
    print(L.fmt_row(name, Sm["ALL"])); print("      ", E.fmt_paired(E.paired(base, a)))
    print("       per-scen SR:", " ".join(f"{k.split(':')[0]}:{100*v['success']:.0f}" for k, v in Sm.items() if ":" in k), flush=True)
L.assert_original_untouched()
