"""EXP-04  Program a LiDAR front-gap braking reflex into the weights by DAgger distillation.
usage: exp04_distill.py SPLIT tau,hw,floor SUBSET ITERS STEPS LR [parent_ckpt]"""
import sys, numpy as np, ws_lib as L, evalcand as E, ws_teachers as T, ws_distill as D
split, spec, subset, iters, steps, lr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
tau, hw, fl = map(float, spec.split(","))
p0 = L.load_params(); base = E.run(split, "original", p0)
name = f"D_distill_shield_t{tau}_hw{hw}_f{fl}_{subset}_i{iters}_s{steps}_lr{lr}"
teacher = T.make_shield(tau, hw, False, fl)
print("==", name, flush=True)
student = D.dagger(p0, teacher, subset, iters=iters, steps=steps, lr=lr, log=lambda s: print(s, flush=True))
L.save_candidate(student, name, dict(family="D: behaviour distillation (DAgger) of an obs-only LiDAR braking shield",
    parent="original", teacher=dict(kind="front-gap speed cap", tau=tau, half_width=hw, v_floor=fl),
    layers_modified=D.SUBSETS[subset], dagger_iters=iters, fit_steps_per_iter=steps, lr=lr))
a = E.run(split, name, student)
S = L.summarize(a, by_scen=True)
print(L.fmt_row(name[:34], S["ALL"])); print("      ", E.fmt_paired(E.paired(base, a)))
print("       per-scen SR:", " ".join(f"{k.split(':')[0]}:{100*v['success']:.0f}" for k, v in S.items() if ":" in k), flush=True)
L.assert_original_untouched()
