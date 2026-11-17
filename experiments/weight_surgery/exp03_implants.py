"""EXP-03  Circuit implants: exact linear skip g_T→raw-action through re-purposed dead neurons."""
import sys, numpy as np, ws_lib as L, evalcand as E, ws_surgery as S
split = sys.argv[1]
p0 = L.load_params(); base = E.run(split, "original", p0)
obs = np.load("results/onpolicy_obs_seed100.npy")
d0, d1 = S.dead_units(p0, obs); print("dead units Dense_0:", d0, " Dense_1:", d1)
# exactness check: zero implant == original ; implant adds exactly A·g
A = np.zeros((2, 11), np.float32); A[1, 1] = 0.37; A[0, 2] = -0.21
q = S.implant_skip(p0, A, d0, d1)
r0, r1 = L.policy_feats(p0, obs[:20000])["raw"], L.policy_feats(q, obs[:20000])["raw"]
g = obs[:20000, :11]
print("implant exactness: max|raw_new-(raw_old+A·g)| =", float(np.abs(r1 - (r0 + g @ A.T)).max()))
cands = {}
for al in (0.2, 0.4):
    cands[f"B_lowpassW_a{al}"] = (S.lowpass_w(p0, al, d0, d1), dict(edit=f"raw_w<-(1-a)raw_w+a*w_prev, a={al}"))
for bta in (0.5, -0.5):   # extra proportional steering on goal bearing
    A = np.zeros((2, 11), np.float32); A[1, 4] = bta
    cands[f"B_goalP_b{bta}"] = (S.implant_skip(p0, A, d0, d1), dict(edit=f"raw_w += {bta}*galign/pi"))
for gm in (-0.6, -1.2):   # speed-dependent caution: less throttle at high v_max
    A = np.zeros((2, 11), np.float32); A[0, 2] = gm
    cands[f"B_vmaxCaution_g{gm}"] = (S.implant_skip(p0, A, d0, d1), dict(edit=f"raw_v += {gm}*vmax_norm"))
for name, (p, meta) in cands.items():
    L.save_candidate(p, name, dict(family="B: dead-neuron skip implant", parent="original", carriers=dict(dense0=[int(x) for x in d0[:2]], dense1=[int(x) for x in d1[:2]]),
                                   layers_modified=["Dense_0", "Dense_1", "Dense_2", "Dense_3(row zeroed)"], **meta))
    a = E.run(split, name, p)
    print(L.fmt_row(name, L.summarize(a)["ALL"])); print("      ", E.fmt_paired(E.paired(base, a)), flush=True)
L.assert_original_untouched()
