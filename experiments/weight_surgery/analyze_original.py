"""Step 3 — analysis of the ORIGINAL policy: weights, activations, behaviour."""
import json, numpy as np, jax, jax.numpy as jnp
import ws_lib as L, ws_trace as T
L.assert_original_untouched()
p = L.load_params()
rep = {}

# ---- static weight statistics -------------------------------------------------
def walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from walk(v, pre + k + "/")
        else: yield pre + k, np.asarray(v)
ws = {}
for k, v in walk(p):
    d = dict(shape=list(v.shape), n=int(v.size), mean=float(v.mean()), std=float(v.std()),
             absmax=float(np.abs(v).max()), frac_small=float((np.abs(v) < 1e-3).mean()))
    if v.ndim >= 2:
        m = v.reshape(-1, v.shape[-1]); s = np.linalg.svd(m, compute_uv=False)
        d["sv_max"] = float(s[0]); d["eff_rank"] = float((s.sum() ** 2) / (s ** 2).sum()); d["rank_full"] = int(min(m.shape))
    ws[k] = d
rep["weights"] = ws; rep["n_params"] = int(sum(d["n"] for d in ws.values()))
print("total params", rep["n_params"])
for k, d in ws.items():
    print(f"{k:45s} {str(d['shape']):16s} std={d['std']:.3f} max={d['absmax']:.2f}" + (f" svmax={d['sv_max']:.2f} effrank={d['eff_rank']:.1f}/{d['rank_full']}" if 'sv_max' in d else ""))
ls = np.asarray(p["log_std"]); print("log_std param", ls, "-> sigma", np.exp(-4 + 2 * (np.tanh(ls) + 1)))

# ---- activations on on-policy states -----------------------------------------
rec, st = T.trace(p, seed=100, n_per_scen=28, T=500)
act = rec["active"]
obs = rec["obs"][act]                      # (M, 668) on-policy states
print("on-policy states:", obs.shape)
np.save("results/onpolicy_obs_seed100.npy", obs[::3].astype(np.float32))
obs_s = obs[:: max(1, len(obs) // 40000)]
F = L.policy_feats(p, obs_s)
h0 = np.maximum(F["d0"], 0); h1 = np.maximum(F["d1"], 0); raw = F["raw"]; cnn = [F["cnn"]]
for nm, h in [("Dense_0(256)", h0), ("Dense_1(128)", h1), ("CNN-dense(64) newest", cnn[-1])]:
    fr = (h > 0).mean(0)
    print(f"{nm:22s} dead(never>0)={int((fr==0).sum())} rare(<1%)={int((fr<0.01).sum())} always-on(>99%)={int((fr>0.99).sum())} mean-act-frac={fr.mean():.2f}")
    rep[f"act_{nm}"] = dict(dead=int((fr == 0).sum()), rare=int((fr < 0.01).sum()), always=int((fr > 0.99).sum()))
np.savez("results/activations_orig.npz", h0_fr=(h0 > 0).mean(0), h1_fr=(h1 > 0).mean(0), h0_mean=h0.mean(0), h1_mean=h1.mean(0), h1_std=h1.std(0))
tv, tw = np.tanh(raw[:, 0]), np.tanh(raw[:, 1])
print(f"raw_v: mean={raw[:,0].mean():.2f} std={raw[:,0].std():.2f} | v-cmd frac: mean={(tv*.5+.5).mean():.3f}  saturated-high(>0.98)={((tv*.5+.5)>0.98).mean():.3f} saturated-low(<0.02)={((tv*.5+.5)<0.02).mean():.3f}")
print(f"raw_w: mean={raw[:,1].mean():.3f} std={raw[:,1].std():.2f} | |w|>0.95: {(np.abs(tw)>0.95).mean():.3f}  |w|<0.05: {(np.abs(tw)<0.05).mean():.3f}  mean w={tw.mean():+.4f} (left/right bias)")
rep["v_frac_mean"] = float((tv * .5 + .5).mean()); rep["w_bias"] = float(tw.mean())

# ---- left/right mirror symmetry test -----------------------------------------
def mirror(o):
    o = np.array(o); kin = o[:, :5].copy(); kin[:, 1] *= -1; kin[:, 4] *= -1
    goal = o[:, 5:11].reshape(-1, 3, 2).copy(); goal[:, :, 1] *= -1
    d = o[:, 11:20].reshape(-1, 3, 3).copy(); d[:, :, 1] *= -1; d[:, :, 2] *= -1
    lid = o[:, 20:].reshape(-1, 3, 216)[:, :, ::-1]
    return np.concatenate([kin, goal.reshape(-1, 6), d.reshape(-1, 9), lid.reshape(-1, 648)], 1)
m1 = L.policy_mean(p, obs_s); m2 = L.policy_mean(p, mirror(obs_s))
m1, m2 = np.asarray(m1), np.asarray(m2)
print(f"mirror asymmetry: mean|v(o)-v(Mo)|={np.abs(m1[:,0]-m2[:,0]).mean():.4f}  mean|w(o)+w(Mo)|={np.abs(m1[:,1]+m2[:,1]).mean():.4f}  (mean|w|={np.abs(m1[:,1]).mean():.3f})")
rep["mirror_v_err"] = float(np.abs(m1[:, 0] - m2[:, 0]).mean()); rep["mirror_w_err"] = float(np.abs(m1[:, 1] + m2[:, 1]).mean())

# ---- input-group sensitivity of the fused trunk ------------------------------
W0 = np.asarray(p["Dense_0"]["kernel"])
names = ["attn_tok_t-4 (0:64)", "attn_tok_t-2 (64:128)", "attn_tok_t (128:192)", "v", "w_prev", "max_v", "goal_dist", "goal_align",
         "g(t-4)x", "g(t-4)y", "g(t-2)x", "g(t-2)y", "g(t)x", "g(t)y"]
sl = [slice(0, 64), slice(64, 128), slice(128, 192)] + [slice(192 + i, 193 + i) for i in range(11)]
print("Dense_0 input-group column norms (Frobenius):")
for n, s in zip(names, sl): print(f"   {n:24s} {np.linalg.norm(W0[s]):.3f}")
json.dump(rep, open("results/original_analysis.json", "w"), indent=1)
