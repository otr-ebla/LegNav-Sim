"""Evaluate one or more checkpoints on a seed set, paired against the original.
usage: evalcand.py {dev|test} name=path [name=path ...]   (name 'original' is implicit)"""
import sys, os, json, time, numpy as np
import ws_lib as L
SEEDS = {"dev": [101, 102, 103], "test": [9001, 9002, 9003, 9004, 9005], "smoke": [7], "s1": [101], "s2": [102], "s3": [103]}
NPS = {"dev": 280, "test": 280, "smoke": 70, "s1": 280, "s2": 280, "s3": 280}


def run(split, name, params, force=False, chunk_fn=None):
    out = os.path.join(L.RES_DIR, f"{split}__{name}.npz")
    if os.path.exists(out) and not force:
        return dict(np.load(out))
    if split == "dev":   # dev = s1+s2+s3 (reuse per-seed caches)
        parts = [run(s, name, params, force, chunk_fn) for s in ("s1", "s2", "s3")]
        a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez_compressed(out, **a)
        return a
    t = time.time()
    a = L.evaluate_seeds(params, SEEDS[split], n_per_scen=NPS[split], chunk_fn=chunk_fn)
    np.savez_compressed(out, **a)
    print(f"[{split}] {name}: {len(a['scen'])} episodes in {time.time()-t:.0f}s", flush=True)
    return a


def paired(a, b, nboot=2000, seed=0):
    """b - a paired differences with bootstrap 95% CI (episodes are paired by seed/index)."""
    ma, mb = L.episode_metrics(a), L.episode_metrics(b)
    rng = np.random.default_rng(seed); n = len(a["scen"]); idx = rng.integers(0, n, (nboot, n))
    r = {}
    for k in ["success", "collision", "timeout", "reward", "spl", "dw", "ang_jerk", "abs_w"]:
        d = mb[k] - ma[k]; bs = d[idx].mean(1)
        r[k] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return r


def fmt_paired(r):
    s = []
    for k, sc in [("success", 100), ("collision", 100), ("reward", 1), ("spl", 1), ("dw", 1)]:
        m, lo, hi = r[k]; star = "*" if (lo > 0 or hi < 0) else " "
        s.append(f"Δ{k}={m*sc:+.3f}[{lo*sc:+.3f},{hi*sc:+.3f}]{star}")
    return "  ".join(s)


if __name__ == "__main__":
    split = sys.argv[1]
    L.assert_original_untouched()
    base = run(split, "original", L.load_params())
    print(L.fmt_row("original", L.summarize(base)["ALL"]))
    for arg in sys.argv[2:]:
        name, path = arg.split("=")
        a = run(split, name, L.load_params(path))
        S = L.summarize(a)
        print(L.fmt_row(name, S["ALL"])); print("    paired vs original:", fmt_paired(paired(base, a)), flush=True)
    L.assert_original_untouched()
