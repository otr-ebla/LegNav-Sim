"""
train_multiseed.py — sequential multi-seed training launcher (resumable).

Runs the existing training scripts one after another, once per (method, pedestrian
model, training seed), each in its own checkpoint sandbox, and stores the final
checkpoint as  checkpoints/multiseed/<method>_<ped>_s<seed>.msgpack
(the names paper_comparison_eval.py discovers).

  methods : calf_ppo calf_sac calf_tqc vanilla
            (TAGD is NOT retrained: the paper uses the authors' released weights as is)
  ped     : nsg  (leg-pair pedestrian model, USE_LEGS=1 — the default of all trainers)
            disc (disc/cylinder pedestrians, USE_LEGS=0)

Suites
  main     : 4 methods x nsg x seeds                                  (paper table)
  ablation : vanilla + calf_ppo, each x {nsg,disc}  (2x2)
  all      : main + ablation, de-duplicated (the nsg ablation cells ARE main runs)

A run is DONE only if its final checkpoint exists in checkpoints/multiseed/, so an
interrupted launcher can simply be restarted. Hyper-parameters are the unmodified
defaults of each training script (env vars only select seed/ped/output dir).

    2drlenv/bin/python -m legnav.algorithms.train_multiseed --suite all --dry-run
    2drlenv/bin/python -m legnav.algorithms.train_multiseed --suite all

Other robot footprints use their own config + output dir (never mixed with the paper runs), e.g. TIAGo OMNI:
    LEGNAV_CONFIG=configs/revision_tiago.yaml 2drlenv/bin/python -m legnav.algorithms.train_multiseed \
        --suite main --methods calf_ppo --seeds 0 --out checkpoints/multiseed_tiago
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# The launcher only checks the config (rc.verify_env imports the env modules, i.e. JAX): keep it on the
# CPU so it does not pre-allocate 75% of the GPU that the child trainer needs. Children get the original value.
_USER_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
os.environ["JAX_PLATFORMS"] = "cpu"

from legnav import runconfig as rc

# Footprint comes from the config (env.robot_footprint, default "circle") and is passed to the children.
os.environ["LEGNAV_ROBOT_FOOTPRINT"] = rc.load()["env"].get("robot_footprint", "circle")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "checkpoints" / "multiseed"
WORK = OUT / "_work"

# method -> (python module, extra CLI args(seed), final-checkpoint relpath(ped), env, label)
# Budgets, hyper-parameters, seeds and early-stop settings all come from the frozen config
# (configs/revision.yaml); this launcher only sets seed / pedestrian model / output dir.
METHODS = {
    "calf_ppo": dict(mod="legnav.algorithms.jax_ppo",
                     final=lambda ped: f"ppo/ppo_{'legs' if ped == 'nsg' else 'circles'}_final.msgpack",
                     env={}),
    "calf_sac": dict(mod="legnav.algorithms.SACjax", final=lambda ped: "sac/sac_final.msgpack", env={}),
    "calf_tqc": dict(mod="legnav.algorithms.TQCjac", final=lambda ped: "tqc/tqc_final.msgpack", env={}),
    "vanilla":  dict(mod="legnav.baselines.ppo_mlp_baseline",
                     final=lambda ped: "vanilla_ppo/ppo_mlp_final.msgpack", env={}),
}
MAIN_METHODS = ["calf_ppo", "calf_sac", "calf_tqc", "vanilla"]


def build_runs(suite, ablation_baseline, seeds, methods=None):
    runs = []                                    # (method, ped, seed), ordered, de-duplicated
    def add(m, p, s):
        if (m, p, s) not in runs and (methods is None or m in methods): runs.append((m, p, s))
    if suite in ("main", "all"):
        for m in MAIN_METHODS:
            for s in seeds: add(m, "nsg", s)
    if suite in ("ablation", "all"):
        base = {"vanilla": ["vanilla"]}[ablation_baseline]
        for m in base + ["calf_ppo"]:
            for p in ("nsg", "disc"):
                for s in seeds: add(m, p, s)
    return runs


SIDECARS = {"run_config.yaml": "{n}_config.yaml", "run_meta.json": "{n}_meta.json", "code_diff.patch": "{n}_code.patch"}


def is_done(name):
    """Done = final checkpoint exists AND its sidecar was produced with the CURRENT config and code."""
    target, meta_p = OUT / f"{name}.msgpack", OUT / f"{name}_meta.json"
    if not target.exists():
        return False, "no checkpoint"
    if not meta_p.exists():
        return False, "no sidecar (old run)"
    import json
    m = json.load(open(meta_p)); ci = rc.code_identity()
    if m.get("config_hash") != rc.config_hash():
        return False, "config changed since this run"
    if (m.get("commit"), m.get("diff_hash")) != (ci["commit"], ci["diff_hash"]):
        return False, "code changed since this run (commit/diff differ)"
    if "final_env_steps" not in m:
        return False, "run did not finish cleanly"
    return True, ""


def run_one(m, ped, seed, dry, early_stop=True):
    name = f"{m}_{ped}_s{seed}"
    ok, why = is_done(name)
    if ok:
        print(f"[skip] {name} (done, config+code match)"); return True
    if (OUT / f"{name}.msgpack").exists():
        print(f"[redo] {name}: {why}")
    spec = METHODS[m]
    work = WORK / name
    cmd = [sys.executable, "-u", "-m", spec["mod"]]
    if spec.get("seed_arg"): cmd += ["--seed", str(seed)]
    env = dict(os.environ, LEGNAV_CKPT_ROOT=str(work), LEGNAV_SEED=str(seed), LEGNAV_RUN_NAME=name,
               LEGNAV_USE_LEGS="1" if ped == "nsg" else "0", LEGNAV_ALLOW_ANY_HOST="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""), **spec["env"])
    env.pop("JAX_PLATFORMS", None)
    if _USER_JAX_PLATFORMS is not None: env["JAX_PLATFORMS"] = _USER_JAX_PLATFORMS
    if not early_stop: env["LEGNAV_EARLY_STOP"] = "0"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") +
                        " --xla_gpu_experimental_enable_fusion_autotuner=false").strip()
    print(f"[run ] {name}: {' '.join(cmd)}  (ckpt root {work})", flush=True)
    if dry: return True
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(work / "train.log", "w") as lf:
        rcode = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    final = work / spec["final"](ped)
    if rcode != 0 or not final.exists():
        print(f"[FAIL] {name}: rc={rcode}, final ckpt present={final.exists()} (see {work/'train.log'})")
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    src = final.parent
    best = final.with_name(final.name.replace("_final", "_best"))
    if best.exists(): os.replace(best, OUT / f"{name}_best.msgpack")
    curve = src / f"{name}_curve.csv"
    if curve.exists(): os.replace(curve, OUT / f"{name}_curve.csv")
    for k, v in SIDECARS.items():
        if (src / k).exists(): os.replace(src / k, OUT / v.format(n=name))
    os.replace(final, OUT / f"{name}.msgpack")   # last: its presence + sidecars mark completion; rename, never copy (NFS)
    print(f"[done] {name} in {(time.time()-t0)/60:.1f} min -> {OUT / (name + '.msgpack')}", flush=True)
    return True


def main():
    cfg = rc.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["main", "ablation", "all"], default="all")
    ap.add_argument("--ablation-baseline", choices=["vanilla"], default=cfg["launch"]["ablation_baseline"])
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["launch"]["seeds"])
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS), help="restrict to these methods")
    ap.add_argument("--no-early-stop", action="store_true", help="always train the full budget")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (default checkpoints/multiseed)")
    a = ap.parse_args()
    if a.out:
        global OUT, WORK
        OUT = Path(a.out).resolve(); WORK = OUT / "_work"
    if OUT == REPO / "checkpoints" / "multiseed" and os.environ["LEGNAV_ROBOT_FOOTPRINT"] != "circle":
        sys.exit("non-default robot footprint: pass --out (keep checkpoints/multiseed for the paper runs)")
    rc.verify_env()
    runs = build_runs(a.suite, a.ablation_baseline, a.seeds, a.methods)
    todo = [r for r in runs if not is_done(f"{r[0]}_{r[1]}_s{r[2]}")[0]]
    print(f"{len(runs)} runs in suite, {len(todo)} still to do (config hash {rc.config_hash()})\n")
    failed = [r for r in runs if not run_one(*r, a.dry_run, not a.no_early_stop)]
    if failed: print("FAILED:", failed); sys.exit(1)


if __name__ == "__main__":
    main()
