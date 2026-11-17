"""
runconfig.py — frozen run configuration (configs/revision.yaml).

* `load()`        parsed config (path: $LEGNAV_CONFIG or configs/revision.yaml), cached.
* `verify_env()`  aborts if the environment / observation / reward constants in the code differ from
                  the `env_expect` block of the config (this is how the env section is enforced: the
                  env module constants stay in code, the config documents and guards them).
* `write_sidecar` saves the effective config + code identity (commit, dirty flag, diff hash) next to a run.
"""
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "configs" / "revision.yaml"
_CACHE = {}


def path():
    return Path(os.environ.get("LEGNAV_CONFIG") or DEFAULT_PATH)


def load():
    p = str(path())
    if p not in _CACHE:
        with open(p) as f:
            _CACHE[p] = yaml.safe_load(f)
    return _CACHE[p]


def section(name):
    return load()[name]


def config_hash():
    return hashlib.sha256(json.dumps(load(), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def code_identity():
    """HEAD commit, dirty flag and a hash of the uncommitted diff (the commit alone does not identify
    this code while changes are uncommitted)."""
    diff = _git("diff", "HEAD", "--", "legnav", "configs")
    for f in sorted(_git("ls-files", "--others", "--exclude-standard", "--", "legnav", "configs").split()):
        if f.endswith((".pyc", ".pyo")) or "__pycache__" in f: continue
        try:   # untracked files are part of the code identity too (git diff does not show them)
            diff += f"\n### UNTRACKED {f}\n" + (REPO / f).read_text(errors="replace")
        except Exception:
            pass
    return {"commit": _git("rev-parse", "HEAD").strip(), "dirty": bool(diff.strip()),
            "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:16], "_diff": diff}


def verify_env(effective_use_legs=None):
    """Raise RuntimeError if code constants differ from config['env_expect']."""
    bad = []
    for key, want in load()["env_expect"].items():
        mod, attr = key.split(":")
        try:
            holder = importlib.import_module(mod)
        except ModuleNotFoundError:               # "pkg.module.Class:ATTR"
            m, cls = mod.rsplit(".", 1)
            holder = getattr(importlib.import_module(m), cls)
        got = getattr(holder, attr)
        got = float(got) if hasattr(got, "item") else got
        if isinstance(want, bool) or isinstance(got, bool):
            ok = bool(got) == bool(want)
        else:
            ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
        if not ok: bad.append(f"{key}: code={got} config={want}")
    if bad:
        raise RuntimeError("Code differs from the frozen config:\n  " + "\n  ".join(bad))
    if effective_use_legs is not None:
        want = load()["env"]["use_legs"]
        os_val = os.environ.get("LEGNAV_USE_LEGS")
        if os_val is None and bool(effective_use_legs) != bool(want):
            raise RuntimeError("USE_LEGS differs from config.env.use_legs")


def write_sidecar(out_dir, method, ped, seed, extra=None):
    """Write run_config.yaml (full effective config) + run_meta.json (+ code diff) into out_dir."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ci = code_identity()
    diff = ci.pop("_diff")
    cfg = dict(load())
    cfg["effective"] = {"method": method, "pedestrian_model": ped, "seed": seed,
                        "use_legs": os.environ.get("LEGNAV_USE_LEGS", "1") != "0",
                        "total_steps_override": os.environ.get("LEGNAV_TOTAL_STEPS"),
                        "early_stop_override": os.environ.get("LEGNAV_EARLY_STOP")}
    with open(out / "run_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {"method": method, "pedestrian_model": ped, "seed": seed, "config_hash": config_hash(),
            "config_path": str(path()), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **ci, **(extra or {})}
    with open(out / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if ci["dirty"]:
        with open(out / "code_diff.patch", "w") as f:
            f.write(diff)
    return meta
