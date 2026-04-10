#!/usr/bin/env python3
"""
run_all_trainings.py — Train PPO, SAC and TQC sequentially with the same
env-step budget so their training curves are directly comparable in
benchmark_eval.py.

Usage:
    python3 run_all_trainings.py                    # 20M steps each (default)
    python3 run_all_trainings.py --steps 30000000   # custom budget
    python3 run_all_trainings.py --only ppo,sac     # subset
    python3 run_all_trainings.py --skip tqc         # subset

Run from src/jax_env/ so the trainers' relative checkpoint paths
(checkpoints/, checkpoints_sac/, checkpoints_tqc/) land in the right place.
"""

import os
import sys
import gc
import time
import argparse

# JAX env vars must be set BEFORE the first `import jax`. Each trainer module
# uses os.environ.setdefault, so values set here win.
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")
os.environ.setdefault("JAX_PLATFORMS",               "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR",            "cuda_malloc_async")


# ── Editable budget ──────────────────────────────────────────────────────────
TOTAL_ENV_STEPS = 20_000_000


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Train PPO, SAC and TQC sequentially.")
    parser.add_argument("--steps", type=int, default=TOTAL_ENV_STEPS,
                        help=f"Env-step budget per algorithm (default: {TOTAL_ENV_STEPS:,}).")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated algos to skip: ppo,sac,tqc")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated algos to run (overrides --skip).")
    args = parser.parse_args()

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}

    def should_run(name: str) -> bool:
        if only:
            return name in only
        return name not in skip

    plan = [name for name in ("ppo", "sac", "tqc") if should_run(name)]
    if not plan:
        print("Nothing to run (check --only / --skip).", file=sys.stderr)
        return 1

    print("=" * 70)
    print("  Sequential RL training")
    print(f"   Budget per algo : {args.steps:,} env steps")
    print(f"   Order           : {' -> '.join(p.upper() for p in plan)}")
    print("=" * 70)

    # Single jax import — all trainers will share the same JAX runtime.
    import jax  # noqa: F401  (import after env vars)

    overall_start = time.time()
    timings: dict[str, float] = {}

    for name in plan:
        print("\n" + "=" * 70)
        print(f"  Training {name.upper()}  —  {args.steps:,} env steps")
        print("=" * 70 + "\n")
        t0 = time.time()
        if name == "ppo":
            import jax_ppo
            jax_ppo.train(total_env_steps=args.steps)
        elif name == "sac":
            import SACjax
            SACjax.train(total_env_steps=args.steps)
        elif name == "tqc":
            import TQCjac
            TQCjac.train(total_env_steps=args.steps)
        timings[name] = time.time() - t0

        # Release device memory before the next trainer instantiates its own
        # replay buffer / networks. XLA holds compiled-kernel workspace until
        # caches are explicitly cleared.
        jax.clear_caches()
        gc.collect()

    print("\n" + "=" * 70)
    print(f"  All done in {_fmt_hms(time.time() - overall_start)}")
    for name in plan:
        print(f"   {name.upper():>3}: {_fmt_hms(timings[name])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
