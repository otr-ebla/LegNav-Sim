"""
perception/eval_perception.py — Detection & tracking evaluation
================================================================
Loads a trained PerceptionNet checkpoint, runs stationary-robot episodes and
reports person-level metrics: detections are decoded from the per-ray votes
and greedily associated to *visible* ground-truth people within ASSOC_DIST.

  precision / recall / F1   — person-level detection quality
  position RMSE             — localisation error of matched detections (m)

Also saves a qualitative figure (scan + GT people + detections) to figures/.

Run from the repo root:
    python -m perception.eval_perception [--ckpt path] [--episodes N]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from legnav import paths
from perception.config import PerceptionConfig as PC
from perception.perception_env import reset_perception, step_perception
from perception.perception_network import PerceptionNet, decode_detections
from perception.train_perception import load_checkpoint, CKPT_BEST


def _greedy_match(dets, gts):
    """Greedy nearest association within ASSOC_DIST. Returns matched index pairs."""
    pairs = []
    used_d, used_g = set(), set()
    if len(dets) and len(gts):
        cost = np.linalg.norm(dets[:, None, :] - gts[None, :, :], axis=-1)
        for _ in range(min(len(dets), len(gts))):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[i, j] > PC.ASSOC_DIST:
                break
            pairs.append((i, j, cost[i, j]))
            used_d.add(i); used_g.add(j)
            cost[i, :] = np.inf
            cost[:, j] = np.inf
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_BEST)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    net = PerceptionNet()
    rng = jax.random.PRNGKey(args.seed)
    rng, k_init = jax.random.split(rng)
    template = net.init(k_init, jnp.zeros((1, PC.STACK_DIM, PC.NUM_RAYS)))["params"]
    params = load_checkpoint(template, args.ckpt)
    print(f"Loaded checkpoint: {args.ckpt}")

    apply_jit = jax.jit(lambda s: net.apply({"params": params}, s))
    step_jit  = jax.jit(step_perception)
    reset_jit = jax.jit(reset_perception)

    tp = fp = fn = 0
    near_tp = near_fn = 0          # GT people within NEAR_RANGE only
    NEAR_RANGE = 4.0
    sq_err, n_match = 0.0, 0
    last_snapshot = None

    for ep in range(args.episodes):
        rng, k_reset = jax.random.split(rng)
        state, gt = reset_jit(k_reset)
        for t in range(args.steps):
            rng, k = jax.random.split(rng)
            state, gt = step_jit(k, state)
            if t < PC.STACK_DIM:           # let the stack fill with real history
                continue

            logits, offsets, leg_offsets = apply_jit(state.lidar_stack)
            dets, scores = decode_detections(
                logits, offsets, leg_offsets, state.lidar_stack[-1])

            visible = np.asarray(gt["visible"])
            gts = np.asarray(gt["centers"])[visible]
            pairs = _greedy_match(dets, gts)

            tp += len(pairs)
            fp += len(dets) - len(pairs)
            fn += len(gts) - len(pairs)
            for _, _, d in pairs:
                sq_err += d ** 2
                n_match += 1

            # Range-binned recall: the leg-pair decoder needs ≥2 rays per
            # leg, so distant single-ray people are unreachable by design.
            near = np.linalg.norm(gts, axis=1) < NEAR_RANGE
            matched_g = {j for _, j, _ in pairs}
            near_tp += sum(1 for j in range(len(gts)) if near[j] and j in matched_g)
            near_fn += sum(1 for j in range(len(gts)) if near[j] and j not in matched_g)

            last_snapshot = (np.asarray(state.lidar_stack[-1]), gts, dets)
        print(f"episode {ep + 1}/{args.episodes} done")

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    rmse      = np.sqrt(sq_err / max(n_match, 1))

    near_recall = near_tp / max(near_tp + near_fn, 1)

    print("\n── Perception evaluation ─────────────────────────")
    print(f"  precision      : {precision:.3f}")
    print(f"  recall         : {recall:.3f}   (all visible people)")
    print(f"  recall <{NEAR_RANGE:.0f} m   : {near_recall:.3f}")
    print(f"  F1             : {f1:.3f}")
    print(f"  position RMSE  : {rmse:.3f} m   ({n_match} matches)")

    if last_snapshot is not None:
        _save_figure(*last_snapshot)


def _save_figure(inv_scan, gts, dets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legnav.core.jax_env import ROBOT_RADIUS

    dist = PC.MAX_DIST - inv_scan * (PC.MAX_DIST - ROBOT_RADIUS)
    ang = -PC.FOV * 0.5 + np.arange(PC.NUM_RAYS) * (PC.FOV / (PC.NUM_RAYS - 1))
    pts = dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="gray", label="LiDAR")
    ax.scatter(0, 0, marker="s", c="black", s=60, label="robot (still)")
    if len(gts):
        ax.scatter(gts[:, 0], gts[:, 1], marker="o", facecolors="none",
                   edgecolors="green", s=180, label="GT people (visible)")
    if len(dets):
        ax.scatter(dets[:, 0], dets[:, 1], marker="x", c="red", s=80,
                   label="detections")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Perception module — people detection from 2D LiDAR")
    out = paths.figure("perception_eval.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  figure saved   : {out}")


if __name__ == "__main__":
    main()
