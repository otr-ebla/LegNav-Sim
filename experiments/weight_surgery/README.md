# Weight-space programming of the CALF PPO Legs policy

Original (never modified): `checkpoints/ppo/ppo_legs_best.msgpack`, sha256 `4b1dec6b…fe6568` (see `ORIGINAL_SHA256.txt`; re-verified after every run).
`ppo_legs_final.msgpack` in the same folder is a 5-update aborted run (≈ random weights) and was not used.

## Result (held-out test seeds, 18,200 paired episodes per policy, 13 scenarios)

| policy | success | collision | timeout | reward | nav time | SPL | ang. jerk | status |
|---|---|---|---|---|---|---|---|---|
| original | 95.80 % | 3.95 % | 0.26 % | 48.07 | 18.5 s | 0.871 | 4.14 | baseline |
| **C_c2_4** (primary pick, chosen on dev before test) | **96.90 %** | 2.74 % | 0.36 % | **49.21** | 19.6 s | 0.889 | 3.64 | validated |
| C_c3_1 | 96.95 % | 2.55 % | 0.49 % | 48.29 | 20.6 s | 0.891 | 3.39 | validated |
| C_c2_6 | 96.82 % | 2.43 % | 0.75 % | 47.64 | 20.0 s | 0.888 | 3.42 | validated (reward slightly lower) |
| A_headcal_bv-0.7_gw1.15 | 96.95 % | 2.37 % | 0.68 % | 48.33 | 19.8 s | 0.893 | 4.55 | validated (less smooth) |
| B_lowpassW_a0.2 | 95.52 % | 4.20 % | 0.27 % | 47.75 | 18.7 s | 0.865 | 3.27 | smoother only; not better |

Full table with confidence intervals and per-scenario numbers: `results/RANKING_test.md`. Screening of all 37 candidates + 8 probes: `results/RANKING_s1.md`.
Checkpoints of the validated candidates: `best/`. Every candidate: `candidates/<name>.msgpack` + `.json` (edit, parent, sha256).
Costs of the winners: ~1 s longer episodes (6 %), slightly more timeouts, Parallel scenario about 1 pp lower.

## What the primary pick changes (C_c2_4)

Only 1,434 of 243,909 weights differ from the original (Dense_0/1/2, one zeroed critic row). Same file format; loads with the repo loaders, e.g.
`python -m legnav.evaluation.jax_eval_multi --algo ppo --ckpt experiments/weight_surgery/best/C_c2_4.msgpack`

    raw_v ← raw_v + 0.3 − 1.2·v_prev − 0.6·vmax_norm      (throttle: unchanged when starting from rest, more cautious while moving / at high speed limits)
    raw_w ← 0.8·1.15·raw_w + 0.2·w_prev                    (slightly stronger steering, low-pass filtered)

The terms in `v_prev`, `vmax_norm`, `w_prev` do not exist in the original head; they are an exact linear skip connection from the
kinematic inputs to the action, implanted by re-wiring two dead ReLU units in Dense_0 and two in Dense_1 as always-on carriers
(`ws_surgery.implant_skip`; verified error 6e-4 against the analytic formula).

## What helped / what failed

Helped
- Lower speed-head bias ("slow down when hesitant"): the single most effective knob; fixes U-turn, long-corridor and intersection collisions.
- Steering gain 1.15–1.25: small extra gain in success.
- Skip implant on `v_prev` (and `vmax_norm`): keeps the safety gain while removing most freezing/timeouts and raising reward.
- Steering low-pass implant: −12 … −20 % angular jerk at no cost when combined with the above.

Failed or neutral
- Removing the measured left-turn bias (mean w = +0.033): success −0.85 pp. Reverse sign: neutral.
- Speed gain changes, positive speed bias, steering gain 0.75 or ≥ 1.3: worse.
- Bias −1.2 / −1.8 alone: fewest collisions (1.3–1.6 %) but 2–5 % timeouts (freezing; 0.2 m/s runs out of time).
- Extra goal-bearing steering term: better SPL, more collisions.
- Low-pass alone: smoother, not safer (test: −0.27 pp success, n.s.).
- Left/right symmetrised policy (probe): smoother, slower, same success.
- LiDAR front-gap braking shield (probe): collisions 3.4 → 1.0 % but 7 % deadlocks; with a creep floor it drives into door frames in Sequential Rooms.
- DAgger distillation of that shield into the actor weights: the network learns it (imitation error 0.32 → 0.07) and inherits its flaw: −2.0 pp success.

Not attempted: PPO fine-tuning, scenarios 13–17, real-robot checks.

## Protocol
- `ws_lib.evaluate`: deterministic policy, ghost_prob 0, sensor noise on; scenarios 0–6 with v_max ∈ {0.2…2.0}, scenarios 7–12 multi-waypoint as in `test_scenarios_eval.py`. Every policy sees identical initial conditions and noise keys (paired).
- Seeds: screening `s1`=101 (3,640 ep.), dev = 101–103 (10,920 ep.), test = 9001–9005 (18,200 ep.). Test seeds were run once, after finalists were fixed (`results/PREREGISTRATION.txt`).
- CIs: paired bootstrap over episodes.

## Files
`ws_lib.py` eval harness · `ws_surgery.py` edits/implants · `ws_teachers.py`, `ws_distill.py` probes + DAgger · `exp0*.py` experiments ·
`evalcand.py`, `make_report.py`, `make_viz.py` · `logs/` full history · `results/*.npz` per-episode data · `figures/` plots and GIF · `analyze_original.py` + `results/original_analysis.json` weight/activation analysis.
