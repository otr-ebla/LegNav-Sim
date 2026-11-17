"""
run_log.py — common training-curve logger for ALL algorithms (PPO, SAC, TQC, VanillaE2E, TAGD).

One CSV per run, x-axis = environment steps (every real env interaction counts, warm-up included),
identical logging grid for every algorithm (config: logging.interval_steps). Each row aggregates the
episodes that FINISHED since the previous row; episode statistics are carried across update/chunk
boundaries (an episode that spans two chunks is counted once, with its full return and length).

Columns
  step_nominal   k * interval_steps  (common grid: use this to average across seeds / algorithms)
  step_actual    env steps really taken when the row was written (differs per algorithm: PPO advances
                 131,072 steps per update, SAC/TQC 102,400 per chunk, TAGD n_envs per step)
  wall_s         wall-clock seconds since the run started
  phase          warmup | train
  n_episodes     finished episodes in this interval
  mean_return    mean RAW environment return per episode (never the normalised PPO reward)
  success_rate, active_col_rate, passive_col_rate, static_col_rate, timeout_rate   (sum to 1)
                 outcome precedence follows the env reward cascade: goal > static(obstacle|wall) > active > passive > timeout
  mean_ep_len    mean episode length in env steps
  rew_<term>     per-episode SUM of each reward term, averaged over episodes; the terms add up to mean_return
  terms_sum_gap  mean_return - sum(rew_*)  (should be ~0: consistency check)
  cur_*          curriculum state (goal distance, ghost prob, max scenario) when provided by the trainer

Also writes run_config.yaml / run_meta.json (+ code_diff.patch if the tree is dirty) next to the curve, and
on close() records the stop step / reason, so cross-seed aggregation can handle runs that stopped early.
"""
import csv
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legnav import paths, runconfig as rc

TERMS = ["progress", "step", "smooth", "rot", "stop", "heading", "yield", "terminal"]
K = len(TERMS)
_N_FIX = 8          # n_ep, sum_ret, sum_len, goal, static, active, passive, timeout


@jax.jit
def _accumulate(carry, rew, done, goal, static, active, passive, terms):
    """rew/done/goal/static/active/passive: (T,N); terms: (T,N,K). carry=(ret(N),len(N),run_terms(N,K),acc(8+K))."""
    def step(c, x):
        ret, ln, rt, acc = c
        r, d, g, s, a, p, t = x
        ret = ret + r; ln = ln + 1.0; rt = rt + t
        tm = d & ~g & ~s & ~a & ~p
        gg, ss = g & d, s & d & ~g
        aa = a & d & ~g & ~s
        pp = p & d & ~g & ~s & ~a
        df = d.astype(jnp.float32)
        add = jnp.concatenate([jnp.stack([df.sum(), (df * ret).sum(), (df * ln).sum(), gg.sum().astype(jnp.float32), ss.sum().astype(jnp.float32),
                                aa.sum().astype(jnp.float32), pp.sum().astype(jnp.float32), tm.sum().astype(jnp.float32)]),
                               (df[:, None] * rt).sum(0)])
        keep = 1.0 - df
        return (ret * keep, ln * keep, rt * keep[:, None], acc + add.astype(jnp.float32)), None
    out, _ = jax.lax.scan(step, carry, (rew, done, goal, static, active, passive, terms))
    return out


class RunLogger:
    def __init__(self, method, ped, seed, n_envs, out_dir, interval=None, extra_meta=None):
        cfg = rc.load()
        self.interval = int(interval or cfg["logging"]["interval_steps"])
        self.name = os.environ.get("LEGNAV_RUN_NAME") or f"{method}_{ped}_s{seed}"
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.name}_curve.csv"
        self.n_envs = n_envs
        self.carry = (jnp.zeros(n_envs), jnp.zeros(n_envs), jnp.zeros((n_envs, K)), jnp.zeros(_N_FIX + K))
        self.phase = "train"
        self.t0 = time.time()
        self.next_b = self.interval
        self.steps = 0
        self.meta = rc.write_sidecar(self.dir, method, ped, seed, extra_meta)
        self.meta_path = self.dir / "run_meta.json"
        self._f = open(self.path, "w", newline="")
        self._w = None
        self.last_extra = {}

    # ---- feeding -------------------------------------------------------------------------------
    def update(self, rew, done, goal, static, active, passive, terms, n_steps, extra=None):
        """Arrays are (T,N[,K]) (device or numpy). n_steps = env steps this call represents (T*N)."""
        asf = lambda x: jnp.asarray(x)
        self.carry = _accumulate(self.carry, asf(rew).astype(jnp.float32), asf(done).astype(bool), asf(goal).astype(bool),
                                 asf(static).astype(bool), asf(active).astype(bool), asf(passive).astype(bool),
                                 asf(terms).astype(jnp.float32))
        self.steps += int(n_steps)
        if extra: self.last_extra = extra
        if self.steps >= self.next_b:
            self._flush(self.next_b)
            while self.next_b <= self.steps: self.next_b += self.interval

    def add_steps_only(self, n_steps):
        """Count env steps that produce no episode statistics (not normally needed)."""
        self.steps += int(n_steps)

    def set_phase(self, phase):
        if phase != self.phase:
            self._flush(None)                      # close the previous phase with a partial row
            self.phase = phase

    # ---- output --------------------------------------------------------------------------------
    def _flush(self, nominal):
        acc = np.asarray(jax.device_get(self.carry[3]), dtype=np.float64)
        n = acc[0]
        row = {"step_nominal": nominal if nominal is not None else -1, "step_actual": self.steps,
               "wall_s": round(time.time() - self.t0, 2), "phase": self.phase, "n_episodes": int(n)}
        if n > 0:
            row.update(mean_return=acc[1] / n, success_rate=acc[3] / n, static_col_rate=acc[4] / n, active_col_rate=acc[5] / n,
                       passive_col_rate=acc[6] / n, timeout_rate=acc[7] / n, mean_ep_len=acc[2] / n)
            for i, t in enumerate(TERMS): row[f"rew_{t}"] = acc[_N_FIX + i] / n
            row["terms_sum_gap"] = row["mean_return"] - sum(row[f"rew_{t}"] for t in TERMS)
        else:
            row.update({k: float("nan") for k in ["mean_return", "success_rate", "static_col_rate", "active_col_rate", "passive_col_rate",
                                                   "timeout_rate", "mean_ep_len", "terms_sum_gap"]})
            for t in TERMS: row[f"rew_{t}"] = float("nan")
        for k, v in self.last_extra.items(): row[f"cur_{k}"] = v
        if self._w is None:
            self._w = csv.DictWriter(self._f, fieldnames=list(row.keys())); self._w.writeheader()
        self._w.writerow(row); self._f.flush()
        # reset only the interval accumulator (per-env running sums persist across the boundary)
        self.carry = (self.carry[0], self.carry[1], self.carry[2], jnp.zeros(_N_FIX + K))

    def close(self, stopped_early=False, reason=""):
        self._flush(None)
        self._f.close()
        meta = json.load(open(self.meta_path))
        meta.update({"final_env_steps": self.steps, "wall_s_total": round(time.time() - self.t0, 1),
                     "stopped_early": bool(stopped_early), "stop_reason": reason, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        json.dump(meta, open(self.meta_path, "w"), indent=1)
