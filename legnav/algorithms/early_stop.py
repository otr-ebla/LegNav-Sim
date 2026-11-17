"""
early_stop.py — plateau-based early stopping shared by all training scripts.

The same rule is used for PPO / SAC / TQC / VanillaE2E / TAGD, expressed as
FRACTIONS of each run's env-step budget so that runs with different budgets (and
different algorithms) are stopped on a comparable scale.

Signal: exponentially smoothed *training success rate* (%). The curriculum raises
difficulty as success grows, so a flat success curve means the policy has stopped
improving at the current task difficulty, not that the task became trivial.

Rule: stop when, after `min_frac` of the budget, the smoothed success has not
beaten its previous best by at least `min_delta` percentage points for
`patience_frac` of the budget.

Parameters (min_frac, patience_frac, min_delta, smooth_frac) and the on/off switch live in the frozen
config (configs/revision.yaml: early_stop). LEGNAV_EARLY_STOP=0|1 overrides `enabled`.
NOTE for the cross-seed figure: seeds that stop early end at different steps; record is in run_meta.json.
"""
import math
import os


class EarlyStopper:
    def __init__(self, total_steps, min_frac=0.30, patience_frac=0.20,
                 min_delta=1.0, smooth_frac=0.03):
        self.total = float(total_steps)
        self.min_steps = min_frac * self.total
        self.patience = patience_frac * self.total
        self.min_delta = min_delta
        self.tau = max(smooth_frac * self.total, 1.0)
        self.ema = None
        self.last_step = 0
        self.best = -math.inf
        self.best_step = 0

    @classmethod
    def from_env(cls, total_steps):
        """Return an EarlyStopper, or None when disabled. Parameters come from the frozen config
        (configs/revision.yaml: early_stop); LEGNAV_EARLY_STOP=0/1 overrides `enabled` (e.g. --no-early-stop)."""
        from legnav import runconfig as rc
        c = rc.load()["early_stop"]
        on = os.environ.get("LEGNAV_EARLY_STOP")
        enabled = (on == "1") if on is not None else bool(c["enabled"])
        if not enabled:
            return None
        return cls(total_steps, c["min_frac"], c["patience_frac"], c["min_delta"], c["smooth_frac"])

    def update(self, step, success_pct):
        """Feed (env steps so far, success %). Returns True when training should stop."""
        if self.ema is None:
            self.ema = success_pct
        else:
            a = 1.0 - math.exp(-max(step - self.last_step, 0) / self.tau)
            self.ema += a * (success_pct - self.ema)
        self.last_step = step
        if self.ema > self.best + self.min_delta:
            self.best, self.best_step = self.ema, step
        return step >= self.min_steps and (step - self.best_step) >= self.patience

    def reason(self, step):
        return (f"early stop at {step:,} steps: smoothed success {self.ema:.1f}% has not improved by "
                f">{self.min_delta:g}pp since step {self.best_step:,} (best {self.best:.1f}%)")
