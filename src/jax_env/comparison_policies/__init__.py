"""
comparison_policies — Baseline policies for benchmarking against PPO / SAC / TQC.

Model-based (zero-shot, no training):
  - DWA  : Dynamic Window Approach
  - MPPI : Model Predictive Path Integral Control  [TODO]

RL-based (JAX reimplementations, same SharedEncoder):
  - A2C  : Advantage Actor-Critic                  [TODO]
  - TD3  : Twin Delayed Deep Deterministic PG       [TODO]
"""

from comparison_policies.dwa_planner import DWA
from comparison_policies.mppi_planner import MPPI
from comparison_policies.vanilla_mlp_network import VanillaMLPActorCritic

__all__ = ["DWA", "MPPI", "VanillaMLPActorCritic"]

