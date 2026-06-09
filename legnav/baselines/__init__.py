"""
comparison_policies — Baseline policies for benchmarking against PPO / SAC / TQC.

Model-based (zero-shot, no training):
  - DWA        : Dynamic Window Approach
  - HumanPilot : JHSFM Social Force Model toward goal
  - MPPI       : Model Predictive Path Integral Control  [TODO]

RL-based (JAX reimplementations, same SharedEncoder):
  - A2C  : Advantage Actor-Critic                  [TODO]
  - TD3  : Twin Delayed Deep Deterministic PG       [TODO]
"""

from legnav.baselines.dwa_planner import DWA
from legnav.baselines.jhsfm_planner import HumanPilot
from legnav.baselines.mppi_planner import MPPI
from legnav.baselines.vanilla_mlp_network import VanillaMLPActorCritic

__all__ = ["DWA", "HumanPilot", "MPPI", "VanillaMLPActorCritic"]

