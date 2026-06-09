# LegNav-Sim

A fast, JAX-native 2D LiDAR simulation for training Deep RL agents to navigate a
mobile robot through dynamic human crowds — modelled down to individual **legs and
feet** for realistic LiDAR returns.

<img src="assets/indoor_gif.gif" alt="Environment Demo" width="70%"/>

## Overview

**LegNav-Sim** solves dynamic-environment navigation from sparse sensor data. The
agent perceives the world through a simulated 360° 2D LiDAR (ray-casts) and must
reach target coordinates while avoiding static geometry (walls, obstacles) and
moving pedestrians driven by a Headed Social Force Model (HSFM).

The whole simulation — physics, LiDAR, crowd dynamics and the training loop — is
written in **JAX**, so thousands of environments step in parallel on the GPU. The
repository ships PPO, SAC and TQC implementations plus a suite of classical and
learned baselines (DWA, MPPI, HSFM planner, NavRep, TAGD, vanilla-MLP PPO) for
paper-grade comparison.

## Key Features

- **Leg-level LiDAR perception** — pedestrians are simulated as moving legs/feet,
  producing realistic sparse 2D scans (configurable ray count, default 216).
- **Dynamic crowds** — pedestrians follow an HSFM social-force model, forcing
  predictive rather than memorized navigation.
- **Massively parallel** — fully vectorized JAX environment (`legnav.core`) for
  high-throughput on-GPU training.
- **Batteries included** — PPO / SAC / TQC trainers and a baseline zoo for
  benchmarking and reproduction.

## Repository Structure

```text
.
├── legnav/                 # Core installable package
│   ├── config.py           # Global robot / sim / LiDAR configuration
│   ├── paths.py            # Central path resolver (checkpoints / data / figures)
│   ├── core/               # JAX simulation engine (env, physics, humans, legs, scenarios, network)
│   ├── algorithms/         # Trainers: PPO, SAC, TQC, TAGD-DDPG
│   ├── evaluation/         # Single/multi-env eval, benchmarks, paper comparisons
│   ├── baselines/          # Comparison policies: DWA, MPPI, HSFM, NavRep, TAGD, vanilla-MLP
│   ├── plotting/           # Plot & scenario-visualization scripts
│   ├── deployment/         # Real-robot (TurtleBot4) inference scripts
│   ├── dreamer/            # Experimental world-model agent
│   └── jhsfm/              # Vendored JHSFM social-force utilities
├── checkpoints/            # Trained weights, one subdir per algorithm (.msgpack)
│   ├── ppo/  sac/  tqc/  tagd/  navrep/  vanilla_ppo/
├── data/                   # Generated CSV result dumps (git-ignored)
├── figures/                # Generated plots & dashboards
├── assets/                 # README visuals
├── legacy/                 # Older, non-JAX code (Gymnasium/SB3) — kept for reference
└── requirements.txt
```

## Getting Started

Create a virtual environment and install dependencies:

```bash
python3 -m venv 2drlenv
source 2drlenv/bin/activate
pip install -r requirements.txt
pip install -e .            # install the `legnav` package
```

All commands below are run from the repository root. Paths to checkpoints, data
and figures are resolved automatically (via `legnav.paths`), so scripts work
regardless of the current working directory.

Train an agent (PPO / SAC / TQC):

```bash
python -m legnav.algorithms.jax_ppo      # PPO
python -m legnav.algorithms.SACjax       # SAC
python -m legnav.algorithms.TQCjac       # TQC
```

Evaluate / visualize a trained policy:

```bash
python -m legnav.evaluation.jax_eval_multi --algo sac
python -m legnav.evaluation.benchmark_eval
```

> **GPU required** for training and most evaluation scripts — they request the
> JAX `gpu` backend at import time.

## Legacy code

The `legacy/` directory holds the original, pre-JAX implementation
(Gymnasium + stable-baselines3 environments and trainers, real-robot SB3 scripts,
and plotting scratch). It is retained for reference and is **not** part of the
`legnav` package.
