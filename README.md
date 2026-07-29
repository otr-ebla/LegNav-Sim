<h1>LegNav-Sim</h1>

Official implementation of

[Learning Robot Social Navigation By Sensing Human Legs](#) by

[Alberto Vaglio](#), [Andrea Garulli](#), [Antonio Giannitrapani](#), [Renato Quartullo](#), [Tommaso Van Der Meer](#) (University of Siena & Uninettuno University)</br>

<img src="assets/indoor_gif.gif" width="70%"></br>

[[Paper]](#) [[Project Page]](#) [[Models]](#)

----

## Overview

**LegNav-Sim** is a fast, JAX-native 2D LiDAR simulation for training Deep RL
agents to navigate a mobile robot through dynamic human crowds — modelled down to
individual **legs and feet** for realistic LiDAR returns. The agent perceives the
world through a simulated 360° 2D LiDAR (ray-casts) and must reach target
coordinates while avoiding static geometry (walls, obstacles) and moving
pedestrians driven by a Headed Social Force Model (HSFM).

The core motivation is a sim-to-real gap that most social-navigation simulators
ignore: ankle-height 2D LiDAR sensors (mounted 10–20 cm above the ground) never
see a pedestrian as a filled disc — they see two small, independently moving leg
clusters, with the forward-protruding shoe sitting in a blind zone below the scan
plane. LegNav models this explicitly via a **Planted-Foot Gait Model**, and trains
policies with **CALF** (Convolutional Attention Leg Features), a hybrid
CNN-Attention-MLP architecture designed to robustly process the alternating leg
signature of ankle-height scans.

The whole simulation — physics, LiDAR, crowd dynamics and the training loop — is
written in **JAX**, so thousands of environments step in parallel on the GPU
(~135k steps/s on a single RTX 3080, training a deployable policy in ~30 minutes).
The repository ships PPO, SAC and TQC implementations plus a suite of classical and
learned baselines (DWA, MPPI, HSFM planner, NavRep, TAGD, vanilla-MLP PPO) for
paper-grade comparison. The trained policy was zero-shot deployed on a real
**TurtleBot 4**, producing smooth, socially compliant trajectories without any
domain adaptation.

## Key Features

- **Planted-Foot Gait Model** — each pedestrian is simulated as two independently
  planted/swinging feet with shoe-shaped footprints, reproducing both the
  two-cluster LiDAR signature and the forward shoe blind zone of real ankle-height
  observations.
- **Leg-level LiDAR perception** — realistic sparse 2D scans (configurable ray
  count, default 216) with Gaussian and salt-and-pepper noise.
- **CALF architecture** — a weight-shared 1D-CNN encoder over stacked LiDAR frames
  followed by temporal multi-head self-attention, implicitly inferring obstacle
  motion without explicit detection or tracking.
- **Social compliance** — yielding behavior (stopping and waiting for nearby
  pedestrians) is explicitly shaped into the reward, and evaluated via a
  **Yielding Score** metric.
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

## Citation

If you use LegNav-Sim in your research, please cite our paper (arXiv preprint
coming soon):

```bibtex
@article{vaglio2026legnav,
  title   = {Learning Robot Social Navigation among Human Legs},
  author  = {Vaglio, Alberto and Garulli, Andrea and Giannitrapani, Antonio
             and Quartullo, Renato and Van Der Meer, Tommaso},
  year    = {2026},
}
```
