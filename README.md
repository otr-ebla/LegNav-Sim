<h1>LegNav-Sim</h1>

**Accepted at an IROS 2026 workshop.**

[![JAX](https://img.shields.io/badge/JAX-Enabled-orange?style=flat-square&logo=google)](https://github.com/google/jax)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)

Official implementation of

[Learning Robot Social Navigation By Sensing Human Legs](https://arxiv.org/abs/2607.27922) by

[Alberto Vaglio](https://scholar.google.com/citations?user=zfT7om8AAAAJ&hl=en&oi=ao) & [Andrea Garulli](https://scholar.google.com/citations?user=4rFwUskAAAAJ&hl=en&oi=ao), [Antonio Giannitrapani](https://scholar.google.com/citations?user=0eeBLTYAAAAJ&hl=en&oi=ao), [Renato Quartullo](https://scholar.google.com/citations?user=Dl4lZcEAAAAJ&hl=en&oi=ao) and [Tommaso Van Der Meer](https://scholar.google.com/citations?user=-o4MmxgAAAAJ&hl=en&oi=ao) (Department of Information Engineering and Mathematics (DIISM), University of Siena)</br>

<img src="assets/indoor_gif.gif" width="70%"></br>

[[Paper]](https://arxiv.org/abs/2607.27922) [[Video]](https://youtu.be/P6gFTvi3k7w) [[Models]](#download-pretrained-policies-and-test-them-live)

### Video of Experiments

<a href="https://youtu.be/P6gFTvi3k7w">
  <img src="https://img.youtube.com/vi/P6gFTvi3k7w/maxresdefault.jpg" alt="Video of Experiments" width="70%">
</a>

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

Use **Python 3.11–3.13** (Python 3.12 recommended). CPU evaluation works on
Linux, Windows x86-64 and Apple Silicon macOS; NVIDIA training uses Linux
(or a suitably configured WSL2 installation). Other platforms are not validated.

### Install on a participant's computer

Use a shallow clone to avoid downloading the large experiment history. Do not
use `--recurse-submodules`: historical optional gitlinks lack submodule metadata
and are not needed for the core workshop.

```bash
git clone --depth 1 https://github.com/otr-ebla/LegNav-Sim.git
cd LegNav-Sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

On Windows PowerShell, use `py -3.12 -m venv .venv` and
`.venv\Scripts\Activate.ps1` instead of the two environment commands above.
If script activation is disabled, invoke `.venv\Scripts\python.exe` directly.
Do not copy a virtual environment from another computer.

### Check the installation before the workshop

```bash
python -m legnav.evaluation.jax_eval_multi --algo sac --headless --steps 10
python -m legnav.evaluation.jax_eval_multi --algo sac
```

The first command loads the committed SAC checkpoint and checks a short rollout
for finite observations, actions and rewards without a display. It must end in
`PASS`. Initial JAX compilation can take a few minutes. The second command opens
the interactive viewer: `0`–`6` select scenarios, `R` resets, and `Q` exits.
PPO and TQC can also be selected with `--algo ppo` / `--algo tqc`.
The viewer defaults to CPU, so no CUDA installation is needed.

The editable install resolves maps and checkpoints independently of your working
directory. A wheel contains simulator maps but **not pretrained checkpoints**;
when using a wheel, pass `--ckpt /path/to/model.msgpack` from a downloaded clone.
Missing checkpoints stop evaluation with an error instead of using random weights.
SHAC, PPO circles and asymmetric PPO weights are not included; those options
require your own compatible checkpoint.

### Download pretrained policies and test them live

**No training is needed.** The clone above already includes the pretrained
`.msgpack` weights in `checkpoints/`. After installation, run the following
commands from the repository root to open the live simulator with each saved
`best` policy:

```bash
python -m legnav.evaluation.jax_eval_multi --algo ppo --ckpt checkpoints/ppo/ppo_legs_best.msgpack
python -m legnav.evaluation.jax_eval_multi --algo sac --ckpt checkpoints/sac/sac_best.msgpack
python -m legnav.evaluation.jax_eval_multi --algo tqc --ckpt checkpoints/tqc/tqc_best.msgpack
```

To download a weight file separately, use these direct links, then pass its
local path with `--ckpt`. The simulator package must still be installed.

| Policy | Saved best weights | Weights selected by `benchmark_eval` |
| --- | --- | --- |
| PPO | [ppo_legs_best.msgpack](https://github.com/otr-ebla/LegNav-Sim/raw/refs/heads/master/checkpoints/ppo/ppo_legs_best.msgpack) | Same best checkpoint |
| SAC | [sac_best.msgpack](https://github.com/otr-ebla/LegNav-Sim/raw/refs/heads/master/checkpoints/sac/sac_best.msgpack) | [sac_final.msgpack](https://github.com/otr-ebla/LegNav-Sim/raw/refs/heads/master/checkpoints/sac/sac_final.msgpack) |
| TQC | [tqc_best.msgpack](https://github.com/otr-ebla/LegNav-Sim/raw/refs/heads/master/checkpoints/tqc/tqc_best.msgpack) | [tqc_final.msgpack](https://github.com/otr-ebla/LegNav-Sim/raw/refs/heads/master/checkpoints/tqc/tqc_final.msgpack) |

`best` and `final` are different saved snapshots; the filenames do not guarantee
which performs best on every test scenario. To watch **the exact weights used
by the current statistics benchmark**, use:

```bash
python -m legnav.evaluation.jax_eval_multi --algo ppo --ckpt checkpoints/ppo/ppo_legs_best.msgpack
python -m legnav.evaluation.jax_eval_multi --algo sac --ckpt checkpoints/sac/sac_final.msgpack
python -m legnav.evaluation.jax_eval_multi --algo tqc --ckpt checkpoints/tqc/tqc_final.msgpack
```

Run one viewer at a time. Controls: `0`–`6` select a scenario, `7` restores random
selection, `R` resets, `Space` pauses, `L` toggles LiDAR, and `Q`/`Esc` quits.
This is a live **simulator** demo, not a real-robot deployment. It defaults to CPU.
For a terminal-only check, append `--headless --steps 10` to any command above.
That short check validates loading and execution; it does not compute benchmark
statistics.

### Generate evaluation statistics

On a Linux NVIDIA machine, install the GPU and plotting extras and run:

```bash
python -m pip install -e ".[cuda12,plots]"
python -c "import jax; print(jax.devices('gpu'))"
python -m legnav.evaluation.benchmark_eval
```

The benchmark loads the three files in the table's final column, evaluates test
scenarios at multiple speeds, and writes:

- `data/test_evaluation_raw_data.csv`: episode metrics, including success,
  collisions, timeouts, path efficiency, travel time and yielding score.
- `figures/Evaluation_Dashboard.png`: the evaluation dashboard.

Existing outputs at these paths are overwritten. Check the startup messages:
all three policies should report that they loaded; missing or incompatible
checkpoints are skipped by the benchmark. This is a large parallel evaluation,
requires a working GPU, and has not been validated by the CPU installation check.
Live demos use interactive scenarios and are not a reproduction of aggregate
benchmark statistics.

For reproducible comparisons, record `git rev-parse HEAD` and the checkpoint
filenames with your results. The download links follow `master`; use that commit
hash in place of `refs/heads/master` in the links to retain a fixed version.

### NVIDIA GPU training (optional)

In the same environment, on Linux with a compatible NVIDIA driver:

```bash
python -m pip install -e ".[cuda12]"
python -c "import jax; print(jax.devices('gpu'))"
python -m legnav.algorithms.jax_ppo
# Alternatives:
python -m legnav.algorithms.SACjax
python -m legnav.algorithms.TQCjac
```

The GPU check must list a GPU before starting a trainer. Trainers explicitly
select CUDA and use large batches; a CPU demo passing does not validate training
or available VRAM. The CUDA extra installs JAX's CUDA 12 runtime dependencies;
see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for driver/platform requirements. This does not install the NVIDIA driver.

For plotting and benchmarks, install `python -m pip install -e ".[plots]"`.
The SB3 comparison trainers additionally need `python -m pip install -e ".[baselines]"`.
ROS deployment and experimental/legacy agents have separate requirements and
are outside this workshop installation path.

See [the workshop readiness notes](WORKSHOP.md) for verification and remaining limits.

## Legacy code

The `legacy/` directory holds the original, pre-JAX implementation
(Gymnasium + stable-baselines3 environments and trainers, real-robot SB3 scripts,
and plotting scratch). It is retained for reference and is **not** part of the
`legnav` package.

## Citation

If you use LegNav-Sim in your research, please cite our [arXiv preprint](https://arxiv.org/abs/2607.27922):

```bibtex
@misc{vaglio2026learningsocialrobotnavigation,
      title={Learning Social Robot Navigation By Sensing Human Legs}, 
      author={Alberto Vaglio and Andrea Garulli and Antonio Giannitrapani and Renato Quartullo and Tommaso Van Der Meer},
      year={2026},
      eprint={2607.27922},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.27922}, 
}
```
