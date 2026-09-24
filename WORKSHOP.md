# Workshop readiness

Audit date: 2026-09-24.

## Repository status

The local `master` was 12 commits behind GitHub; it was fast-forwarded to
`208ad8d`. Those upstream changes only affected README and ignore rules.
The pre-existing reward plotting script and two PDF modifications were preserved.
The workshop installation fixes, policy download instructions and acceptance
notice are included with this readiness report.

## Problems fixed

- The requirements were a legacy environment freeze, missing JAX, Flax and
  Optax while requiring Linux/NVIDIA/PyTorch packages for every participant.
  Core dependencies now live in package metadata; CUDA, plotting and SB3 are
  optional extras. The old freeze is retained in `legacy/requirements.txt` for
  reference, not as a supported workshop installation.
- Package metadata previously claimed Python 3.10 support despite incompatible
  dependencies. The new minimum is Python 3.11.
- Wheels omitted the NPZ map loaded when importing the simulator. Map data is
  now packaged, and the unrelated nested NavRep checkout is excluded.
- Missing evaluation checkpoints silently selected random weights. They now
  produce a failing exit status and an actionable message.
- A headless rollout checks actual checkpoint inference and simulator steps.
- Benchmark output directories were absent on fresh clones; the benchmark now
  creates them before running.
- README now includes cloning, Windows activation, CPU validation, GPU setup,
  checkpoint limitations and shallow-clone guidance.
- An installation workflow covers Python 3.11–3.13 on Linux, Windows and macOS.

## Validation performed

In a new temporary Python 3.13.5 environment on Linux x86-64:

- Installed `.[plots]` from package metadata; `pip check` passed.
- Editable installation: SAC checkpoint, 10 CPU steps, finite outputs.
- Built and installed a wheel from a copy containing tracked source/assets only.
- Ran outside the repository with explicit checkpoints from that copy:
  SAC 10 steps, PPO 3 steps, TQC 3 steps; all passed with 668-element observations.
- Additional README checkpoint checks: TQC best and SAC final passed three CPU
  steps. SAC best was rechecked after the compatibility fix. Both SAC variants
  also passed a benchmark actor forward pass on CPU.
- Deterministic SAC evaluation now accepts both historical global exploration
  vectors and newer state-dependent exploration layers; neither is needed to
  compute the deterministic action.
- A missing checkpoint exited with status 1 and an explanatory message.

## Remaining limits and workshop preparation

- GPU training is unverified here: `nvidia-smi` cannot communicate with a driver.
  Run the README GPU check and a rehearsal on the actual training machine.
- Windows/macOS and other Python versions await the new CI workflow; the workflow
  has not run online. GUI rendering requires a desktop and a participant rehearsal.
- Short rollouts validate installation and inference, not policy success rates or
  full training convergence. ROS, experimental agents and legacy code are outside
  this validated path.
- SHAC, PPO circles and asymmetric PPO default checkpoints are absent from Git.
  Use the validated SAC/PPO/TQC defaults for the workshop.
- Historical gitlinks `.claude/worktrees/great-cori` and
  `legnav/baselines/navrep` have no `.gitmodules` definitions. Avoid recursive
  submodule cloning. The core simulator does not need them; repository cleanup
  is still outstanding.
- There is no top-level license file. The vendored JHSFM license does not specify
  the license of the entire project; the owner should choose one before wider
  redistribution.
- Local Git storage is roughly 3.6 GiB; tracked checkpoints total about 99 MiB.
  Use the documented shallow clone and install dependencies before arriving.
- Core JAX/Flax/Optax versions are pinned. Transitive dependencies are resolved
  by pip; this is not a fully locked, offline distribution.

Participant preflight (after publishing these changes):

```bash
python -m pip check
python -m legnav.evaluation.jax_eval_multi --algo sac --headless --steps 10
python -m legnav.evaluation.jax_eval_multi --algo sac
```
