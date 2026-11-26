#!/bin/bash
# Run on a compute node. Builds the conda env with CUDA JAX.
set -euo pipefail
module load conda
eval "$(conda shell.bash hook)"
conda create -y -n legnav python=3.12
conda activate legnav
cd "$HOME/legnav"
python -m pip install -e ".[cuda12]"
python -c "import jax; print(jax.devices())"
conda clean -a -y
