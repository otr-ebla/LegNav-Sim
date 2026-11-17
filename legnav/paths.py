"""Central filesystem paths for the ``legnav`` package.

All checkpoint / data / figure I/O should resolve through these helpers so that
scripts work regardless of the current working directory (previously everything
assumed it was launched from inside ``src/jax_env``).
"""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent          # .../legnav
REPO_ROOT = PKG_ROOT.parent                          # repository root

import os
CHECKPOINTS_DIR = Path(os.environ.get("LEGNAV_CKPT_ROOT") or REPO_ROOT / "checkpoints")  # override: multi-seed runs
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"


def checkpoint(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``checkpoints/``."""
    p = CHECKPOINTS_DIR.joinpath(*parts)
    if os.environ.get("LEGNAV_CKPT_ROOT"):      # redirected root (multi-seed runs): make sure the dir exists
        p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def data(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``data/``."""
    return str(DATA_DIR.joinpath(*parts))


def figure(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``figures/``."""
    return str(FIGURES_DIR.joinpath(*parts))
