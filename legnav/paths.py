"""Central filesystem paths for the ``legnav`` package.

All checkpoint / data / figure I/O should resolve through these helpers so that
scripts work regardless of the current working directory (previously everything
assumed it was launched from inside ``src/jax_env``).
"""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent          # .../legnav
REPO_ROOT = PKG_ROOT.parent                          # repository root

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"


def checkpoint(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``checkpoints/``."""
    return str(CHECKPOINTS_DIR.joinpath(*parts))


def data(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``data/``."""
    return str(DATA_DIR.joinpath(*parts))


def figure(*parts) -> str:
    """Absolute path (as ``str``) to a file under ``figures/``."""
    return str(FIGURES_DIR.joinpath(*parts))
