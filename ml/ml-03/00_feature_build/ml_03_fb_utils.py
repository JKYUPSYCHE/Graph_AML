"""ML-03 feature build utility functions and project paths.

Code map:
- Input: optional seed and local git repository path.
- Output: deterministic seed setup and project path constants.
- Public: set_seed, BASE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, EXPERIMENTS_DIR.
- Leakage guard: none; this module only provides environment utilities.
- Notes: BASE_DIR is resolved with git rev-parse at import time.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42, use_torch: bool = False) -> None:
    """Set Python, NumPy, and optionally torch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


UTILS_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(
    subprocess.check_output(
        ["git", "-C", str(UTILS_DIR), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
).resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
