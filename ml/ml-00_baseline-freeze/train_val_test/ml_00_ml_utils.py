from __future__ import annotations
import random
import subprocess
from pathlib import Path
import numpy as np

# Seed 고정
def set_seed(seed: int = 42, use_torch: bool = False) -> None:
    """
    공통 seed 고정 함수.
    기본은 XGBoost/NumPy/Python random 기준으로만 고정한다.
    torch는 ML-00 XGBoost 실행에 필수가 아니므로 use_torch=True일 때만 import한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ml_00_ml_utils.py 파일 위치
UTILS_DIR = Path(__file__).resolve().parent

# Git 프로젝트 루트를 기준으로 경로 계산
BASE_DIR = Path(
    subprocess.check_output(
        ["git", "-C", str(UTILS_DIR), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
).resolve()

# 프로젝트 상대 경로
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
