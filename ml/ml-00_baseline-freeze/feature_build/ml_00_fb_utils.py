"""
Feature build 공통 유틸리티 모듈

이 파일의 역할
----------------
1. feature build 실험에서 사용할 난수 seed를 고정한다.
2. 현재 파일 위치를 기준으로 Git 프로젝트 루트(BASE_DIR)를 찾는다.
3. data/raw, data/processed, experiments 같은 프로젝트 공통 경로를 제공한다.

중요한 전제
-----------
- 노트북 실행 위치가 달라도 경로가 흔들리지 않도록 Git 루트를 기준으로 삼는다.
- 이 구조는 `ml/ml-00_baseline/train_val_test/ml_00_ml_utils.py`의 경로 처리 방식을 그대로 따른다.
- torch는 기본 의존성이 아니므로 `use_torch=True`일 때만 import한다.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# 1. 난수 seed 고정
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42, use_torch: bool = False) -> None:
    """
    feature 실험 재현성을 위해 Python random, NumPy, 선택적으로 torch seed를 고정한다.

    Parameters
    ----------
    seed:
        고정할 난수 seed. 프로젝트 기본값은 42다.
    use_torch:
        True이면 torch seed까지 고정한다. 현재 feature build에는 torch가 필수가 아니므로
        기본값은 False로 둔다.
    """

    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        # torch가 설치되지 않은 환경에서도 feature build는 동작해야 하므로 지연 import한다.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# 2. 프로젝트 기준 경로
# -----------------------------------------------------------------------------
# ml_00_fb_utils.py 파일 위치다. cwd가 아니라 이 파일 위치에서 Git 루트를 찾는다.
UTILS_DIR = Path(__file__).resolve().parent

# Git 프로젝트 루트다. 노트북 실행 위치와 무관하게 항상 같은 루트를 반환한다.
# `git -C <UTILS_DIR> rev-parse --show-toplevel`은 해당 경로가 속한 Git 루트를 출력한다.
BASE_DIR = Path(
    subprocess.check_output(
        ["git", "-C", str(UTILS_DIR), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
).resolve()

# 프로젝트 공통 디렉터리다. 다른 모듈은 문자열 경로를 직접 만들지 말고 이 상수를 우선 사용한다.
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
