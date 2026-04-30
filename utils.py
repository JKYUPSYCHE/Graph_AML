import random
import numpy as np
import torch
from pathlib import Path


# Seed 고정
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 프로젝트 상대 경로
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
EXPERIMENTS_DIR = BASE_DIR / 'experiments'