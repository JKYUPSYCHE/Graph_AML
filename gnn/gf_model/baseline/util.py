import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from utils import set_seed  # noqa: E402

_BASELINE_DIR = Path(__file__).resolve().parent


def logger_setup(log_dir=None, log_name='run'):
    if log_dir is None:
        log_dir = _BASELINE_DIR / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{log_name}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
