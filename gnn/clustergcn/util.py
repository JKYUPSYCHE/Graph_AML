import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import set_seed  # noqa: E402

_CLUSTERGCN_DIR = Path(__file__).resolve().parent


def logger_setup(log_dir=None, log_name='logs'):
    log_directory = Path(log_dir) if log_dir else _CLUSTERGCN_DIR / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fh = logging.FileHandler(log_directory / f"{log_name}.log", encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def create_parser():
    parser = argparse.ArgumentParser()

    # Adaptations
    parser.add_argument("--emlps",       action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp",  action='store_true', help="Use reverse MP in GNN training")
    parser.add_argument("--ports",       action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds",         action='store_true', help="Use time deltas in GNN training")
    parser.add_argument("--ego",         action='store_true', help="Use ego IDs in GNN training")

    # Model parameters
    parser.add_argument("--n_epochs",    default=100,  type=int,   help="Number of training epochs")
    parser.add_argument("--num_parts",   default=300,  type=int,   help="Number of METIS graph partitions")
    parser.add_argument("--clusters_per_batch", default=10, type=int, help="Number of clusters to merge per mini-batch")

    # Misc
    parser.add_argument("--seed",        default=42,   type=int,   help="Random seed")
    parser.add_argument("--tqdm",        action='store_true',      help="Use tqdm progress bar")
    parser.add_argument("--data",        default=None, type=str,   required=True, help="AML dataset (small or medium)")
    parser.add_argument("--model",       default=None, type=str,   required=True, help="Model architecture [gin, gat, rgcn, pna]")
    parser.add_argument("--save_model",  action='store_true',      help="Save the best model")
    parser.add_argument("--unique_name", default=None, type=str,   help="Unique name for model checkpoint")
    parser.add_argument("--finetune",    action='store_true',      help="Fine-tune a pre-trained model")
    parser.add_argument("--inference",   action='store_true',      help="Inference only (skip training)")
    parser.add_argument("--patience",    default=None, type=int,   help="Early stopping patience in epochs")

    return parser
