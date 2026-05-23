import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import set_seed  # noqa: E402

_BASELINE_DIR = Path(__file__).resolve().parent


def logger_setup(log_dir=None, log_name='logs'):
    log_directory = Path(log_dir) if log_dir else _BASELINE_DIR / "logs"
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
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")

    # Model parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    # Misc
    parser.add_argument("--seed", default=42, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--unique_name", default=None, type=str, help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")
    parser.add_argument("--patience", default=None, type=int, help="Early stopping patience in epochs (disabled if not set).")
    parser.add_argument("--weighted_sampler", action='store_true', help="Use WeightedRandomSampler for training (oversamples minority class). CE loss weights are set to [1,1] when enabled.")

    return parser
