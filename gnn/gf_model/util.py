import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import set_seed  # noqa: E402

_BASELINE_DIR = Path(__file__).resolve().parent


def logger_setup():
    log_directory = _BASELINE_DIR / "logs"
    log_directory.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(log_directory / "logs.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_parser():
    parser = argparse.ArgumentParser()

    # Adaptations
    parser.add_argument("--node_features", action='store_true', help="Use account-level node features from account_node_features.csv")
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
    parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")
    parser.add_argument("--patience", default=None, type=int, help="Early stopping patience in epochs (disabled if not set).")

    return parser
