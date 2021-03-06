import torch
import argparse

parser = argparse.ArgumentParser(description="GAT")

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=False,
    help="GAT with sparse version or not.",
)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs to train."
)
parser.add_argument("--save_every", type=int, default=10, help="Save every n epochs")
parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
parser.add_argument("--n_heads", type=int, default=8, help="Number of head attentions.")
parser.add_argument(
    "--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer"],
    help="Dataset to train.",
)
parser.add_argument(
    "--model", type=str, default="GAT", choices=["GCN", "GAT"], help="Model"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
