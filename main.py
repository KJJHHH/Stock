import argparse
import json
import os

import torch

from backtestors import ResnetBacktestor, TransformerBacktestor
from trainers import TransformerTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer_config(epochs_override=None):
    config_path = "configs/Transformer.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("configs/Transformer.json not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if epochs_override is not None:
        config["epochs"] = int(epochs_override)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock model trainer and backtester")
    parser.add_argument("task", choices=["train", "test"], help="Run training or backtesting")
    parser.add_argument("-m", "--model", required=True, help="Model name: Transformer or Resnet")
    parser.add_argument(
        "-st", "--stock_target", required=True, help="Target stock symbol, e.g. 2884.TW"
    )
    parser.add_argument(
        "-sp",
        "--stock_pool",
        nargs="*",
        default=[],
        help="Optional pretrain pool symbols, e.g. 2881.TW 2882.TW",
    )
    parser.add_argument("-e", "--epochs", default=None, help="Optional epoch override")

    args = parser.parse_args()

    stock_list = [args.stock_target] + [s for s in args.stock_pool if s != args.stock_target]
    dirs = {
        "ckpt_dir": f"results/{args.model}-temp/",
        "performance_dir": f"results/{args.model}-result/",
        "file_prefix": "-".join(stock_list),
    }

    if args.model == "Transformer":
        config = load_transformer_config(args.epochs)

        if args.task == "train":
            trainer = TransformerTrainer(stock_list=stock_list, config=config, dirs=dirs)
            trainer.training()
        else:
            testor = TransformerBacktestor(stock_list=stock_list, config=config, dirs=dirs)
            testor.plot()

    elif args.model == "Resnet":
        raise NotImplementedError(
            "Resnet pipeline needs a dedicated config file and trainer wiring; use Transformer for now."
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
