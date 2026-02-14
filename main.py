import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from backtestors import TransformerBacktestor
from constants import ModelName
from mechanism_demo import main as run_mechanism_demo
from rl_runner import run_rl
from supervised import run_supervised
from trainers import TransformerTrainer


ROOT_DIR = Path(__file__).resolve().parent
RL_MODEL_MAP = {
    "RL_DQN": "dqn",
    "RL_PPO": "ppo",
    "RL_PPO_TUNED": "ppo_tuned",
    "RL_A2C": "a2c",
    "RL_SAC": "sac",
    "RL_TD3": "td3",
}
SUPERVISED_MODELS = {m.value for m in ModelName}
ALL_MODELS = {"Transformer", "Resnet", "MECHANISM"} | SUPERVISED_MODELS | set(RL_MODEL_MAP.keys())
AUTO_DEFAULTS = {
    "model": "Transformer",
    "stock_target": "2330.TW",
    "stock_pool": [],
    "epochs": None,
}


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_runtime_precision(device):
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")


def load_transformer_config(epochs_override=None):
    config_path = ROOT_DIR / "configs" / "Transformer.json"
    if not config_path.exists():
        raise FileNotFoundError("configs/Transformer.json not found")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if epochs_override is not None:
        config["epochs"] = int(epochs_override)

    return config


def load_stockai_config():
    paths = [
        ROOT_DIR / "config" / "core.yaml",
        ROOT_DIR / "config" / "backtest.yaml",
        ROOT_DIR / "config" / "rl.yaml",
        ROOT_DIR / "config" / "model_structure.yaml",
        ROOT_DIR / "config" / "training.yaml",
    ]
    merged = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def run_transformer(args):
    device = select_device()
    print_runtime_precision(device)

    result_root = ROOT_DIR / "result"
    legacy_root = ROOT_DIR / "results"
    stock_list = [args.stock_target] + [s for s in args.stock_pool if s != args.stock_target]
    ckpt_dir = result_root / f"{args.model}-temp"
    perf_dir = result_root / f"{args.model}-result"

    # Backward-compatible checkpoint lookup: if new dir has no checkpoints,
    # read checkpoints from legacy `results/`.
    if args.task == "test":
        has_new_ckpt = ckpt_dir.exists() and any(ckpt_dir.glob("*.pth"))
        legacy_ckpt_dir = legacy_root / f"{args.model}-temp"
        has_legacy_ckpt = legacy_ckpt_dir.exists() and any(legacy_ckpt_dir.glob("*.pth"))
        if not has_new_ckpt and has_legacy_ckpt:
            ckpt_dir = legacy_ckpt_dir

    dirs = {
        "ckpt_dir": str(ckpt_dir) + "/",
        "performance_dir": str(perf_dir) + "/",
        "file_prefix": "-".join(stock_list),
    }

    if args.model == "Transformer":
        config = load_transformer_config(args.epochs)
        print("Transformer mode: always run train + backtest")
        trainer = TransformerTrainer(stock_list=stock_list, config=config, dirs=dirs)
        trainer.training()
        testor = TransformerBacktestor(stock_list=stock_list, config=config, dirs=dirs)
        testor.plot()
        return

    if args.model == "Resnet":
        raise NotImplementedError(
            "Resnet pipeline needs a dedicated config file and trainer wiring; use Transformer for now."
        )
    raise ValueError(f"Unsupported model for task={args.task}: {args.model}")


def run_unified_model(args):
    if args.model == "MECHANISM":
        run_mechanism_demo()
        return

    config = load_stockai_config()

    if args.stock_target:
        config["ticker"] = args.stock_target
    if args.train_start:
        config["train_start"] = args.train_start
    if args.train_end:
        config["train_end"] = args.train_end
    if args.backtest_start:
        config["backtest_start"] = args.backtest_start
    if args.backtest_end:
        config["backtest_end"] = args.backtest_end

    device = select_device()
    print_runtime_precision(device)

    ticker = config["ticker"]
    train_start = config["train_start"]
    train_end = config["train_end"]
    backtest_start = config["backtest_start"]
    backtest_end = config["backtest_end"]
    seq_length = config["seq_length"]
    transaction_cost = float(config.get("transaction_cost", 0.0))

    if args.model in RL_MODEL_MAP:
        config.setdefault("rl", {})
        config["rl"]["algorithm"] = RL_MODEL_MAP[args.model]
        run_rl(
            ticker,
            train_start,
            train_end,
            backtest_start,
            backtest_end,
            config,
            device,
            transaction_cost,
        )
        return

    if args.model not in SUPERVISED_MODELS:
        raise ValueError(
            f"Model {args.model} is not supported for unified run. Supported: {sorted(SUPERVISED_MODELS | set(RL_MODEL_MAP.keys()) | {'MECHANISM'})}"
        )

    model_name = ModelName(args.model)
    task = (args.prediction_task or config.get("task", "R")).upper()
    if task == "R":
        task = "regression"
    elif task == "C":
        task = "classification"
    elif task not in {"regression", "classification"}:
        raise ValueError("prediction_task must be one of: R, C, regression, classification")

    run_supervised(
        ticker,
        train_start,
        train_end,
        backtest_start,
        backtest_end,
        seq_length,
        model_name,
        model_name.value,
        task,
        device,
        config,
        transaction_cost,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Unified Stock project entrypoint")
    parser.add_argument(
        "task",
        nargs="?",
        default="run",
        choices=["train", "test", "backtest", "run"],
        help="Ignored in unified mode. Kept only for backward compatibility.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=AUTO_DEFAULTS["model"],
        choices=sorted(ALL_MODELS),
        help="Model name (always runs full pipeline: train + backtest where applicable).",
    )
    parser.add_argument("-st", "--stock_target", default=AUTO_DEFAULTS["stock_target"], help="Ticker, e.g. 2330.TW")
    parser.add_argument(
        "-sp",
        "--stock_pool",
        nargs="*",
        default=[],
        help="Optional pretrain pool symbols (Transformer train/test)",
    )
    parser.add_argument("-e", "--epochs", default=None, help="Transformer epoch override")

    parser.add_argument("--train_start", default=None)
    parser.add_argument("--train_end", default=None)
    parser.add_argument("--backtest_start", default=None)
    parser.add_argument("--backtest_end", default=None)
    parser.add_argument(
        "--prediction_task",
        default=None,
        help="StockAI supervised task: R/C/regression/classification",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print("Unified mode: always run complete flow")

    if args.model in {"Transformer", "Resnet"}:
        run_transformer(args)
        return

    run_unified_model(args)


if __name__ == "__main__":
    main()
