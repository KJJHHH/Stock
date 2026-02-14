from __future__ import annotations

import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import torch


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class Backtestor:
    def __init__(self, stock_list, config, dirs):
        self.stock = stock_list
        self.stock_target = stock_list[0]
        self.config = config
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        self.file_prefix = dirs["file_prefix"]
        os.makedirs(self.performance_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = self._init_model().to(device)
        self.data = self._init_data()

        self.loader = self.data.testloader
        self.len = self.__get_datalen()
        self.dates = self.__get_time()

        print(f"Backtesting {self.model.__name__} on {self.stock_target}")

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _data_obj(self, stock):
        raise NotImplementedError

    @abstractmethod
    def _init_data(self):
        raise NotImplementedError

    def __get_datalen(self):
        batch_x, _ = next(iter(self.loader))
        return batch_x.shape[0]

    def __get_time(self):
        if hasattr(self.data, "test_dates") and len(self.data.test_dates) == self.len:
            return self.data.test_dates
        return self.data.data.index[-self.len :]

    def __load_model(self, epoch):
        if epoch == "best":
            path = os.path.join(self.ckpt_dir, f"{self.file_prefix}-best.pth")
        else:
            path = os.path.join(self.ckpt_dir, f"{self.file_prefix}-{epoch}.pth")

        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        return True

    def __list_ckpt_epochs(self, prefix):
        if not os.path.isdir(self.ckpt_dir):
            return []
        epochs = []
        for name in os.listdir(self.ckpt_dir):
            if not name.startswith(prefix + "-") or not name.endswith(".pth"):
                continue
            if name.endswith("-best.pth"):
                continue
            stem = name[:-4]
            try:
                epoch = int(stem.split("-")[-1])
                epochs.append(epoch)
            except ValueError:
                continue
        return sorted(set(epochs))

    def single_stock_training(self):
        single_prefix = self.stock_target
        epochs = self.__list_ckpt_epochs(single_prefix)
        ckpts = []
        if epochs:
            ckpts.append(epochs[0])
            ckpts.append(epochs[-1])
        if os.path.exists(os.path.join(self.ckpt_dir, f"{single_prefix}-best.pth")):
            ckpts.append("best")
        if ckpts:
            original_prefix = self.file_prefix
            self.file_prefix = single_prefix
            self._plot(ckpts, "single")
            self.file_prefix = original_prefix
        else:
            print(f"No single-stock checkpoints found for prefix: {single_prefix}")

    def multi_stock_training(self):
        has_best = os.path.exists(os.path.join(self.ckpt_dir, f"{self.file_prefix}-best.pth"))
        if has_best:
            self._plot(["best"], "multiple")
        else:
            epochs = self.__list_ckpt_epochs(self.file_prefix)
            if epochs:
                self._plot([epochs[-1]], "multiple")
            else:
                print(f"No multi-stock checkpoints found for prefix: {self.file_prefix}")

    def buy_hold(self):
        self._plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        self.buy_hold()
        self.multi_stock_training()
        self.single_stock_training()

        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {self.stock_target}")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.performance_dir, f"{self.stock_target}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved backtest plot: {save_path}")
        plt.show()

    def _plot(self, ckpts: list = None, train_method: str = None):
        ckpts = ckpts or []
        if train_method is None:
            label = "BuyHold"
        elif train_method == "multiple":
            label = "Model multiple "
        else:
            label = "Model single epoch "

        print(f"Predicting {label}...")
        if ckpts:
            for epoch in ckpts:
                if self.__load_model(epoch):
                    asset_hist = self.backtesting(
                        self._test_method,
                        self.model,
                        (self.loader, self.data.src),
                        verbose=False,
                    )
                    curve_df = pd.DataFrame(asset_hist, index=pd.Index(self.dates))
                    plt.plot(curve_df, label=label + str(epoch))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return

        asset_hist = self.backtesting(
            self._test_method,
            self.model,
            (self.loader, self.data.src),
            use_model=False,
            verbose=False,
        )
        curve_df = pd.DataFrame(asset_hist, index=pd.Index(self.dates))
        plt.plot(curve_df, linestyle="dashed", label=label)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def backtesting(_test_method, model, data, use_model=True, short=True, verbose=True):
        result, truth = _test_method(model, data)
        truth = truth.clone()

        if not use_model:
            asset_hold_hist = torch.cumprod(1 + truth, dim=0).cpu().numpy()
            return asset_hold_hist

        long_mask = result >= 0
        short_mask = result < 0

        truth[long_mask] = 1 + truth[long_mask]
        if short:
            truth[short_mask] = 1 - truth[short_mask]
        else:
            truth[short_mask] = 1

        asset_hist = torch.cumprod(truth, dim=0).cpu().numpy()

        if verbose:
            print(result)
            plt.plot(asset_hist)
            plt.show()

        return asset_hist

    @staticmethod
    @abstractmethod
    def _test_method(model, data):
        raise NotImplementedError
