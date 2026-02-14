import os
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BaseTrainer:
    def __init__(
        self,
        stock_list,
        config,
        dirs,
        short=True,
        device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    ) -> None:
        self.stock = None
        self.stock_target = stock_list[0]
        self.stock_list = stock_list
        self.stock_trained = []

        self.file_prefix = dirs["file_prefix"]
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        self.device = device
        self._check_dir()

        self.config = config
        self.epochs = config["epochs"]
        self.val_type = config["val_type"]
        self.lr = config["optimizer"]["args"]["lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.amsgrad = config["optimizer"]["args"]["amsgrad"]
        self.scheduler_step = config["lr_scheduler"]["args"]["step_size"]
        self.scheduler_gamma = config["lr_scheduler"]["args"]["gamma"]

        self.start_epoch = 0
        self.not_improve_cnt = 0
        self.best_val_result = float("inf") if self.val_type == "loss" else float("-inf")

        self.data = self._init_data()

        self.model = self._init_model().to(self.device)
        self.model_best = self.model
        self.criterion = nn.MSELoss()
        self.use_fp16 = self.device.type in {"cuda", "mps"}
        self.autocast_device_type = self.device.type if self.use_fp16 else "cpu"
        self.compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        self.short = short
        print(f"Validation method: {self.val_type}")
        print(f"Mixed precision FP16: {self.use_fp16}")
        print(f"Compute dtype: {self.compute_dtype}")

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _data_obj(self, stock):
        raise NotImplementedError

    @abstractmethod
    def _init_data(self):
        raise NotImplementedError

    @abstractmethod
    def _update_data(self):
        raise NotImplementedError

    def training(self):
        self.pretrain()
        self.train()

    def pretrain(self):
        print("TRAINING multiple stock ...")
        self._resume_checkpoint()

        for stock in self.stock_list:
            if stock in self.stock_trained:
                print(f"Stock {stock} already trained")
                continue
            self._train_stock(stock)

    def train(self):
        print("TRAINING target stock ...")
        self._train_stock(self.stock_target)

    def _train_stock(self, stock):
        print(f"Start training stock {stock}")
        self.stock = stock
        self.model = self.model_best
        self._init_training_control()
        self._update_data()

        for epoch in range(self.start_epoch, self.epochs):
            loss_train_mean = self._model_train()

            if epoch > 100:
                self.scheduler.step()

            self._save_checkpoint(epoch)
            if epoch == self.epochs - 1 and self.stock not in self.stock_trained:
                self.stock_trained.append(self.stock)
                self._save_checkpoint(epoch)

            stop = self._select_best_model(epoch)
            if stop:
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.6f}")

    def _select_best_model(self, epoch):
        if self.val_type == "asset":
            val_return = self._model_backtest()
            if val_return > self.best_val_result:
                print(f"New best model found with return {val_return}")
                self.not_improve_cnt = 0
                self.best_val_result = val_return
                self.model_best = self.model
                self._save_checkpoint(epoch, save_best=True)
            else:
                self.not_improve_cnt += 1
                if self.not_improve_cnt >= 100:
                    print(f"Early stopping at epoch {epoch}")
                    return True
            return False

        loss_valid_mean = self._model_validate()
        if loss_valid_mean < self.best_val_result:
            print(f"New best model found in epoch {epoch} with val loss: {loss_valid_mean}")
            self.not_improve_cnt = 0
            self.best_val_result = loss_valid_mean
            self.model_best = self.model
            self._save_checkpoint(epoch, save_best=True)
        else:
            self.not_improve_cnt += 1
            if self.not_improve_cnt >= 100:
                print(f"Early stopping at epoch {epoch}")
                return True
        return False

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "stock_trained": self.stock_trained,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "not_improve_cnt": self.not_improve_cnt,
            "best_val_result": self.best_val_result,
        }

        if save_best:
            file = f"{self.file_prefix}-best.pth"
        else:
            file = f"{self.file_prefix}-{epoch}.pth"

        torch.save(state, os.path.join(self.ckpt_dir, file))

    def _resume_checkpoint(self):
        ckpts = [
            f
            for f in os.listdir(self.ckpt_dir)
            if os.path.isfile(os.path.join(self.ckpt_dir, f))
            and self.file_prefix in f
            and "best" not in f
        ]
        if not ckpts:
            return

        ckpt_epochs = [int(file.split("-")[-1].split(".")[0]) for file in ckpts]
        resume_file = f"{self.file_prefix}-{max(ckpt_epochs)}.pth"
        resume_path = os.path.join(self.ckpt_dir, resume_file)

        print(f"Loading checkpoint: {resume_path}")
        if not os.path.exists(resume_path):
            return

        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)

        if checkpoint["arch"] != self.config["name"]:
            raise ValueError("Checkpoint architecture does not match current config.")

        self.model.load_state_dict(checkpoint["state_dict"])

        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            raise ValueError("Checkpoint optimizer type does not match current config.")

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.not_improve_cnt = checkpoint["not_improve_cnt"]
        self.best_val_result = checkpoint["best_val_result"]
        self.stock_trained = checkpoint["stock_trained"]

    @abstractmethod
    def _model_train(self):
        raise NotImplementedError

    @abstractmethod
    def _model_validate(self):
        raise NotImplementedError

    @abstractmethod
    def _model_backtest(self):
        raise NotImplementedError

    def _check_dir(self):
        for path in [self.ckpt_dir, self.performance_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def _init_training_control(self):
        self.start_epoch = 0
        self.not_improve_cnt = 0
        self.best_val_result = float("inf") if self.val_type == "loss" else float("-inf")
