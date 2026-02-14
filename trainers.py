import torch
from tqdm import tqdm

from backtestors import TransformerBacktestor
from base.base_trainer import BaseTrainer
from datas import TransformerData
from models import Transformer


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class TransformerTrainer(BaseTrainer):
    def __init__(self, stock_list, config, dirs) -> None:
        self.src_len = None
        super().__init__(stock_list, config, dirs)

    def _init_model(self):
        return Transformer(
            d_model=self.config["d_model"],
            dropout=self.config["dropout"],
            d_hid=self.config["d_hid"],
            nhead=self.config["nhead"],
            nlayers_e=self.config["nlayers_e"],
            nlayers_d=self.config["nlayers_d"],
            ntoken=self.config["ntoken"],
        ).to(device)

    def _data_obj(self, stock):
        return TransformerData(stock=stock, config=self.config)

    def _init_data(self):
        data = self._data_obj(self.stock_target)
        self.src_len = data.src.shape[1]
        return data

    def _update_data(self):
        self.data = self._data_obj(self.stock)

        if self.data.src.shape[1] > self.src_len:
            self.data.src = self.data.src[:, : self.src_len, :]
        elif self.data.src.shape[1] < self.src_len:
            pad_len = self.src_len - self.data.src.shape[1]
            padding = torch.zeros(
                (self.data.src.shape[0], pad_len, self.data.src.shape[2]),
                device=self.data.src.device,
            )
            self.data.src = torch.cat((self.data.src, padding), dim=1)

    def _model_train(self):
        loss_train_mean = 0.0
        self.model.train()

        for x, y in tqdm(self.data.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            src = self.data.src.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.autocast_device_type,
                dtype=torch.float16,
                enabled=self.use_fp16,
            ):
                outputs = self.model(src=src, tgt=x)
                loss = self.criterion(outputs, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_train_mean += loss.item()

        return loss_train_mean / len(self.data.trainloader)

    def _model_validate(self):
        loss_valid_mean = 0.0
        self.model.eval()

        with torch.no_grad():
            for x_val, y_val in self.data.validloader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                src = self.data.src.to(self.device)

                with torch.autocast(
                    device_type=self.autocast_device_type,
                    dtype=torch.float16,
                    enabled=self.use_fp16,
                ):
                    outputs_val = self.model(src=src, tgt=x_val)
                    loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()

        return loss_valid_mean / len(self.data.validloader)

    def _model_backtest(self):
        asset_hist = TransformerBacktestor.backtesting(
            TransformerBacktestor._test_method,
            self.model,
            (self.data.validloader, self.data.src),
            short=True,
            verbose=False,
        )
        return asset_hist[-1]
