import torch

from base.base_testor import Backtestor
from datas import CVBasedData, TransformerData
from models import BasicBlock, ResNet, Transformer


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class TransformerBacktestor(Backtestor):
    def __init__(self, stock_list, config, dirs):
        super().__init__(stock_list, config, dirs)

    def _data_obj(self, stock):
        return TransformerData(stock=stock, config=self.config)

    def _init_data(self):
        data = self._data_obj(self.stock_target)
        self.src_len = data.src.shape[1]
        return data

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

    @staticmethod
    def _test_method(model, data):
        loader, src = data
        use_fp16 = device.type in {"cuda", "mps"}
        autocast_device_type = device.type if use_fp16 else "cpu"

        model.eval()
        with torch.no_grad():
            for x_test, y_test in loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                src = src.to(device)

                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=torch.float16,
                    enabled=use_fp16,
                ):
                    result = model(src=src, tgt=x_test)
                result = result[:, -1, -1].detach()
                truth = y_test[:, -1, -1].detach()

        return result, truth


class ResnetBacktestor(Backtestor):
    def _data_obj(self, stock):
        return CVBasedData(stock=stock, config=self.config)

    def _init_data(self):
        return self._data_obj(self.stock_target)

    def _init_model(self):
        return ResNet(BasicBlock, [3, 4, 6, 3], 1).to(device)

    @staticmethod
    def _test_method(model, data):
        loader, _ = data

        model.eval()
        with torch.no_grad():
            for x_test, y_test in loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                result = model(x_test).squeeze(-1)
                truth = y_test.squeeze(-1)

        return result, truth
