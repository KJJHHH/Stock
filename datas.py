import warnings

import torch
from tqdm import tqdm

from base.base_data import BaseData

warnings.filterwarnings("ignore")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class TransformerData(BaseData):
    def __init__(self, stock: str, config: dict, src_len=None) -> None:
        super().__init__(stock, config)
        self.src = None
        self.src_len = src_len

        self.data = self._fetch_price()
        self.data_origin = self.data.copy()
        self._create_var_target()
        self._clean()
        self._normalize()
        self.prepare_data()

    def __get_src(self, x_train: torch.Tensor) -> None:
        if self.src_len is None:
            self.src_len = len(x_train)

        if self.src_len <= 0 or self.src_len > len(x_train):
            raise ValueError("src_len must be in range [1, len(x_train)].")

        self.src = x_train[: self.src_len][:, -1, :].unsqueeze(0).to(device)

    def prepare_data(self, verbose: bool = False) -> None:
        def append_data(samples, lists):
            for sample, target_list in zip(samples, lists):
                target_list.append(sample)

        x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
        train_dates, valid_dates, test_dates = [], [], []

        for i in range(len(self.data) - self.window):
            x_window = self.data.iloc[i : i + self.window][
                ["do", "dh", "dl", "dv", "dac", "return"]
            ].values
            y_window = self.data.iloc[i + 1 : i + self.window + 1][
                ["do", "dh", "dl", "dv", "dac", "return"]
            ].values
            date = self.data.index[i + self.window]

            if date < self.valid_start:
                append_data((x_window, y_window, date), (x_train, y_train, train_dates))
            elif date < self.test_start:
                append_data((x_window, y_window, date), (x_valid, y_valid, valid_dates))
            else:
                append_data((x_window, y_window, date), (x_test, y_test, test_dates))

        self.train_dates = train_dates
        self.valid_dates = valid_dates
        self.test_dates = test_dates

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        self.__get_src(x_train)
        self.getLoaders((x_train, y_train, x_valid, y_valid, x_test, y_test), self.batch_size)

        if verbose:
            print(
                "Data Shape:\n"
                f"x_train: {x_train.shape}\n"
                f"x_valid: {x_valid.shape}\n"
                f"x_test: {x_test.shape}\n"
                f"y_train: {y_train.shape}\n"
                f"y_valid: {y_valid.shape}\n"
                f"y_test: {y_test.shape}\n"
                f"src: {self.src.shape}"
            )


class CVBasedData(BaseData):
    def __init__(self, stock: str, config: dict) -> None:
        super().__init__(stock, config)
        self.data = self._fetch_price()
        self.data_origin = self.data.copy()
        self._create_var_target()
        self._clean()
        self._normalize()
        self.prepare_data()

    def _gaf(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        x_diff = x.unsqueeze(0) - x.unsqueeze(1)
        return torch.cos(x_diff).unsqueeze(0)

    def _window_xy(self):
        x_list, y_list = [], []
        dates = []

        for i in range(len(self.data) - self.window):
            x_window = self.data.iloc[i : i + self.window]
            y_window = self.data.iloc[i + 1 : i + self.window + 1]
            date = self.data.index[i + self.window]

            x_values = x_window[["do", "dh", "dl", "dc", "dv", "dac"]].values
            y_value = y_window[["return"]].values[-1][0]

            x_list.append(x_values)
            y_list.append(y_value)
            dates.append(date)

        return x_list, y_list, dates

    def prepare_data(self, verbose: bool = False) -> None:
        x_list, y_list, dates = self._window_xy()

        x = torch.tensor(x_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1)

        x_gaf = torch.tensor([])
        for i in tqdm(range(len(x)), disable=not verbose):
            x_element = torch.cat([self._gaf(x[i, :, j]) for j in range(x.shape[-1])], dim=0).unsqueeze(0)
            x_gaf = torch.cat((x_gaf, x_element), dim=0)

        x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
        train_dates, valid_dates, test_dates = [], [], []

        for i, date in enumerate(dates):
            if date < self.valid_start:
                x_train.append(x_gaf[i])
                y_train.append(y[i])
                train_dates.append(date)
            elif date < self.test_start:
                x_valid.append(x_gaf[i])
                y_valid.append(y[i])
                valid_dates.append(date)
            else:
                x_test.append(x_gaf[i])
                y_test.append(y[i])
                test_dates.append(date)

        self.train_dates = train_dates
        self.valid_dates = valid_dates
        self.test_dates = test_dates

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)
        x_valid = torch.stack(x_valid)
        y_valid = torch.stack(y_valid)
        x_test = torch.stack(x_test)
        y_test = torch.stack(y_test)

        self.src = None
        self.getLoaders((x_train, y_train, x_valid, y_valid, x_test, y_test), self.batch_size)
