import warnings
from abc import abstractmethod
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class BaseData:
    def __init__(self, stock: str, config: dict):
        self.stock = stock
        self.start_date = config["train_start"]
        self.end_date = config["end_date"]
        self.window = config["ntoken"]
        self.batch_size = config["batch_size"]

        self.train_start = pd.Timestamp(config["train_start"])
        self.valid_start = pd.Timestamp(config["valid_start"])
        self.test_start = pd.Timestamp(config["test_start"])

        self.train_dates = []
        self.valid_dates = []
        self.test_dates = []

        self.trainloader = None
        self.validloader = None
        self.testloader = None

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self) -> Path:
        stock_safe = self.stock.replace("/", "_")
        start_safe = str(self.start_date)
        end_safe = str(self.end_date)
        return self.cache_dir / f"{stock_safe}_{start_safe}_{end_safe}_1d.csv"

    def _fetch_price(self) -> pd.DataFrame:
        cache_path = self._cache_path()

        # Cache-first avoids unnecessary calls that can trigger Yahoo rate limits.
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return self._normalize_datetime_index(cached)

        last_exc = None
        for attempt in range(3):
            try:
                ticker = yf.Ticker(self.stock)
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    auto_adjust=False,
                )
                if data.empty:
                    raise ValueError(f"No price data fetched for {self.stock}.")

                data.columns = [str(c).strip() for c in data.columns]
                if "Adj Close" not in data.columns and "Close" in data.columns:
                    data["Adj Close"] = data["Close"]

                required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                missing = [c for c in required if c not in data.columns]
                if missing:
                    raise ValueError(f"Missing columns for {self.stock}: {missing}")

                data = self._normalize_datetime_index(data)
                data.to_csv(cache_path, index=True)
                return data
            except Exception as exc:
                last_exc = exc
                # Exponential backoff on transient failures / rate limits.
                time.sleep(2**attempt)

        # Secondary provider fallback for when Yahoo blocks/rate-limits.
        try:
            data = self._fetch_from_stooq()
            data = self._normalize_datetime_index(data)
            data.to_csv(cache_path, index=True)
            return data
        except Exception as exc:
            last_exc = exc

        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return self._normalize_datetime_index(cached)

        raise ValueError(
            f"Download failed and no valid cache found for {self.stock} at {cache_path}."
        ) from last_exc

    def _fetch_from_stooq(self) -> pd.DataFrame:
        symbol = quote(self.stock.lower())
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        data = pd.read_csv(url)
        if data.empty:
            raise ValueError(f"No data from Stooq for {self.stock}.")

        if "Date" not in data.columns:
            raise ValueError(f"Stooq response missing Date column for {self.stock}.")

        data["Date"] = pd.to_datetime(data["Date"])
        data = data[(data["Date"] >= pd.Timestamp(self.start_date)) & (data["Date"] <= pd.Timestamp(self.end_date))]
        data = data.set_index("Date").sort_index()

        # Stooq doesn't provide adjusted close; align schema for downstream features.
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]

        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Stooq missing columns for {self.stock}: {missing}")

        return data[required]

    def _normalize_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data

    def _create_var_target(self) -> None:
        self.data["do"] = self.data["Open"].pct_change() * 100
        self.data["dh"] = self.data["High"].pct_change() * 100
        self.data["dl"] = self.data["Low"].pct_change() * 100
        self.data["dc"] = self.data["Close"].pct_change() * 100
        self.data["dac"] = self.data["Adj Close"].pct_change() * 100
        self.data["dv"] = self.data["Volume"].pct_change() * 100
        self.data["return"] = (
            (self.data["Adj Close"] - self.data["Adj Close"].shift(1))
            / self.data["Adj Close"].shift(1)
        )
        self.data = self.data[["do", "dh", "dl", "dc", "dv", "dac", "return"]]

    def _clean(self) -> None:
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)

    def _normalize(self) -> None:
        columns = ["do", "dh", "dl", "dc", "dv", "dac"]
        train_slice = self.data[self.data.index < self.valid_start][columns]
        if train_slice.empty:
            raise ValueError(
                f"No training rows before valid_start={self.valid_start.date()} for {self.stock}."
            )

        scaler = StandardScaler()
        scaler.fit(train_slice)
        self.data[columns] = scaler.transform(self.data[columns])

    def getLoaders(self, datas, batch_size: int) -> None:
        def loader(x: torch.Tensor, y: torch.Tensor, bs: int, drop_last: bool) -> DataLoader:
            dataset = TensorDataset(x, y)
            return DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=drop_last)

        x_train, y_train, x_valid, y_valid, x_test, y_test = datas
        if len(x_train) == 0 or len(x_valid) == 0 or len(x_test) == 0:
            raise ValueError("Train/valid/test split produced empty dataset.")

        self.trainloader = loader(x_train, y_train, batch_size, drop_last=True)
        self.validloader = loader(x_valid, y_valid, len(y_valid), drop_last=False)
        self.testloader = loader(x_test, y_test, len(y_test), drop_last=False)

    @abstractmethod
    def prepare_data(self, verbose: bool = False):
        raise NotImplementedError
