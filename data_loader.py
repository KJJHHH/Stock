
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

RETURN_SCALE = 100.0

def load_raw_close(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = data["Close"].dropna().astype(np.float32).to_numpy()
    dates = data.index.to_numpy()
    return close_prices, dates

def build_features(close):
    returns = close.pct_change()
    features = pd.DataFrame(index=returns.index)
    features["ret_1"] = returns
    day_of_week = features.index.dayofweek.to_numpy(dtype=np.float32)
    month_of_year = features.index.month.to_numpy(dtype=np.float32)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    features["moy_sin"] = np.sin(2 * np.pi * month_of_year / 12.0)
    features["moy_cos"] = np.cos(2 * np.pi * month_of_year / 12.0)
    day_of_year = features.index.dayofyear.to_numpy(dtype=np.float32)
    for k in (1, 2, 4):
        features[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)
    for window in (5, 20, 60):
        ma = close.rolling(window).mean()
        features[f"ma_{window}_dev"] = (close / ma) - 1.0
    for window in (5, 20):
        features[f"vol_{window}"] = returns.rolling(window).std()

    # Momentum indicators
    for window in (5, 10, 20):
        features[f"roc_{window}"] = close.pct_change(window)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd - macd_signal

    low_14 = close.rolling(14).min()
    high_14 = close.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_k.rolling(3).mean()
    features["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    features["rsi_14"] = 100 - (100 / (1 + rs))

    features = features.dropna()
    returns = returns.loc[features.index]
    return features, returns


def load_features(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    close = data["Close"].astype(np.float32)
    features, returns = build_features(close)
    raw_returns = returns.to_numpy(dtype=np.float32).reshape(-1, 1)
    dates = features.index.to_numpy()
    return features.to_numpy(dtype=np.float32), raw_returns, dates


def load_execution_prices(ticker, start_date, end_date, dates):
    data = yf.download(ticker, start=start_date, end=end_date)
    frame = data[["Open", "Close"]].astype(np.float32)
    idx = pd.DatetimeIndex(dates)
    aligned = frame.reindex(idx).ffill().bfill()
    return (
        aligned["Open"].to_numpy(dtype=np.float32).reshape(-1, 1),
        aligned["Close"].to_numpy(dtype=np.float32).reshape(-1, 1),
    )


def load_data(ticker, start_date, end_date, scaler=None):
    features, raw_returns, dates = load_features(ticker, start_date, end_date)
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(features)
    else:
        data_scaled = scaler.transform(features)
    return data_scaled, scaler, dates, raw_returns

def create_sequences(
    data, seq_length, dates=None, raw_returns=None, classify=False
):
    xs, ys = [], []
    y_dates = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        if raw_returns is not None:
            if classify:
                y = 1.0 if raw_returns[i + seq_length] > 0 else 0.0
            else:
                y = raw_returns[i + seq_length] * RETURN_SCALE
        else:
            y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
        if dates is not None:
            y_dates.append(dates[i + seq_length])
    ys = np.array(ys).reshape(-1, 1)
    return np.array(xs), ys, np.array(y_dates) if dates is not None else None

def get_data(ticker, start_date, end_date, seq_length, classify=False):
    data, scaler, dates, raw_returns = load_data(ticker, start_date, end_date)
    X, y, y_dates = create_sequences(
        data,
        seq_length,
        dates=dates,
        raw_returns=raw_returns,
        classify=classify,
    )

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    dates_train, dates_test = y_dates[0:train_size], y_dates[train_size:len(y_dates)]

    return X_train, y_train, X_test, y_test, scaler, dates_train, dates_test

def get_sequences(ticker, start_date, end_date, seq_length, scaler=None, classify=False):
    data, scaler, dates, raw_returns = load_data(
        ticker, start_date, end_date, scaler=scaler
    )
    X, y, y_dates = create_sequences(
        data,
        seq_length,
        dates=dates,
        raw_returns=raw_returns,
        classify=classify,
    )

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    return X, y, scaler, y_dates, raw_returns[seq_length:]
