import numpy as np

from predict import predict, predict_sklearn


class Backtester:
    def __init__(self, threshold: float = 0.5, trading_cost: float = 0.0):
        self.threshold = threshold
        self.trading_cost = trading_cost

    def run(
        self,
        model,
        X_test,
        scaler,
        y_test,
        y_dates=None,
        device=None,
        return_mean=0.0,
        return_std=1.0,
        task="regression",
        raw_returns=None,
    ):
        predicted, y_test = predict(
            model,
            X_test,
            scaler,
            y_test,
            device=device,
            return_mean=return_mean,
            return_std=return_std,
            task=task,
        )
        actual_return = raw_returns
        if task == "classification":
            position = np.where(predicted > self.threshold, 1.0, 0.0)
        else:
            position = np.where(predicted > 0, 1.0, 0.0)
        strategy_return = position * actual_return
        if self.trading_cost:
            position_change = np.abs(np.diff(position, axis=0, prepend=0.0))
            strategy_return = strategy_return - (position_change * self.trading_cost)
        cumulative_return = np.cumprod(1 + strategy_return, axis=0) - 1
        return predicted, y_test, float(cumulative_return[-1][0]), cumulative_return, y_dates


class SklearnBacktester:
    def __init__(self, threshold: float = 0.5, trading_cost: float = 0.0):
        self.threshold = threshold
        self.trading_cost = trading_cost

    def run(
        self,
        model,
        X_test,
        y_test,
        y_dates=None,
        return_mean=0.0,
        return_std=1.0,
        task="regression",
        raw_returns=None,
    ):
        predicted, y_test = predict_sklearn(
            model,
            X_test,
            y_test,
            return_mean=return_mean,
            return_std=return_std,
            task=task,
        )
        actual_return = raw_returns.reshape(-1, 1)
        if task == "classification":
            position = np.where(predicted > self.threshold, 1.0, 0.0)
        else:
            position = np.where(predicted > 0, 1.0, 0.0)
        strategy_return = position * actual_return
        if self.trading_cost:
            position_change = np.abs(np.diff(position, axis=0, prepend=0.0))
            strategy_return = strategy_return - (position_change * self.trading_cost)
        cumulative_return = np.cumprod(1 + strategy_return, axis=0) - 1
        return predicted, y_test, float(cumulative_return[-1][0]), cumulative_return, y_dates


class SignalBacktester:
    def __init__(self, threshold: float = 0.0, trading_cost: float = 0.0):
        self.threshold = threshold
        self.trading_cost = trading_cost

    def run(self, signals: np.ndarray, raw_returns: np.ndarray):
        signals = np.asarray(signals).reshape(-1)
        actual_return = np.asarray(raw_returns).reshape(-1)
        min_len = min(len(signals), len(actual_return))
        signals = signals[:min_len].reshape(-1, 1)
        actual_return = actual_return[:min_len].reshape(-1, 1)
        position = np.where(signals > self.threshold, 1.0, 0.0)
        strategy_return = position * actual_return
        if self.trading_cost:
            position_change = np.abs(np.diff(position, axis=0, prepend=0.0))
            strategy_return = strategy_return - (position_change * self.trading_cost)
        cumulative_return = np.cumprod(1 + strategy_return, axis=0) - 1
        return signals, float(cumulative_return[-1][0]), cumulative_return


def backtest(
    model,
    X_test,
    scaler,
    y_test,
    y_dates=None,
    device=None,
    return_mean=0.0,
    return_std=1.0,
    task="regression",
    raw_returns=None,
    threshold=0.5,
    trading_cost=0.0,
):
    return Backtester(threshold=threshold, trading_cost=trading_cost).run(
        model,
        X_test,
        scaler,
        y_test,
        y_dates=y_dates,
        device=device,
        return_mean=return_mean,
        return_std=return_std,
        task=task,
        raw_returns=raw_returns,
    )


def backtest_sklearn(
    model,
    X_test,
    y_test,
    y_dates=None,
    return_mean=0.0,
    return_std=1.0,
    task="regression",
    raw_returns=None,
    threshold=0.5,
    trading_cost=0.0,
):
    return SklearnBacktester(threshold=threshold, trading_cost=trading_cost).run(
        model,
        X_test,
        y_test,
        y_dates=y_dates,
        return_mean=return_mean,
        return_std=return_std,
        task=task,
        raw_returns=raw_returns,
    )


def backtest_signals(
    signals: np.ndarray,
    raw_returns: np.ndarray,
    threshold: float = 0.0,
    trading_cost: float = 0.0,
):
    return SignalBacktester(threshold=threshold, trading_cost=trading_cost).run(
        signals, raw_returns
    )
