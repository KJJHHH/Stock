import numpy as np

from predict import predict, predict_sklearn
from strategy import StrategyConfig, run_strategy, run_strategy_ohlc


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
        open_prices=None,
        close_prices=None,
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
        threshold = self.threshold if task == "classification" else 0.0
        if open_prices is not None and close_prices is not None:
            result = run_strategy_ohlc(
                predicted,
                open_prices,
                close_prices,
                StrategyConfig(
                    threshold=threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
                dates=y_dates,
            )
        else:
            result = run_strategy(
                predicted,
                raw_returns,
                StrategyConfig(
                    threshold=threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
            )
        return predicted, y_test, result["final_return"], result["equity_curve"], y_dates, result.get("trades")


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
        open_prices=None,
        close_prices=None,
    ):
        predicted, y_test = predict_sklearn(
            model,
            X_test,
            y_test,
            return_mean=return_mean,
            return_std=return_std,
            task=task,
        )
        threshold = self.threshold if task == "classification" else 0.0
        if open_prices is not None and close_prices is not None:
            result = run_strategy_ohlc(
                predicted,
                open_prices,
                close_prices,
                StrategyConfig(
                    threshold=threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
                dates=y_dates,
            )
        else:
            result = run_strategy(
                predicted,
                raw_returns.reshape(-1, 1),
                StrategyConfig(
                    threshold=threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
            )
        return predicted, y_test, result["final_return"], result["equity_curve"], y_dates, result.get("trades")


class SignalBacktester:
    def __init__(self, threshold: float = 0.0, trading_cost: float = 0.0):
        self.threshold = threshold
        self.trading_cost = trading_cost

    def run(
        self,
        signals: np.ndarray,
        raw_returns: np.ndarray,
        y_dates=None,
        open_prices=None,
        close_prices=None,
    ):
        signals = np.asarray(signals).reshape(-1, 1)
        actual_return = np.asarray(raw_returns).reshape(-1, 1)
        if open_prices is not None and close_prices is not None:
            result = run_strategy_ohlc(
                signals,
                open_prices,
                close_prices,
                StrategyConfig(
                    threshold=self.threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
                dates=y_dates,
            )
        else:
            result = run_strategy(
                signals,
                actual_return,
                StrategyConfig(
                    threshold=self.threshold,
                    trading_cost=self.trading_cost,
                    slippage=0.0,
                    allow_short=False,
                    max_position=1.0,
                ),
            )
        return signals, result["final_return"], result["equity_curve"], result.get("trades")


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
    open_prices=None,
    close_prices=None,
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
        open_prices=open_prices,
        close_prices=close_prices,
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
    open_prices=None,
    close_prices=None,
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
        open_prices=open_prices,
        close_prices=close_prices,
    )


def backtest_signals(
    signals: np.ndarray,
    raw_returns: np.ndarray,
    threshold: float = 0.0,
    trading_cost: float = 0.0,
    y_dates=None,
    open_prices=None,
    close_prices=None,
):
    return SignalBacktester(threshold=threshold, trading_cost=trading_cost).run(
        signals,
        raw_returns,
        y_dates=y_dates,
        open_prices=open_prices,
        close_prices=close_prices,
    )
