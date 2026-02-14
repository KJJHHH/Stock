from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    threshold: float = 0.0
    trading_cost: float = 0.0
    slippage: float = 0.0
    allow_short: bool = False
    max_position: float = 1.0


def _to_position(
    signals: np.ndarray,
    threshold: float,
    allow_short: bool,
    max_position: float,
) -> np.ndarray:
    s = np.asarray(signals, dtype=np.float64).reshape(-1)
    position = np.zeros_like(s, dtype=np.float64)

    if allow_short:
        position[s > threshold] = 1.0
        position[s < -threshold] = -1.0
    else:
        position[s > threshold] = 1.0

    position = np.clip(position, -max_position, max_position)
    return position.reshape(-1, 1)


def run_strategy(
    signals: np.ndarray,
    raw_returns: np.ndarray,
    config: StrategyConfig,
):
    ret = np.asarray(raw_returns, dtype=np.float64).reshape(-1, 1)
    pos = _to_position(
        signals=signals,
        threshold=config.threshold,
        allow_short=config.allow_short,
        max_position=config.max_position,
    )

    n = min(len(pos), len(ret))
    pos = pos[:n]
    ret = ret[:n]

    gross = pos * ret
    turnover = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, 1))))
    total_cost = turnover * (config.trading_cost + config.slippage)
    net = gross - total_cost
    equity = np.cumprod(1.0 + net, axis=0) - 1.0

    return {
        "position": pos,
        "strategy_return": net,
        "equity_curve": equity,
        "final_return": float(equity[-1][0]) if len(equity) else 0.0,
    }


def run_strategy_ohlc(
    signals: np.ndarray,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    config: StrategyConfig,
    dates=None,
):
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    opn = np.asarray(open_prices, dtype=np.float64).reshape(-1)
    cls = np.asarray(close_prices, dtype=np.float64).reshape(-1)

    n = min(len(sig), len(opn), len(cls))
    sig = sig[:n]
    opn = opn[:n]
    cls = cls[:n]

    pos = _to_position(
        signals=sig,
        threshold=config.threshold,
        allow_short=config.allow_short,
        max_position=config.max_position,
    ).reshape(-1)

    prev_pos = 0.0
    returns = []
    rows = []

    for i in range(n):
        day_ret = (cls[i] / opn[i] - 1.0) if opn[i] != 0 else 0.0
        gross = pos[i] * day_ret
        turnover = abs(pos[i] - prev_pos)
        cost = turnover * (config.trading_cost + config.slippage)
        net = gross - cost
        returns.append(net)

        action = "HOLD"
        if prev_pos <= 0 and pos[i] > 0:
            action = "BUY"
        elif prev_pos >= 0 and pos[i] < 0:
            action = "SHORT"
        elif prev_pos != 0 and pos[i] == 0:
            action = "FLAT"
        elif prev_pos != pos[i]:
            action = "REBAL"

        rows.append(
            {
                "date": dates[i] if dates is not None and i < len(dates) else i,
                "signal": float(sig[i]),
                "position": float(pos[i]),
                "action": action,
                "entry_open": float(opn[i]),
                "exit_close": float(cls[i]),
                "gross_return": float(gross),
                "cost": float(cost),
                "net_return": float(net),
            }
        )
        prev_pos = pos[i]

    ret = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
    equity = np.cumprod(1.0 + ret, axis=0) - 1.0

    return {
        "position": pos.reshape(-1, 1),
        "strategy_return": ret,
        "equity_curve": equity,
        "final_return": float(equity[-1][0]) if len(equity) else 0.0,
        "trades": pd.DataFrame(rows),
    }
