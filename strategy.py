from dataclasses import dataclass
import numpy as np


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
