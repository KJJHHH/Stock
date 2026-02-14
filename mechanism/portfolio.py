from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    target_gross: float = 1.0
    max_weight: float = 0.1
    long_short: bool = True
    risk_aversion: float = 1.0


def build_target_weights(
    signal: pd.Series,
    volatility: pd.Series,
    config: PortfolioConfig,
) -> pd.Series:
    """Convert alpha signal to portfolio weights with risk-scaling and caps."""
    s = signal.astype(float).copy()
    vol = volatility.astype(float).replace(0, np.nan).fillna(volatility.median())

    # Risk-adjusted score (closer to mean-variance style sizing)
    raw = (s / (config.risk_aversion * vol + 1e-8)).replace([np.inf, -np.inf], 0.0)

    if config.long_short:
        raw = raw - raw.mean()
    else:
        raw = raw.clip(lower=0.0)

    if raw.abs().sum() == 0:
        return pd.Series(0.0, index=raw.index)

    w = raw / raw.abs().sum() * config.target_gross
    w = w.clip(lower=-config.max_weight, upper=config.max_weight)

    gross = w.abs().sum()
    if gross > 0:
        w = w / gross * config.target_gross

    return w.fillna(0.0)
