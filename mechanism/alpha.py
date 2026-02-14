import numpy as np
import pandas as pd


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def build_alpha_signal(close: pd.DataFrame) -> pd.DataFrame:
    """Build cross-sectional alpha scores from price history.

    Args:
        close: DataFrame indexed by date, columns are tickers, values are close prices.

    Returns:
        DataFrame of alpha scores (higher = stronger long signal).
    """
    ret_1 = close.pct_change(1)
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    vol_20 = ret_1.rolling(20).std()

    # Simple but effective composite: trend + short-term reversal + risk adjustment.
    momentum = ret_20
    reversal = -ret_5
    quality = (ret_20 / (vol_20 + 1e-8)).replace([np.inf, -np.inf], np.nan)

    score = 0.5 * momentum + 0.3 * reversal + 0.2 * quality
    score = score.apply(_zscore)

    # Cross-sectional normalize each day.
    score = score.sub(score.mean(axis=1), axis=0)
    denom = score.std(axis=1).replace(0, np.nan)
    score = score.div(denom, axis=0)
    return score.fillna(0.0)
