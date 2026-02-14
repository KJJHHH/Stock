from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    fee_bps: float = 2.0
    spread_bps: float = 1.0
    impact_coef: float = 10.0  # scales with participation
    max_participation: float = 0.1


def simulate_execution(
    target_weights: pd.Series,
    prev_weights: pd.Series,
    dollar_adv: pd.Series,
    portfolio_value: float,
    config: ExecutionConfig,
):
    """Apply simple transaction-cost model and return executed weights + cost."""
    tw = target_weights.astype(float)
    pw = prev_weights.reindex(tw.index).fillna(0.0).astype(float)
    adv = dollar_adv.reindex(tw.index).replace(0, np.nan).fillna(dollar_adv.median())

    trade_w = tw - pw
    trade_dollar = trade_w.abs() * portfolio_value
    participation = (trade_dollar / (adv + 1e-8)).clip(upper=config.max_participation)

    linear_bps = config.fee_bps + config.spread_bps
    impact_bps = config.impact_coef * np.sqrt(participation)
    total_bps = linear_bps + impact_bps

    cost_dollar = (trade_dollar * total_bps / 10000.0).sum()

    # Damp target by participation pressure to mimic partial fill/urgency control.
    fill_ratio = 1.0 - (participation / (config.max_participation + 1e-8)) * 0.2
    executed = pw + trade_w * fill_ratio

    gross = executed.abs().sum()
    if gross > 0:
        executed = executed / gross * tw.abs().sum()

    return executed.fillna(0.0), float(cost_dollar)
