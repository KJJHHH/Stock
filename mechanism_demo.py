import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mechanism.execution import ExecutionConfig
from mechanism.live_loop import DailyTradingEngine, LiveState
from mechanism.portfolio import PortfolioConfig
from mechanism.risk import RiskConfig


def main():
    np.random.seed(42)
    tickers = ["2330.TW", "2317.TW", "2454.TW", "2881.TW", "6505.TW"]
    dates = pd.date_range("2022-01-01", periods=300, freq="B")

    rets = np.random.normal(0.0003, 0.015, (len(dates), len(tickers)))
    close = 100 * (1 + pd.DataFrame(rets, index=dates, columns=tickers)).cumprod()
    adv = pd.Series(5_000_000.0, index=tickers)

    engine = DailyTradingEngine(
        portfolio_cfg=PortfolioConfig(target_gross=1.0, max_weight=0.2, long_short=True),
        execution_cfg=ExecutionConfig(fee_bps=2.0, spread_bps=1.0, impact_coef=8.0),
        risk_cfg=RiskConfig(max_gross=1.0, max_net=0.2, max_single_name=0.2),
    )

    initial_value = 1_000_000.0
    state = LiveState(portfolio_value=initial_value, weights=pd.Series(0.0, index=tickers))

    history_dates = []
    history_portfolio = []
    history_buyhold = []

    start_idx = 120
    for i in range(start_idx, len(close)):
        # Daily rebalance using history up to current day, apply next-day realized return.
        close_hist = close.iloc[: i + 1]
        state, report = engine.step(close_hist, adv, state)

        if i + 1 < len(close):
            next_ret = (close.iloc[i + 1] / close.iloc[i] - 1.0).fillna(0.0)
            gross_ret = float((report["executed_weights"] * next_ret).sum())
            state.portfolio_value *= (1.0 + gross_ret)

            bh_ret = float(next_ret.mean())
            buyhold_value = initial_value if not history_buyhold else history_buyhold[-1]
            buyhold_value *= (1.0 + bh_ret)
        else:
            buyhold_value = history_buyhold[-1] if history_buyhold else initial_value

        history_dates.append(close.index[i])
        history_portfolio.append(state.portfolio_value)
        history_buyhold.append(buyhold_value)

    print("Mechanism backtest report")
    print(f"final_portfolio_value: {history_portfolio[-1]:.2f}")
    print(f"final_buyhold_value: {history_buyhold[-1]:.2f}")

    out_dir = Path("result/mechanism-result")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "portfolio_equity.png"

    plt.figure(figsize=(10, 5))
    plt.plot(history_dates, history_portfolio, label="Mechanism Portfolio")
    plt.plot(history_dates, history_buyhold, label="Equal-weight Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Mechanism Portfolio Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved portfolio plot: {plot_path}")


if __name__ == "__main__":
    main()
