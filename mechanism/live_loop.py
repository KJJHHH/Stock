from dataclasses import dataclass
import pandas as pd

from .alpha import build_alpha_signal
from .execution import ExecutionConfig, simulate_execution
from .portfolio import PortfolioConfig, build_target_weights
from .risk import RiskConfig, RiskGuard


@dataclass
class LiveState:
    portfolio_value: float
    weights: pd.Series


class DailyTradingEngine:
    def __init__(
        self,
        portfolio_cfg: PortfolioConfig,
        execution_cfg: ExecutionConfig,
        risk_cfg: RiskConfig,
    ):
        self.portfolio_cfg = portfolio_cfg
        self.execution_cfg = execution_cfg
        self.risk_guard = RiskGuard(risk_cfg)

    def step(
        self,
        close_history: pd.DataFrame,
        dollar_adv: pd.Series,
        state: LiveState,
    ) -> tuple[LiveState, dict]:
        signal_df = build_alpha_signal(close_history)
        signal = signal_df.iloc[-1]
        vol = close_history.pct_change().rolling(20).std().iloc[-1].replace(0, 1e-4)

        target = build_target_weights(signal, vol, self.portfolio_cfg)
        target = self.risk_guard.enforce(target)

        executed, cost = simulate_execution(
            target,
            state.weights,
            dollar_adv,
            state.portfolio_value,
            self.execution_cfg,
        )

        out_state = LiveState(portfolio_value=state.portfolio_value - cost, weights=executed)
        report = {
            "target_weights": target,
            "executed_weights": executed,
            "cost": cost,
            "portfolio_value": out_state.portfolio_value,
        }
        return out_state, report
