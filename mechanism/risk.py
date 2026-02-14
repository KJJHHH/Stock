from dataclasses import dataclass
import pandas as pd


@dataclass
class RiskConfig:
    max_gross: float = 1.2
    max_net: float = 0.2
    max_single_name: float = 0.1


class RiskGuard:
    def __init__(self, config: RiskConfig):
        self.config = config

    def enforce(self, w: pd.Series) -> pd.Series:
        out = w.astype(float).clip(
            lower=-self.config.max_single_name,
            upper=self.config.max_single_name,
        )

        gross = out.abs().sum()
        if gross > self.config.max_gross and gross > 0:
            out = out / gross * self.config.max_gross

        net = out.sum()
        if abs(net) > self.config.max_net:
            out = out - (net - (self.config.max_net if net > 0 else -self.config.max_net)) / len(out)

        return out.fillna(0.0)
