import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class MSKalmanConfig:
    transition: np.ndarray  # shape (K, K)
    a: np.ndarray  # state transition per regime
    q: np.ndarray  # process noise variance per regime
    r: np.ndarray  # observation noise variance per regime
    init_state: float = 0.0
    init_var: float = 1.0


def default_config_from_returns(returns: np.ndarray) -> MSKalmanConfig:
    var = float(np.var(returns))
    if not math.isfinite(var) or var <= 0:
        var = 1e-6
    q = np.array([0.1 * var, 5.0 * var], dtype=np.float64)
    r = np.array([0.5 * var, 2.0 * var], dtype=np.float64)
    transition = np.array([[0.97, 0.03], [0.03, 0.97]], dtype=np.float64)
    a = np.array([1.0, 1.0], dtype=np.float64)
    return MSKalmanConfig(
        transition=transition,
        a=a,
        q=q,
        r=r,
        init_state=0.0,
        init_var=max(var, 1e-6),
    )


class MSKalmanFilter:
    def __init__(self, config: MSKalmanConfig):
        self.config = config
        self.num_regimes = len(config.a)
        self.state = np.full(self.num_regimes, config.init_state, dtype=np.float64)
        self.var = np.full(self.num_regimes, config.init_var, dtype=np.float64)
        self.regime_prob = np.full(self.num_regimes, 1.0 / self.num_regimes, dtype=np.float64)

    @staticmethod
    def _normal_pdf(x: float, mean: float, var: float) -> float:
        var = max(var, 1e-12)
        denom = math.sqrt(2.0 * math.pi * var)
        return math.exp(-0.5 * (x - mean) ** 2 / var) / denom

    def step(self, observation: float) -> Tuple[float, np.ndarray]:
        trans = self.config.transition
        prior_prob = self.regime_prob
        mix_prob = prior_prob[:, None] * trans
        norm = mix_prob.sum(axis=0)
        norm = np.where(norm == 0, 1e-12, norm)
        mixing = mix_prob / norm

        mixed_state = (mixing * self.state[:, None]).sum(axis=0)
        mixed_var = np.zeros(self.num_regimes, dtype=np.float64)
        for j in range(self.num_regimes):
            diff = self.state - mixed_state[j]
            mixed_var[j] = (mixing[:, j] * (self.var + diff ** 2)).sum()

        new_state = np.zeros(self.num_regimes, dtype=np.float64)
        new_var = np.zeros(self.num_regimes, dtype=np.float64)
        likelihood = np.zeros(self.num_regimes, dtype=np.float64)

        for j in range(self.num_regimes):
            a = self.config.a[j]
            q = self.config.q[j]
            r = self.config.r[j]
            pred_state = a * mixed_state[j]
            pred_var = a * a * mixed_var[j] + q
            innov = observation - pred_state
            s = pred_var + r
            k = pred_var / max(s, 1e-12)
            new_state[j] = pred_state + k * innov
            new_var[j] = (1.0 - k) * pred_var
            likelihood[j] = self._normal_pdf(observation, pred_state, s)

        updated_prob = norm * likelihood
        prob_sum = updated_prob.sum()
        if prob_sum <= 0:
            updated_prob = np.full(self.num_regimes, 1.0 / self.num_regimes, dtype=np.float64)
        else:
            updated_prob = updated_prob / prob_sum

        self.state = new_state
        self.var = new_var
        self.regime_prob = updated_prob
        combined_state = float(np.dot(self.regime_prob, self.state))
        return combined_state, self.regime_prob.copy()

    def run(self, observations: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        signals = []
        probs = []
        for obs in observations:
            signal, prob = self.step(float(obs))
            signals.append(signal)
            probs.append(prob)
        return np.array(signals, dtype=np.float64), np.array(probs, dtype=np.float64)

