from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.svm import SVR


@dataclass
class SVRBounds:
    c_min: float = 0.01
    c_max: float = 1.0
    eps_min: float = 0.1
    eps_max: float = 1.0
    gamma_min: float = 1e-6
    gamma_max: float = 1e-3


def minmax_scale(values: np.ndarray, data_min=None, data_max=None) -> Tuple[np.ndarray, float, float]:
    if data_min is None:
        data_min = float(np.min(values))
    if data_max is None:
        data_max = float(np.max(values))
    denom = data_max - data_min
    if denom == 0:
        denom = 1.0
    scaled = (values - data_min) / denom
    return scaled, data_min, data_max


def time_index_from_dates(dates: np.ndarray) -> np.ndarray:
    try:
        days = (dates.astype("datetime64[D]") - dates.astype("datetime64[D]")[0]).astype(np.int64)
        return days.reshape(-1, 1).astype(np.float64)
    except Exception:
        return np.arange(len(dates), dtype=np.float64).reshape(-1, 1)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _evaluate_svr(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    c: float,
    eps: float,
    gamma: float,
    recent_years: int = 5,
) -> float:
    model = SVR(C=c, epsilon=eps, gamma=gamma)
    y = np.asarray(y).reshape(-1)
    model.fit(X, y)
    pred = model.predict(X)
    full_mape = mape(y, pred)
    recent_mask = None
    if dates is not None and len(dates) > 0:
        cutoff = dates[-1] - np.timedelta64(recent_years * 365, "D")
        recent_mask = dates >= cutoff
    if recent_mask is not None and np.any(recent_mask):
        recent_mape = mape(y[recent_mask], pred[recent_mask])
    else:
        recent_mape = full_mape
    return 0.5 * (full_mape + recent_mape)


def ga_optimize_svr(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    bounds: SVRBounds,
    generations: int = 30,
    population_size: int = 30,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    random_state: int = 42,
    recent_years: int = 5,
):
    rng = np.random.default_rng(random_state)
    pop = np.column_stack(
        [
            rng.uniform(bounds.c_min, bounds.c_max, population_size),
            rng.uniform(bounds.eps_min, bounds.eps_max, population_size),
            rng.uniform(bounds.gamma_min, bounds.gamma_max, population_size),
        ]
    )

    fitness = np.array(
        [
            _evaluate_svr(X, y, dates, c, eps, gamma, recent_years=recent_years)
            for c, eps, gamma in pop
        ],
        dtype=np.float64,
    )

    def tournament() -> np.ndarray:
        idx = rng.choice(len(pop), size=3, replace=False)
        best = idx[np.argmin(fitness[idx])]
        return pop[best]

    for _ in range(generations):
        new_pop = []
        elite_idx = int(np.argmin(fitness))
        new_pop.append(pop[elite_idx].copy())

        while len(new_pop) < population_size:
            parent1 = tournament()
            parent2 = tournament()
            if rng.random() < crossover_rate:
                alpha = rng.random()
                child = alpha * parent1 + (1.0 - alpha) * parent2
            else:
                child = parent1.copy()

            if rng.random() < mutation_rate:
                ranges = np.array(
                    [
                        bounds.c_max - bounds.c_min,
                        bounds.eps_max - bounds.eps_min,
                        bounds.gamma_max - bounds.gamma_min,
                    ],
                    dtype=np.float64,
                )
                noise = rng.normal(0.0, 0.1, size=3) * ranges
                child = child + noise

            child[0] = np.clip(child[0], bounds.c_min, bounds.c_max)
            child[1] = np.clip(child[1], bounds.eps_min, bounds.eps_max)
            child[2] = np.clip(child[2], bounds.gamma_min, bounds.gamma_max)
            new_pop.append(child)

        pop = np.array(new_pop, dtype=np.float64)
        fitness = np.array(
            [
                _evaluate_svr(X, y, dates, c, eps, gamma, recent_years=recent_years)
                for c, eps, gamma in pop
            ],
            dtype=np.float64,
        )

    best_idx = int(np.argmin(fitness))
    best = pop[best_idx]
    return {
        "C": float(best[0]),
        "epsilon": float(best[1]),
        "gamma": float(best[2]),
        "fitness": float(fitness[best_idx]),
    }


def train_predict_iga_svr(
    train_close: np.ndarray,
    train_dates: np.ndarray,
    backtest_close: np.ndarray,
    backtest_dates: np.ndarray,
    generations: int = 30,
    population_size: int = 30,
    recent_years: int = 5,
    random_state: int = 42,
):
    full_dates = np.concatenate([train_dates, backtest_dates])
    full_close = np.concatenate([train_close, backtest_close])
    X_full = time_index_from_dates(full_dates)
    X_train = X_full[: len(train_close)]
    X_backtest = X_full[len(train_close) :]

    y_train, data_min, data_max = minmax_scale(train_close)
    y_train = y_train.reshape(-1)
    y_backtest, _, _ = minmax_scale(backtest_close, data_min=data_min, data_max=data_max)
    y_backtest = y_backtest.reshape(-1)

    scale = 1.0 / (X_train.shape[1] * np.var(X_train))
    bounds = SVRBounds(gamma_max=max(scale / 10.0, 1e-6))

    best = ga_optimize_svr(
        X_train,
        y_train,
        train_dates,
        bounds,
        generations=generations,
        population_size=population_size,
        recent_years=recent_years,
        random_state=random_state,
    )

    model = SVR(C=best["C"], epsilon=best["epsilon"], gamma=best["gamma"])
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_backtest)
    pred = pred_scaled * (data_max - data_min) + data_min
    return pred, y_backtest, best
