import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from backtest import backtest, backtest_sklearn, backtest_signals
from data_loader import create_sequences, load_execution_prices, load_features, load_raw_close
from sa_models.ms_kalman import MSKalmanConfig, MSKalmanFilter, default_config_from_returns
from sa_models.ga_svr import mape, train_predict_iga_svr
from sa_models import (
    DecoderOnly,
    LSTMModel,
    GRUModel,
    TransformerEncoderModel,
    TCNModel,
)
from train import train_model


def _plot_backtest(dates, equity_curve, buy_hold, ticker, model_name_str, task=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity_curve, label=f"{model_name_str} strategy equity")
    plt.plot(dates, buy_hold, label="Buy and hold equity")
    plt.xlabel("Date")
    plt.ylabel("Asset Value")
    if task is None:
        title = f"{ticker} Backtest Equity ({model_name_str})"
        plot_name = f"backtest_{ticker.replace('^', '')}_{model_name_str}.png"
    else:
        title = f"{ticker} Backtest Equity ({model_name_str})"
        plot_name = f"backtest_{ticker.replace('^', '')}_{model_name_str}_{task}.png"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    backend = plt.get_backend().lower()
    model_dir = os.path.join("result", f"{model_name_str}-result")
    os.makedirs(model_dir, exist_ok=True)
    if "agg" in backend:
        plot_path = os.path.join(model_dir, plot_name)
        plt.savefig(plot_path, dpi=150)
        print(f"Saved backtest plot to {plot_path}")
    else:
        plot_path = os.path.join(model_dir, plot_name)
        plt.savefig(plot_path, dpi=150)
        print(f"Saved backtest plot to {plot_path}")
        plt.show()


def run_supervised(
    ticker,
    train_start,
    train_end,
    backtest_start,
    backtest_end,
    seq_length,
    model_name,
    model_name_str,
    task,
    device,
    config,
    transaction_cost,
):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("backtest", exist_ok=True)
    full_start = train_start
    full_end = backtest_end
    features_all, raw_returns_all, dates_all = load_features(
        ticker, full_start, full_end
    )
    open_all, close_all = load_execution_prices(ticker, full_start, full_end, dates_all)
    train_mask = (dates_all >= np.datetime64(train_start)) & (
        dates_all < np.datetime64(train_end)
    )
    backtest_mask = (dates_all >= np.datetime64(backtest_start)) & (
        dates_all < np.datetime64(backtest_end)
    )
    train_features = features_all[train_mask]
    train_returns = raw_returns_all[train_mask]
    train_dates = dates_all[train_mask]
    backtest_features = features_all[backtest_mask]
    backtest_returns = raw_returns_all[backtest_mask]
    backtest_dates = dates_all[backtest_mask]
    backtest_open = open_all[backtest_mask]
    backtest_close = close_all[backtest_mask]

    if model_name.value == "MSKF":
        train_returns = train_returns.reshape(-1)
        backtest_returns = backtest_returns.reshape(-1)
        model_config = config.get("model_structure", {}).get(model_name.value, {})
        training_cfg = config.get("training", {})
        model_training = training_cfg.get("models", {}).get(model_name.value, {})
        if model_config:
            base_var = float(np.var(train_returns))
            if not np.isfinite(base_var) or base_var <= 0:
                base_var = 1e-6
            q_scale = np.array(model_config.get("q_scale", [0.1, 5.0]), dtype=np.float64)
            r_scale = np.array(model_config.get("r_scale", [0.5, 2.0]), dtype=np.float64)
            config_mskf = MSKalmanConfig(
                transition=np.array(model_config.get("transition", [[0.97, 0.03], [0.03, 0.97]]), dtype=np.float64),
                a=np.array(model_config.get("a", [1.0, 1.0]), dtype=np.float64),
                q=q_scale * base_var,
                r=r_scale * base_var,
                init_state=float(model_config.get("init_state", 0.0)),
                init_var=float(model_config.get("init_var_scale", 1.0)) * base_var,
            )
        else:
            config_mskf = default_config_from_returns(train_returns)
        mskf = MSKalmanFilter(config_mskf)
        _ = mskf.run(train_returns)
        signals, _ = mskf.run(backtest_returns)
        warmup_steps = int(model_training.get("warmup_steps", 0))
        if warmup_steps > 0:
            signals = signals[warmup_steps:]
            backtest_returns = backtest_returns[warmup_steps:]
            backtest_dates = backtest_dates[warmup_steps:]
            backtest_open = backtest_open[warmup_steps:]
            backtest_close = backtest_close[warmup_steps:]
        _, backtest_return, cumulative_return, trades = backtest_signals(
            signals,
            backtest_returns,
            threshold=0.0,
            trading_cost=transaction_cost,
            y_dates=backtest_dates,
            open_prices=backtest_open,
            close_prices=backtest_close,
        )
        if trades is not None:
            os.makedirs("result/trades", exist_ok=True)
            ticker_tag = ticker.replace("^", "")
            trades_path = os.path.join("result", "trades", f"trades_{ticker_tag}_{model_name_str}.csv")
            trades.to_csv(trades_path, index=False)
            print(f"Saved trade log to {trades_path}")
        print(f"Backtest strategy return: {backtest_return:.2%}")

        initial_asset = 100.0
        equity_curve = initial_asset * (1 + cumulative_return)
        buy_hold = initial_asset * np.cumprod(1 + backtest_returns, axis=0)
        _plot_backtest(backtest_dates, equity_curve, buy_hold, ticker, model_name_str)
        return

    if model_name.value == "IGA_SVR":
        train_close, train_close_dates = load_raw_close(ticker, train_start, train_end)
        backtest_close, backtest_close_dates = load_raw_close(ticker, backtest_start, backtest_end)
        model_config = config.get("model_structure", {}).get(model_name.value, {})
        training_cfg = config.get("training", {})
        model_training = training_cfg.get("models", {}).get(model_name.value, {})
        pred_close, _, best = train_predict_iga_svr(
            train_close,
            train_close_dates,
            backtest_close,
            backtest_close_dates,
            generations=int(model_config.get("generations", 30)),
            population_size=int(model_config.get("population_size", 30)),
            recent_years=int(model_training.get("recent_years", 5)),
            random_state=int(model_training.get("random_state", 42)),
        )
        actual_close = backtest_close
        min_len = min(len(pred_close), len(actual_close))
        pred_close = pred_close[:min_len]
        actual_close = actual_close[:min_len]
        pred_close = np.asarray(pred_close).reshape(-1)
        actual_close = np.asarray(actual_close).reshape(-1)
        price_mape = mape(actual_close, pred_close)
        predicted_return = (pred_close[1:] / actual_close[:-1]) - 1.0
        actual_return = (actual_close[1:] / actual_close[:-1]) - 1.0
        min_ret_len = min(len(predicted_return), len(actual_return))
        predicted_return = predicted_return[:min_ret_len]
        actual_return = actual_return[:min_ret_len]
        _, backtest_return, cumulative_return, _ = backtest_signals(
            predicted_return,
            actual_return,
            threshold=0.0,
            trading_cost=transaction_cost,
        )
        print(
            f"IGA_SVR best params: C={best['C']:.4f}, "
            f"epsilon={best['epsilon']:.4f}, gamma={best['gamma']:.6f}"
        )
        print(f"IGA_SVR price MAPE: {price_mape:.2f}%")
        print(f"Backtest strategy return: {backtest_return:.2%}")

        initial_asset = 100.0
        equity_curve = initial_asset * (1 + cumulative_return)
        buy_hold = initial_asset * np.cumprod(1 + actual_return, axis=0)
        plot_dates = backtest_close_dates[1:][: len(equity_curve)]
        _plot_backtest(plot_dates, equity_curve, buy_hold, ticker, model_name_str)
        return

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    backtest_features = scaler.transform(backtest_features)

    X_train, y_train, _ = create_sequences(
        train_features,
        seq_length,
        dates=train_dates,
        raw_returns=train_returns,
        classify=task == "classification",
    )
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    val_size = max(1, int(len(X_train) * 0.2))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
    print(f"Train samples: {len(X_tr)}, Validation samples: {len(X_val)}")
    weights = None
    return_mean = None
    return_std = None
    if task == "regression":
        return_mean = y_tr.mean().item()
        return_std = y_tr.std().item()
        if return_std == 0:
            return_std = 1.0
        y_tr = (y_tr - return_mean) / return_std
        y_val = (y_val - return_mean) / return_std

    X_backtest, y_backtest, _ = create_sequences(
        backtest_features,
        seq_length,
        dates=backtest_dates,
        raw_returns=backtest_returns,
        classify=task == "classification",
    )
    X_backtest = torch.from_numpy(X_backtest).float()
    y_backtest = torch.from_numpy(y_backtest).float()
    raw_returns = backtest_returns[seq_length:]
    backtest_open = backtest_open[seq_length:]
    backtest_close = backtest_close[seq_length:]
    backtest_dates = backtest_dates[seq_length:]
    if task == "regression":
        y_backtest = (y_backtest - return_mean) / return_std

    input_dim = X_tr.shape[-1]
    trades = None

    if model_name.value == "GBDT":
        model_config = config.get("model_structure", {}).get(model_name.value, {})
        training_cfg = config.get("training", {})
        model_training = training_cfg.get("models", {}).get(model_name.value, {})
        random_state = int(model_training.get("random_state", 42))
        early_cfg = model_config.get("early_stopping", {}) or {}
        n_iter_no_change = early_cfg.get("n_iter_no_change")
        validation_fraction = early_cfg.get("validation_fraction")
        tol = early_cfg.get("tol")
        X_tr_np = X_tr.numpy().reshape(len(X_tr), -1)
        y_tr_np = y_tr.numpy().reshape(-1)
        X_val_np = X_val.numpy().reshape(len(X_val), -1)
        y_val_np = y_val.numpy().reshape(-1)
        if task == "classification":
            model = GradientBoostingClassifier(
                random_state=random_state,
                verbose=1,
                n_estimators=model_config.get("n_estimators", 100),
                learning_rate=model_config.get("learning_rate", 0.1),
                max_depth=model_config.get("max_depth", 3),
                subsample=model_config.get("subsample", 1.0),
                n_iter_no_change=n_iter_no_change,
                validation_fraction=validation_fraction,
                tol=tol,
            )
            model.fit(X_tr_np, y_tr_np.astype(int))
            val_probs = model.predict_proba(X_val_np)[:, 1]
            target_rate = float((y_val_np == 1).sum()) / max(1, len(y_val_np))
            threshold = np.quantile(val_probs, 1.0 - target_rate)
        else:
            model = GradientBoostingRegressor(
                random_state=random_state,
                verbose=1,
                n_estimators=model_config.get("n_estimators", 100),
                learning_rate=model_config.get("learning_rate", 0.1),
                max_depth=model_config.get("max_depth", 3),
                subsample=model_config.get("subsample", 1.0),
                n_iter_no_change=n_iter_no_change,
                validation_fraction=validation_fraction,
                tol=tol,
            )
            model.fit(X_tr_np, y_tr_np)
            threshold = 0.0

        predicted, y_backtest, backtest_return, cumulative_return, backtest_dates, trades = backtest_sklearn(
            model,
            X_backtest.numpy(),
            y_backtest.numpy(),
            y_dates=backtest_dates,
            return_mean=return_mean,
            return_std=return_std,
            task=task,
            raw_returns=raw_returns,
            threshold=threshold,
            trading_cost=transaction_cost,
            open_prices=backtest_open,
            close_prices=backtest_close,
        )
    else:
        model_config = config.get("model_structure", {}).get(model_name.value, {})
        if model_name.value == "DECODER_ONLY":
            model = DecoderOnly(
                input_dim=input_dim,
                d_model=model_config.get("d_model", 64),
                nhead=model_config.get("nhead", 4),
                num_layers=model_config.get("num_layers", 6),
                dim_feedforward=model_config.get("dim_feedforward", 512),
                dropout=model_config.get("dropout", 0.1),
            )
        elif model_name.value == "LSTM":
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=model_config.get("hidden_dim", 128),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.1),
            )
        elif model_name.value == "GRU":
            model = GRUModel(
                input_dim=input_dim,
                hidden_dim=model_config.get("hidden_dim", 128),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.1),
            )
        elif model_name.value == "TRANSFORMER_ENCODER":
            model = TransformerEncoderModel(
                input_dim=input_dim,
                d_model=model_config.get("d_model", 64),
                nhead=model_config.get("nhead", 4),
                num_layers=model_config.get("num_layers", 3),
                dim_feedforward=model_config.get("dim_feedforward", 256),
                dropout=model_config.get("dropout", 0.1),
                max_len=model_config.get("max_len", 5000),
            )
        elif model_name.value == "TCN":
            channels = tuple(model_config.get("channels", [64, 64, 64]))
            model = TCNModel(
                input_dim=input_dim,
                channels=channels,
                kernel_size=model_config.get("kernel_size", 3),
                dropout=model_config.get("dropout", 0.1),
            )
        else:
            raise ValueError("Invalid model name")

        training_cfg = config.get("training", {})
        defaults = training_cfg.get("defaults", {})
        model_training = training_cfg.get("models", {}).get(model_name.value, {})
        epochs = model_training.get("epochs", defaults.get("epochs", 50))
        batch_size = model_training.get("batch_size", defaults.get("batch_size", 32))
        lr = model_training.get("lr", defaults.get("lr", 0.001))
        weight_decay = model_training.get("weight_decay", defaults.get("weight_decay", 1e-4))
        early_stopping_patience = model_training.get(
            "early_stopping_patience",
            defaults.get("early_stopping_patience", 15),
        )
        scheduler_type = model_training.get(
            "scheduler_type",
            defaults.get("scheduler_type", "plateau"),
        )

        pos_weight = None
        if task == "classification":
            positives = float((y_tr == 1).sum().item())
            negatives = float((y_tr == 0).sum().item())
            if positives > 0:
                pos_weight = negatives / positives
                pos_weight = min(pos_weight, 5.0)

        model = train_model(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            model_path=os.path.join("checkpoints", f"best_model_{model_name_str}_{task}.pt"),
            model_name=model_name,
            device=device,
            task=task,
            pos_weight=pos_weight,
            early_stopping_patience=early_stopping_patience,
            sample_weights=weights,
            debug=True,
            scheduler_type=scheduler_type,
        )

        threshold = 0.5
        if task == "classification":
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(device)).cpu().numpy().reshape(-1)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            target_rate = float((y_val == 1).sum().item()) / max(1, len(y_val))
            threshold = np.quantile(val_probs, 1.0 - target_rate)

        predicted, y_backtest, backtest_return, cumulative_return, backtest_dates, trades = backtest(
            model,
            X_backtest,
            scaler,
            y_backtest,
            y_dates=backtest_dates,
            device=device,
            return_mean=return_mean,
            return_std=return_std,
            task=task,
            raw_returns=raw_returns,
            threshold=threshold,
            trading_cost=transaction_cost,
            open_prices=backtest_open,
            close_prices=backtest_close,
        )

    if trades is not None:
        os.makedirs("result/trades", exist_ok=True)
        ticker_tag = ticker.replace("^", "")
        trades_path = os.path.join("result", "trades", f"trades_{ticker_tag}_{model_name_str}_{task}.csv")
        trades.to_csv(trades_path, index=False)
        print(f"Saved trade log to {trades_path}")

    print(f"Backtest strategy return: {backtest_return:.2%}")
    initial_asset = 100.0
    equity_curve = initial_asset * (1 + cumulative_return)
    buy_hold = initial_asset * np.cumprod(1 + raw_returns, axis=0)
    _plot_backtest(backtest_dates, equity_curve, buy_hold, ticker, model_name_str, task=task)
