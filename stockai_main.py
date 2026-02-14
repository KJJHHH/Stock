"""Compatibility wrapper for old StockAI entrypoint.

Use `uv run main.py run --model <MODEL>` instead.
"""

from main import run_unified_model


class _Args:
    task = "run"
    model = "LSTM"
    stock_target = None
    stock_pool = []
    epochs = None
    train_start = None
    train_end = None
    backtest_start = None
    backtest_end = None
    prediction_task = None


if __name__ == "__main__":
    run_unified_model(_Args())
