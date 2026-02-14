#!/usr/bin/env python3
"""Minimal yfinance connectivity and schema diagnostic."""

import sys
import time

import pandas as pd
import yfinance as yf


REQUIRED = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
SYMBOL = "2330.TW"
START = "2010-01-01"
END = "2026-03-31"
RETRIES = 3
DELAY = 1.0
SAVE_PATH = ""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).strip() for c in df.columns]
        return df

    levels = df.columns.nlevels
    for level in range(levels):
        candidate = [str(c).strip() for c in df.columns.get_level_values(level)]
        if REQUIRED.issubset(set(candidate)):
            df.columns = candidate
            return df

    df.columns = ["_".join([str(part).strip() for part in col]) for col in df.columns]
    return df


def fetch(symbol: str, start: str, end: str, retries: int, delay: float) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # Official docs style: https://ranaroussi.github.io/yfinance/
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
            )
            if df.empty:
                raise ValueError("yfinance returned empty DataFrame")
            return normalize_columns(df)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            print(f"Attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(delay * attempt)

    raise RuntimeError(f"All attempts failed for {symbol}") from last_exc


def main() -> int:
    df = fetch(SYMBOL, START, END, RETRIES, DELAY)

    print("Fetch succeeded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    missing = sorted(REQUIRED.difference(set(df.columns)))
    if missing:
        print(f"Missing required columns: {missing}")
    else:
        print("Required OHLCV columns present")

    print("Head:")
    print(df.head(3).to_string())

    if SAVE_PATH:
        df.to_csv(SAVE_PATH, index=True)
        print(f"Saved CSV: {SAVE_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
