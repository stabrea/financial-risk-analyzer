"""Load portfolio and price data from CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_analyzer.portfolio import Portfolio


def load_portfolio(
    holdings_path: str | Path,
    prices_path: str | Path,
) -> Portfolio:
    """
    Build a Portfolio from two CSV files.

    Args:
        holdings_path: Path to CSV with columns [ticker, shares, purchase_price].
        prices_path: Path to CSV with a 'date' column and one column per ticker.

    Returns:
        A fully initialised Portfolio instance.
    """
    holdings = _read_csv(holdings_path)
    prices = _read_csv(prices_path)
    return Portfolio(holdings=holdings, historical_prices=prices)


def load_holdings(path: str | Path) -> pd.DataFrame:
    """Load a holdings CSV into a DataFrame."""
    df = _read_csv(path)
    required = {"ticker", "shares", "purchase_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Holdings CSV missing columns: {missing}")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a historical prices CSV into a DataFrame."""
    df = _read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Prices CSV must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, raising a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)
