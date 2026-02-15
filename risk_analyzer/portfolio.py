"""Portfolio management and return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Portfolio:
    """Represents a financial portfolio with holdings and historical price data."""

    def __init__(
        self,
        holdings: pd.DataFrame,
        historical_prices: pd.DataFrame,
    ) -> None:
        """
        Initialize a Portfolio.

        Args:
            holdings: DataFrame with columns [ticker, shares, purchase_price].
            historical_prices: DataFrame with a 'date' column and one column
                per ticker containing daily closing prices.
        """
        self._validate_holdings(holdings)
        self._validate_prices(historical_prices, holdings)

        self.holdings: pd.DataFrame = holdings.copy()
        self.prices: pd.DataFrame = historical_prices.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"])
        self.prices = self.prices.sort_values("date").reset_index(drop=True)

        self.tickers: list[str] = self.holdings["ticker"].tolist()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_holdings(holdings: pd.DataFrame) -> None:
        required = {"ticker", "shares", "purchase_price"}
        missing = required - set(holdings.columns)
        if missing:
            raise ValueError(f"Holdings DataFrame missing columns: {missing}")
        if holdings.empty:
            raise ValueError("Holdings DataFrame is empty")

    @staticmethod
    def _validate_prices(prices: pd.DataFrame, holdings: pd.DataFrame) -> None:
        if "date" not in prices.columns:
            raise ValueError("Historical prices must contain a 'date' column")
        tickers = holdings["ticker"].tolist()
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Historical prices missing tickers: {missing}")

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @property
    def market_values(self) -> pd.Series:
        """Current market value of each holding using the latest price."""
        latest = self.prices.iloc[-1]
        values = self.holdings.apply(
            lambda row: row["shares"] * latest[row["ticker"]],
            axis=1,
        )
        return pd.Series(values.values, index=self.tickers, name="market_value")

    @property
    def total_value(self) -> float:
        """Total portfolio market value."""
        return float(self.market_values.sum())

    @property
    def weights(self) -> np.ndarray:
        """Portfolio weights as a 1-D numpy array aligned with self.tickers."""
        mv = self.market_values.values.astype(float)
        return mv / mv.sum()

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the portfolio."""
        return float(
            (self.holdings["shares"] * self.holdings["purchase_price"]).sum()
        )

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.total_value - self.cost_basis

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def daily_returns(self) -> pd.DataFrame:
        """Calculate daily simple returns for each ticker."""
        price_cols = self.prices[self.tickers]
        return price_cols.pct_change().dropna()

    def portfolio_daily_returns(self) -> pd.Series:
        """Weighted portfolio daily returns."""
        dr = self.daily_returns()
        return dr.dot(self.weights).rename("portfolio_return")

    def cumulative_returns(self) -> pd.Series:
        """Cumulative portfolio returns."""
        pr = self.portfolio_daily_returns()
        return ((1 + pr).cumprod() - 1).rename("cumulative_return")

    def annualized_return(self, trading_days: int = 252) -> float:
        """Annualized portfolio return."""
        pr = self.portfolio_daily_returns()
        total = (1 + pr).prod()
        n_years = len(pr) / trading_days
        return float(total ** (1 / n_years) - 1)

    def covariance_matrix(self) -> pd.DataFrame:
        """Annualized covariance matrix of asset returns."""
        return self.daily_returns().cov() * 252

    def correlation_matrix(self) -> pd.DataFrame:
        """Correlation matrix of asset returns."""
        return self.daily_returns().corr()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dictionary of the portfolio."""
        return {
            "tickers": self.tickers,
            "weights": dict(zip(self.tickers, self.weights.tolist())),
            "total_value": round(self.total_value, 2),
            "cost_basis": round(self.cost_basis, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "annualized_return": round(self.annualized_return(), 4),
        }
