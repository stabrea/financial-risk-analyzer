"""Risk metric calculations for portfolio analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_analyzer.monte_carlo import SimulationResult
from risk_analyzer.portfolio import Portfolio


@dataclass(frozen=True)
class RiskReport:
    """Immutable container for all computed risk metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    max_drawdown: float
    annualized_volatility: float
    annualized_return: float
    calmar_ratio: float
    information_ratio: float | None

    def to_dict(self) -> dict:
        return {
            "VaR_95": round(self.var_95, 4),
            "VaR_99": round(self.var_99, 4),
            "CVaR_95": round(self.cvar_95, 4),
            "CVaR_99": round(self.cvar_99, 4),
            "Sharpe_Ratio": round(self.sharpe_ratio, 4),
            "Sortino_Ratio": round(self.sortino_ratio, 4),
            "Beta": round(self.beta, 4),
            "Alpha": round(self.alpha, 4),
            "Max_Drawdown": round(self.max_drawdown, 4),
            "Annualized_Volatility": round(self.annualized_volatility, 4),
            "Annualized_Return": round(self.annualized_return, 4),
            "Calmar_Ratio": round(self.calmar_ratio, 4),
            "Information_Ratio": (
                round(self.information_ratio, 4)
                if self.information_ratio is not None
                else None
            ),
        }


class RiskCalculator:
    """Compute standard risk metrics for a portfolio."""

    def __init__(
        self,
        portfolio: Portfolio,
        risk_free_rate: float = 0.05,
        benchmark_returns: pd.Series | None = None,
        trading_days: int = 252,
    ) -> None:
        """
        Args:
            portfolio: Portfolio instance.
            risk_free_rate: Annual risk-free rate (default 5 %).
            benchmark_returns: Optional daily returns of a benchmark index.
            trading_days: Trading days per year.
        """
        self.portfolio = portfolio
        self.rf = risk_free_rate
        self.benchmark = benchmark_returns
        self.td = trading_days

        self._daily = portfolio.portfolio_daily_returns().values
        self._rf_daily = (1 + self.rf) ** (1 / self.td) - 1

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical Value at Risk (daily)."""
        return float(-np.percentile(self._daily, 100 * (1 - confidence)))

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric (variance-covariance) VaR assuming normal distribution."""
        from scipy.stats import norm  # type: ignore[import-untyped]

        z = norm.ppf(1 - confidence)
        return float(-(np.mean(self._daily) + z * np.std(self._daily)))

    def var_montecarlo(
        self,
        sim_result: SimulationResult,
        confidence: float = 0.95,
    ) -> float:
        """Monte Carlo VaR from simulation final returns."""
        return float(-np.percentile(sim_result.final_returns, 100 * (1 - confidence)))

    # ------------------------------------------------------------------
    # Conditional VaR (Expected Shortfall)
    # ------------------------------------------------------------------

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) — mean of losses beyond VaR."""
        cutoff = np.percentile(self._daily, 100 * (1 - confidence))
        tail = self._daily[self._daily <= cutoff]
        return float(-np.mean(tail)) if len(tail) > 0 else 0.0

    # ------------------------------------------------------------------
    # Return / risk ratios
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        excess = self._daily - self._rf_daily
        if np.std(excess) == 0:
            return 0.0
        return float(np.mean(excess) / np.std(excess) * np.sqrt(self.td))

    def sortino_ratio(self) -> float:
        """Annualized Sortino Ratio (downside deviation only)."""
        excess = self._daily - self._rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return float(np.mean(excess) / np.std(downside) * np.sqrt(self.td))

    # ------------------------------------------------------------------
    # Beta and Alpha (CAPM)
    # ------------------------------------------------------------------

    def beta(self) -> float:
        """Portfolio beta relative to benchmark."""
        if self.benchmark is None:
            return 1.0  # assume market portfolio
        bench = self.benchmark.values[: len(self._daily)]
        cov = np.cov(self._daily, bench)
        if cov[1, 1] == 0:
            return 1.0
        return float(cov[0, 1] / cov[1, 1])

    def alpha(self) -> float:
        """Jensen's Alpha (annualized)."""
        port_ann = self.portfolio.annualized_return(self.td)
        b = self.beta()
        if self.benchmark is not None:
            bench_ann = float(
                (1 + self.benchmark.mean()) ** self.td - 1
            )
        else:
            bench_ann = port_ann  # self-benchmark → alpha = 0
        return port_ann - (self.rf + b * (bench_ann - self.rf))

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self) -> float:
        """Maximum drawdown of the portfolio."""
        cum = (1 + pd.Series(self._daily)).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        """Full drawdown time series."""
        cum = (1 + pd.Series(self._daily)).cumprod()
        running_max = cum.cummax()
        return ((cum - running_max) / running_max).rename("drawdown")

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def annualized_volatility(self) -> float:
        """Annualized portfolio volatility."""
        return float(np.std(self._daily) * np.sqrt(self.td))

    # ------------------------------------------------------------------
    # Information Ratio
    # ------------------------------------------------------------------

    def information_ratio(self) -> float | None:
        """Information Ratio relative to benchmark."""
        if self.benchmark is None:
            return None
        bench = self.benchmark.values[: len(self._daily)]
        active = self._daily - bench
        te = np.std(active) * np.sqrt(self.td)
        if te == 0:
            return 0.0
        return float(np.mean(active) * self.td / te)

    # ------------------------------------------------------------------
    # Calmar Ratio
    # ------------------------------------------------------------------

    def calmar_ratio(self) -> float:
        """Calmar Ratio = annualized return / |max drawdown|."""
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return self.portfolio.annualized_return(self.td) / mdd

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def compute_all(self) -> RiskReport:
        """Compute all risk metrics and return a RiskReport."""
        return RiskReport(
            var_95=self.var_historical(0.95),
            var_99=self.var_historical(0.99),
            cvar_95=self.cvar(0.95),
            cvar_99=self.cvar(0.99),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            beta=self.beta(),
            alpha=self.alpha(),
            max_drawdown=self.max_drawdown(),
            annualized_volatility=self.annualized_volatility(),
            annualized_return=self.portfolio.annualized_return(self.td),
            calmar_ratio=self.calmar_ratio(),
            information_ratio=self.information_ratio(),
        )
