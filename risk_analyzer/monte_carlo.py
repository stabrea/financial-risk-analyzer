"""Monte Carlo simulation engine for portfolio risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_analyzer.portfolio import Portfolio


@dataclass(frozen=True)
class SimulationResult:
    """Container for Monte Carlo simulation outputs."""

    paths: np.ndarray          # shape (n_simulations, horizon + 1) — portfolio values
    final_values: np.ndarray   # shape (n_simulations,)
    final_returns: np.ndarray  # shape (n_simulations,)
    initial_value: float
    horizon: int
    n_simulations: int
    confidence_levels: dict[float, float]  # percentile → portfolio value

    @property
    def mean_final_value(self) -> float:
        return float(np.mean(self.final_values))

    @property
    def median_final_value(self) -> float:
        return float(np.median(self.final_values))

    @property
    def std_final_value(self) -> float:
        return float(np.std(self.final_values))

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.final_returns))

    @property
    def prob_loss(self) -> float:
        """Probability of ending with a loss."""
        return float(np.mean(self.final_returns < 0))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.final_values, q))


class MonteCarloEngine:
    """Geometric Brownian Motion Monte Carlo simulator for portfolio paths."""

    def __init__(
        self,
        portfolio: Portfolio,
        n_simulations: int = 10_000,
        horizon: int = 252,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            portfolio: Portfolio instance with historical data.
            n_simulations: Number of simulation paths.
            horizon: Number of trading days to simulate forward.
            seed: Random seed for reproducibility.
        """
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """
        Run the Monte Carlo simulation using correlated GBM.

        Steps:
            1. Estimate mean daily returns and covariance from history.
            2. Cholesky-decompose the covariance matrix.
            3. Generate correlated random shocks per asset per day.
            4. Compound the portfolio value along each path.
        """
        daily_ret = self.portfolio.daily_returns()
        mu = daily_ret.mean().values          # (n_assets,)
        cov = daily_ret.cov().values          # (n_assets, n_assets)
        weights = self.portfolio.weights      # (n_assets,)
        initial_value = self.portfolio.total_value

        # Cholesky decomposition for correlated draws
        L = np.linalg.cholesky(cov)

        # Generate uncorrelated standard-normal shocks
        # shape: (n_simulations, horizon, n_assets)
        Z = self.rng.standard_normal(
            (self.n_simulations, self.horizon, len(weights))
        )

        # Correlate the shocks: multiply each (n_assets,) vector by L^T
        correlated = Z @ L.T  # still (n_sim, horizon, n_assets)

        # Daily asset returns: r_i = mu_i + correlated_shock_i
        asset_returns = mu + correlated  # broadcasting mu across sim & horizon

        # Portfolio daily returns: weighted sum across assets
        port_returns = asset_returns @ weights  # (n_sim, horizon)

        # Build value paths by compounding
        growth = 1 + port_returns  # (n_sim, horizon)
        cum = np.cumprod(growth, axis=1)  # cumulative product along time

        paths = np.empty((self.n_simulations, self.horizon + 1))
        paths[:, 0] = initial_value
        paths[:, 1:] = initial_value * cum

        final_values = paths[:, -1]
        final_returns = (final_values - initial_value) / initial_value

        # Confidence intervals
        confidence_levels = {
            q: float(np.percentile(final_values, q))
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return SimulationResult(
            paths=paths,
            final_values=final_values,
            final_returns=final_returns,
            initial_value=initial_value,
            horizon=self.horizon,
            n_simulations=self.n_simulations,
            confidence_levels=confidence_levels,
        )

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    def stress_test(
        self,
        shock_pct: float = -0.20,
    ) -> SimulationResult:
        """
        Run a simulation where day-1 experiences an immediate market shock,
        then resumes normal stochastic behaviour.

        Args:
            shock_pct: Fractional shock on day 1 (e.g. -0.20 = 20 % drop).
        """
        result = self.run()
        # Apply the shock to all paths at t=1, cascade forward
        shocked_paths = result.paths.copy()
        shocked_paths[:, 1] = result.initial_value * (1 + shock_pct)
        # Re-compound from day 1 forward using original daily growth rates
        for t in range(2, self.horizon + 1):
            daily_growth = result.paths[:, t] / result.paths[:, t - 1]
            shocked_paths[:, t] = shocked_paths[:, t - 1] * daily_growth

        final_values = shocked_paths[:, -1]
        final_returns = (final_values - result.initial_value) / result.initial_value

        confidence_levels = {
            q: float(np.percentile(final_values, q))
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return SimulationResult(
            paths=shocked_paths,
            final_values=final_values,
            final_returns=final_returns,
            initial_value=result.initial_value,
            horizon=self.horizon,
            n_simulations=self.n_simulations,
            confidence_levels=confidence_levels,
        )
