"""Tests for the Monte Carlo simulation module."""

import numpy as np
import pandas as pd

from risk_analyzer.portfolio import Portfolio
from risk_analyzer.monte_carlo import MonteCarloEngine, SimulationResult


def _make_portfolio() -> Portfolio:
    rng = np.random.default_rng(42)
    n_days = 100

    holdings = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "shares": [100, 50],
        "purchase_price": [150.0, 300.0],
    })

    dates = pd.bdate_range(start="2024-01-02", periods=n_days)
    aapl = 150.0 + np.cumsum(rng.normal(0.1, 2.0, n_days))
    msft = 300.0 + np.cumsum(rng.normal(0.05, 3.0, n_days))

    prices = pd.DataFrame({
        "date": dates,
        "AAPL": aapl,
        "MSFT": msft,
    })

    return Portfolio(holdings, prices)


def test_simulation_output_shape():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=100, horizon=50, seed=42)
    result = engine.run()

    assert isinstance(result, SimulationResult)
    assert result.paths.shape == (100, 51)  # n_simulations x (horizon + 1)
    assert result.final_values.shape == (100,)
    assert result.final_returns.shape == (100,)
    assert result.n_simulations == 100
    assert result.horizon == 50


def test_simulation_with_seed_reproducible():
    portfolio = _make_portfolio()

    engine1 = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=123)
    result1 = engine1.run()

    engine2 = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=123)
    result2 = engine2.run()

    np.testing.assert_array_almost_equal(result1.final_values, result2.final_values)
    np.testing.assert_array_almost_equal(result1.paths, result2.paths)


def test_simulation_different_seeds_differ():
    portfolio = _make_portfolio()

    engine1 = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=1)
    result1 = engine1.run()

    engine2 = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=2)
    result2 = engine2.run()

    assert not np.allclose(result1.final_values, result2.final_values)


def test_probability_of_loss_between_0_and_1():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=500, horizon=50, seed=42)
    result = engine.run()

    assert 0.0 <= result.prob_loss <= 1.0


def test_initial_value_matches_portfolio():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=42)
    result = engine.run()

    assert abs(result.initial_value - portfolio.total_value) < 1e-6


def test_all_paths_start_at_initial_value():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=42)
    result = engine.run()

    np.testing.assert_array_almost_equal(
        result.paths[:, 0],
        np.full(50, result.initial_value),
    )


def test_confidence_levels_populated():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=200, horizon=50, seed=42)
    result = engine.run()

    assert len(result.confidence_levels) > 0
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        assert q in result.confidence_levels
    # Lower percentiles should be less than or equal to higher ones
    assert result.confidence_levels[5] <= result.confidence_levels[95]


def test_mean_and_median_final_value():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=200, horizon=50, seed=42)
    result = engine.run()

    assert result.mean_final_value > 0
    assert result.median_final_value > 0
    assert result.std_final_value > 0


def test_final_returns_consistent_with_values():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=50, horizon=30, seed=42)
    result = engine.run()

    expected_returns = (result.final_values - result.initial_value) / result.initial_value
    np.testing.assert_array_almost_equal(result.final_returns, expected_returns)


def test_percentile_method():
    portfolio = _make_portfolio()
    engine = MonteCarloEngine(portfolio, n_simulations=200, horizon=50, seed=42)
    result = engine.run()

    p5 = result.percentile(5)
    p50 = result.percentile(50)
    p95 = result.percentile(95)

    assert p5 <= p50 <= p95
