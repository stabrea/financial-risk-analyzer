"""Tests for the portfolio module."""

import numpy as np
import pandas as pd
import pytest

from risk_analyzer.portfolio import Portfolio


def _make_holdings() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "shares": [100, 50, 30],
        "purchase_price": [150.0, 300.0, 100.0],
    })


def _make_prices(n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2024-01-02", periods=n_days)

    aapl = 150.0 + np.cumsum(rng.normal(0.1, 2.0, n_days))
    msft = 300.0 + np.cumsum(rng.normal(0.05, 3.0, n_days))
    goog = 100.0 + np.cumsum(rng.normal(0.08, 1.5, n_days))

    return pd.DataFrame({
        "date": dates,
        "AAPL": aapl,
        "MSFT": msft,
        "GOOG": goog,
    })


def _make_portfolio() -> Portfolio:
    return Portfolio(_make_holdings(), _make_prices())


def test_portfolio_creation():
    portfolio = _make_portfolio()
    assert portfolio.tickers == ["AAPL", "MSFT", "GOOG"]
    assert len(portfolio.holdings) == 3
    assert len(portfolio.prices) == 60


def test_portfolio_missing_columns_raises():
    bad_holdings = pd.DataFrame({"ticker": ["AAPL"], "shares": [10]})
    prices = _make_prices()
    with pytest.raises(ValueError, match="missing columns"):
        Portfolio(bad_holdings, prices)


def test_portfolio_empty_holdings_raises():
    empty = pd.DataFrame({"ticker": [], "shares": [], "purchase_price": []})
    prices = _make_prices()
    with pytest.raises(ValueError, match="empty"):
        Portfolio(empty, prices)


def test_portfolio_missing_price_column_raises():
    holdings = _make_holdings()
    prices = _make_prices().drop(columns=["GOOG"])
    with pytest.raises(ValueError, match="missing tickers"):
        Portfolio(holdings, prices)


def test_portfolio_missing_date_column_raises():
    holdings = _make_holdings()
    prices = _make_prices().drop(columns=["date"])
    with pytest.raises(ValueError, match="date"):
        Portfolio(holdings, prices)


def test_weights_sum_to_one():
    portfolio = _make_portfolio()
    weights = portfolio.weights
    assert abs(np.sum(weights) - 1.0) < 1e-10


def test_weights_are_positive():
    portfolio = _make_portfolio()
    assert all(w > 0 for w in portfolio.weights)


def test_weights_length_matches_tickers():
    portfolio = _make_portfolio()
    assert len(portfolio.weights) == len(portfolio.tickers)


def test_daily_returns_shape():
    portfolio = _make_portfolio()
    dr = portfolio.daily_returns()
    # pct_change drops the first row, so n_days - 1 rows
    assert dr.shape[0] == 59
    assert dr.shape[1] == 3
    assert list(dr.columns) == ["AAPL", "MSFT", "GOOG"]


def test_portfolio_daily_returns_is_series():
    portfolio = _make_portfolio()
    pr = portfolio.portfolio_daily_returns()
    assert isinstance(pr, pd.Series)
    assert len(pr) == 59


def test_cumulative_returns():
    portfolio = _make_portfolio()
    cr = portfolio.cumulative_returns()
    assert isinstance(cr, pd.Series)
    assert len(cr) == 59


def test_total_value_is_positive():
    portfolio = _make_portfolio()
    assert portfolio.total_value > 0


def test_market_values_length():
    portfolio = _make_portfolio()
    mv = portfolio.market_values
    assert len(mv) == 3


def test_cost_basis():
    portfolio = _make_portfolio()
    expected = 100 * 150.0 + 50 * 300.0 + 30 * 100.0
    assert portfolio.cost_basis == expected


def test_annualized_return_type():
    portfolio = _make_portfolio()
    ann_ret = portfolio.annualized_return()
    assert isinstance(ann_ret, float)


def test_covariance_matrix_shape():
    portfolio = _make_portfolio()
    cov = portfolio.covariance_matrix()
    assert cov.shape == (3, 3)


def test_correlation_matrix_diagonal():
    portfolio = _make_portfolio()
    corr = portfolio.correlation_matrix()
    # Diagonal should be 1.0
    for i in range(3):
        assert abs(corr.iloc[i, i] - 1.0) < 1e-10


def test_summary_keys():
    portfolio = _make_portfolio()
    s = portfolio.summary()
    assert "tickers" in s
    assert "weights" in s
    assert "total_value" in s
    assert "cost_basis" in s
    assert "unrealized_pnl" in s
    assert "annualized_return" in s
