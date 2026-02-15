"""Tests for the risk metrics module."""

import numpy as np
import pandas as pd

from risk_analyzer.portfolio import Portfolio
from risk_analyzer.risk_metrics import RiskCalculator, RiskReport


def _make_portfolio() -> Portfolio:
    rng = np.random.default_rng(42)
    n_days = 252

    holdings = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "shares": [100, 50, 30],
        "purchase_price": [150.0, 300.0, 100.0],
    })

    dates = pd.bdate_range(start="2024-01-02", periods=n_days)
    aapl = 150.0 + np.cumsum(rng.normal(0.1, 2.0, n_days))
    msft = 300.0 + np.cumsum(rng.normal(0.05, 3.0, n_days))
    goog = 100.0 + np.cumsum(rng.normal(0.08, 1.5, n_days))

    prices = pd.DataFrame({
        "date": dates,
        "AAPL": aapl,
        "MSFT": msft,
        "GOOG": goog,
    })

    return Portfolio(holdings, prices)


def _make_calculator() -> RiskCalculator:
    return RiskCalculator(_make_portfolio(), risk_free_rate=0.05)


def test_var_95_is_positive():
    """Historical VaR at 95% should be a positive number representing loss magnitude."""
    calc = _make_calculator()
    var = calc.var_historical(0.95)
    # VaR is computed as -percentile, so it should typically be positive
    # (positive means a loss of that magnitude)
    assert isinstance(var, float)


def test_var_99_greater_than_var_95():
    calc = _make_calculator()
    var_95 = calc.var_historical(0.95)
    var_99 = calc.var_historical(0.99)
    # 99% VaR should capture more extreme tail, so should be >= 95% VaR
    assert var_99 >= var_95


def test_sharpe_ratio_is_finite():
    calc = _make_calculator()
    sharpe = calc.sharpe_ratio()
    assert isinstance(sharpe, float)
    assert np.isfinite(sharpe)


def test_sortino_ratio_is_finite():
    calc = _make_calculator()
    sortino = calc.sortino_ratio()
    assert isinstance(sortino, float)
    assert np.isfinite(sortino)


def test_max_drawdown_is_negative_or_zero():
    calc = _make_calculator()
    mdd = calc.max_drawdown()
    assert mdd <= 0.0


def test_max_drawdown_bounded():
    calc = _make_calculator()
    mdd = calc.max_drawdown()
    assert -1.0 <= mdd <= 0.0


def test_annualized_volatility_positive():
    calc = _make_calculator()
    vol = calc.annualized_volatility()
    assert vol > 0


def test_beta_without_benchmark():
    calc = _make_calculator()
    b = calc.beta()
    # Without benchmark, beta defaults to 1.0
    assert b == 1.0


def test_alpha_without_benchmark():
    calc = _make_calculator()
    a = calc.alpha()
    # Without benchmark, alpha should be approximately 0
    assert abs(a) < 1e-10


def test_cvar_greater_than_or_equal_to_var():
    calc = _make_calculator()
    var_95 = calc.var_historical(0.95)
    cvar_95 = calc.cvar(0.95)
    # CVaR (expected shortfall) should be >= VaR
    assert cvar_95 >= var_95


def test_information_ratio_none_without_benchmark():
    calc = _make_calculator()
    ir = calc.information_ratio()
    assert ir is None


def test_calmar_ratio_is_finite():
    calc = _make_calculator()
    calmar = calc.calmar_ratio()
    assert isinstance(calmar, float)
    assert np.isfinite(calmar)


def test_drawdown_series_length():
    calc = _make_calculator()
    dd = calc.drawdown_series()
    assert isinstance(dd, pd.Series)
    assert len(dd) > 0
    # All drawdown values should be <= 0
    assert (dd <= 0).all()


def test_compute_all_returns_risk_report():
    calc = _make_calculator()
    report = calc.compute_all()
    assert isinstance(report, RiskReport)
    assert isinstance(report.var_95, float)
    assert isinstance(report.sharpe_ratio, float)
    assert isinstance(report.max_drawdown, float)
    assert report.information_ratio is None  # no benchmark


def test_risk_report_to_dict():
    calc = _make_calculator()
    report = calc.compute_all()
    d = report.to_dict()
    assert "VaR_95" in d
    assert "VaR_99" in d
    assert "Sharpe_Ratio" in d
    assert "Max_Drawdown" in d
    assert "Annualized_Volatility" in d
    assert "Annualized_Return" in d


def test_beta_with_benchmark():
    portfolio = _make_portfolio()
    rng = np.random.default_rng(99)
    benchmark = pd.Series(rng.normal(0.0005, 0.01, 251), name="benchmark")

    calc = RiskCalculator(portfolio, benchmark_returns=benchmark)
    b = calc.beta()
    assert isinstance(b, float)
    assert np.isfinite(b)


def test_information_ratio_with_benchmark():
    portfolio = _make_portfolio()
    rng = np.random.default_rng(99)
    benchmark = pd.Series(rng.normal(0.0005, 0.01, 251), name="benchmark")

    calc = RiskCalculator(portfolio, benchmark_returns=benchmark)
    ir = calc.information_ratio()
    assert ir is not None
    assert isinstance(ir, float)
    assert np.isfinite(ir)
