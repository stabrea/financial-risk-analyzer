"""Matplotlib-based visualization for portfolio risk analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from risk_analyzer.monte_carlo import SimulationResult
from risk_analyzer.portfolio import Portfolio
from risk_analyzer.risk_metrics import RiskReport


# ------------------------------------------------------------------
# Style defaults
# ------------------------------------------------------------------

COLORS = {
    "primary": "#1a73e8",
    "secondary": "#34a853",
    "danger": "#ea4335",
    "warning": "#fbbc05",
    "neutral": "#5f6368",
    "bg": "#fafafa",
}


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ------------------------------------------------------------------
# Individual charts
# ------------------------------------------------------------------


def plot_simulation_paths(
    sim: SimulationResult,
    n_paths: int = 200,
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot a sample of Monte Carlo simulation paths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(ax)

    idx = np.linspace(0, sim.n_simulations - 1, n_paths, dtype=int)
    for i in idx:
        ax.plot(sim.paths[i], alpha=0.08, color=COLORS["primary"], linewidth=0.6)

    # Highlight percentile bands
    median = np.median(sim.paths, axis=0)
    p5 = np.percentile(sim.paths, 5, axis=0)
    p95 = np.percentile(sim.paths, 95, axis=0)

    ax.plot(median, color=COLORS["secondary"], linewidth=2, label="Median")
    ax.fill_between(
        range(len(median)), p5, p95,
        alpha=0.15, color=COLORS["warning"], label="5th-95th percentile",
    )
    ax.axhline(sim.initial_value, linestyle="--", color=COLORS["danger"], linewidth=1, label="Initial Value")

    ax.set_title("Monte Carlo Simulation — Portfolio Value Paths", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


def plot_return_distribution(
    sim: SimulationResult,
    var_95: float | None = None,
    output: str | Path | None = None,
) -> plt.Figure:
    """Histogram of simulated final returns with optional VaR line."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_style(ax)

    ax.hist(
        sim.final_returns * 100,
        bins=100,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )

    if var_95 is not None:
        ax.axvline(
            -var_95 * 100,
            color=COLORS["danger"],
            linewidth=2,
            linestyle="--",
            label=f"95% VaR ({-var_95 * 100:.1f}%)",
        )
        ax.legend(fontsize=9)

    ax.set_title("Distribution of Simulated Portfolio Returns", fontsize=13, fontweight="bold")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    portfolio: Portfolio,
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot the drawdown curve over time."""
    daily = portfolio.portfolio_daily_returns()
    cum = (1 + daily).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_style(ax)

    ax.fill_between(range(len(dd)), dd.values * 100, 0, color=COLORS["danger"], alpha=0.4)
    ax.plot(dd.values * 100, color=COLORS["danger"], linewidth=0.8)

    ax.set_title("Portfolio Drawdown", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


def plot_weights(
    portfolio: Portfolio,
    output: str | Path | None = None,
) -> plt.Figure:
    """Pie chart of portfolio allocation weights."""
    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        portfolio.weights,
        labels=portfolio.tickers,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)

    ax.set_title("Portfolio Allocation", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    portfolio: Portfolio,
    output: str | Path | None = None,
) -> plt.Figure:
    """Heatmap of asset return correlations."""
    corr = portfolio.correlation_matrix()
    fig, ax = plt.subplots(figsize=(8, 6))

    cax = ax.matshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold", pad=40)
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# Full dashboard
# ------------------------------------------------------------------


def generate_dashboard(
    portfolio: Portfolio,
    sim: SimulationResult,
    risk_report: RiskReport,
    output_dir: str | Path = "output",
) -> list[Path]:
    """Generate all charts and save to output_dir. Returns list of file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths_saved: list[Path] = []

    charts = [
        ("simulation_paths.png", lambda: plot_simulation_paths(sim, output=out / "simulation_paths.png")),
        ("return_distribution.png", lambda: plot_return_distribution(sim, var_95=risk_report.var_95, output=out / "return_distribution.png")),
        ("drawdown.png", lambda: plot_drawdown(portfolio, output=out / "drawdown.png")),
        ("weights.png", lambda: plot_weights(portfolio, output=out / "weights.png")),
        ("correlation.png", lambda: plot_correlation_matrix(portfolio, output=out / "correlation.png")),
    ]

    for name, fn in charts:
        fn()
        plt.close("all")
        paths_saved.append(out / name)

    return paths_saved
