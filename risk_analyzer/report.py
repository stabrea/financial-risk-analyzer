"""Generate risk assessment reports in JSON and HTML formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from risk_analyzer.monte_carlo import SimulationResult
from risk_analyzer.portfolio import Portfolio
from risk_analyzer.risk_metrics import RiskReport


def build_report_data(
    portfolio: Portfolio,
    risk_report: RiskReport,
    sim_result: SimulationResult,
) -> dict:
    """Assemble all analysis data into a single dictionary."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio": portfolio.summary(),
        "risk_metrics": risk_report.to_dict(),
        "monte_carlo": {
            "n_simulations": sim_result.n_simulations,
            "horizon_days": sim_result.horizon,
            "initial_value": round(sim_result.initial_value, 2),
            "mean_final_value": round(sim_result.mean_final_value, 2),
            "median_final_value": round(sim_result.median_final_value, 2),
            "std_final_value": round(sim_result.std_final_value, 2),
            "probability_of_loss": round(sim_result.prob_loss, 4),
            "percentiles": {
                f"p{int(k)}": round(v, 2)
                for k, v in sim_result.confidence_levels.items()
            },
        },
    }


def export_json(
    data: dict,
    output: str | Path = "output/risk_report.json",
) -> Path:
    """Write the report data to a JSON file."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def export_html(
    data: dict,
    chart_dir: str | Path = "output",
    output: str | Path = "output/risk_report.html",
) -> Path:
    """Generate a self-contained HTML risk report."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir = Path(chart_dir)

    portfolio = data["portfolio"]
    metrics = data["risk_metrics"]
    mc = data["monte_carlo"]

    def _pct(val: float) -> str:
        return f"{val * 100:.2f}%"

    def _dollar(val: float) -> str:
        return f"${val:,.2f}"

    def _metric_row(label: str, value: str) -> str:
        return f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"

    holdings_rows = ""
    for ticker, weight in portfolio["weights"].items():
        holdings_rows += f"<tr><td>{ticker}</td><td>{weight * 100:.1f}%</td></tr>\n"

    metrics_rows = "\n".join([
        _metric_row("Value at Risk (95%)", _pct(metrics["VaR_95"])),
        _metric_row("Value at Risk (99%)", _pct(metrics["VaR_99"])),
        _metric_row("Conditional VaR (95%)", _pct(metrics["CVaR_95"])),
        _metric_row("Conditional VaR (99%)", _pct(metrics["CVaR_99"])),
        _metric_row("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.4f}"),
        _metric_row("Sortino Ratio", f"{metrics['Sortino_Ratio']:.4f}"),
        _metric_row("Beta", f"{metrics['Beta']:.4f}"),
        _metric_row("Alpha", _pct(metrics["Alpha"])),
        _metric_row("Max Drawdown", _pct(metrics["Max_Drawdown"])),
        _metric_row("Annualized Volatility", _pct(metrics["Annualized_Volatility"])),
        _metric_row("Annualized Return", _pct(metrics["Annualized_Return"])),
        _metric_row("Calmar Ratio", f"{metrics['Calmar_Ratio']:.4f}"),
    ])

    mc_rows = "\n".join([
        _metric_row("Simulations", f"{mc['n_simulations']:,}"),
        _metric_row("Horizon", f"{mc['horizon_days']} trading days"),
        _metric_row("Initial Value", _dollar(mc["initial_value"])),
        _metric_row("Mean Final Value", _dollar(mc["mean_final_value"])),
        _metric_row("Median Final Value", _dollar(mc["median_final_value"])),
        _metric_row("Probability of Loss", _pct(mc["probability_of_loss"])),
    ])

    # Chart image references (relative)
    charts_html = ""
    chart_files = [
        ("simulation_paths.png", "Monte Carlo Simulation Paths"),
        ("return_distribution.png", "Return Distribution"),
        ("drawdown.png", "Portfolio Drawdown"),
        ("weights.png", "Portfolio Allocation"),
        ("correlation.png", "Correlation Matrix"),
    ]
    for fname, title in chart_files:
        fpath = chart_dir / fname
        if fpath.exists():
            charts_html += f"""
            <div class="chart">
                <h3>{title}</h3>
                <img src="{fname}" alt="{title}">
            </div>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Risk Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5; color: #333; line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
        h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #1a73e8; }}
        h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #202124; border-bottom: 2px solid #1a73e8; padding-bottom: 0.3rem; }}
        .meta {{ color: #5f6368; font-size: 0.9rem; margin-bottom: 2rem; }}
        .summary-cards {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem; margin-bottom: 2rem;
        }}
        .card {{
            background: white; border-radius: 8px; padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .card .label {{ font-size: 0.85rem; color: #5f6368; }}
        .card .value {{ font-size: 1.4rem; font-weight: 700; color: #202124; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }}
        th, td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid #e8eaed; }}
        th {{ background: #f8f9fa; font-size: 0.85rem; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; }}
        .chart {{ background: white; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; }}
        .chart h3 {{ margin-bottom: 0.5rem; font-size: 1rem; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e8eaed; color: #5f6368; font-size: 0.8rem; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Portfolio Risk Report</h1>
    <p class="meta">Generated: {data['generated_at']}</p>

    <div class="summary-cards">
        <div class="card"><div class="label">Portfolio Value</div><div class="value">{_dollar(portfolio['total_value'])}</div></div>
        <div class="card"><div class="label">Unrealized P&amp;L</div><div class="value">{_dollar(portfolio['unrealized_pnl'])}</div></div>
        <div class="card"><div class="label">Annualized Return</div><div class="value">{_pct(portfolio['annualized_return'])}</div></div>
        <div class="card"><div class="label">Sharpe Ratio</div><div class="value">{metrics['Sharpe_Ratio']:.2f}</div></div>
        <div class="card"><div class="label">Max Drawdown</div><div class="value">{_pct(metrics['Max_Drawdown'])}</div></div>
        <div class="card"><div class="label">VaR (95%)</div><div class="value">{_pct(metrics['VaR_95'])}</div></div>
    </div>

    <h2>Holdings</h2>
    <table>
        <thead><tr><th>Ticker</th><th>Weight</th></tr></thead>
        <tbody>{holdings_rows}</tbody>
    </table>

    <h2>Risk Metrics</h2>
    <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{metrics_rows}</tbody>
    </table>

    <h2>Monte Carlo Simulation</h2>
    <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{mc_rows}</tbody>
    </table>

    <h2>Charts</h2>
    {charts_html}

    <div class="footer">
        Financial Risk Analyzer v1.0.0 &mdash; For educational and analytical purposes only. Not financial advice.
    </div>
</div>
</body>
</html>
"""
    path.write_text(html)
    return path
