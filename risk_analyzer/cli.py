"""Command-line interface for the Financial Risk Analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from risk_analyzer.data_loader import load_portfolio
from risk_analyzer.monte_carlo import MonteCarloEngine
from risk_analyzer.report import build_report_data, export_html, export_json
from risk_analyzer.risk_metrics import RiskCalculator
from risk_analyzer.visualizer import generate_dashboard


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-analyzer",
        description="Financial portfolio risk analysis with Monte Carlo simulation.",
    )

    parser.add_argument(
        "--holdings",
        type=str,
        default="data/sample_portfolio.csv",
        help="Path to holdings CSV (default: data/sample_portfolio.csv)",
    )
    parser.add_argument(
        "--prices",
        type=str,
        default="data/historical_prices.csv",
        help="Path to historical prices CSV (default: data/historical_prices.csv)",
    )
    parser.add_argument(
        "--simulations", "-n",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000)",
    )
    parser.add_argument(
        "--horizon", "-d",
        type=int,
        default=252,
        help="Simulation horizon in trading days (default: 252 = 1 year)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Annual risk-free rate (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for reports and charts (default: output/)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON report only (no HTML, no charts)",
    )
    parser.add_argument(
        "--stress-test",
        type=float,
        default=None,
        help="Run stress test with given shock (e.g. -0.20 for -20%%)",
    )

    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the full analysis pipeline."""
    console.print(Panel.fit(
        "[bold blue]Financial Risk Analyzer[/bold blue]\n"
        "Portfolio risk analysis with Monte Carlo simulation",
        border_style="blue",
    ))

    # ------------------------------------------------------------------ Load data
    console.print("\n[bold]Loading data...[/bold]")
    try:
        portfolio = load_portfolio(args.holdings, args.prices)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading data:[/red] {exc}")
        sys.exit(1)

    console.print(f"  Tickers: {', '.join(portfolio.tickers)}")
    console.print(f"  Portfolio value: ${portfolio.total_value:,.2f}")

    # -------------------------------------------------------------- Portfolio table
    weights_table = Table(title="Portfolio Holdings")
    weights_table.add_column("Ticker", style="cyan")
    weights_table.add_column("Weight", justify="right")
    weights_table.add_column("Market Value", justify="right")
    for ticker, w, mv in zip(
        portfolio.tickers, portfolio.weights, portfolio.market_values
    ):
        weights_table.add_row(ticker, f"{w * 100:.1f}%", f"${mv:,.2f}")
    console.print(weights_table)

    # ----------------------------------------------------------- Risk calculations
    console.print("\n[bold]Computing risk metrics...[/bold]")
    calc = RiskCalculator(portfolio, risk_free_rate=args.risk_free_rate)
    risk_report = calc.compute_all()

    metrics_table = Table(title="Risk Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right")
    for label, value in risk_report.to_dict().items():
        display_label = label.replace("_", " ")
        if value is None:
            formatted = "N/A"
        elif "Ratio" in label or label in ("Beta", "Alpha"):
            formatted = f"{value:.4f}"
        else:
            formatted = f"{value * 100:.2f}%"
        metrics_table.add_row(display_label, formatted)
    console.print(metrics_table)

    # ---------------------------------------------------------- Monte Carlo
    console.print(f"\n[bold]Running Monte Carlo simulation ({args.simulations:,} paths, {args.horizon} days)...[/bold]")
    engine = MonteCarloEngine(
        portfolio,
        n_simulations=args.simulations,
        horizon=args.horizon,
        seed=args.seed,
    )
    sim_result = engine.run()

    mc_table = Table(title="Monte Carlo Results")
    mc_table.add_column("Metric", style="cyan")
    mc_table.add_column("Value", justify="right")
    mc_table.add_row("Initial Value", f"${sim_result.initial_value:,.2f}")
    mc_table.add_row("Mean Final Value", f"${sim_result.mean_final_value:,.2f}")
    mc_table.add_row("Median Final Value", f"${sim_result.median_final_value:,.2f}")
    mc_table.add_row("Std Dev", f"${sim_result.std_final_value:,.2f}")
    mc_table.add_row("Probability of Loss", f"{sim_result.prob_loss * 100:.1f}%")
    mc_table.add_row("5th Percentile", f"${sim_result.percentile(5):,.2f}")
    mc_table.add_row("95th Percentile", f"${sim_result.percentile(95):,.2f}")
    console.print(mc_table)

    # ------------------------------------------------------------- Stress test
    if args.stress_test is not None:
        console.print(f"\n[bold]Running stress test (shock: {args.stress_test * 100:.0f}%)...[/bold]")
        stress = engine.stress_test(shock_pct=args.stress_test)
        stress_table = Table(title="Stress Test Results")
        stress_table.add_column("Metric", style="cyan")
        stress_table.add_column("Value", justify="right")
        stress_table.add_row("Mean Final Value", f"${stress.mean_final_value:,.2f}")
        stress_table.add_row("Probability of Loss", f"{stress.prob_loss * 100:.1f}%")
        stress_table.add_row("5th Percentile", f"${stress.percentile(5):,.2f}")
        console.print(stress_table)

    # ----------------------------------------------------------- Export reports
    output_dir = Path(args.output_dir)
    report_data = build_report_data(portfolio, risk_report, sim_result)

    json_path = export_json(report_data, output_dir / "risk_report.json")
    console.print(f"\n[green]JSON report saved:[/green] {json_path}")

    if not args.json_only:
        if not args.no_charts:
            console.print("[bold]Generating charts...[/bold]")
            chart_paths = generate_dashboard(portfolio, sim_result, risk_report, output_dir)
            for p in chart_paths:
                console.print(f"  [green]Saved:[/green] {p}")

        html_path = export_html(report_data, output_dir, output_dir / "risk_report.html")
        console.print(f"[green]HTML report saved:[/green] {html_path}")

    console.print("\n[bold green]Analysis complete.[/bold green]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
