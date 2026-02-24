![CI](https://github.com/stabrea/financial-risk-analyzer/actions/workflows/ci.yml/badge.svg)

# Financial Risk Analyzer

A portfolio risk analysis toolkit that uses Monte Carlo simulation and statistical methods to quantify investment risk. Built for analyzing multi-asset portfolios with configurable parameters, producing detailed reports and visualizations.

## Features

- **Monte Carlo Simulation** -- Correlated Geometric Brownian Motion with Cholesky decomposition for realistic multi-asset path generation
- **Value at Risk (VaR)** -- Historical, parametric, and Monte Carlo VaR at configurable confidence levels
- **Conditional VaR (CVaR)** -- Expected Shortfall beyond the VaR threshold
- **Risk-Adjusted Returns** -- Sharpe Ratio, Sortino Ratio, Calmar Ratio, Information Ratio
- **CAPM Metrics** -- Beta and Jensen's Alpha relative to benchmark
- **Drawdown Analysis** -- Maximum drawdown computation and time-series visualization
- **Stress Testing** -- Simulate immediate market shocks followed by stochastic recovery
- **Report Generation** -- JSON and styled HTML reports with embedded charts
- **CLI Interface** -- Full command-line tool with configurable simulation parameters

## Sample Output

After running the analyzer, the following charts are generated in the `output/` directory:

| Chart | Description |
|-------|-------------|
| `simulation_paths.png` | Monte Carlo paths with median and confidence bands |
| `return_distribution.png` | Histogram of simulated returns with VaR line |
| `drawdown.png` | Portfolio drawdown over time |
| `weights.png` | Portfolio allocation pie chart |
| `correlation.png` | Asset return correlation heatmap |

An HTML report (`risk_report.html`) combines all metrics and charts into a single dashboard.

## Installation

```bash
# Clone the repository
git clone https://github.com/stabrea/financial-risk-analyzer.git
cd financial-risk-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run with the included sample data (10-stock portfolio, 1 year of daily prices):

```bash
python main.py
```

### CLI Options

```bash
# Custom simulation parameters
python main.py --simulations 50000 --horizon 504 --seed 42

# Use your own data
python main.py --holdings path/to/holdings.csv --prices path/to/prices.csv

# JSON-only output (no charts)
python main.py --json-only

# Run with stress test (-20% market shock)
python main.py --stress-test -0.20

# Full options
python main.py --help
```

### Data Format

**Holdings CSV** (`data/sample_portfolio.csv`):

```csv
ticker,shares,purchase_price
AAPL,50,142.50
MSFT,30,285.00
```

**Historical Prices CSV** (`data/historical_prices.csv`):

```csv
date,AAPL,MSFT,...
2024-01-02,150.00,310.00,...
2024-01-03,151.25,312.40,...
```

### As a Python Library

```python
from risk_analyzer.data_loader import load_portfolio
from risk_analyzer.monte_carlo import MonteCarloEngine
from risk_analyzer.risk_metrics import RiskCalculator

portfolio = load_portfolio("data/sample_portfolio.csv", "data/historical_prices.csv")

# Risk metrics
calc = RiskCalculator(portfolio, risk_free_rate=0.05)
report = calc.compute_all()
print(f"VaR (95%): {report.var_95:.4f}")
print(f"Sharpe Ratio: {report.sharpe_ratio:.4f}")

# Monte Carlo
engine = MonteCarloEngine(portfolio, n_simulations=10_000, horizon=252, seed=42)
result = engine.run()
print(f"Mean final value: ${result.mean_final_value:,.2f}")
print(f"Probability of loss: {result.prob_loss:.1%}")
```

## Risk Metrics Explained

### Value at Risk (VaR)

The maximum expected loss over a given time period at a specified confidence level. A 95% daily VaR of 2.1% means there is a 5% chance the portfolio loses more than 2.1% in a single day.

Three calculation methods are supported:
- **Historical** -- Based on the empirical distribution of past returns
- **Parametric** -- Assumes normally distributed returns (variance-covariance method)
- **Monte Carlo** -- Derived from simulated future return distributions

### Conditional VaR (Expected Shortfall)

The average loss in the worst cases beyond the VaR threshold. More informative than VaR for tail risk because it captures the severity of extreme losses, not just their frequency.

### Sharpe Ratio

Measures risk-adjusted return: `(portfolio return - risk-free rate) / portfolio volatility`. Higher values indicate better compensation per unit of risk. Generally, above 1.0 is acceptable, above 2.0 is strong.

### Sortino Ratio

Similar to Sharpe but only penalizes downside volatility. Useful for portfolios where upside variance is desirable.

### Beta

Measures the portfolio's sensitivity to market movements. Beta of 1.0 means the portfolio moves with the market. Below 1.0 is defensive; above 1.0 is aggressive.

### Jensen's Alpha

The excess return above what the CAPM predicts given the portfolio's beta. Positive alpha indicates outperformance relative to the risk taken.

### Maximum Drawdown

The largest peak-to-trough decline in portfolio value. Represents the worst-case historical loss an investor would have experienced.

### Calmar Ratio

Annualized return divided by maximum drawdown. Measures return per unit of drawdown risk.

## Architecture

```
financial-risk-analyzer/
├── main.py                     # Entry point
├── risk_analyzer/
│   ├── __init__.py             # Package metadata
│   ├── portfolio.py            # Portfolio class (holdings, weights, returns)
│   ├── monte_carlo.py          # GBM Monte Carlo engine with Cholesky correlation
│   ├── risk_metrics.py         # VaR, CVaR, Sharpe, Beta, drawdown calculations
│   ├── data_loader.py          # CSV data loading and validation
│   ├── visualizer.py           # Matplotlib chart generation
│   ├── report.py               # JSON + HTML report export
│   └── cli.py                  # Argument parsing and CLI pipeline
├── data/
│   ├── sample_portfolio.csv    # 10-stock sample holdings
│   └── historical_prices.csv   # 1 year of daily price data
├── output/                     # Generated reports and charts (gitignored)
├── requirements.txt
├── setup.py
└── LICENSE
```

### Design Decisions

- **No external API dependencies** -- All data loaded from local CSV files, making the tool portable and reproducible
- **Correlated GBM** -- Monte Carlo uses Cholesky decomposition of the historical covariance matrix to generate correlated asset paths, producing realistic multi-asset simulations
- **Immutable results** -- `SimulationResult` and `RiskReport` are frozen dataclasses to prevent accidental mutation
- **Separation of concerns** -- Data loading, computation, visualization, and reporting are independent modules

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- rich

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*This tool is for educational and analytical purposes. It is not financial advice.*
