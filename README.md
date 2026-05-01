# acc_dual_mom

**Accelerated Dual Momentum on PEA-eligible UCITS ETFs.**

A multi-strategy backtester and weekly signal generator for the Antonacci-style accelerated dual momentum (ADM) approach, restricted to ETFs eligible for the French PEA wrapper. Three Sunday-anchored cadences (weekly, biweekly, monthly first Sunday) are backtested in parallel so the rebalance-frequency effect can be isolated on the same asset slice.

## Features

- 📈 EUR-denominated closes for 12 PEA-eligible Amundi UCITS ETFs (yfinance) + €STR-derived synthetic safe asset (ECB Data Portal)
- 🧮 Accelerated dual-momentum scoring: mean of 1m / 3m / 6m ROC on EUR closes
- 🛡 Absolute-momentum filter vs. the safe asset before any allocation decision
- 🥇 Top-N selection with score-proportional weights, rounded to 10% steps via largest-remainder (Hare quota)
- 📅 Three Sunday-anchored rebalance cadences in one run: `weekly_sunday`, `biweekly_sunday` (anchored), `monthly_first_sunday`
- ⚙️ Vectorized polars backtest engine with one-trading-day execution lag (Friday signal → Monday fill) and turnover-based transaction costs
- 📊 Overlaid Plotly equity curves + drawdown chart + metrics table (CAGR / Sharpe / Sortino / MaxDD / Calmar / hit rate)
- 🤖 GitHub Actions cron (Saturday 04:00 UTC) → fetch → backtest → render → deploy to GitHub Pages
- 🗄 Orphan parquet branches (`prices-data`, `backtest-results`) keep binary history out of `main`

## Quick Start

```bash
# Install
poetry install

# Fetch ETF closes from yfinance + €STR fixings from ECB
poetry run pea-mom fetch

# Run all configured strategies
poetry run pea-mom backtest

# Print this week's signals
poetry run pea-mom signal

# Build the HTML report into ./site
poetry run pea-mom render

# Full pipeline: fetch → backtest → render
poetry run pea-mom run
```

Strategy config lives in [`strategies.yaml`](strategies.yaml) — universe slice + rebalance cadence per strategy, shared scoring / allocation / cost knobs above.

## Documentation

### 📊 Live Data & Charts

- **[Live dashboard](https://yohplala.github.io/acc_dual_mom/)** — overlaid equity curves, drawdown, current signals, performance metrics. Refreshed every Saturday 04:00 UTC.
- **[Correlation matrix](https://yohplala.github.io/acc_dual_mom/correlations.html)** — pairwise correlation across the broad PEA-eligible Amundi universe + redundancy groups. Useful for picking non-overlapping assets when constructing strategy universes.

### 📋 References

- **[Strategy configuration](strategies.yaml)** — universe (12 Amundi PEA ETFs + safe asset), shared scoring / allocation / cost / execution settings, per-strategy rebalance cadence
- **[Weekly update workflow](.github/workflows/weekly-update.yml)** — fetch → backtest → build → deploy pipeline
- **[Tests workflow](.github/workflows/tests.yml)** — ruff + mypy + pytest on every PR

## Branch model

Mirrors the [halvix](https://github.com/yohplala/halvix) pattern — orphan branches keep parquet binaries out of `main`'s git history while remaining cheap to clone in CI.

| Branch | Contents |
|--------|----------|
| `main` | Source code |
| `prices-data` | Orphan, force-pushed: `prices.parquet` (ETF closes + €STR synthetic) |
| `backtest-results` | Orphan, force-pushed: per-strategy equity curves + rebalance logs |
| `gh-pages` | Auto-deployed by `actions/deploy-pages@v4` |

## Project Status

| Module | Status |
|--------|--------|
| Universe & config loader | ✅ Complete |
| Price fetching (yfinance + ECB €STR) | ✅ Complete |
| Scoring (1m / 3m / 6m ROC) | ✅ Complete |
| Allocation (filter + top-N + rounding) | ✅ Complete |
| Sunday-anchored rebalance schedules | ✅ Complete |
| Polars backtest engine | ✅ Complete |
| Performance metrics | ✅ Complete |
| Plotly + Jinja2 HTML report | ✅ Complete |
| CLI (`pea-mom`) | ✅ Complete |
| Weekly cron + GitHub Pages deploy | ✅ Complete |
| Pre-ETF index stitching | ⏳ Planned |
| Per-strategy detail pages | ⏳ Planned |
| EONIA pre-2019 €STR splice | ⏳ Planned |

## License

[MIT](LICENSE)
