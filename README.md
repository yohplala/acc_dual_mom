# acc_dual_mom

**Accelerated Dual Momentum on PEA-eligible UCITS ETFs.**

A multi-strategy backtester and weekly signal generator for the Antonacci-style accelerated dual momentum (ADM) approach, restricted to ETFs eligible for the French PEA wrapper. Two Sunday-anchored cadences (weekly + monthly first Sunday) plus a buy-and-hold mode are backtested in parallel so the rebalance-frequency effect can be isolated on the same asset slice. Live ETF history is stitched onto pre-inception index-proxy data so backtests cover 2012-02 onward, gated by EEMA's first trading day.

## Features

- 📈 EUR-denominated closes for 9 PEA-eligible Amundi UCITS ETFs (yfinance) + €STR-derived synthetic safe asset (ECB Data Portal, with EONIA splice for pre-2019 history)
- 🪡 Index-proxy stitching: every strategy ETF has a TR proxy (CSPX.AS / IWM / IWDA.AS / IMEU.AS / SXRJ.DE / IJPE.L / EEMA / CSX5.AS) covering pre-inception history back to ~2010
- 🧮 Accelerated dual-momentum scoring: mean of 1m / 3m / 6m ROC on EUR closes; per-strategy override supports classic 12m and 6m Antonacci variants
- 🛡 Absolute-momentum filter vs. the safe asset before any allocation decision
- 🥇 Top-N selection with score-proportional weights, rounded to 10% steps via largest-remainder (Hare quota); buy-and-hold mode for zero-cost reference benchmarks
- 📅 Sunday-anchored rebalance cadences available: `weekly_sunday`, `biweekly_sunday`, `monthly_first_sunday`, plus `buy_and_hold` mode
- ⚙️ Vectorized polars backtest engine with one-trading-day execution lag (Friday signal → Monday fill) and turnover-based transaction costs
- 📊 Overlaid Plotly equity curves + drawdown chart + metrics table (CAGR / Sharpe / Sortino / MaxDD / Calmar / hit rate / avg correlation / total cost)
- 🔍 Discovery universe: 107-entry catalogue of all PEA-eligible Amundi ETFs with correlation matrix, perimeter-aware redundancy groups, best-in-group representative (CAGR − TER), and per-strategy `remove` / `replace` diagnostics
- 🤖 GitHub Actions cron (Saturday 02:15 UTC) → fetch → discover → backtest → render → deploy to GitHub Pages
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

- **[Live dashboard](https://yohplala.github.io/acc_dual_mom/)** — overlaid equity curves, drawdown, current signals, performance metrics. Refreshed every Saturday 02:15 UTC.
- **[Correlation matrix](https://yohplala.github.io/acc_dual_mom/correlations.html)** — pairwise correlation across the broad PEA-eligible Amundi universe + perimeter-aware redundancy groups, best-in-group picks (CAGR − TER), and per-strategy `remove` / `replace` diagnostics. ETF names link directly to their Amundi product pages.

### 📋 References

- **[Strategy configuration](strategies.yaml)** — universe (9 Amundi PEA ETFs + safe asset), shared scoring / allocation / cost settings, per-strategy rebalance cadence and optional lookback override
- **[Discovery universe](pea_universe.yaml)** — 107-entry catalogue of all Amundi PEA-eligible UCITS ETFs (~42 with confirmed Yahoo tickers, growing iteratively)
- **[Weekly update workflow](.github/workflows/weekly-update.yml)** — fetch → discover → backtest → build → deploy pipeline
- **[Tests workflow](.github/workflows/tests.yml)** — ruff + mypy + pytest on every PR

## Branch model

Mirrors the [halvix](https://github.com/yohplala/halvix) pattern — orphan branches keep parquet binaries out of `main`'s git history while remaining cheap to clone in CI.

| Branch | Contents |
|--------|----------|
| `main` | Source code |
| `prices-data` | Orphan, force-pushed: `prices.parquet` (ETF closes + €STR synthetic) |
| `backtest-results` | Orphan, force-pushed: per-strategy equity curves + rebalance logs |
| `gh-pages` | Auto-deployed by `actions/deploy-pages@v5` |

## Project Status

### Core engine
| Module | Status |
|--------|--------|
| Universe & config loader | ✅ Complete |
| Price fetching (yfinance + ECB €STR) | ✅ Complete |
| EONIA pre-2019 €STR splice | ✅ Complete |
| Index-proxy stitching (TR proxies, EUR conversion) | ✅ Complete |
| Scoring (mean of 1m/3m/6m ROC, per-strategy lookback override) | ✅ Complete |
| Allocation (filter + top-N + score-proportional + rounding) | ✅ Complete |
| Sunday-anchored rebalance schedules | ✅ Complete |
| Polars backtest engine (rotation + buy-and-hold modes) | ✅ Complete |
| Performance metrics (incl. avg correlation, total cost) | ✅ Complete |

### Reporting & automation
| Module | Status |
|--------|--------|
| Plotly + Jinja2 strategy dashboard | ✅ Complete |
| Discovery universe + correlation matrix page | ✅ Complete |
| Per-strategy `remove` / `replace` diagnostics | ✅ Complete |
| Clickable Amundi product page links | ✅ Complete |
| CLI (`pea-mom`: fetch / backtest / signal / render / discover / render-correlations / run) | ✅ Complete |
| Weekly cron + GitHub Pages deploy | ✅ Complete |

### Backlog
| Item | Status |
|--------|--------|
| Per-strategy detail pages | ⏳ Planned |
| Continue filling `pea_universe.yaml` Yahoo tickers (currently 42 / 107) | ⏳ In progress |

## License

[MIT](LICENSE)
