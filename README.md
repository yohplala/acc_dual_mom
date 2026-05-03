# acc_dual_mom

**Rank-only momentum rotation on PEA-eligible UCITS ETFs (no positive-momentum filter — rank decides everything).**

A multi-strategy backtester and weekly signal generator inspired by Antonacci's accelerated dual momentum (ADM), restricted to ETFs eligible for the French PEA wrapper. Sunday-anchored rotation cadences (weekly / biweekly / monthly-first-Sunday / quarterly-first-Sunday / semiannual-first-Sunday) plus buy-and-hold benchmarks are backtested in parallel so the rebalance-frequency, lookback, and concentration effects can each be isolated on the same asset slice. Live ETF history is stitched onto pre-inception index-proxy data — single-stage or multi-stage chains where a longer-history proxy extends a cleaner one's pre-handoff range — so backtests span the GFC (2008-08+ across the active universe). See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the canonical methodology reference.

## Features

- 📈 EUR-denominated closes for 7 PEA-eligible Amundi UCITS ETFs (yfinance) + €STR-derived synthetic safe asset (ECB Data Portal, with EONIA splice for pre-2019 history)
- 🪡 Index-proxy stitching: TR proxies extend every strategy ETF's history pre-inception, with **multi-stage chains** for `world` (IWDA.AS → IWRD.L → ACWI) and `em_asia` (EEMA → AAXJ) and `eurostoxx50` (CSX5.AS → MSE.PA); single proxies for `sp500` (SPY since 1993), `russell2000` (IWM since 2000), `topix` (EWJ since 1996), `eu_banks` (EXX1.DE since 2008). Active-universe coverage spans 2008-08 onward, gated by AAXJ.
- 🧮 Accelerated scoring: mean of 1m / 3m / 6m ROC on EUR closes; per-strategy `lookbacks_days` override supports classic single-12m / single-6m variants. `mean` / `median` / `min` aggregations.
- 🛡 Defensive cash via ranking, not filtering. There is **no positive-momentum floor** — the top-N picks the highest-scoring assets regardless of sign (in synchronised bears, the strategy holds the least-negative leader). To get cash defence in crashes, include `cash_estr` in the strategy's `assets:` — its small €STR-positive score naturally wins top-N when every equity score collapses below zero.
- 🥇 Top-N selection with `equal_weight` (default) or `score_proportional` weighting; largest-remainder rounding to 10% steps. Buy-and-hold mode for zero-cost benchmarks.
- 📅 Sunday-anchored cadences: `weekly_sunday`, `biweekly_sunday`, `monthly_first_sunday`, `quarterly_first_sunday`, `semiannual_first_sunday`, plus `buy_and_hold` mode
- ⚙️ Vectorized polars backtest engine with one-trading-day execution lag (Friday signal → Monday fill); per-asset cost model: shared broker fee + per-asset bid-ask spread half (configurable via `est_spread_bps`)
- 📊 Overlaid Plotly equity curves + drawdown chart + sortable metrics table (CAGR / Sharpe / Sortino / MaxDD / Calmar / daily & rebal hit / turnover/yr / avg correlation / total cost) + sortable signal table (4 region columns + Cash + Scoring + Allocation + non-zero allocations).
- 🔍 Discovery universe: 107-entry catalogue of all PEA-eligible Amundi ETFs with correlation matrix, complete-link redundancy groups, lowest-TER representative pick, and per-strategy `remove` / `replace` diagnostics
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
- **[Correlation matrix](https://yohplala.github.io/acc_dual_mom/correlations.html)** — pairwise correlation across the broad PEA-eligible Amundi universe + perimeter-aware complete-link redundancy groups, lowest-TER representative pick, and per-strategy `remove` / `replace` diagnostics. ETF names link directly to their Amundi product pages.

### 📋 References

- **[Methodology](docs/METHODOLOGY.md)** — strategy modes, scoring, allocation rules, cost model (broker fee + per-asset spread), stitching, metrics, redundancy analysis. The canonical "what does the dashboard show me" reference.
- **[Strategy configuration](strategies.yaml)** — active universe (7 Amundi PEA ETFs + €STR-derived safe asset), shared scoring / allocation / cost settings, per-strategy cadence + optional `lookbacks_days` / `allocation_rule` overrides
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
| Scoring (mean / median / min over `lookbacks_days`, per-strategy override) | ✅ Complete |
| Allocation (positive-momentum floor + top-N + equal_weight / score-prop + rounding) | ✅ Complete |
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

## License

[MIT](LICENSE)
