# acc_dual_mom

Accelerated Dual Momentum on PEA-eligible UCITS ETFs.

A multi-strategy backtester and live signal generator for the Antonacci-style
accelerated dual momentum (ADM) approach, restricted to ETFs eligible for the
French PEA wrapper. Built around polars + parquet for storage and GitHub
Actions for scheduled execution.

## What it does

- Fetches monthly EUR-denominated closes for a curated universe of PEA-eligible ETFs
- Scores each asset using mean of 1m / 3m / 6m ROC
- Filters by absolute momentum vs. an €STR-based synthetic safe asset
- Selects top-N and assigns score-proportional weights, rounded to 10% steps
- Backtests multiple strategies in parallel (different rebalance frequencies)
- Renders an interactive HTML report with overlaid equity curves, deployed to GitHub Pages

## Strategies

Strategies share scoring/allocation/cost logic but vary in universe slice and
rebalance cadence. Three Sunday-anchored cadences are supported:

- `weekly_sunday` — every Sunday
- `biweekly_sunday` — every other Sunday (anchored to a `reference_date`)
- `monthly_first_sunday` — first Sunday of each month

The signal is computed using the preceding Friday's close; execution is assumed
at the following Monday's close (one-trading-day lag, no lookahead).

## Branches

Mirrors the halvix pattern:

- `main` — source code
- `prices-data` — orphan, parquet inputs (ETF closes, €STR fixings)
- `backtest-results` — orphan, parquet+json outputs (equity curves, signals history)
- `gh-pages` — auto-deployed HTML report

## Quick start

```bash
poetry install
poetry run pea-mom fetch          # populate prices-data
poetry run pea-mom backtest       # run all strategies
poetry run pea-mom signal         # show this week's signals
poetry run pea-mom render         # build HTML report
```
