"""Parquet-based storage for prices and backtest results.

Layout under the data root (typically `./data` locally, or a checked-out
data branch in CI):

    prices.parquet         long-format closes for every asset and the safe asset
    history/<strategy>.parquet   per-strategy weight + return time series
    metrics.json           cross-strategy summary statistics
    last_signals.json      most-recent rebalance per strategy (for the live CLI)

Schema for prices.parquet:

    date      Date
    asset_id  Utf8     (matches `id` from strategies.yaml; "safe" for the safe asset)
    close     Float64  EUR-denominated close
    source    Utf8     "yfinance" | "estr_synthetic" | "stitched_index"
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PRICES_FILE = "prices.parquet"
HISTORY_DIR = "history"
METRICS_FILE = "metrics.json"
LAST_SIGNALS_FILE = "last_signals.json"

PRICES_SCHEMA = {
    "date": pl.Date,
    "asset_id": pl.Utf8,
    "close": pl.Float64,
    "source": pl.Utf8,
}


def prices_path(data_root: str | Path) -> Path:
    return Path(data_root) / PRICES_FILE


def history_path(data_root: str | Path, strategy_name: str) -> Path:
    return Path(data_root) / HISTORY_DIR / f"{strategy_name}.parquet"


def metrics_path(data_root: str | Path) -> Path:
    return Path(data_root) / METRICS_FILE


def last_signals_path(data_root: str | Path) -> Path:
    return Path(data_root) / LAST_SIGNALS_FILE


def write_prices(df: pl.DataFrame, data_root: str | Path) -> Path:
    path = prices_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.select(list(PRICES_SCHEMA)).cast(PRICES_SCHEMA).sort(["asset_id", "date"])
    out.write_parquet(path, compression="zstd")
    return path


def read_prices(data_root: str | Path) -> pl.DataFrame:
    path = prices_path(data_root)
    if not path.exists():
        return pl.DataFrame(schema=PRICES_SCHEMA)
    return pl.read_parquet(path).cast(PRICES_SCHEMA)


def upsert_prices(new: pl.DataFrame, data_root: str | Path) -> pl.DataFrame:
    """Merge new rows with existing prices, dedup on (asset_id, date), latest wins."""
    if new.is_empty():
        return read_prices(data_root)
    existing = read_prices(data_root)
    merged = (
        pl.concat([existing, new.select(list(PRICES_SCHEMA)).cast(PRICES_SCHEMA)])
        .unique(subset=["asset_id", "date"], keep="last")
        .sort(["asset_id", "date"])
    )
    write_prices(merged, data_root)
    return merged


def write_history(df: pl.DataFrame, data_root: str | Path, strategy_name: str) -> Path:
    path = history_path(data_root, strategy_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return path


def read_history(data_root: str | Path, strategy_name: str) -> pl.DataFrame | None:
    path = history_path(data_root, strategy_name)
    if not path.exists():
        return None
    return pl.read_parquet(path)


def prices_wide(df: pl.DataFrame, asset_ids: list[str] | None = None) -> pl.DataFrame:
    """Pivot the long prices table to wide [date | id1 | id2 | ...]."""
    selected = df if asset_ids is None else df.filter(pl.col("asset_id").is_in(asset_ids))
    return selected.pivot(values="close", index="date", on="asset_id").sort("date")
