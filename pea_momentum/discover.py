"""Fetch close prices for the broad PEA-eligible Amundi ETF discovery universe.

Independent of `fetch.py` (which fetches the strategy universe). The
discovery universe lives in `pea_universe.yaml` and contains every
PEA-eligible Amundi ETF — used for correlation-matrix exploration and
redundancy detection, not for active strategies.

Loud failures are *partial* here — a single bad ticker should not break
the discovery fetch. Each per-asset failure is logged and skipped; the
caller gets back whatever data was successfully retrieved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import yaml

from . import fetch
from .errors import FetchError
from .universe import Asset

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveryEntry:
    id: str
    name: str
    isin: str
    currency: str
    ter_pct: float
    sfdr: str
    category: str
    yahoo: str | None
    leveraged: bool = False


def load_discovery_universe(path: str | Path = "pea_universe.yaml") -> list[DiscoveryEntry]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        DiscoveryEntry(
            id=e["id"],
            name=e["name"],
            isin=e["isin"],
            currency=e["currency"],
            ter_pct=float(e["ter_pct"]),
            sfdr=e.get("sfdr", "Article 6"),
            category=e.get("category", "Other"),
            yahoo=e.get("yahoo"),
            leveraged=bool(e.get("leveraged", False)),
        )
        for e in raw["universe"]
    ]


def fetch_discovery_universe(
    entries: list[DiscoveryEntry],
    start: date,
    *,
    skip_leveraged: bool = True,
    skip_non_eur: bool = True,
) -> pl.DataFrame:
    """Fetch every entry with a Yahoo ticker. Returns long-format
    `[date, asset_id, close, source]` covering all successfully-fetched
    assets. Per-asset failures log a warning and are skipped — the
    discovery fetch never crashes the whole run because of one bad ticker.

    `skip_leveraged`: omit 2x / inverse ETFs (they distort correlation).
    `skip_non_eur`: omit USD/CHF/GBP-denominated ETFs (would need FX
    conversion which is out of scope for the discovery view).
    """
    frames: list[pl.DataFrame] = []
    skipped_no_ticker = 0
    skipped_leveraged = 0
    skipped_non_eur = 0
    failed = 0

    for entry in entries:
        if entry.yahoo is None:
            skipped_no_ticker += 1
            continue
        if skip_leveraged and entry.leveraged:
            skipped_leveraged += 1
            continue
        if skip_non_eur and entry.currency != "EUR":
            skipped_non_eur += 1
            continue

        as_asset = _entry_to_asset(entry)
        try:
            df = fetch.fetch_yahoo(as_asset, start=start)
            frames.append(df)
        except FetchError as exc:
            log.warning("discovery fetch failed for %s (%s): %s", entry.id, entry.yahoo, exc)
            failed += 1

    log.info(
        "discovery: %d fetched, %d no-ticker, %d leveraged, %d non-EUR, %d failed",
        len(frames),
        skipped_no_ticker,
        skipped_leveraged,
        skipped_non_eur,
        failed,
    )

    if not frames:
        return pl.DataFrame(
            schema={"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64, "source": pl.Utf8}
        )
    return pl.concat(frames)


def _entry_to_asset(entry: DiscoveryEntry) -> Asset:
    """Adapt a `DiscoveryEntry` into the `Asset` shape that `fetch.fetch_yahoo`
    expects. We don't use the proxy / inception fields here — discovery
    fetches live ETF history only."""
    return Asset(
        id=entry.id,
        name=entry.name,
        isin=entry.isin,
        yahoo=entry.yahoo or "",
        ter_pct=entry.ter_pct,
        replication="synthetic",
        region=entry.category.lower(),
    )
