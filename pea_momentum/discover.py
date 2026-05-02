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
import re
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
    category: str
    yahoo: str | None
    leveraged: bool = False
    amundi_url: str | None = None

    @property
    def region(self) -> str:
        """Coarse geographic / asset-class perimeter, derived from `category`.

        Used by `find_groups()` to skip unioning two assets whose daily
        correlation exceeds the threshold but which sit in distinct
        perimeters (e.g. MSCI World vs S&P 500: same beta, different
        underlying universes — should not be flagged as redundant)."""
        return _coarse_region(self.category)


def _coarse_region(category: str) -> str:
    cat = category.upper()
    if cat.startswith("USA"):
        return "USA"
    if cat.startswith("WORLD"):
        return "WORLD"
    if cat.startswith("JAPAN"):
        return "JAPAN"
    if cat in ("EMERGING-ASIA", "CHINA", "INDIA"):
        return "EM-ASIA"
    if cat.startswith("EMERGING"):
        return "EM"
    if cat.startswith("ASIA-PACIFIC"):
        return "ASIA-PACIFIC"
    if cat.startswith("CASH"):
        return "CASH"
    if cat.startswith("THEMATIC"):
        return "THEMATIC"
    # Default bucket: Eurozone, Europe, individual EU countries, EU sectors
    # (all the sector ETFs in our YAML are European-focused)
    return "EUROPE"


def load_discovery_universe(path: str | Path = "pea_universe.yaml") -> list[DiscoveryEntry]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        DiscoveryEntry(
            id=e["id"],
            name=e["name"],
            isin=e["isin"],
            currency=e["currency"],
            ter_pct=float(e["ter_pct"]),
            category=e.get("category", "Other"),
            yahoo=e.get("yahoo"),
            leveraged=bool(e.get("leveraged", False)),
            amundi_url=e.get("amundi_url"),
        )
        for e in raw["universe"]
    ]


_AMUNDI_URL_BASE = "https://www.amundietf.fr/fr/particuliers/produits"

# Share-class suffixes Amundi appends after `-ucits-etf` on the product URL.
# Ordered longest-first so the longest match wins (e.g. `-eur-hedged-acc` beats
# `-acc`). Names that don't end in any of these get a plain `-ucits-etf` append.
_SHARE_CLASS_SUFFIXES: tuple[str, ...] = (
    "-eur-hedged-acc",
    "-eur-hedged-dist",
    "-usd-hedged-acc",
    "-usd-hedged-dist",
    "-eur-acc",
    "-eur-dist",
    "-usd-acc",
    "-usd-dist",
    "-acc",
    "-dist",
    "-dr",
)

_FIXED_INCOME_KEYWORDS = ("money-market", "short-term")
_FIXED_INCOME_PREFIXES = ("cash", "bond")


def amundi_product_url(entry: DiscoveryEntry) -> str:
    """Resolve the Amundi product page URL for `entry`.

    Returns `entry.amundi_url` verbatim when set in the YAML; otherwise
    constructs the URL from name + ISIN using Amundi's observed slug pattern:

        {base}/{asset_class}/{slug}/{isin-lower}

    The slug is the lowercase name with non-alphanumerics dropped and spaces
    replaced by dashes. Amundi inserts `ucits-etf` between the product name
    and the share-class suffix even when the source name omits it, so we
    mirror that.

    Best-effort — set `amundi_url:` in `pea_universe.yaml` to override when
    the heuristic mismatches (e.g. products with expanded English index
    names like `msci-ac-asia-pacific-ex-japan`).
    """
    if entry.amundi_url:
        return entry.amundi_url
    return f"{_AMUNDI_URL_BASE}/{_amundi_asset_class(entry.category)}/{_slug(entry.name)}/{entry.isin.lower()}"


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", name.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    if "ucits-etf" in slug:
        return slug
    for suffix in _SHARE_CLASS_SUFFIXES:
        if slug.endswith(suffix):
            return slug[: -len(suffix)] + "-ucits-etf" + suffix
    return slug + "-ucits-etf"


def _amundi_asset_class(category: str) -> str:
    """Map a YAML `category` to the Amundi URL's asset-class path segment.
    Money-market / cash funds sit under `/fixed-income/`; everything else
    under `/equity/`. Future bond ETFs would also use `/fixed-income/`."""
    cat = category.lower()
    if cat.startswith(_FIXED_INCOME_PREFIXES) or any(k in cat for k in _FIXED_INCOME_KEYWORDS):
        return "fixed-income"
    return "equity"


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
        isin=entry.isin,
        yahoo=entry.yahoo or "",
        region=entry.category.lower(),
    )
