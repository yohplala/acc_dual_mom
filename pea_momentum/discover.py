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


def amundi_product_url(entry: DiscoveryEntry) -> str:
    """Resolve the Amundi product page URL for `entry`.

    Uses `entry.amundi_url` if explicitly set in the YAML; otherwise
    constructs from name + ISIN using Amundi's observed slug pattern:

        https://www.amundietf.fr/fr/professionnels/produits/{asset_class}/{slug}/{isin-lower}

    Two key calibrations from observation:

    * **Audience = ``professionnels``** (B2B). Amundi's site has parallel
      ``/particuliers/`` (B2C) and ``/professionnels/`` (B2B) trees. The
      B2B slugs follow a consistent French-language pattern that matches
      our heuristic; the B2C slugs are inconsistent — sometimes
      English-localised (``Japan`` instead of ``Japon``), sometimes
      retaining a legacy ``lyxor-`` prefix on rebranded products. Linking
      to ``professionnels`` URLs is the more reliable default.

    * **Asset class** = ``equity`` for stock ETFs, ``fixed-income`` for
      money-market / short-term cash funds (e.g. PEA Euro Court Terme,
      which lives at ``/fixed-income/`` not ``/equity/``). Detected from
      the YAML ``category`` field.

    The slug itself: lowercase name, drop non-alphanumeric, dashes for
    spaces. Amundi inserts ``ucits-etf`` between the product name and
    the share-class indicator even when the source name omits it
    (e.g. "Amundi Core EURO STOXX 50 EUR Acc" →
    ``amundi-core-euro-stoxx-50-ucits-etf-eur-acc``); we mirror that.

    Best-effort — set ``amundi_url:`` in ``pea_universe.yaml`` to override
    when the heuristic doesn't match Amundi's actual URL (e.g. products
    whose slug uses an expanded English-language index name like
    ``msci-ac-asia-pacific-ex-japan`` instead of the abbreviated form).
    """
    if entry.amundi_url:
        return entry.amundi_url
    asset_class = _amundi_asset_class(entry.category)
    slug = re.sub(r"[^a-z0-9\s-]", "", entry.name.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    if "ucits-etf" not in slug:
        # Insert "ucits-etf" before the share-class suffix (longest match
        # wins so "-eur-hedged-acc" beats "-acc"). Falls through to a
        # plain append if no suffix matches.
        suffixes = (
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
        for suffix in suffixes:
            if slug.endswith(suffix):
                slug = slug[: -len(suffix)] + "-ucits-etf" + suffix
                break
        else:
            slug += "-ucits-etf"
    return (
        f"https://www.amundietf.fr/fr/professionnels/produits/"
        f"{asset_class}/{slug}/{entry.isin.lower()}"
    )


def _amundi_asset_class(category: str) -> str:
    """Map a YAML ``category`` to the Amundi URL's asset-class path
    segment. Money-market / cash funds sit under ``/fixed-income/``; all
    equity products sit under ``/equity/``. Future bond ETFs (none yet in
    our universe) would also use ``/fixed-income/``.
    """
    cat = category.lower()
    if (
        cat.startswith("cash")
        or "money-market" in cat
        or "short-term" in cat
        or cat.startswith("bond")
    ):
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
