"""Loaders for the strategies.yaml configuration file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Asset:
    id: str
    name: str
    isin: str
    yahoo: str
    ter_pct: float
    replication: str
    region: str
    # Optional Yahoo ticker for the underlying index, used to extend price
    # history backward before the ETF launched. None disables stitching for
    # this asset (history is limited to the ETF's own data).
    index_proxy: str | None = None
    # Currency / return type of the proxy:
    #   "eur_pr"  EUR-denominated price-return index (no FX conversion)
    #   "usd_pr"  USD-denominated; converted via EURUSD=X
    #   "jpy_pr"  JPY-denominated; converted via EURJPY=X
    # NB: the kind tracks ONLY the FX-conversion path. Whether the underlying
    # series is PR or TR depends on the chosen ticker (e.g. ^GSPC is PR, ^SPXTR
    # is TR — both are USD-denominated and use the same FX path).
    index_proxy_kind: str | None = None
    # ETF inception date — splice point between proxy and live ETF history.
    inception: date | None = None
    # Where to fetch the proxy from: "yahoo" (yfinance, default) or "stooq"
    # (CSV via Stooq's Data Portal). Stooq has TR variants for European
    # indices that Yahoo lacks; Yahoo is preferred where TR is available
    # (e.g. ^SPXTR / ^NDXTR / ^RUTTR for US, ^GDAXI for Germany).
    index_proxy_source: str = "yahoo"
    # Optional Yahoo fallback ticker if the primary `index_proxy` fetch fails
    # (typical use: Stooq TR primary → Yahoo PR fallback to gracefully degrade
    # from TR to PR rather than crash). When None, primary fetch failures
    # propagate as FetchError so problems surface loudly.
    index_proxy_fallback: str | None = None


@dataclass(frozen=True, slots=True)
class SafeAsset:
    id: str
    name: str
    isin: str
    proxy: str
    ter_pct: float


@dataclass(frozen=True, slots=True)
class Scoring:
    lookbacks_days: tuple[int, ...]
    aggregation: str


@dataclass(frozen=True, slots=True)
class Allocation:
    rule: str
    granularity_pct: int
    min_weight_pct: float
    rounding: str


@dataclass(frozen=True, slots=True)
class Filter:
    type: str
    benchmark: str


@dataclass(frozen=True, slots=True)
class Costs:
    per_trade_pct: float


@dataclass(frozen=True, slots=True)
class Execution:
    signal_close: str
    fill_close: str


@dataclass(frozen=True, slots=True)
class Shared:
    scoring: Scoring
    allocation: Allocation
    filter: Filter
    costs: Costs
    execution: Execution


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    description: str
    asset_ids: tuple[str, ...]
    rebalance: str
    top_n: int
    reference_date: date | None = None
    # `rotation` runs the score → filter → top-N → score-prop allocation.
    # `buy_and_hold` allocates equal weights to `asset_ids` on day one and
    # never rebalances — useful as a zero-cost reference benchmark (e.g.
    # 100% MSCI World).
    mode: str = "rotation"


@dataclass(frozen=True, slots=True)
class Config:
    shared: Shared
    assets: tuple[Asset, ...]
    safe_asset: SafeAsset
    strategies: tuple[Strategy, ...]
    # Optional rendering layout for the signal table — list of rows, each a
    # list of asset ids. Read by render.py. If None, render falls back to
    # grouping by region. Each strategy uses only its slice of this layout
    # (assets not in the strategy are skipped, empty rows collapse).
    display_layout: tuple[tuple[str, ...], ...] | None = None

    def asset_by_id(self, asset_id: str) -> Asset:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        raise KeyError(f"Unknown asset id: {asset_id}")

    def assets_for(self, strategy: Strategy) -> tuple[Asset, ...]:
        return tuple(self.asset_by_id(aid) for aid in strategy.asset_ids)


def load_config(path: str | Path = "strategies.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> Config:
    shared_raw = raw["shared"]
    universe_raw = raw["universe"]

    shared = Shared(
        scoring=Scoring(
            lookbacks_days=tuple(shared_raw["scoring"]["lookbacks_days"]),
            aggregation=shared_raw["scoring"]["aggregation"],
        ),
        allocation=Allocation(
            rule=shared_raw["allocation"]["rule"],
            granularity_pct=int(shared_raw["allocation"]["granularity_pct"]),
            min_weight_pct=float(shared_raw["allocation"]["min_weight_pct"]),
            rounding=shared_raw["allocation"]["rounding"],
        ),
        filter=Filter(
            type=shared_raw["filter"]["type"],
            benchmark=shared_raw["filter"]["benchmark"],
        ),
        costs=Costs(per_trade_pct=float(shared_raw["costs"]["per_trade_pct"])),
        execution=Execution(
            signal_close=shared_raw["execution"]["signal_close"],
            fill_close=shared_raw["execution"]["fill_close"],
        ),
    )

    assets = tuple(
        Asset(
            id=a["id"],
            name=a["name"],
            isin=a["isin"],
            yahoo=a["yahoo"],
            ter_pct=float(a["ter_pct"]),
            replication=a["replication"],
            region=a["region"],
            index_proxy=a.get("index_proxy"),
            index_proxy_kind=a.get("index_proxy_kind"),
            inception=_parse_date(a.get("inception")),
            index_proxy_source=a.get("index_proxy_source", "yahoo"),
            index_proxy_fallback=a.get("index_proxy_fallback"),
        )
        for a in universe_raw["assets"]
    )

    sa = universe_raw["safe_asset"]
    safe_asset = SafeAsset(
        id=sa["id"],
        name=sa["name"],
        isin=sa["isin"],
        proxy=sa["proxy"],
        ter_pct=float(sa["ter_pct"]),
    )

    strategies = tuple(_parse_strategy(s) for s in raw["strategies"])

    raw_layout = universe_raw.get("display_layout")
    display_layout: tuple[tuple[str, ...], ...] | None = (
        tuple(tuple(row) for row in raw_layout) if raw_layout else None
    )

    return Config(
        shared=shared,
        assets=assets,
        safe_asset=safe_asset,
        strategies=strategies,
        display_layout=display_layout,
    )


VALID_MODES: frozenset[str] = frozenset({"rotation", "buy_and_hold"})


def _parse_strategy(s: dict[str, Any]) -> Strategy:
    mode = s.get("mode", "rotation")
    if mode not in VALID_MODES:
        raise ValueError(
            f"strategy {s.get('name', '?')!r}: unknown mode {mode!r} "
            f"(valid: {sorted(VALID_MODES)})"
        )
    return Strategy(
        name=s["name"],
        description=s.get("description", ""),
        asset_ids=tuple(s["assets"]),
        rebalance=s["rebalance"],
        top_n=int(s["top_n"]),
        reference_date=_parse_date(s.get("reference_date")),
        mode=mode,
    )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
