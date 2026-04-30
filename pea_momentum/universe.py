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


@dataclass(frozen=True, slots=True)
class Config:
    shared: Shared
    assets: tuple[Asset, ...]
    safe_asset: SafeAsset
    strategies: tuple[Strategy, ...]

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

    strategies = tuple(
        Strategy(
            name=s["name"],
            description=s.get("description", ""),
            asset_ids=tuple(s["assets"]),
            rebalance=s["rebalance"],
            top_n=int(s["top_n"]),
            reference_date=_parse_date(s.get("reference_date")),
        )
        for s in raw["strategies"]
    )

    return Config(
        shared=shared,
        assets=assets,
        safe_asset=safe_asset,
        strategies=strategies,
    )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
