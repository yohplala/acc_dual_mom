"""Loaders for the strategies.yaml configuration file."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Asset:
    id: str
    isin: str
    yahoo: str
    region: str
    # Display name (Amundi product name, used in dashboards).
    name: str = ""
    # Annual TER as a percentage (e.g. 0.12 for 12 bps). Used for cost
    # modelling diagnostics; not subtracted from backtest returns since
    # auto_adjust=True closes are already net of fund fees.
    ter_pct: float = 0.0
    # "synthetic" (swap-based) or "physical" (sampling/full replication).
    # Informational only; not used in scoring.
    replication: str | None = None
    # Estimated round-trip bid-ask spread in bps. Half is added to
    # `shared.costs.per_trade_pct` per asset traded at rebalance time.
    # Default 0 keeps cost ≡ shared per_trade_pct for assets without
    # explicit spread data.
    est_spread_bps: float = 0.0
    # Optional Yahoo ticker for the underlying index, used to extend price
    # history backward before the ETF launched. None disables stitching for
    # this asset (history is limited to the ETF's own data).
    index_proxy: str | None = None
    # FX-conversion path for the proxy series:
    #   "eur_tr"  EUR-quoted total-return index (no FX needed)
    #   "usd_tr"  USD-quoted total-return index, converted via EURUSD=X
    # The kind tracks ONLY the FX-conversion path. Whether the underlying
    # series is PR or TR depends on the chosen ticker.
    index_proxy_kind: str | None = None
    # ETF inception date — splice point between proxy and live ETF history.
    inception: date | None = None


@dataclass(frozen=True, slots=True)
class SafeAsset:
    id: str
    proxy: str
    name: str = ""
    ter_pct: float = 0.0


@dataclass(frozen=True, slots=True)
class Scoring:
    lookbacks_days: tuple[int, ...]
    aggregation: str


@dataclass(frozen=True, slots=True)
class Allocation:
    rule: str
    granularity_pct: int
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
    """Sunday-anchored: signal uses preceding Friday close, fill at the
    following Monday close. The values are parsed for transparency; only
    the canonical pair is supported today and any deviation raises at
    config-load time."""

    signal_close: str = "friday"
    fill_close: str = "monday"


@dataclass(frozen=True, slots=True)
class Shared:
    scoring: Scoring
    allocation: Allocation
    filter: Filter
    costs: Costs
    execution: Execution = field(default_factory=Execution)


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    asset_ids: tuple[str, ...]
    rebalance: str
    top_n: int
    description: str = ""
    reference_date: date | None = None
    # `rotation` runs the score → filter → top-N → score-prop allocation.
    # `buy_and_hold` allocates equal weights to `asset_ids` on day one and
    # never rebalances — useful as a zero-cost reference benchmark (e.g.
    # 100% MSCI World).
    mode: str = "rotation"
    # Optional per-strategy scoring lookback override. If None, strategies
    # use `Config.shared.scoring.lookbacks_days` (the default ADM 21/63/126
    # mean). Setting `[126]` gives classic Antonacci dual momentum (single
    # 6-month lookback). Same `aggregation` rule applies (mean for >1
    # lookback; for a single value, mean returns the value as-is).
    lookbacks_days: tuple[int, ...] | None = None
    # Per-strategy override of `shared.allocation.rule`. None = use shared.
    allocation_rule: str | None = None

    def effective_scoring(self, shared_scoring: Scoring) -> Scoring:
        """Return the scoring config this strategy actually uses, applying
        the per-strategy lookback override if set."""
        if self.lookbacks_days is None:
            return shared_scoring
        return Scoring(
            lookbacks_days=self.lookbacks_days,
            aggregation=shared_scoring.aggregation,
        )

    def effective_allocation_rule(self, shared_rule: str) -> str:
        return self.allocation_rule or shared_rule


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


def load_config(path: str | Path = "strategies.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> Config:
    shared_raw = raw["shared"]
    universe_raw = raw["universe"]

    execution = _parse_execution(shared_raw.get("execution"))

    shared = Shared(
        scoring=Scoring(
            lookbacks_days=tuple(shared_raw["scoring"]["lookbacks_days"]),
            aggregation=shared_raw["scoring"]["aggregation"],
        ),
        allocation=Allocation(
            rule=shared_raw["allocation"]["rule"],
            granularity_pct=int(shared_raw["allocation"]["granularity_pct"]),
            rounding=shared_raw["allocation"]["rounding"],
        ),
        filter=Filter(
            type=shared_raw["filter"]["type"],
            benchmark=shared_raw["filter"]["benchmark"],
        ),
        costs=Costs(per_trade_pct=float(shared_raw["costs"]["per_trade_pct"])),
        execution=execution,
    )

    assets = tuple(
        Asset(
            id=a["id"],
            isin=a["isin"],
            yahoo=a["yahoo"],
            region=a["region"],
            name=a.get("name", ""),
            ter_pct=float(a.get("ter_pct", 0.0)),
            replication=a.get("replication"),
            est_spread_bps=float(a.get("est_spread_bps", 0.0)),
            index_proxy=a.get("index_proxy"),
            index_proxy_kind=a.get("index_proxy_kind"),
            inception=_parse_date(a.get("inception")),
        )
        for a in universe_raw["assets"]
    )

    sa = universe_raw["safe_asset"]
    safe_asset = SafeAsset(
        id=sa["id"],
        proxy=sa["proxy"],
        name=sa.get("name", ""),
        ter_pct=float(sa.get("ter_pct", 0.0)),
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
    raw_lookbacks = s.get("lookbacks_days")
    lookbacks: tuple[int, ...] | None = (
        tuple(int(x) for x in raw_lookbacks) if raw_lookbacks is not None else None
    )
    return Strategy(
        name=s["name"],
        asset_ids=tuple(s["assets"]),
        rebalance=s["rebalance"],
        top_n=int(s["top_n"]),
        description=s.get("description", ""),
        reference_date=_parse_date(s.get("reference_date")),
        mode=mode,
        lookbacks_days=lookbacks,
        allocation_rule=s.get("allocation_rule"),
    )


def _parse_execution(raw: dict[str, Any] | None) -> Execution:
    """Validate the execution block. Today only the canonical Friday→Monday
    cadence is supported; raise loud if a deviation is requested so the YAML
    doesn't silently lie about behaviour."""
    if raw is None:
        return Execution()
    signal = str(raw.get("signal_close", "friday")).lower()
    fill = str(raw.get("fill_close", "monday")).lower()
    if signal != "friday" or fill != "monday":
        raise ValueError(
            f"shared.execution: only signal_close=friday/fill_close=monday is "
            f"implemented, got signal_close={signal!r}, fill_close={fill!r}"
        )
    return Execution(signal_close=signal, fill_close=fill)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
