"""Loaders for the strategies + universe config.

Two YAML files combine into a single `Config` at load time:
- `pea_universe.yaml` — the asset catalog. Every Amundi PEA-eligible UCITS
  ETF has one entry (currently ~107). For assets actually used by some
  strategy, optional active-only fields (`region`, `est_spread_bps`,
  `index_proxy`, `inception`, `replication`, `synth_proxy`) are populated.
- `strategies.yaml` — `shared:` config + `strategies:` list. Each strategy
  references catalog ids in its `assets:`.

The "safe asset" stops being a separate config block — it's a regular
`Asset` whose `synth_proxy: estr` field signals the fetcher to compound
ECB €STR fixings instead of pulling yfinance.

`Config.assets` carries the active subset (i.e. only catalog entries
referenced by at least one strategy). The full catalog is available via
`load_full_universe(path)` for discovery / correlation flows.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    name: str = ""
    currency: str = "EUR"
    # Annual TER as a percentage (e.g. 0.12 for 12 bps). Cost diagnostics
    # only — backtest closes use yfinance auto_adjust=True, already net of
    # fund fees.
    ter_pct: float = 0.0
    # SFDR classification (`Article 6` / `Article 8` / `Article 9`).
    sfdr: str = ""
    # Free-form geographic / asset-class tag from the discovery YAML
    # (e.g. `USA`, `Eurozone-Small`, `Cash-Eurozone`). Single source of
    # truth for region grouping: drives the correlation-page perimeter
    # (via `discover.coarse_region`) and the signal-table dashboard
    # bucket (via `discover.dashboard_bucket`). No separate `region`
    # field — both consumers derive from `category`.
    category: str = ""
    leveraged: bool = False
    amundi_url: str | None = None
    # ── Active-asset fields (only set on assets used by some strategy) ──
    # Estimated round-trip bid-ask spread in bps. Half is added per traded
    # notional in the backtest cost model.
    est_spread_bps: float = 0.0
    # "synthetic" (swap-based) or "physical" (sampling/full replication).
    # Informational.
    replication: str | None = None
    # ETF inception date — splice point between proxy and live ETF history.
    inception: date | None = None
    # Optional Yahoo ticker for the underlying index, used to extend price
    # history backward before the ETF launched. None disables stitching.
    index_proxy: str | None = None
    # FX-conversion path for the proxy series:
    #   "eur_tr"  EUR-quoted total-return index (no FX needed)
    #   "usd_tr"  USD-quoted total-return index, converted via EURUSD=X
    index_proxy_kind: str | None = None
    # Multi-stage proxy chain: an ordered tuple of (ticker, kind) pairs,
    # CLEANEST first (latest start, best methodological match), DIRTIEST
    # last (earliest start, more composition / FX drift). Each successor
    # in the chain extends history pre-handoff by being level-rescaled
    # to match its predecessor at the predecessor's first available
    # date. When set, supersedes `index_proxy` / `index_proxy_kind` and
    # the splice-at-inception algorithm is fed the assembled chain as
    # a single EUR-denominated series.
    index_proxy_chain: tuple[tuple[str, str], ...] | None = None
    # Synthetic-price flag: when set, fetch.py compounds the named source
    # instead of pulling yfinance for `yahoo`. Currently `estr` is the only
    # supported value (compounds ECB €STR fixings, with EONIA splice for
    # pre-2019 history). The asset becomes the "safe sleeve" — competes in
    # rank-only top-N selection like any other asset.
    synth_proxy: str | None = None


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
    # `rotation` runs the score → filter → top-N → weight allocation.
    # `buy_and_hold` allocates equal weights (or `static_weights` if set)
    # to `asset_ids` on day one and never rebalances.
    mode: str = "rotation"
    # Optional per-strategy scoring lookback override.
    lookbacks_days: tuple[int, ...] | None = None
    # Per-strategy override of `shared.allocation.rule`.
    allocation_rule: str | None = None
    # Optional explicit weights (asset_id → weight) for buy_and_hold mode.
    static_weights: tuple[tuple[str, float], ...] | None = None
    # True when this Strategy was synthesised at config-load time from a
    # catalog asset (vs declared in strategies.yaml). Auto-generated
    # entries are routed onto regional pages only — never the main
    # dashboard, which is reserved for the user's curated lineup.
    auto_generated: bool = False
    # Optional user-facing label (legend, tables, chip text). When None,
    # consumers fall back to `name`. Auto-generated strategies set this
    # to the bare asset id so the regional pages display "em_asia"
    # instead of the internal "asia/em_asia" namespaced name.
    display_name: str | None = None

    def effective_scoring(self, shared_scoring: Scoring) -> Scoring:
        if self.lookbacks_days is None:
            return shared_scoring
        return Scoring(
            lookbacks_days=self.lookbacks_days,
            aggregation=shared_scoring.aggregation,
        )

    def effective_allocation_rule(self, shared_rule: str) -> str:
        return self.allocation_rule or shared_rule

    @property
    def label(self) -> str:
        """User-facing label — `display_name` when set, else `name`."""
        return self.display_name or self.name


@dataclass(frozen=True, slots=True)
class Config:
    shared: Shared
    # Active assets only: those referenced by `strategies[*].asset_ids`.
    # The full catalog is in `pea_universe.yaml`; load it via
    # `load_full_universe()` when you need every entry (discovery flows).
    assets: tuple[Asset, ...]
    strategies: tuple[Strategy, ...]

    def asset_by_id(self, asset_id: str) -> Asset:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        raise KeyError(f"Unknown asset id: {asset_id}")

    @property
    def safe_asset(self) -> Asset | None:
        """The asset with `synth_proxy: estr` set (or `None` if none).
        Convenience for callers that previously used `Config.safe_asset`."""
        for a in self.assets:
            if a.synth_proxy == "estr":
                return a
        return None

    @property
    def safe_asset_id(self) -> str | None:
        sa = self.safe_asset
        return sa.id if sa else None


VALID_MODES: frozenset[str] = frozenset({"rotation", "buy_and_hold"})


def load_config(
    path: str | Path = "strategies.yaml",
    universe_path: str | Path = "pea_universe.yaml",
) -> Config:
    """Load `strategies.yaml` and join it with `pea_universe.yaml` by id.

    The active-asset subset is the union of every `assets:` list across
    all strategies (plus any id used in `static_weights`). Each id must
    resolve to an entry in `pea_universe.yaml`; unknown ids raise a
    clear error at config-load time.
    """
    catalog = load_full_universe(universe_path)
    catalog_by_id = {a.id: a for a in catalog}

    raw = yaml.safe_load(Path(path).read_text())
    return _parse(raw, catalog_by_id)


def load_full_universe(path: str | Path = "pea_universe.yaml") -> tuple[Asset, ...]:
    """Load every entry in the discovery YAML as an `Asset`. Active-only
    fields are populated where the YAML has them; otherwise default. Each
    entry's `amundi_url` is resolved via the slug heuristic if not
    explicitly set in the YAML."""
    # Lazy import to avoid universe → discover → fetch → universe cycle.
    from .discover import amundi_product_url

    raw = yaml.safe_load(Path(path).read_text())
    bare = [_asset_from_yaml(e) for e in raw["universe"]]
    # Fill amundi_url where not explicit — the slug derivation needs the
    # parsed Asset, so do it as a second pass.
    return tuple(a if a.amundi_url else _replace_amundi_url(a, amundi_product_url(a)) for a in bare)


def _asset_from_yaml(e: dict[str, Any]) -> Asset:
    raw_chain = e.get("index_proxy_chain")
    chain: tuple[tuple[str, str], ...] | None = None
    if raw_chain is not None:
        chain = tuple((str(x["ticker"]), str(x["kind"])) for x in raw_chain)
    return Asset(
        id=e["id"],
        isin=e["isin"],
        yahoo=e.get("yahoo") or "",
        name=e.get("name", ""),
        currency=e.get("currency", "EUR"),
        ter_pct=float(e.get("ter_pct", 0.0)),
        sfdr=e.get("sfdr", ""),
        category=e.get("category", ""),
        leveraged=bool(e.get("leveraged", False)),
        amundi_url=e.get("amundi_url"),
        est_spread_bps=float(e.get("est_spread_bps", 0.0)),
        replication=e.get("replication"),
        inception=_parse_date(e.get("inception")),
        index_proxy=e.get("index_proxy"),
        index_proxy_kind=e.get("index_proxy_kind"),
        index_proxy_chain=chain,
        synth_proxy=e.get("synth_proxy"),
    )


def _replace_amundi_url(a: Asset, url: str) -> Asset:
    """Frozen-dataclass-friendly clone with `amundi_url` set."""
    from dataclasses import replace as _dc_replace

    return _dc_replace(a, amundi_url=url)


def _parse(raw: dict[str, Any], catalog_by_id: dict[str, Asset]) -> Config:
    shared_raw = raw["shared"]
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
        costs=Costs(per_trade_pct=float(shared_raw["costs"]["per_trade_pct"])),
        execution=execution,
    )

    manual_strategies = tuple(_parse_strategy(s, catalog_by_id) for s in raw["strategies"])

    # Auto-generated single-asset B&H strategies for the regional pages.
    # The user populates strategies.yaml with cross-region / curated entries
    # for the main dashboard; the regional pages auto-generate one B&H
    # per eligible catalog asset (yahoo-tickered, single-region category,
    # not leveraged, with `_acc` preferred over `_dist` siblings).
    auto_strategies = auto_bh_strategies(catalog_by_id.values())
    manual_names = {s.name for s in manual_strategies}
    collisions = manual_names & {a.name for a in auto_strategies}
    if collisions:
        # Shouldn't happen — auto names are namespaced as `<region>/<id>`
        # and manual names follow the user's free convention. Loud-fail
        # if a user happens to write `asia/em_asia` manually so the
        # collision is visible.
        raise ValueError(
            f"Auto-generated strategy names collide with manual entries: {sorted(collisions)}. "
            f"Auto names use the `<region>/<asset_id>` namespace — pick a different "
            f"name for the manual entry."
        )
    strategies = manual_strategies + auto_strategies

    # Active asset set: union of strategy assets + static_weights keys.
    active_ids: set[str] = set()
    for s in strategies:
        active_ids.update(s.asset_ids)
        if s.static_weights is not None:
            active_ids.update(k for k, _ in s.static_weights)
    missing = active_ids - catalog_by_id.keys()
    if missing:
        raise ValueError(
            f"Strategies reference unknown asset ids (not in pea_universe.yaml): "
            f"{sorted(missing)}"
        )
    assets = tuple(catalog_by_id[i] for i in sorted(active_ids))

    return Config(
        shared=shared,
        assets=assets,
        strategies=strategies,
    )


def auto_bh_strategies(catalog: Iterable[Asset]) -> tuple[Strategy, ...]:
    """Generate one buy-and-hold `Strategy` per eligible catalog asset.

    Eligibility (filters compounded):
    - Has a `yahoo` ticker (so a price series can be fetched)
    - Not leveraged (B&H comparison is between unlevered building blocks)
    - Maps to a region bucket via `discover.dashboard_bucket(category)` —
      i.e. `us` / `europe` / `asia` (world bucket is handled separately
      via the regional-page reference line, not auto-gen).
    - Not in a category that dashboard_bucket misroutes (Emerging-LatAm
      and Emerging-EMEA both match the "EMERGING" prefix and would land
      on the asia page despite not being Asia exposure).
    - Not a Thematic-* category (off-topic for a regional ETF
      comparison — the asia page is for regional benchmarks, not
      single-theme bets).
    - Not a `_dist` share-class twin of an `_acc` sibling — duplicate
      curves muddy the regional-page comparison.

    Naming:
    - Internal `name`: `<region>/<asset_id>` (e.g. `asia/em_asia`).
      Namespaced so it can never collide with a manual `strategies.yaml`
      entry, and persisted to `data/history/<region>/<asset_id>.parquet`
      for tidy filesystem grouping.
    - User-facing `display_name`: `<asset_id>` only (e.g. `em_asia`).
      Drives the chart legend, metrics-table row, and chip text on the
      regional page.

    The `auto_generated=True` flag is what render.py uses to keep these
    off the main dashboard.
    """
    # Lazy import: avoid the universe → discover → fetch → universe cycle.
    from .discover import dashboard_bucket

    by_id = {a.id: a for a in catalog}
    out: list[Strategy] = []
    for asset in by_id.values():
        if not asset.yahoo:
            continue
        if asset.leveraged:
            continue
        if asset.category in _AUTO_BH_SKIP_CATEGORIES:
            continue
        if any(asset.category.startswith(p) for p in _AUTO_BH_SKIP_PREFIXES):
            continue
        if _is_dist_share_class_with_acc_twin(asset, by_id):
            continue
        bucket = dashboard_bucket(asset.category)
        # Only auto-route into the three regional pages we render. World
        # auto-B&H (e.g. `world` / `world_pea_monde`) skipped — they live
        # on the main page only as multi-asset strategies and serve as
        # the cross-page reference line via render._REGIONAL_REFERENCE_STRATEGIES.
        if bucket not in ("us", "europe", "asia"):
            continue
        out.append(
            Strategy(
                name=f"{bucket}/{asset.id}",
                asset_ids=(asset.id,),
                rebalance="monthly_first_sunday",  # ignored by buy_and_hold mode
                top_n=1,
                description=f"100% {asset.id} buy-and-hold (auto-generated for {bucket} page)",
                mode="buy_and_hold",
                auto_generated=True,
                display_name=asset.id,
            )
        )
    return tuple(sorted(out, key=lambda s: s.name))


# Categories `dashboard_bucket` lumps into a region that don't actually
# belong on that regional page. Emerging-LatAm and Emerging-EMEA both
# match the "EMERGING" prefix and otherwise land on the asia page.
_AUTO_BH_SKIP_CATEGORIES: frozenset[str] = frozenset(
    {
        "Emerging-LatAm",
        "Emerging-EMEA",
    }
)

# Category prefixes that don't fit a regional B&H comparison page.
# Thematic ETFs (Thematic-Water, Thematic-Climate, etc.) are
# single-theme bets, not regional benchmarks — the regional pages
# stay focused on broad-market and sector candidates.
_AUTO_BH_SKIP_PREFIXES: tuple[str, ...] = ("Thematic",)


def _is_dist_share_class_with_acc_twin(asset: Asset, by_id: dict[str, Asset]) -> bool:
    """True iff `asset.id` ends in `_dist` AND `<id-without-_dist>_acc`
    exists in the catalog. Heuristic dedup so the regional pages don't
    show two essentially-identical curves for the same fund's Acc and
    Dist share classes (the Acc twin is preferred — it absorbs the
    distributions and is what most PEA holders use)."""
    if not asset.id.endswith("_dist"):
        return False
    acc_twin_id = asset.id[: -len("_dist")] + "_acc"
    return acc_twin_id in by_id


def _parse_strategy(s: dict[str, Any], catalog_by_id: dict[str, Asset]) -> Strategy:
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
    raw_weights = s.get("static_weights")
    static_weights: tuple[tuple[str, float], ...] | None = None
    if raw_weights is not None:
        items = tuple((str(k), float(v)) for k, v in raw_weights.items())
        total = sum(w for _, w in items)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"strategy {s['name']!r}: static_weights must sum to 1.0 (got {total:.6f})"
            )
        # `static_weights` keys must match the strategy's `assets:` exactly.
        # Two places of truth otherwise (and the dashboard had to merge them
        # to render — see the recent world_60_40_bh fix). The strategy's
        # `assets:` IS the universe; for buy-and-hold with explicit weights,
        # every weighted asset must appear in `assets:`.
        weight_ids = {k for k, _ in items}
        asset_ids = set(s["assets"])
        if weight_ids != asset_ids:
            missing_in_assets = weight_ids - asset_ids
            extra_in_assets = asset_ids - weight_ids
            details: list[str] = []
            if missing_in_assets:
                details.append(f"static_weights keys not in assets: {sorted(missing_in_assets)}")
            if extra_in_assets:
                details.append(f"assets entries not in static_weights: {sorted(extra_in_assets)}")
            raise ValueError(
                f"strategy {s['name']!r}: static_weights keys must match assets: list "
                f"exactly. " + "; ".join(details)
            )
        static_weights = items
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
        static_weights=static_weights,
    )


def _parse_execution(raw: dict[str, Any] | None) -> Execution:
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
