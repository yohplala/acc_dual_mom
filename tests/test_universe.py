"""Tests for the strategies.yaml + pea_universe.yaml config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pea_momentum.universe import (
    Asset,
    auto_bh_strategies,
    load_config,
    load_full_universe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_full_universe_returns_every_catalog_entry() -> None:
    catalog = load_full_universe(REPO_ROOT / "pea_universe.yaml")
    # Catalog has ~107 entries; lower bound 100 to remain robust as it grows.
    assert len(catalog) >= 100


def test_load_config_joins_strategies_with_universe() -> None:
    """Active assets (those referenced by some strategy) are populated from
    pea_universe.yaml — name, isin, ter_pct, amundi_url, category, etc. all
    come from there."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", REPO_ROOT / "pea_universe.yaml")
    assert len(cfg.assets) > 0
    for asset in cfg.assets:
        assert asset.name, f"{asset.id} has no name from catalog"
        assert asset.isin, f"{asset.id} has no isin"
        assert asset.amundi_url is not None, f"{asset.id} has no amundi_url"
        assert asset.category, f"{asset.id} has no category (drives dashboard bucket)"


def test_safe_asset_property_finds_synth_proxy_asset() -> None:
    """The safe asset is identified by `synth_proxy: estr` in
    pea_universe.yaml. `Config.safe_asset` is a convenience property that
    returns it (or None if no asset is so marked)."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", REPO_ROOT / "pea_universe.yaml")
    safe = cfg.safe_asset
    assert safe is not None
    assert safe.synth_proxy == "estr"
    assert cfg.safe_asset_id == safe.id


def test_unknown_asset_id_in_strategy_raises() -> None:
    """If a strategy references an asset id that isn't in pea_universe.yaml,
    config load fails loud rather than silently dropping the asset."""
    bad_strategies = REPO_ROOT / "strategies.yaml"
    # We can't easily produce a synthetic strategies.yaml here without
    # writing a tmp file. Reuse the real one but override pea_universe.yaml
    # with a sparse file missing every entry.
    sparse_universe = REPO_ROOT / "tests" / "_fixtures" / "_sparse_universe.yaml"
    sparse_universe.parent.mkdir(parents=True, exist_ok=True)
    sparse_universe.write_text("universe: []\n")
    try:
        with pytest.raises(ValueError, match="unknown asset ids"):
            load_config(bad_strategies, sparse_universe)
    finally:
        sparse_universe.unlink(missing_ok=True)
        if sparse_universe.parent.exists() and not any(sparse_universe.parent.iterdir()):
            sparse_universe.parent.rmdir()


# ── static_weights / assets: consistency validation ──────────────────────


_MINIMAL_UNIVERSE_YAML = """\
universe:
  - { id: a, name: A, isin: A1, currency: EUR, ter_pct: 0.1, sfdr: Article 6, category: USA, yahoo: A.PA }
  - { id: b, name: B, isin: B1, currency: EUR, ter_pct: 0.1, sfdr: Article 6, category: World, yahoo: B.PA }
  - { id: c, name: C, isin: C1, currency: EUR, ter_pct: 0.1, sfdr: Article 6, category: Cash-Eurozone, yahoo: C.PA, synth_proxy: estr }
"""


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_static_weights_keys_must_match_assets_exactly(tmp_path: Path) -> None:
    """static_weights keys must match the strategy's `assets:` list — no
    extra weights, no extra assets — so the visible universe and the
    weighted positions stay aligned."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    # Case 1: static_weights references an asset NOT in `assets:`
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  filter: { type: positive_momentum }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: bad
    assets: [a]
    rebalance: monthly_first_sunday
    top_n: 1
    mode: buy_and_hold
    static_weights: { a: 0.6, b: 0.4 }
""",
    )
    with pytest.raises(ValueError, match="static_weights keys not in assets"):
        load_config(strategies, universe)


def test_static_weights_assets_subset_also_raises(tmp_path: Path) -> None:
    """The reverse direction: an asset listed in `assets:` but NOT given a
    weight in `static_weights:` is also rejected — no implicit zero."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  filter: { type: positive_momentum }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: bad
    assets: [a, b]
    rebalance: monthly_first_sunday
    top_n: 1
    mode: buy_and_hold
    static_weights: { a: 1.0 }
""",
    )
    with pytest.raises(ValueError, match="assets entries not in static_weights"):
        load_config(strategies, universe)


def test_regional_weights_requires_top_1_per_region(tmp_path: Path) -> None:
    """`regional_weights` only makes sense paired with the per-region
    selection rule — without it, the fixed split has no asset slot to
    map onto. Loud-fail at config-load time rather than letting
    ambiguous configs slip through."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: bad
    assets: [a]
    rebalance: monthly_first_sunday
    top_n: 1
    regional_weights: { us: 0.6, europe: 0.1, asia: 0.3 }
""",
    )
    with pytest.raises(ValueError, match="regional_weights requires"):
        load_config(strategies, universe)


def test_regional_weights_must_sum_to_one(tmp_path: Path) -> None:
    """Sum-to-1 invariant — weights are interpreted as final portfolio
    fractions, so anything else would silently re-scale or leave a gap
    routed to the residual holder. Reject up front."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: bad
    assets: [a]
    rebalance: monthly_first_sunday
    top_n: 1
    selection_rule: top_1_per_region
    regional_weights: { us: 0.5, europe: 0.1, asia: 0.3 }
""",
    )
    with pytest.raises(ValueError, match="must sum to 1"):
        load_config(strategies, universe)


def test_regional_weights_unknown_region_raises(tmp_path: Path) -> None:
    """Only us/europe/asia are valid keys — `world` and free-form
    strings are rejected. Mirrors the regional-page lineup."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: bad
    assets: [a]
    rebalance: monthly_first_sunday
    top_n: 1
    selection_rule: top_1_per_region
    regional_weights: { us: 0.6, world: 0.1, asia: 0.3 }
""",
    )
    with pytest.raises(ValueError, match="unknown regions"):
        load_config(strategies, universe)


def test_regional_weights_valid_config_loads(tmp_path: Path) -> None:
    """Happy path — valid 60/10/30 split with selection_rule set is
    accepted and persisted on the parsed Strategy."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: ok
    assets: [a]
    rebalance: monthly_first_sunday
    top_n: 3
    selection_rule: top_1_per_region
    regional_weights: { us: 0.6, europe: 0.1, asia: 0.3 }
""",
    )
    cfg = load_config(strategies, universe)
    manual = next(s for s in cfg.strategies if s.name == "ok")
    assert manual.regional_weights is not None
    assert dict(manual.regional_weights) == {"us": 0.6, "europe": 0.1, "asia": 0.3}


def test_static_weights_matching_assets_is_valid(tmp_path: Path) -> None:
    """The good case: keys exactly match `assets:` (set equality). Loads
    without raising."""
    universe = tmp_path / "universe.yaml"
    _write_yaml(universe, _MINIMAL_UNIVERSE_YAML)
    strategies = tmp_path / "strategies.yaml"
    _write_yaml(
        strategies,
        """\
shared:
  scoring: { lookbacks_days: [21], aggregation: mean }
  allocation: { rule: equal_weight, granularity_pct: 10, rounding: largest_remainder }
  filter: { type: positive_momentum }
  costs: { per_trade_pct: 0.0 }
strategies:
  - name: ok
    assets: [a, b]
    rebalance: monthly_first_sunday
    top_n: 1
    mode: buy_and_hold
    static_weights: { a: 0.6, b: 0.4 }
""",
    )
    cfg = load_config(strategies, universe)
    assert cfg.strategies[0].static_weights == (("a", 0.6), ("b", 0.4))


# ── auto-generated B&H per catalog asset ─────────────────────────────


def _asset(
    asset_id: str,
    *,
    yahoo: str = "X.PA",
    category: str = "USA",
    leveraged: bool = False,
    synth: str | None = None,
) -> Asset:
    return Asset(
        id=asset_id,
        isin=f"ISIN_{asset_id}",
        yahoo=yahoo,
        category=category,
        leveraged=leveraged,
        synth_proxy=synth,
    )


class TestAutoBHStrategies:
    def test_yields_one_bh_per_eligible_asset_with_namespaced_name(self) -> None:
        catalog = [
            _asset("sp500", category="USA"),
            _asset("eurostoxx50", category="Eurozone"),
            _asset("em_asia", category="Emerging-Asia"),
        ]
        out = auto_bh_strategies(catalog)
        names = [s.name for s in out]
        assert names == ["asia/em_asia", "europe/eurostoxx50", "us/sp500"]
        # Display name is the bare asset id (no namespace, no _bh suffix).
        display = {s.name: s.display_name for s in out}
        assert display["asia/em_asia"] == "em_asia"
        assert display["us/sp500"] == "sp500"
        assert display["europe/eurostoxx50"] == "eurostoxx50"
        # Every auto-gen entry is buy_and_hold + auto_generated=True.
        for s in out:
            assert s.mode == "buy_and_hold"
            assert s.auto_generated is True
            assert s.asset_ids == (s.display_name,)

    def test_skips_assets_without_yahoo_ticker(self) -> None:
        catalog = [
            _asset("sp500", yahoo="SP500.AS", category="USA"),
            _asset("em_emea", yahoo="", category="Emerging-EMEA"),  # no ticker
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/sp500"]

    def test_skips_leveraged_assets(self) -> None:
        catalog = [
            _asset("sp500", category="USA"),
            _asset("dax_2x", category="Leveraged-Germany", leveraged=True),
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/sp500"]

    def test_skips_thematic_categories(self) -> None:
        catalog = [
            _asset("sp500", category="USA"),
            _asset("thm_water", category="Thematic-Water"),
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/sp500"]

    def test_skips_emerging_latam_and_emea(self) -> None:
        """LatAm and EMEA both match the dashboard_bucket "EMERGING" prefix
        but they're not Asia exposures — explicitly skipped."""
        catalog = [
            _asset("em_asia", category="Emerging-Asia"),
            _asset("em_latam", category="Emerging-LatAm"),
            _asset("em_emea", category="Emerging-EMEA"),
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["asia/em_asia"]

    def test_dedups_dist_share_class_when_acc_twin_exists(self) -> None:
        """When both `<base>_acc` and `<base>_dist` exist, the `_dist`
        sibling is dropped — they track the same fund and produce
        near-identical equity curves on the regional page."""
        catalog = [
            _asset("ftse_mib_acc", category="Italy"),
            _asset("ftse_mib_dist", category="Italy"),
            _asset("dax_acc", category="Germany"),  # no dist twin
        ]
        out = auto_bh_strategies(catalog)
        names = [s.name for s in out]
        assert "europe/ftse_mib_acc" in names
        assert "europe/ftse_mib_dist" not in names
        assert "europe/dax_acc" in names

    def test_keeps_dist_when_no_acc_twin(self) -> None:
        catalog = [_asset("solo_dist", category="USA")]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/solo_dist"]

    def test_world_bucket_assets_excluded(self) -> None:
        """World-bucket assets aren't auto-generated — they live on the
        main page only as part of the curated strategies."""
        catalog = [
            _asset("world", category="World"),
            _asset("sp500", category="USA"),
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/sp500"]

    def test_safe_asset_excluded(self) -> None:
        """The synthetic safe sleeve has a Cash-* category; not auto-gen-able."""
        catalog = [
            _asset("cash_estr", yahoo="", category="Cash-Eurozone", synth="estr"),
            _asset("sp500", category="USA"),
        ]
        out = auto_bh_strategies(catalog)
        assert [s.name for s in out] == ["us/sp500"]
