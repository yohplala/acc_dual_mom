"""Tests for the strategies.yaml + pea_universe.yaml config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pea_momentum.universe import load_config, load_full_universe

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
