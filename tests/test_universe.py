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
