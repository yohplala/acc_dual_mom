"""Tests for the strategies.yaml + pea_universe.yaml config loader."""

from __future__ import annotations

from pathlib import Path

from pea_momentum.universe import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_amundi_url_resolved_for_active_assets() -> None:
    """Every Asset in strategies.yaml whose ISIN appears in pea_universe.yaml
    must have its amundi_url populated by load_config."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", REPO_ROOT / "pea_universe.yaml")
    for asset in cfg.assets:
        assert asset.amundi_url is not None, f"{asset.id} has no amundi_url"
        assert asset.amundi_url.startswith("https://www.amundietf.fr/"), asset.amundi_url


def test_safe_asset_amundi_url_resolved() -> None:
    """Safe asset's ISIN is in pea_universe.yaml, so its URL must resolve."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", REPO_ROOT / "pea_universe.yaml")
    assert cfg.safe_asset.amundi_url is not None
    assert cfg.safe_asset.amundi_url.startswith("https://www.amundietf.fr/")


def test_load_config_without_discovery_path() -> None:
    """Passing discovery_path=None disables the cross-reference; assets get
    amundi_url=None. The framework must still load and run without it."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", discovery_path=None)
    for asset in cfg.assets:
        assert asset.amundi_url is None
    assert cfg.safe_asset.amundi_url is None


def test_load_config_with_missing_discovery_file_falls_back_silently() -> None:
    """A typo'd or absent discovery_path shouldn't crash config load — the
    URL cross-reference is best-effort."""
    cfg = load_config(REPO_ROOT / "strategies.yaml", discovery_path="/nonexistent.yaml")
    assert all(a.amundi_url is None for a in cfg.assets)
