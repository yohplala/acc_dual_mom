"""Command-line interface.

pea-mom fetch              # download/refresh strategy prices
pea-mom backtest           # run all strategies, persist equity + rebalances
pea-mom signal             # print the most recent rebalance per strategy
pea-mom render             # build the strategy HTML report
pea-mom run                # full pipeline: fetch → backtest → render

pea-mom discover           # fetch the broad PEA-eligible Amundi universe
pea-mom render-correlations  # build the correlation-matrix HTML page
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import click
import polars as pl

from . import backtest, correlations, discover, fetch, render, store
from .diagnostics import diagnose_strategies
from .universe import Config, load_config

DEFAULT_CONFIG = "strategies.yaml"
DEFAULT_DATA_ROOT = "data"
DEFAULT_RESULTS_ROOT = "data/results"
DEFAULT_SITE_ROOT = "site"


@click.group()
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--data-root", default=DEFAULT_DATA_ROOT, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, config: str, data_root: str, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["data_root"] = Path(data_root)


@cli.command(name="fetch")
@click.option("--start", default=None, help="ISO date (YYYY-MM-DD); default = ~12y back")
@click.pass_context
def cmd_fetch(ctx: click.Context, start: str | None) -> None:
    """Fetch ETF closes from yfinance + €STR from ECB; merge into prices.parquet."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    start_date = date.fromisoformat(start) if start else None
    new = fetch.fetch_all(cfg, start=start_date)
    merged = store.upsert_prices(new, data_root)
    n_assets = merged.get_column("asset_id").n_unique()
    click.echo(
        f"OK · {merged.height} rows · {n_assets} assets · saved to {store.prices_path(data_root)}"
    )


@cli.command(name="backtest")
@click.option(
    "--start",
    default="2008-08-15",
    show_default=True,
    help="ISO date for backtest start. Default 2008-08-15 — AAXJ's first "
    "trading day (the iShares MSCI All-Country Asia ex-Japan ETF that "
    "extends em_asia history pre-2012-02). em_asia is the binding "
    "constraint among the active universe's stitched proxies; every "
    "active strategy spans 2008-08+ at this default.",
)
@click.option("--end", default=None, help="ISO date for backtest end")
@click.pass_context
def cmd_backtest(ctx: click.Context, start: str | None, end: str | None) -> None:
    """Run every configured strategy and persist equity curves + rebalance logs."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    _run_backtests(cfg, data_root, start=start_d, end=end_d)


def _run_backtests(
    cfg: Config,
    data_root: Path,
    *,
    start: date | None,
    end: date | None,
) -> list[backtest.BacktestResult]:
    """Run every configured strategy, persist artefacts, return results in-memory."""
    prices = store.read_prices(data_root)
    if prices.is_empty():
        raise click.ClickException("No prices found; run `pea-mom fetch` first.")
    results_root = data_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    results: list[backtest.BacktestResult] = []
    summary: list[dict[str, object]] = []
    for strategy in cfg.strategies:
        click.echo(f"running {strategy.name} ({strategy.rebalance})…")
        result = backtest.run(prices, strategy, cfg, start=start, end=end)
        store.write_history(result.equity, data_root, strategy_name=strategy.name)
        rebal_path = results_root / f"{strategy.name}.rebalances.json"
        # Auto-generated strategies use namespaced names like
        # `asia/em_asia` → the rebalances JSON path becomes
        # `data/results/asia/em_asia.rebalances.json`. Ensure the
        # subdirectory exists before writing.
        rebal_path.parent.mkdir(parents=True, exist_ok=True)
        rebal_path.write_text(backtest.rebalances_to_json(result.rebalances))
        last_w = result.rebalances[-1].weights if result.rebalances else {}
        click.echo(
            f"  {strategy.name}: {len(result.rebalances)} rebalances, "
            f"{result.n_fill_skips} fill-skips, {result.n_signal_skips} signal-skips"
        )
        summary.append(
            {
                "strategy": strategy.name,
                "n_rebalances": len(result.rebalances),
                "n_fill_skips": result.n_fill_skips,
                "n_signal_skips": result.n_signal_skips,
                "final_equity": float(result.equity.get_column("equity")[-1])
                if not result.equity.is_empty()
                else 1.0,
                "last_weights": last_w,
            }
        )
        results.append(result)

    (results_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    click.echo(f"OK · {len(summary)} strategies · results in {results_root}")
    return results


@cli.command(name="signal")
@click.pass_context
def cmd_signal(ctx: click.Context) -> None:
    """Print the most recent rebalance per strategy."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    results_root = data_root / "results"
    if not (results_root / "summary.json").exists():
        raise click.ClickException("No backtest results found; run `pea-mom backtest` first.")

    for strategy in cfg.strategies:
        path = results_root / f"{strategy.name}.rebalances.json"
        if not path.exists():
            click.echo(f"{strategy.name}: no rebalances")
            continue
        data = json.loads(path.read_text())
        if not data:
            click.echo(f"{strategy.name}: no rebalances")
            continue
        last = data[-1]
        weights = ", ".join(
            f"{a}={w * 100:.0f}%" for a, w in sorted(last["weights"].items(), key=lambda kv: -kv[1])
        )
        click.echo(
            f"{strategy.name:24s}  {last['rebalance_date']}  →  {weights}"
            f"  (turnover {last['turnover'] * 100:.1f}%)"
        )


@cli.command(name="render")
@click.option("--site-root", default=DEFAULT_SITE_ROOT, show_default=True)
@click.pass_context
def cmd_render(ctx: click.Context, site_root: str) -> None:
    """Build the HTML report from persisted backtest results."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    results = _load_results(cfg, data_root)
    if not results:
        raise click.ClickException("No equity histories to render.")
    out = _render_results(cfg, data_root, results, site_root)
    click.echo(f"OK · wrote {out}")


def _load_results(cfg: Config, data_root: Path) -> list[backtest.BacktestResult]:
    """Reconstruct BacktestResults from persisted equity + rebalance artefacts."""
    results_root = data_root / "results"
    if not results_root.exists():
        raise click.ClickException("No backtest results found; run `pea-mom backtest` first.")
    results: list[backtest.BacktestResult] = []
    for strategy in cfg.strategies:
        equity = store.read_history(data_root, strategy.name)
        if equity is None or equity.is_empty():
            continue
        rebal_path = results_root / f"{strategy.name}.rebalances.json"
        rebals = (
            backtest.rebalances_from_json(rebal_path.read_text()) if rebal_path.exists() else []
        )
        results.append(
            backtest.BacktestResult(strategy_name=strategy.name, equity=equity, rebalances=rebals)
        )
    return results


def _render_results(
    cfg: Config,
    data_root: Path,
    results: list[backtest.BacktestResult],
    site_root: str,
) -> Path:
    """Render the main dashboard plus the three regional pages (US /
    Europe / Asia). World is intentionally not a separate page —
    `world_bh` is included as a reference line on each regional page
    instead. Every page shares the same fixed 5-button top nav."""
    prices = store.read_prices(data_root)
    prices_long = prices if not prices.is_empty() else None

    main_nav = render._build_nav_links(current_page="index")
    main_out = render.render(
        results,
        cfg,
        site_root,
        prices_long=prices_long,
        nav_links=main_nav,
    )
    for region in ("us", "europe", "asia"):
        out = render.render_region(results, cfg, site_root, region=region, prices_long=prices_long)
        if out is not None:
            click.echo(f"  · regional page: {out}")
    return main_out


@cli.command(name="run")
@click.option("--site-root", default=DEFAULT_SITE_ROOT, show_default=True)
@click.option("--start", default=None)
@click.pass_context
def cmd_run(ctx: click.Context, site_root: str, start: str | None) -> None:
    """Full pipeline: fetch → backtest → render. Backtest results stay in-memory
    between the steps; the on-disk artefacts are still written for downstream
    consumers (signal command, future re-renders)."""
    ctx.invoke(cmd_fetch, start=start)
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    results = _run_backtests(cfg, data_root, start=None, end=None)
    if not results:
        raise click.ClickException("No backtest results to render.")
    out = _render_results(cfg, data_root, results, site_root)
    click.echo(f"OK · wrote {out}")


@cli.command(name="discover")
@click.option("--start", default=None, help="ISO date for fetch start; default = ~3y back")
@click.option(
    "--universe",
    default="pea_universe.yaml",
    show_default=True,
    help="Path to the discovery universe YAML",
)
@click.pass_context
def cmd_discover(ctx: click.Context, start: str | None, universe: str) -> None:
    """Fetch close prices for every ticker-confirmed entry in the broad
    PEA-eligible Amundi universe (no proxies, no FX conversion). Writes to
    `<data-root>/discover.parquet`."""
    data_root: Path = ctx.obj["data_root"]
    entries = discover.load_discovery_universe(universe)
    start_d = (
        date.fromisoformat(start) if start else (datetime.now(UTC).date() - timedelta(days=365 * 3))
    )
    prices = discover.fetch_discovery_universe(entries, start=start_d)

    out_path = data_root / store.DISCOVERY_PRICES_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if prices.is_empty():
        click.echo("WARN · discovery fetched zero rows; nothing to persist")
        return
    prices.write_parquet(out_path, compression="zstd")
    n_assets = prices.get_column("asset_id").n_unique()
    click.echo(f"OK · {prices.height} rows · {n_assets} assets · saved to {out_path}")


@cli.command(name="render-correlations")
@click.option("--site-root", default=DEFAULT_SITE_ROOT, show_default=True)
@click.option(
    "--universe",
    default="pea_universe.yaml",
    show_default=True,
    help="Path to the discovery universe YAML (used for TER values)",
)
@click.option("--window-days", default=252, show_default=True, type=int)
@click.option(
    "--threshold",
    default=0.90,
    show_default=True,
    type=float,
    help="Correlation threshold for group identification",
)
@click.pass_context
def cmd_render_correlations(
    ctx: click.Context,
    site_root: str,
    universe: str,
    window_days: int,
    threshold: float,
) -> None:
    """Render the correlation-matrix HTML page from `discover.parquet`."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    entries = discover.load_discovery_universe(universe)
    prices_path = data_root / store.DISCOVERY_PRICES_FILE
    if not prices_path.exists():
        raise click.ClickException(
            f"No discovery prices at {prices_path}; run `pea-mom discover` first."
        )
    prices = pl.read_parquet(prices_path)

    asset_ids = prices.get_column("asset_id").unique().to_list()
    cm = correlations.compute_correlation_matrix(prices, asset_ids, window_days=window_days)
    # Coarse perimeter from each entry's `category` field (strategy `region`
    # is populated only for active assets and would be empty for ~96 of the
    # 107 catalog entries — using `coarse_region(category)` covers all).
    region_by_id = {e.id: discover.coarse_region(e.category) for e in entries}
    grouped = correlations.find_groups(cm, threshold=threshold, region_by_id=region_by_id)
    ter_pct_by_id = {e.id: e.ter_pct for e in entries}
    reps = [correlations.best_in_group(g, ter_pct_by_id) for g in grouped]

    diagnostics = diagnose_strategies(cfg, entries, reps)

    out = render.render_correlations(
        cm,
        reps,
        threshold=threshold,
        output_dir=site_root,
        diagnostics=diagnostics,
        entries=entries,
    )
    click.echo(
        f"OK · wrote {out} ({len(asset_ids)} assets, {len(grouped)} groups, "
        f"{len(diagnostics)} strategy diagnostics)"
    )


if __name__ == "__main__":
    cli()
