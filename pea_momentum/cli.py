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
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import polars as pl

from . import backtest, correlations, discover, fetch, render, store
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
    default="2012-01-01",
    show_default=True,
    help="ISO date for backtest start. Default 2012-01-01 — earliest date "
    "where every strategy in strategies.yaml has usable data (gated by "
    "EEMA's 2012 inception which proxies em_asia).",
)
@click.option("--end", default=None, help="ISO date for backtest end")
@click.pass_context
def cmd_backtest(ctx: click.Context, start: str | None, end: str | None) -> None:
    """Run every configured strategy and persist equity curves + rebalance logs."""
    cfg: Config = ctx.obj["config"]
    data_root: Path = ctx.obj["data_root"]
    prices = store.read_prices(data_root)
    if prices.is_empty():
        raise click.ClickException("No prices found; run `pea-mom fetch` first.")

    results_root = data_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None

    summary: list[dict[str, object]] = []
    for strategy in cfg.strategies:
        click.echo(f"running {strategy.name} ({strategy.rebalance})…")
        result = backtest.run(prices, strategy, cfg, start=start_d, end=end_d)
        store.write_history(result.equity, data_root, strategy_name=strategy.name)
        rebal_path = results_root / f"{strategy.name}.rebalances.json"
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

    (results_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    click.echo(f"OK · {len(summary)} strategies · results in {results_root}")


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
    results_root = data_root / "results"
    if not results_root.exists():
        raise click.ClickException("No backtest results found; run `pea-mom backtest` first.")

    results = []
    for strategy in cfg.strategies:
        equity = store.read_history(data_root, strategy.name)
        if equity is None or equity.is_empty():
            continue
        rebal_path = results_root / f"{strategy.name}.rebalances.json"
        rebals = _load_rebalances(rebal_path) if rebal_path.exists() else []
        results.append(
            backtest.BacktestResult(
                strategy_name=strategy.name,
                equity=equity,
                rebalances=rebals,
            )
        )

    if not results:
        raise click.ClickException("No equity histories to render.")

    prices = store.read_prices(data_root)
    out = render.render(
        results, cfg, site_root, prices_long=prices if not prices.is_empty() else None
    )
    click.echo(f"OK · wrote {out}")


@cli.command(name="run")
@click.option("--site-root", default=DEFAULT_SITE_ROOT, show_default=True)
@click.option("--start", default=None)
@click.pass_context
def cmd_run(ctx: click.Context, site_root: str, start: str | None) -> None:
    """Full pipeline: fetch → backtest → render."""
    ctx.invoke(cmd_fetch, start=start)
    ctx.invoke(cmd_backtest, start=None, end=None)
    ctx.invoke(cmd_render, site_root=site_root)


def _load_rebalances(path: Path) -> list[backtest.Rebalance]:
    """Strict deserializer for `<strategy>.rebalances.json` artifacts.

    Every field is written unconditionally by `backtest.rebalances_to_json`,
    so a missing key indicates a corrupted artifact rather than a legacy
    schema. We index strictly so corruption surfaces as a `KeyError`.
    """
    raw = json.loads(path.read_text())
    return [
        backtest.Rebalance(
            rebalance_date=date.fromisoformat(r["rebalance_date"]),
            signal_date=date.fromisoformat(r["signal_date"]),
            fill_date=date.fromisoformat(r["fill_date"]),
            scores=r["scores"],
            weights=r["weights"],
            turnover=float(r["turnover"]),
            cost=float(r["cost"]),
        )
        for r in raw
    ]


DISCOVERY_PRICES_FILE = "discover.parquet"


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
        date.fromisoformat(start) if start else (datetime.utcnow().date() - timedelta(days=365 * 3))
    )
    prices = discover.fetch_discovery_universe(entries, start=start_d)

    out_path = data_root / DISCOVERY_PRICES_FILE
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
    default=0.85,
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
    prices_path = data_root / DISCOVERY_PRICES_FILE
    if not prices_path.exists():
        raise click.ClickException(
            f"No discovery prices at {prices_path}; run `pea-mom discover` first."
        )
    prices = pl.read_parquet(prices_path)

    asset_ids = prices.get_column("asset_id").unique().to_list()
    cm = correlations.compute_correlation_matrix(prices, asset_ids, window_days=window_days)
    grouped = correlations.find_groups(cm, threshold=threshold)
    ter_pct_by_id = {e.id: e.ter_pct for e in entries}
    reps = [
        correlations.best_in_group(g, prices, ter_pct_by_id, window_days=window_days)
        for g in grouped
    ]

    diagnostics = correlations.diagnose_strategies(cfg, entries, reps)

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
