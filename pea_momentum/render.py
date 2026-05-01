"""HTML report rendering: plotly charts + Jinja2 template."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .backtest import BacktestResult
from .metrics import avg_pairwise_correlation, compute, drawdown_series
from .universe import Config

TEMPLATES_DIR = Path(__file__).parent / "templates"

PLOT_LAYOUT_BASE = {
    "paper_bgcolor": "#161b22",
    "plot_bgcolor": "#161b22",
    "font": {"color": "#e6edf3", "family": "system-ui"},
    "margin": {"t": 24, "r": 16, "b": 40, "l": 56},
    "xaxis": {"gridcolor": "#30363d", "zerolinecolor": "#30363d"},
    "yaxis": {"gridcolor": "#30363d", "zerolinecolor": "#30363d"},
    "legend": {"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center"},
    "hovermode": "x unified",
}

PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#79c0ff"]


def render(
    results: list[BacktestResult],
    config: Config,
    output_dir: str | Path,
    prices_long: pl.DataFrame | None = None,
    template_name: str = "index.html.j2",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_name)

    summary = _summary(results, config)
    signals = [_signal_row(r, config) for r in results]
    metrics_rows = [_metrics_row(r, config, prices_long) for r in results]

    equity_traces, equity_layout = _equity_figure(results)
    drawdown_traces, drawdown_layout = _drawdown_figure(results)

    rendered = template.render(
        summary=summary,
        signals=signals,
        metrics_rows=metrics_rows,
        equity_traces=json.dumps(equity_traces),
        equity_layout=json.dumps(equity_layout),
        drawdown_traces=json.dumps(drawdown_traces),
        drawdown_layout=json.dumps(drawdown_layout),
    )

    out_file = output_path / "index.html"
    out_file.write_text(rendered)
    return out_file


def _summary(results: list[BacktestResult], config: Config) -> dict[str, object]:
    all_dates = [d for r in results for d in r.equity.get_column("date").to_list()]
    return {
        "n_strategies": len(results),
        "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "data_start": min(all_dates).isoformat() if all_dates else "—",
        "data_end": max(all_dates).isoformat() if all_dates else "—",
        "cost_bps": int(config.shared.costs.per_trade_pct * 100),
    }


def _signal_row(result: BacktestResult, config: Config) -> dict[str, object]:
    cadence_map = {
        "weekly_sunday": "weekly",
        "biweekly_sunday": "biweekly",
        "monthly_first_sunday": "monthly",
    }
    strategy = next(s for s in config.strategies if s.name == result.strategy_name)
    asset_rows = _asset_rows(strategy, config)

    last = result.rebalances[-1] if result.rebalances else None
    prev = result.rebalances[-2] if len(result.rebalances) >= 2 else None

    return {
        "name": result.strategy_name,
        "cadence": cadence_map.get(strategy.rebalance, strategy.rebalance),
        "last_rebalance": last.rebalance_date.isoformat() if last else "—",
        "previous_rows": _chip_rows(asset_rows, prev.weights if prev else {}),
        "current_rows": _chip_rows(asset_rows, last.weights if last else {}),
    }


# Region groupings, ordered top-to-bottom in the holdings cell. Assets that
# don't fit a known region get appended below the recognized rows.
REGION_ORDER = (
    "us",
    "world",
    "europe",
    "eurozone",
    "france",
    "germany",
    "japan",
    "em_asia",
    "em",
)


def _asset_rows(strategy: Any, config: Config) -> list[list[Any]]:
    """Layout the strategy's universe as rows of assets.

    The chip layout only contains assets that the user explicitly listed
    under `strategy.assets` in the YAML — no implicit additions. If a
    strategy needs `safe` shown alongside its risky sleeves, it has to be
    listed there explicitly.

    If `universe.display_layout` is set, that list-of-lists drives the row
    structure (each row filtered to the strategy's `asset_ids`, empty rows
    collapse). Otherwise we group by `region` in REGION_ORDER.
    """
    if config.display_layout is not None:
        strategy_ids = set(strategy.asset_ids)
        rows: list[list[Any]] = []
        for row_ids in config.display_layout:
            row_assets: list[Any] = []
            for aid in row_ids:
                if aid not in strategy_ids:
                    continue
                row_assets.append(
                    config.safe_asset if aid == config.safe_asset.id else config.asset_by_id(aid)
                )
            if row_assets:
                rows.append(row_assets)
        return rows

    by_region: dict[str, list[Any]] = {}
    for asset_id in strategy.asset_ids:
        a = config.asset_by_id(asset_id)
        by_region.setdefault(a.region, []).append(a)
    rows = []
    for region in REGION_ORDER:
        if region in by_region:
            rows.append(by_region.pop(region))
    for assets in by_region.values():  # any unknown regions
        rows.append(assets)
    return rows


def _chip_rows(
    asset_rows: list[list[Any]], weights: dict[str, float]
) -> list[list[dict[str, object]]]:
    """Convert each row of assets into a row of chip descriptors,
    each carrying the asset id, weight pct, and color bracket."""
    out: list[list[dict[str, object]]] = []
    for row in asset_rows:
        chips: list[dict[str, object]] = []
        for asset in row:
            w = weights.get(asset.id, 0.0)
            pct = round(w * 100)
            chips.append(
                {
                    "id": asset.id,
                    "pct": pct,
                    "bracket": _weight_bracket(pct),
                }
            )
        out.append(chips)
    return out


def _weight_bracket(pct: int) -> str:
    if pct == 0:
        return "zero"
    if pct < 25:
        return "low"
    if pct < 50:
        return "mid"
    return "high"


def _metrics_row(
    result: BacktestResult,
    config: Config,
    prices_long: pl.DataFrame | None,
) -> dict[str, object]:
    m = compute(result.equity)
    # Sum of per-rebalance cost values (each = turnover_l1 * per_trade_pct).
    # In the same dimensionless unit as the equity multiple: `Final` plus
    # `total_cost` is roughly the no-cost final equity.
    total_cost = sum(r.cost for r in result.rebalances)
    # Average pairwise correlation of the strategy's risky universe (excludes
    # safe asset since the absolute filter already gates on it). Lower = more
    # decorrelated = better diversification potential.
    strategy = next(s for s in config.strategies if s.name == result.strategy_name)
    avg_corr = (
        avg_pairwise_correlation(prices_long, list(strategy.asset_ids))
        if prices_long is not None
        else None
    )
    return {
        "name": result.strategy_name,
        **m.to_dict(),
        "total_cost": total_cost,
        "avg_corr": avg_corr,
    }


def _equity_figure(results: list[BacktestResult]) -> tuple[list[dict], dict]:
    traces: list[dict] = []
    for i, r in enumerate(results):
        if r.equity.is_empty():
            continue
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": r.strategy_name,
                "x": [d.isoformat() for d in r.equity.get_column("date").to_list()],
                "y": r.equity.get_column("equity").to_list(),
                "line": {"width": 2, "color": PALETTE[i % len(PALETTE)]},
            }
        )
    layout = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = {**PLOT_LAYOUT_BASE["yaxis"], "title": "Equity (multiple)", "type": "log"}
    layout["xaxis"] = {**PLOT_LAYOUT_BASE["xaxis"], "title": ""}
    return traces, layout


def _drawdown_figure(results: list[BacktestResult]) -> tuple[list[dict], dict]:
    traces: list[dict] = []
    for i, r in enumerate(results):
        if r.equity.is_empty():
            continue
        dd = drawdown_series(r.equity)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": r.strategy_name,
                "x": [d.isoformat() for d in dd.get_column("date").to_list()],
                "y": [v * 100 for v in dd.get_column("drawdown").to_list()],
                "line": {"width": 1.5, "color": PALETTE[i % len(PALETTE)]},
                "fill": "tozeroy" if i == 0 else "none",
                "fillcolor": "rgba(248, 81, 73, 0.1)" if i == 0 else None,
            }
        )
    layout = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = {**PLOT_LAYOUT_BASE["yaxis"], "title": "Drawdown (%)"}
    return traces, layout
