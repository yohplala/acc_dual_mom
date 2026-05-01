"""HTML report rendering: plotly charts + Jinja2 templates.

Two pages produced:
- `index.html`            strategy dashboard (equity / metrics / drawdown)
- `correlations.html`     PEA-universe correlation matrix + redundancy groups
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .backtest import BacktestResult
from .correlations import CorrelationMatrix, GroupRepresentative
from .metrics import avg_pairwise_correlation, compute, drawdown_series
from .universe import Config

TEMPLATES_DIR = Path(__file__).parent / "templates"

PLOT_LAYOUT_BASE: dict[str, Any] = {
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

_ENV = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(
    results: list[BacktestResult],
    config: Config,
    output_dir: str | Path,
    prices_long: pl.DataFrame | None = None,
    template_name: str = "index.html.j2",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    template = _ENV.get_template(template_name)

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
    if strategy.mode == "buy_and_hold":
        # Buy-and-hold never rebalances after the initial allocation. The
        # "Previous allocation" chips would otherwise all show 0% / grey,
        # which is misleading — the static weights ARE the previous (and
        # current, and forever) allocation. Mirror current onto previous.
        prev = last
    else:
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

    The chip layout shows whatever the user listed under `strategy.assets`
    in YAML — plus, for rotation strategies, the safe asset as a final row
    (because rotation can hold safe when no candidate beats the absolute
    filter). Buy-and-hold strategies never rotate, so the safe row is
    omitted there.

    If `universe.display_layout` is set, that list-of-lists drives the row
    structure (each row filtered to the strategy's `asset_ids` plus safe
    for rotation; empty rows collapse). Otherwise we group by `region` in
    REGION_ORDER.
    """
    include_safe = strategy.mode == "rotation"

    if config.display_layout is not None:
        strategy_ids = set(strategy.asset_ids)
        if include_safe:
            strategy_ids.add(config.safe_asset.id)
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
    if include_safe:
        rows.append([config.safe_asset])
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


def _equity_figure(
    results: list[BacktestResult],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
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
    layout: dict[str, Any] = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = {**PLOT_LAYOUT_BASE["yaxis"], "title": "Equity (multiple)", "type": "log"}
    layout["xaxis"] = {**PLOT_LAYOUT_BASE["xaxis"], "title": ""}
    return traces, layout


def _drawdown_figure(
    results: list[BacktestResult],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
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
    layout: dict[str, Any] = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = {**PLOT_LAYOUT_BASE["yaxis"], "title": "Drawdown (%)"}
    return traces, layout


# ─── Correlation matrix page ──────────────────────────────────────────


def render_correlations(
    cm: CorrelationMatrix,
    groups: list[GroupRepresentative],
    threshold: float,
    output_dir: str | Path,
    diagnostics: list[Any] | None = None,
    entries: list[Any] | None = None,
    template_name: str = "correlations.html.j2",
) -> Path:
    """Render the PEA-universe correlation matrix to `correlations.html`.

    `cm`           the n-by-n matrix with asset_ids in heatmap order
    `groups`       redundancy groups (each with a representative pick)
    `threshold`    the correlation cutoff used to form groups (for display)
    `diagnostics`  optional list of `StrategyDiagnostic` rows to render in
                   the per-strategy "remove / replace" recommendations table
    `entries`      optional list of `DiscoveryEntry` — used to render each
                   asset id in the redundancy-groups table as a link to
                   its Amundi product page
    """
    from .discover import amundi_product_url

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    template = _ENV.get_template(template_name)

    summary = {
        "n_assets": len(cm.asset_ids),
        "window_days": cm.window_days,
        "threshold": threshold,
        "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

    heatmap_traces, heatmap_layout = _heatmap_figure(cm)

    url_by_id: dict[str, str] = {e.id: amundi_product_url(e) for e in entries} if entries else {}

    groups_view = [
        {
            "group": list(g.group),
            "representative": g.representative,
            "representative_score": g.representative_score,
        }
        for g in groups
    ]

    diagnostics_view = [
        {
            "strategy_name": d.strategy_name,
            "issue": d.issue,
            "detail": d.detail,
            "suggestion": d.suggestion,
        }
        for d in (diagnostics or [])
    ]

    rendered = template.render(
        summary=summary,
        groups=groups_view,
        diagnostics=diagnostics_view,
        url_by_id=url_by_id,
        heatmap_traces=json.dumps(heatmap_traces),
        heatmap_layout=json.dumps(heatmap_layout),
    )

    out_file = output_path / "correlations.html"
    out_file.write_text(rendered)
    return out_file


def _heatmap_figure(cm: CorrelationMatrix) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Triangular Plotly heatmap. Upper triangle masked with NaN so only the
    lower triangle (incl. diagonal) renders."""
    if not cm.asset_ids:
        return [], dict(PLOT_LAYOUT_BASE)

    z = cm.matrix.copy()
    mask = np.triu(np.ones_like(z, dtype=bool), k=1)
    z[mask] = np.nan

    trace = {
        "type": "heatmap",
        "x": cm.asset_ids,
        "y": cm.asset_ids,
        "z": [[None if math_isnan(v) else float(v) for v in row] for row in z],
        "zmin": -1,
        "zmax": 1,
        "colorscale": [
            [0.0, "#58a6ff"],  # blue: -1
            [0.5, "#161b22"],  # dark: 0
            [0.85, "#d29922"],  # amber: 0.7
            [1.0, "#f85149"],  # red: 1
        ],
        "hoverongaps": False,
        "hovertemplate": "%{y} vs %{x}<br>corr = %{z:.2f}<extra></extra>",
    }

    layout = dict(PLOT_LAYOUT_BASE)
    layout["xaxis"] = {
        **PLOT_LAYOUT_BASE["xaxis"],
        "tickangle": -45,
        "tickfont": {"size": 9},
    }
    layout["yaxis"] = {
        **PLOT_LAYOUT_BASE["yaxis"],
        "autorange": "reversed",
        "tickfont": {"size": 9},
    }
    layout["margin"] = {"t": 16, "r": 16, "b": 120, "l": 140}
    return [trace], layout


def math_isnan(x: object) -> bool:
    """True if x is a Python float NaN (or numpy NaN coerced to float)."""
    return isinstance(x, float) and x != x  # NaN != NaN by IEEE 754
