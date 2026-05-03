"""HTML report rendering: plotly charts + Jinja2 templates.

Two pages produced:
- `index.html`            strategy dashboard (equity / metrics / drawdown)
- `correlations.html`     PEA-universe correlation matrix + redundancy groups
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .backtest import BacktestResult
from .correlations import CorrelationMatrix, GroupRepresentative
from .discover import dashboard_bucket
from .metrics import (
    avg_pairwise_correlation,
    compute,
    drawdown_series,
    rebalance_hit_rate,
    turnover_per_year,
)
from .universe import Config, Strategy

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

    strategy_by_name: dict[str, Strategy] = {s.name: s for s in config.strategies}
    asset_meta = _build_asset_meta(config)

    # Default presentation order: highest CAGR first. Drives the signal
    # table, the metrics table (rendered in this same loop), and the
    # equity / drawdown chart trace order — Plotly's legend follows the
    # traces array so the legend ranks the strategies by CAGR too.
    sorted_results = sorted(results, key=_result_cagr, reverse=True)

    summary = _summary(sorted_results, config)
    signals = [_signal_row(r, config, strategy_by_name, asset_meta) for r in sorted_results]
    metrics_rows = [_metrics_row(r, strategy_by_name, prices_long) for r in sorted_results]

    equity_traces, equity_layout = _equity_figure(sorted_results, strategy_by_name)
    drawdown_traces, drawdown_layout = _drawdown_figure(sorted_results, strategy_by_name)

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


def _result_cagr(result: BacktestResult) -> float:
    """CAGR sort key used to order the dashboard. Empty equity curves sort
    last (treated as -inf so they don't displace ranked strategies)."""
    if result.equity.is_empty():
        return float("-inf")
    return float(compute(result.equity).cagr)


def _summary(results: list[BacktestResult], config: Config) -> dict[str, object]:
    all_dates = [d for r in results for d in r.equity.get_column("date").to_list()]
    return {
        "n_strategies": len(results),
        "last_update": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "data_start": min(all_dates).isoformat() if all_dates else "—",
        "data_end": max(all_dates).isoformat() if all_dates else "—",
        "cost_bps": int(config.shared.costs.per_trade_pct * 100),
    }


_CADENCE_LABELS: dict[str, str] = {
    "weekly_sunday": "weekly",
    "biweekly_sunday": "biweekly",
    "monthly_first_sunday": "monthly",
    "quarterly_first_sunday": "quarterly",
    "semiannual_first_sunday": "semiannual",
}

# Order of the 5 dashboard region columns rendered in the signal table.
# Membership rules live in `discover.dashboard_bucket(category)`.
_REGION_BUCKETS_ORDER: tuple[str, ...] = ("world", "us", "europe", "asia", "cash")


def _scoring_label(strategy: Strategy, shared_lookbacks: tuple[int, ...]) -> str:
    """Human-readable description of the strategy's scoring spec."""
    if strategy.mode == "buy_and_hold":
        return "Buy & Hold"
    lookbacks = strategy.lookbacks_days or shared_lookbacks
    if len(lookbacks) == 1:
        return f"DM {lookbacks[0] // 21}m"
    months_str = "/".join(f"{lb // 21}m" for lb in lookbacks)
    return f"ADM {months_str} mean"


def _allocation_label(strategy: Strategy, shared_rule: str) -> tuple[str, str]:
    """Two-line allocation spec: (line1, line2).

    Rotation: ('top-N', 'equal' | 'score-prop')
    Buy-and-hold: ('equal' | 'static', '')  — `B&H` is dropped from the
    label because the Scoring column already says `Buy & Hold` for these
    strategies; only the weighting kind ('equal' = implicit 1/N across
    `assets:`, 'static' = explicit `static_weights:` from YAML) is
    informative here.
    """
    if strategy.mode == "buy_and_hold":
        return ("static" if strategy.static_weights is not None else "equal", "")
    rule = strategy.allocation_rule or shared_rule
    rule_short = "equal" if rule == "equal_weight" else "score-prop"
    return (f"top-{strategy.top_n}", rule_short)


def _signal_row(
    result: BacktestResult,
    config: Config,
    strategy_by_name: dict[str, Strategy],
    asset_meta: dict[str, dict[str, str | None]],
) -> dict[str, object]:
    strategy = strategy_by_name[result.strategy_name]

    last = result.rebalances[-1] if result.rebalances else None
    if strategy.mode == "buy_and_hold":
        # Buy-and-hold never rebalances after the initial allocation. The
        # "Previous allocation" chips would otherwise all show 0% / grey,
        # which is misleading — the static weights ARE the previous (and
        # current, and forever) allocation. Mirror current onto previous.
        prev = last
    else:
        prev = result.rebalances[-2] if len(result.rebalances) >= 2 else None

    alloc_top, alloc_weight = _allocation_label(strategy, config.shared.allocation.rule)
    current_chips = _alloc_chips(last.weights if last else {}, asset_meta)
    universe_buckets = _universe_buckets(strategy, config, asset_meta)
    return {
        "name": result.strategy_name,
        "cadence": _CADENCE_LABELS.get(strategy.rebalance, strategy.rebalance),
        "scoring": _scoring_label(strategy, config.shared.scoring.lookbacks_days),
        "allocation_top": alloc_top,
        "allocation_weight": alloc_weight,
        "cagr": _result_cagr(result),
        "last_rebalance": last.rebalance_date.isoformat() if last else "—",
        "universe_buckets": universe_buckets,
        # Prev alloc ordered by region bucket (world / cash / us / europe /
        # asia) so the rows align visually with the consolidated universe
        # columns to the left.
        "previous_alloc": _ordered_alloc_chips(
            prev.weights if prev else {}, universe_buckets, asset_meta
        ),
        # Lookup map keyed by asset_id, used by the consolidated region
        # columns to overlay current allocation weight + bracket on the
        # corresponding universe chip (chip text becomes "id pct%" with
        # weight-bracket coloring; chips with no live allocation render
        # neutral).
        "current_alloc_by_id": {c["id"]: c for c in current_chips},
    }


def _build_asset_meta(config: Config) -> dict[str, dict[str, str | None]]:
    """Pre-compute id -> {name, amundi_url} for every asset including safe.
    Used to attach display name + Amundi product URL to chip descriptors so
    the signal table can render asset chips as clickable links."""
    return {a.id: {"name": a.name or a.id, "url": a.amundi_url} for a in config.assets}


def _universe_buckets(
    strategy: Strategy, config: Config, meta: dict[str, dict[str, str | None]]
) -> dict[str, list[dict[str, str | None]]]:
    """Group the strategy's universe assets into the 5 dashboard buckets
    (world / us / europe / asia / cash). Each bucket value is a list of
    `{id, name, url}` dicts; empty buckets render as `—` in the template.

    `strategy.asset_ids` is the single source of truth for the visible
    universe — config-load validation ensures any `static_weights:` keys
    are also in `assets:`, so we can iterate `asset_ids` alone."""
    buckets: dict[str, list[dict[str, str | None]]] = {b: [] for b in _REGION_BUCKETS_ORDER}
    for asset_id in strategy.asset_ids:
        info = meta.get(asset_id, {"name": asset_id, "url": None})
        a = config.asset_by_id(asset_id)
        # synth_proxy assets (the €STR cash sleeve) drop into the Cash
        # bucket regardless of `category` — chip layout follows yield
        # behaviour, not literal geography.
        bucket = "cash" if a.synth_proxy is not None else dashboard_bucket(a.category)
        buckets[bucket].append({"id": asset_id, "name": info["name"], "url": info["url"]})
    return buckets


_PREV_ALLOC_BUCKET_ORDER: tuple[str, ...] = ("world", "cash", "us", "europe", "asia")


def _ordered_alloc_chips(
    weights: dict[str, float],
    universe_buckets: dict[str, list[dict[str, str | None]]],
    meta: dict[str, dict[str, str | None]],
) -> list[dict[str, object]]:
    """Non-zero allocation chips ordered by region bucket — same display
    order the consolidated signal-table columns use (World, Cash, then
    US, Europe, Asia). Within each bucket, the order matches
    `universe_buckets[bucket]` (i.e. the YAML `assets:` order filtered
    by region).

    Any asset with a non-zero weight that doesn't belong to any of the
    strategy's universe buckets (e.g. the `cash` CASH_KEY residual when
    no synth_proxy asset is listed) is appended at the end, sorted by
    weight desc, so it still shows up.
    """
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for bucket_name in _PREV_ALLOC_BUCKET_ORDER:
        for asset_info in universe_buckets.get(bucket_name, []):
            asset_id = asset_info["id"]
            assert isinstance(asset_id, str)
            seen.add(asset_id)
            w = weights.get(asset_id, 0.0)
            pct = round(w * 100)
            if pct == 0:
                continue
            info = meta.get(asset_id, {"name": asset_id, "url": None})
            out.append(
                {
                    "id": asset_id,
                    "pct": pct,
                    "bracket": _weight_bracket(pct),
                    "name": info["name"],
                    "url": info["url"],
                }
            )
    # Off-universe non-zero weights (typically CASH_KEY residual)
    for asset_id, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if asset_id in seen:
            continue
        pct = round(w * 100)
        if pct == 0:
            continue
        info = meta.get(asset_id, {"name": asset_id, "url": None})
        out.append(
            {
                "id": asset_id,
                "pct": pct,
                "bracket": _weight_bracket(pct),
                "name": info["name"],
                "url": info["url"],
            }
        )
    return out


def _alloc_chips(
    weights: dict[str, float], meta: dict[str, dict[str, str | None]]
) -> list[dict[str, object]]:
    """Non-zero allocation chips sorted descending by weight. Used to build
    the per-asset lookup dict that overlays current allocation onto the
    consolidated universe columns; ordering is irrelevant since the dict
    is keyed by asset_id."""
    out: list[dict[str, object]] = []
    for asset_id, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        pct = round(w * 100)
        if pct == 0:
            continue
        info = meta.get(asset_id, {"name": asset_id, "url": None})
        out.append(
            {
                "id": asset_id,
                "pct": pct,
                "bracket": _weight_bracket(pct),
                "name": info["name"],
                "url": info["url"],
            }
        )
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
    strategy_by_name: dict[str, Strategy],
    prices_long: pl.DataFrame | None,
) -> dict[str, object]:
    m = compute(result.equity)
    # Sum of per-rebalance cost values (each = turnover_l1 * per_trade_pct).
    # In the same dimensionless unit as the equity multiple: `Final` plus
    # `total_cost` is roughly the no-cost final equity.
    total_cost = sum(r.cost for r in result.rebalances)
    turnovers = [r.turnover for r in result.rebalances]
    fill_dates = [r.fill_date for r in result.rebalances]
    # Average pairwise correlation of the strategy's risky universe (excludes
    # safe asset — its return profile is qualitatively different and would
    # depress the average without telling us about equity diversification).
    # Lower = more decorrelated = better diversification potential.
    strategy = strategy_by_name[result.strategy_name]
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
        "turnover_per_year": turnover_per_year(result.equity, turnovers),
        "rebalance_hit_rate": rebalance_hit_rate(result.equity, fill_dates),
    }


def _strategy_line_style(strategy: Strategy | None) -> dict[str, Any]:
    """Return Plotly `line` properties keyed by strategy family.

    All lines share the same 1px width — wider lines were hard to read
    when many strategies overlay. Family is encoded by dash pattern only:
        buy-and-hold              solid    (zero-cost reference)
        classic dual momentum     dash     (single lookback override)
        accelerated dual momentum dot      (mean of 1m/3m/6m, default)
    """
    if strategy is None or strategy.mode == "buy_and_hold":
        return {"width": 1, "dash": "solid"}
    if strategy.lookbacks_days is not None:
        return {"width": 1, "dash": "dash"}
    return {"width": 1, "dash": "dot"}


def _equity_figure(
    results: list[BacktestResult],
    strategy_by_name: dict[str, Strategy],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        if r.equity.is_empty():
            continue
        style = _strategy_line_style(strategy_by_name.get(r.strategy_name))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": r.strategy_name,
                "x": [d.isoformat() for d in r.equity.get_column("date").to_list()],
                "y": r.equity.get_column("equity").to_list(),
                "line": {**style, "color": PALETTE[i % len(PALETTE)]},
            }
        )
    layout: dict[str, Any] = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = {**PLOT_LAYOUT_BASE["yaxis"], "title": "Equity (multiple)", "type": "log"}
    layout["xaxis"] = {**PLOT_LAYOUT_BASE["xaxis"], "title": ""}
    return traces, layout


def _drawdown_figure(
    results: list[BacktestResult],
    strategy_by_name: dict[str, Strategy],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        if r.equity.is_empty():
            continue
        dd = drawdown_series(r.equity)
        style = _strategy_line_style(strategy_by_name.get(r.strategy_name))
        # Drawdown lines stay slimmer than equity-curve lines so the chart
        # doesn't get visually cluttered, but the dash pattern is preserved.
        dd_style = {"width": max(1.0, style["width"] * 0.75), "dash": style["dash"]}
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": r.strategy_name,
                "x": [d.isoformat() for d in dd.get_column("date").to_list()],
                "y": [v * 100 for v in dd.get_column("drawdown").to_list()],
                "line": {**dd_style, "color": PALETTE[i % len(PALETTE)]},
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
        "last_update": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
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
        "z": [
            [None if isinstance(v, float) and math.isnan(v) else float(v) for v in row] for row in z
        ],
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
