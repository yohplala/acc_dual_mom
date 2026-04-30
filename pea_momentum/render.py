"""HTML report rendering: plotly charts + Jinja2 template."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .backtest import BacktestResult
from .metrics import compute, drawdown_series
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
    metrics_rows = [_metrics_row(r) for r in results]

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
    if result.rebalances:
        last = result.rebalances[-1]
        holdings = ", ".join(
            f"{a} {int(round(w * 100))}%"
            for a, w in sorted(last.weights.items(), key=lambda kv: -kv[1])
        )
        return {
            "name": result.strategy_name,
            "cadence": cadence_map.get(strategy.rebalance, strategy.rebalance),
            "last_rebalance": last.rebalance_date.isoformat(),
            "holdings_str": holdings,
            "turnover": last.turnover,
        }
    return {
        "name": result.strategy_name,
        "cadence": cadence_map.get(strategy.rebalance, strategy.rebalance),
        "last_rebalance": "—",
        "holdings_str": "—",
        "turnover": 0.0,
    }


def _metrics_row(result: BacktestResult) -> dict[str, object]:
    m = compute(result.equity)
    return {"name": result.strategy_name, **m.to_dict()}


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
