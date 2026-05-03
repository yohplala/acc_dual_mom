# Methodology

This document describes how strategies are scored, allocated, costed, and stitched in the `pea_momentum` framework. It is the canonical reference for "what is the dashboard actually showing me". Everything here is a deliberate choice — change behaviour by editing [`strategies.yaml`](../strategies.yaml) and the asset YAML metadata, never the source code.

## Framework label

The framework is **rank-only momentum rotation — no positive-momentum filter, no absolute-momentum filter**. It is *inspired by* Antonacci's accelerated dual momentum (ADM) and his GEM (Global Equities Momentum), but it is **not strict Antonacci**: there's no two-stage filter against a benchmark and no `score > 0` floor. Every asset (including the safe sleeve, when listed) competes purely on rank. References to "ADM" / "DM" elsewhere in the doc and dashboard label the *score recipe* (e.g. accelerated 1m/3m/6m mean vs single 12-month lookback), not any filter design.

## Strategy modes

Two execution modes coexist:

- **`rotation`** — the default. At each rebalance, score every listed asset, pick the top-N highest scores, and weight them. The cadence (`weekly_sunday`, `biweekly_sunday`, `monthly_first_sunday`, `quarterly_first_sunday`, `semiannual_first_sunday`) drives when this fires.
- **`buy_and_hold`** — equal-weight (or explicit `static_weights`) on `assets`, allocated on day one and never rebalanced. No transaction cost. Used as zero-cost reference benchmarks (`world_bh`, `sp500_bh`, `wld3_noeu_uh_bh`, `world_60_40_bh`). Buy-and-hold ignores the cadence field.

### Rotation pipeline (per rebalance day)

```
raw close prices ─▶ score (per asset, including safe if listed)
  ──▶ top-N selection (by score, descending — sign-agnostic)
  ──▶ weighting rule (equal_weight or score_proportional)
  ──▶ largest-remainder rounding to granularity_pct
  ──▶ rounding shortfall ─▶ residual_holder
       (= safe asset if listed in `assets:`, else CASH 0%-return placeholder)
```

### Scoring (`pea_momentum/score.py`)

Score at signal date `t`:

```
score(asset) = aggregate_over_lookbacks(close(t) / close(t - L) - 1)
```

- **Lookbacks** are configured in trading days via the `lookbacks_days` field. The default accelerated set is `[21, 63, 126]` (≈ 1, 3, 6 months). A per-strategy override `lookbacks_days: [252]` produces the single-12-month variant; `[126]` produces the single-6-month variant.
- **Aggregations**: `mean` (default), `median` (robust to a single anomalous lookback), `min` (pessimistic — every lookback must be positive). The latter two are documented sensitivity exhibits, not the recommended live setup.

### Selection: rank-only top-N (no filter)

The top-N highest-scoring assets are picked **regardless of the sign of their scores**. Even in a synchronised bear regime where every score is negative, the strategy still allocates to the least-negative top-N — "hold the leader" rather than "go to cash via filter".

Defensive cash behaviour comes from **ranking**, not filtering. List `cash_estr` (or any low-volatility yielding sleeve) in a strategy's `assets:` and its small €STR-positive score will naturally win the top-N rank when every equity score collapses below zero, producing a 100%-cash defensive allocation organically. Strategies that don't list `cash_estr` forgo this defence and stay equity-allocated through bears (holding the least-bad equity sleeve).

This was a deliberate methodology choice: the previous version had a `score > 0` filter that dumped 100% to the residual holder when nothing passed. Removing the filter means the strategy always commits to a top-N allocation — defensive behaviour is opt-in via the universe composition, not enforced by the engine. Strategies that *want* cash defence are explicit about it (`cash_estr` in `assets:`); strategies that don't get full pass-through to whichever equity is least-bad in stress.

### Allocation rules (`pea_momentum/allocate.py`)

Two rules are supported:

- **`equal_weight`** (the framework default): `1/N` across the selected top-N assets. This is the canonical Antonacci-style weighting and the recommended live setup. Robust to negative scores.
- **`score_proportional`**: |score|-proportional weights with **rank-based attribution** — the highest-scoring asset always gets the largest weight, the lowest-scoring the smallest. For all-positive selections this collapses to the canonical `w_i = s_i / Σ s_j` formula (because score-rank == |score|-rank). For mixed-sign or all-negative selections (reachable now that the top-N has no filter), the formula stays well-defined: no negative weights (shorts), no inverted ranks. Concretely with selected scores `[-0.05, -0.10, -0.20]`, the pure formula `s/Σs = [0.143, 0.286, 0.571]` would give the *worst* score the largest weight; we instead assign the |score|-magnitudes by rank → `[0.571, 0.286, 0.143]` so the *least-crashing* asset gets the highest conviction. Kept as an opt-in sensitivity exhibit (`core_sp_m_t3`).

After raw weights are computed, they are rounded via **largest-remainder (Hare quota)** to integer multiples of `granularity_pct` (10% by default). Largest-remainder is exact-summing by construction, so when there are selected candidates the rounded weights total 100% — no residual.

The **residual holder** catches both rounding shortfalls (rare under largest-remainder) and the early-history "no scores yet at this rebalance" path (when no asset has enough lookback to produce a signal):
- the **safe asset** if it's listed in the strategy's `assets:` (residual earns €STR yield), OR
- a **`cash`** placeholder (0%-return, no yield) when the strategy has no yielding cash sleeve listed.

The "list safe explicitly to get the defensive crash behaviour" pattern is the YAML-honesty corollary: what you list is what you get.

## Stitching (`pea_momentum/stitching.py`, `pea_momentum/fetch.py`)

Most PEA-eligible UCITS ETFs in the universe launched between 2014 and 2024, so a 2008-onward backtest needs pre-launch history. The splice approach:

1. Fetch the live ETF series (post-inception only).
2. Fetch the configured proxy (a longer-history total-return index ETF in the same family — e.g. `SPY` for S&P 500, `EWJ` for unhedged Japan, `EXX1.DE` for STOXX Europe 600 Banks). Two YAML shapes are supported:
   - **Single proxy** via `index_proxy` + `index_proxy_kind` (`eur_tr` or `usd_tr`).
   - **Multi-stage chain** via `index_proxy_chain` — see the next subsection.
3. If the proxy is USD-denominated (`index_proxy_kind: usd_tr`), convert to EUR via daily `EURUSD=X`.
4. **Rescale the proxy levels** so the proxy's level on the inception date matches the ETF's level on that date. Pre-inception levels are walked back from there using the proxy's own returns.
5. Concatenate: scaled proxy for `date < inception`, ETF for `date >= inception` (the post side filters to `date >= inception` so any pre-inception synthetic NAV rows yfinance might return for renamed share classes are dropped — see the C50.PA back-history bug). The `source` column on `prices.parquet` records provenance (`yfinance` vs `stitched_index_proxy`).

The result is continuous in level (no jump on the splice date) and continuous in return.

### Multi-stage proxy chain

Some assets benefit from chaining: a methodologically-pure proxy with a shorter history (cleanest), then a less-clean one with longer history that extends only the pre-handoff segment. `index_proxy_chain` is an ordered list of `{ticker, kind}` pairs, **cleanest first**. Each successor is level-rescaled to match its predecessor at the predecessor's first available date. The chain extends history only as much as the dirtier proxies can — without contaminating the clean recent segment.

Active examples in [`pea_universe.yaml`](../pea_universe.yaml):

- `world` (DCAM.PA): `IWDA.AS` (cleanest, EUR Acc UCITS since 2009-09-25) → `IWRD.L` (same fund, USD Dist share class, +9 months back to 2009-01-02) → `ACWI` (broader MSCI ACWI, ~12% EM contamination, extends to 2008-03-28 capturing the Lehman peak).
- `em_asia` (PAASI.PA): `EEMA` (pure MSCI EM Asia since 2012-02-09) → `AAXJ` (Asia ex-Japan, ~10% developed-Asia drift, extends to 2008-08-15).
- `eurostoxx50` (C50.PA): `CSX5.AS` (iShares Core EUR Acc since 2014-04-28, low TER same as live) → `MSE.PA` (Amundi II share class, higher TER, extends to 2008-01-02).

### Caveat: DCAM.PA

Amundi PEA Monde (`DCAM.PA`, `world` in the universe) launched 2025-03-04. The entire pre-2025 backtest history for the `world` sleeve is therefore the spliced proxy chain (`IWDA.AS` from 2009-09-25 onwards, then `IWRD.L`, then `ACWI` for the deepest pre-2009 segment). Methodologically the closest ones in each window — but the equity curve before 2025-03-04 reflects the proxies' tracking, not DCAM's. Use of the actual DCAM tracker on PEA accounts may differ marginally.

### Bad-data scrubbing

yfinance has been observed to return spurious "round-trip" closes (price drops 50%, recovers next day, sometimes over multiple days). These are detected at two layers:

1. **Splice-time scrub** (`stitching._scrub_round_trip_spikes`) on each individual `[date, close]` series before concatenation. Catches single-day spikes plus the FX-channel spikes that propagate across every cross-currency proxy.
2. **Backtest-time scrub** (`stitching.scrub_long_format`) on the full long-format `prices.parquet`. Catches multi-day round-trip patterns up to 3 days, plus sustained "flat run" forward-fill artefacts of length ≥ 3 trading days.

Both layers null the corrupt close so the daily-return chain shows 0% on the bad day, then forward-fill replaces it with the previous valid close. Genuine non-round-trip outliers above the 30% plausibility ceiling raise `FetchError` for human investigation.

## Cost model (`pea_momentum/backtest.py`)

Per-rebalance cost is computed **per-asset**, not as a flat rate on aggregate turnover:

```
cost_at_fill = Σ_i |delta_w_i| * (per_trade_pct/100 + est_spread_bps_i / 2 / 10000)
```

Charged as a negative return on the fill day.

- `per_trade_pct` (shared, currently `0.05%` one-way) covers the broker commission (Boursorama / Interactive Brokers tier).
- `est_spread_bps` per asset is the **estimated round-trip bid-ask spread**. Half of it is added per traded notional (the strategy buys at ask and sells at bid).

Currently configured on the active assets (see [`pea_universe.yaml`](../pea_universe.yaml)):

| Asset             | Spread (bps) | Why                                      |
|-------------------|--------------|------------------------------------------|
| `sp500` (PSP5.PA)        | 5     | High-AUM flagship, tight book            |
| `eurostoxx50` (C50.PA)   | 5     | Eurozone large-cap flagship              |
| `world` (DCAM.PA)        | 10    | Mid sleeve, recent launch                |
| `eu_banks` (BNK.PA)      | 20    | Sector ETF, mid liquidity                |
| `em_asia` (PAASI.PA)     | 25    | EM exposure, thinner book                |
| `russell2000` (RS2K.PA)  | 30    | Satellite, small-cap                     |
| `topix` (PTPXE.PA)       | 30    | Satellite, low AUM                       |

Spread numbers are derived by hand from Boursorama and Amundi factsheets; refresh them when the universe materially shifts.

### Why estimate spreads in YAML rather than fetch live?

`yfinance`'s `Ticker.info["bid"]/["ask"]` is intraday-only and frequently null for thinly-traded EU tickers. Boursorama scraping is not API-stable. A static per-asset estimate is the only honest backtesting approach — refresh sporadically as the live spreads shift.

## Execution timing

All rotation strategies share the same Sunday-anchored cadence:

- **Rebalance day R** is a Sunday (the rotation cadence determines which ones).
- **Signal date S = R - 2d**: the preceding Friday's close is what scoring sees.
- **Fill date F = R + 1d**: the following Monday's close is the assumed execution price. Costs are charged on F as a negative return; new weights take effect from F+1.

This guarantees no look-ahead and matches the realistic case for a French PEA holder reviewing weekly signals over the weekend.

### Cross-cadence start alignment

Every cadence rebalances **unconditionally on the first Sunday of the backtest range**, even when that Sunday isn't a calendar-anchor day for the cadence's natural rule (e.g. `quarterly_first_sunday` with a backtest starting mid-Feb still fires its first rebalance on the first Sunday of February, then resumes the calendar-quarter cadence — Apr / Jul / Oct first Sundays). Every subsequent rebalance follows the cadence's own rule. The result: regardless of whether you compare a `weekly_sunday`, `quarterly_first_sunday`, or `semiannual_first_sunday` strategy, all of them have their first rebalance on the same day-1 Sunday and start trading from the same allocation date — making cross-cadence equity curves directly comparable from t=0.

**Buy-and-hold benchmarks share the same alignment**: a `buy_and_hold` strategy's equity curve is also slid forward to the first Sunday of the backtest range (same day rotations fire), so static benchmarks plot apples-to-apples against the rotation lineup on the dashboard. Falls back to the latest-listed-sleeve's first available date if some held asset has no data on that Sunday — alignment is achievable only when the data permits.

## Performance metrics (`pea_momentum/metrics.py`)

Computed on the daily equity curve plus the rebalance log:

| Metric             | Definition                                                                 |
|--------------------|----------------------------------------------------------------------------|
| `cagr`             | `(equity[-1] / equity[0]) ** (1/years) - 1`                                |
| `vol_ann`          | Daily-return std × √252                                                    |
| `sharpe`           | `mean_daily_return / vol_daily × √252`. **rf=0**: see note below.          |
| `sortino`          | Same but only counts downside daily returns in the volatility denominator. |
| `max_drawdown`     | Worst peak-to-trough decline                                               |
| `calmar`           | `cagr / |max_drawdown|`                                                    |
| `hit_rate`         | Fraction of trading days with positive return — correlated with vol more than skill |
| `rebalance_hit_rate` | Fraction of rebalance intervals where equity at the next rebalance is above this one |
| `turnover_per_year` | Annualised L1 turnover (sum of per-rebalance turnovers / years)           |
| `total_cost`       | Sum of per-rebalance costs over the run                                    |
| `avg_corr`         | Average pairwise daily-return correlation across the strategy's risky universe |

### `rf=0` in Sharpe / Sortino

The framework already absorbs the risk-free rate at the strategy level: a strategy that lists `safe` in its `assets:` will rotate into it during crashes (€STR yield wins the rank when equity scores collapse), and one that doesn't list `safe` falls through to a 0%-return CASH placeholder. Subtracting `rf` again at the metric layer would double-count the safe-asset return for safe-listing strategies. The dashboard's Sharpe is therefore directly comparable across rotation strategies with similar `safe` treatment.

### Per-rebalance hit rate

Per-day hit rate is near-meaningless for a slow rotation strategy — sample size is dominated by quiet drift days. The per-rebalance hit rate ("did the equity at the next rebalance exceed the equity at this one?") is the meaningful skill measure: sample size = number of rebalances, and that's exactly the bet the strategy is making each cycle.

## Discovery & redundancy

The broad PEA-eligible Amundi universe in [`pea_universe.yaml`](../pea_universe.yaml) is fetched independently and used for correlation-matrix exploration. Two diagnostics are produced:

- **Complete-link clustering** (`correlations.find_groups`) groups assets where *every* member-to-member daily correlation exceeds the threshold (default 0.90). This avoids the chaining property of single-link clustering, where two weakly correlated assets get lumped together via a high-correlation bridge pair.
- **Best-in-group representative** (`correlations.best_in_group`) picks the **lowest-TER** member of each group. TER is decided ex-ante, so the choice is immune to in-sample selection bias — unlike a 1-year CAGR ranking, which would be momentum-on-momentum and defeat the prospective spirit of dual momentum.

`diagnostics.diagnose_strategies` then cross-references each strategy's asset slice against these groups and emits two diagnostic types: **`remove`** (the strategy uses 2+ assets in the same redundancy group — keep one, drop the others) and **`replace`** (the strategy uses an asset that isn't the lowest-TER pick in its group).

## Glossary

For future readers: the canonical vocabulary used across YAML, code, and dashboard. If a term doesn't appear here, it's not part of the framework's API surface.

| Concept | YAML | Code field | Dashboard label | Notes |
|---|---|---|---|---|
| Score lookback windows (trading days) | `lookbacks_days: [21, 63, 126]` | `Strategy.lookbacks_days`, `Scoring.lookbacks_days` | `Scoring` (e.g. `ADM 1m/3m/6m mean`, `DM 12m`) | Days, not months. 21 ≈ 1 month. |
| Score aggregation across lookbacks | `aggregation: mean` | `Scoring.aggregation` | (folded into Scoring label) | `mean` / `median` / `min` |
| Allocation weighting rule | `rule: equal_weight` (shared) or `allocation_rule: score_proportional` (per strategy) | `Allocation.rule`, `Strategy.allocation_rule` | `Allocation` (e.g. `top-3 equal`, `top-1 score-prop`) | `equal_weight` is canonical; `score_proportional` is the opt-in sensitivity exhibit. |
| Number of selected sleeves | `top_n: 3` | `Strategy.top_n` | (folded into Allocation label) | |
| Rebalance cadence | `rebalance: monthly_first_sunday` | `Strategy.rebalance` | `Cadence` (e.g. `monthly`, `weekly`, `biweekly`, `quarterly`, `semiannual`) | Sunday-anchored only. |
| Strategy mode | `mode: rotation` (default) or `mode: buy_and_hold` | `Strategy.mode` | Reflected in `Scoring` label (`Buy & Hold`) | |
| Static buy-and-hold weights | `static_weights: {world: 0.6, safe: 0.4}` | `Strategy.static_weights` | (folded into `B&H static` label) | Optional; absent ⇒ equal-weight. |
| Pre-selection filter | (removed) | (removed) | — | The historical `positive_momentum` (`score > 0`) and `absolute_momentum` (vs safe-score) filters are both gone — selection is rank-only. |
| Safe asset | `universe.safe_asset:` block | `Config.safe_asset`, `SafeAsset` | `safe` chip in the **Cash** column | Just another listable asset under the rank-only methodology — list it in a strategy's `assets:` to give it a chance to win the top-N. |
| Cash residual | (n/a) | `allocate.CASH_KEY` | `cash` chip | 0%-return placeholder for rounding-shortfall residuals (rare under largest-remainder) and the early-history "no scores yet at this rebalance" path when `safe` isn't listed. |
| Per-asset bid-ask spread (bps) | `est_spread_bps: 5` | `Asset.est_spread_bps` | (folded into `total_cost`) | Half-spread added per traded notional. |
| Asset region (geographic bucket) | `region: us` | `Asset.region` | Drives the **World / US / Europe / Asia** signal-table columns. | Coarse buckets: `world` / `us` / `europe` (incl. eurozone/france/germany) / `asia` (incl. japan/em_asia/em). Safe ⇒ `cash` column. |

## Pointers

- [`strategies.yaml`](../strategies.yaml) — universe, shared scoring/allocation/cost knobs, per-strategy entries
- [`pea_universe.yaml`](../pea_universe.yaml) — broad discovery catalogue
- [`pea_momentum/score.py`](../pea_momentum/score.py)
- [`pea_momentum/allocate.py`](../pea_momentum/allocate.py)
- [`pea_momentum/backtest.py`](../pea_momentum/backtest.py)
- [`pea_momentum/stitching.py`](../pea_momentum/stitching.py)
- [`pea_momentum/metrics.py`](../pea_momentum/metrics.py)
- [`pea_momentum/correlations.py`](../pea_momentum/correlations.py)
