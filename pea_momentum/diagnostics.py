"""Strategy diagnostics — cross-reference strategies.yaml against correlation
groups built from the discovery universe.

Two diagnostic types:
- **remove**: a strategy uses 2+ assets that fall in the same correlation
  group (drop the weaker).
- **replace**: a strategy uses an asset that isn't the best-in-group rep
  (consider swapping for the representative).

Strategy assets ↔ discovery entries are matched by ISIN — `strategies.yaml`
and `pea_universe.yaml` use independent id slugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .correlations import GroupRepresentative
    from .universe import Asset, Config


@dataclass(frozen=True, slots=True)
class StrategyDiagnostic:
    strategy_name: str
    issue: str  # "replace" | "remove"
    detail: str  # human-readable summary of what's flagged
    suggestion: str


def diagnose_strategies(
    config: Config,
    discovery_entries: list[Asset],
    groups: list[GroupRepresentative],
) -> list[StrategyDiagnostic]:
    """Cross-reference active strategies against correlation groups.

    Matches strategy assets ↔ discovery assets by ISIN (the two YAMLs use
    different `id` slugs). For each strategy:
    - "remove" diagnostic if 2+ of its assets fall in the same correlation
      group → keep one, drop the others.
    - "replace" diagnostic if an asset used isn't the best-in-group rep
      → consider swapping for the representative.

    Singleton groups (no correlation pairs above threshold) are ignored —
    they're not redundancies.
    """
    isin_to_disc_id: dict[str, str] = {e.isin: e.id for e in discovery_entries}
    disc_id_to_group: dict[str, GroupRepresentative] = {}
    for g in groups:
        if len(g.group) < 2:  # ignore singletons
            continue
        for member in g.group:
            disc_id_to_group[member] = g

    safe_id = config.safe_asset_id  # None if no synth_proxy asset listed

    out: list[StrategyDiagnostic] = []
    for strategy in config.strategies:
        # strategy_id → discovery_id (only for assets matched + grouped)
        per_asset: dict[str, str] = {}
        # group_rep → list of strategy assets in that group
        groups_in_strategy: dict[str, list[str]] = {}

        for asset_id in strategy.asset_ids:
            # Skip the safe asset: it's a synthetic €STR series, not a real
            # PEA ETF, so the discovery universe doesn't carry it in any
            # correlation group — no diagnostics to emit either way.
            if asset_id == safe_id:
                continue
            asset = config.asset_by_id(asset_id)
            disc_id = isin_to_disc_id.get(asset.isin)
            if disc_id is None or disc_id not in disc_id_to_group:
                continue
            group = disc_id_to_group[disc_id]
            per_asset[asset_id] = disc_id
            groups_in_strategy.setdefault(group.representative, []).append(asset_id)

        # Issue 1: redundant pair — 2+ assets in same group
        for rep, members in groups_in_strategy.items():
            if len(members) >= 2:
                out.append(
                    StrategyDiagnostic(
                        strategy_name=strategy.name,
                        issue="remove",
                        detail=(
                            f"{', '.join(members)} all fall in the same correlation "
                            f"group (representative: {rep})"
                        ),
                        suggestion=f"keep one (suggest the rep: {rep}), drop the others",
                    )
                )

        # Issue 2: suboptimal — used asset isn't the rep, AND no redundancy
        # (if it's flagged as redundant, the "remove" diagnostic above already
        # implicitly recommends the rep)
        flagged_for_remove = {
            m for members in groups_in_strategy.values() if len(members) >= 2 for m in members
        }
        for asset_id, disc_id in per_asset.items():
            if asset_id in flagged_for_remove:
                continue
            group = disc_id_to_group[disc_id]
            if disc_id != group.representative:
                out.append(
                    StrategyDiagnostic(
                        strategy_name=strategy.name,
                        issue="replace",
                        detail=(
                            f"{asset_id} is in a correlation group whose best member "
                            f"is {group.representative} (CAGR/TER score)"
                        ),
                        suggestion=f"replace {asset_id} with {group.representative}",
                    )
                )

    return out
