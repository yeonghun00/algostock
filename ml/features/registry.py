"""Feature group base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureGroup(ABC):
    """Base class for feature groups. Subclass + @register to add features.

    phase=1: runs before universe filters and external merges (price-based features)
    phase=2: runs after universe filters and external merges (needs sector/market data)

    columns:       columns exposed as model features (included in FEATURE_COLUMNS).
    produced_cols: ALL columns written by compute(), including intermediates that are
                   NOT model features (e.g. ret_1d, mom_Nd).  Used by resolve_order()
                   so downstream groups can declare dependencies on these intermediates.
                   Defaults to columns when not overridden.
    """

    name: str = ""
    columns: list[str] = []
    produced_cols: list[str] = []   # override to declare intermediate outputs
    dependencies: list[str] = []
    phase: int = 2  # default: runs after all merges

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to df. Mutate in-place is OK."""
        ...


_FEATURE_GROUPS: list[type[FeatureGroup]] = []


def register(cls: type[FeatureGroup]) -> type[FeatureGroup]:
    """Decorator: @register on a FeatureGroup subclass adds it to the global registry."""
    _FEATURE_GROUPS.append(cls)
    return cls


def get_all_groups() -> list[type[FeatureGroup]]:
    """Return all registered feature groups."""
    return list(_FEATURE_GROUPS)


def get_all_feature_columns() -> list[str]:
    """Return flat list of all feature columns from all registered groups."""
    cols: list[str] = []
    for group_cls in _FEATURE_GROUPS:
        cols.extend(group_cls.columns)
    return cols


def get_feature_group_map() -> dict[str, str]:
    """Return mapping of feature column name → group name."""
    mapping: dict[str, str] = {}
    for group_cls in _FEATURE_GROUPS:
        for col in group_cls.columns:
            mapping[col] = group_cls.name
    return mapping


def resolve_order(groups: list[type[FeatureGroup]]) -> list[type[FeatureGroup]]:
    """Topological sort: run groups whose dependencies are satisfied first.

    Groups with no dependencies come first. Groups that depend on columns
    produced by earlier groups come later.
    """
    remaining = list(groups)
    ordered: list[type[FeatureGroup]] = []
    available_cols: set[str] = set()

    max_iterations = len(remaining) * len(remaining) + 1
    iteration = 0
    while remaining:
        iteration += 1
        if iteration > max_iterations:
            # Break cycle: just append remaining in original order
            ordered.extend(remaining)
            break
        progress = False
        next_remaining = []
        for g in remaining:
            if all(dep in available_cols for dep in g.dependencies):
                ordered.append(g)
                # Track both model features (columns) and intermediate outputs
                # (produced_cols) so downstream dependency declarations work.
                available_cols.update(g.columns)
                available_cols.update(getattr(g, "produced_cols", []))
                progress = True
            else:
                next_remaining.append(g)
        remaining = next_remaining
        if not progress:
            # No group can be resolved; append all remaining
            ordered.extend(remaining)
            break

    return ordered
