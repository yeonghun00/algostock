"""Momentum features."""

from __future__ import annotations

import pandas as pd

from .registry import FeatureGroup, register


@register
class MomentumFeatures(FeatureGroup):
    name = "momentum"
    # Raw momentum columns are intermediates — excluded from model features
    # (sector-neutral versions sector_zscore_mom_* are used instead).
    # produced_cols lists what compute() actually writes so that downstream
    # groups (VolumeFeatures, SectorFeatures, etc.) can declare dependencies
    # on ret_1d / mom_Nd and have resolve_order() satisfy them correctly.
    columns = []
    produced_cols = ["ret_1d", "mom_5d", "mom_21d", "mom_63d", "mom_126d"]
    dependencies = ["closing_price"]
    phase = 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use adj_closing_price so splits mid-window don't create fake momentum.
        # Fallback to raw closing_price if adj table hasn't been built yet.
        p = "adj_closing_price" if "adj_closing_price" in df.columns else "closing_price"
        g = df.groupby("stock_code")
        df["ret_1d"]   = g[p].pct_change()
        df["mom_5d"]   = g[p].pct_change(5)
        df["mom_21d"]  = g[p].pct_change(21)
        df["mom_63d"]  = g[p].pct_change(63)
        df["mom_126d"] = g[p].pct_change(126)
        return df
