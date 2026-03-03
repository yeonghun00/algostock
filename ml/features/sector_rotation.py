"""Sector rotation signal features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class SectorRotationFeatures(FeatureGroup):
    name = "sector_rotation"
    columns = [
        "sector_dispersion",
        "sector_dispersion_21d",
        "sector_rotation_signal",
    ]
    dependencies = ["ret_1d", "sector_relative_momentum_21d"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # .copy() so we don't mutate the caller's DataFrame or corrupt row order
        out = df.copy()
        out["sector_dispersion"] = out.groupby(["date", "sector"])["ret_1d"].transform("std").fillna(0.0)
        # Sort by sector+date for the rolling window, then restore stock_code+date order
        out = out.sort_values(["sector", "date"])
        out["sector_dispersion_21d"] = (
            out.groupby("sector")["sector_dispersion"]
            .rolling(21, min_periods=10)
            .mean()
            .droplevel(0)
            .fillna(0.0)
        )

        disp_rank = out.groupby("date")["sector_dispersion_21d"].rank(pct=True)
        mom = out["sector_relative_momentum_21d"].fillna(0.0)
        out["sector_rotation_signal"] = (mom * (1.0 - disp_rank)).clip(-1.0, 1.0)
        return out.sort_values(["stock_code", "date"])
