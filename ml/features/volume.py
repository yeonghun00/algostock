"""Volume and liquidity features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class VolumeFeatures(FeatureGroup):
    name = "volume"
    # turnover_21d has a sector-neutral version; amihud and volume_ratio are kept raw
    # as pure liquidity signals (less sector-contaminated).
    columns = ["volume_ratio_21d", "amihud_21d"]
    dependencies = ["ret_1d", "closing_price"]
    phase = 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("stock_code")
        ma_vol = g["volume"].rolling(21, min_periods=10).mean().droplevel(0)
        ma_turnover = g["value"].rolling(21, min_periods=10).mean().droplevel(0)
        df["avg_value_20d"] = g["value"].rolling(20, min_periods=10).mean().droplevel(0)
        df["volume_ratio_21d"] = df["volume"] / ma_vol.replace(0, np.nan)
        df["turnover_21d"] = ma_turnover / df["market_cap"].replace(0, np.nan)

        abs_ret = df["ret_1d"].abs().replace(0, np.nan)
        df["amihud_21d"] = abs_ret / df["value"].replace(0, np.nan)
        df["amihud_21d"] = g["amihud_21d"].rolling(21, min_periods=10).mean().droplevel(0)
        return df
