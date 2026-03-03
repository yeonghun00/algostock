"""Volatility features and rolling beta."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class VolatilityFeatures(FeatureGroup):
    name = "volatility"
    # Raw vol/drawdown are intermediates for sector-neutral features; excluded from model.
    # rolling_beta_60d is a market-level signal (not sector-contaminated), kept raw.
    columns = ["rolling_beta_60d"]
    dependencies = ["ret_1d", "closing_price"]
    phase = 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("stock_code")
        df["volatility_21d"] = g["ret_1d"].rolling(21, min_periods=10).std().droplevel(0)
        df["volatility_63d"] = g["ret_1d"].rolling(63, min_periods=21).std().droplevel(0)

        # adj_closing_price: drawdown correctly handles splits within the 252-day window.
        p = "adj_closing_price" if "adj_closing_price" in df.columns else "closing_price"
        roll_max = g[p].rolling(252, min_periods=60).max().droplevel(0)
        df["drawdown_252d"] = df[p] / roll_max - 1

        # Rolling beta computed later after market_ret_1d is available
        # Placeholder — overwritten by _compute_rolling_beta in pipeline
        if "rolling_beta_60d" not in df.columns:
            df["rolling_beta_60d"] = 1.0
        return df
