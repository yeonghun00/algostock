"""Distress and trap detection features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class DistressFeatures(FeatureGroup):
    name = "distress"
    columns = ["distress_composite_score", "liquidity_decay_score", "low_price_trap"]
    dependencies = ["closing_price"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["stock_code", "date"])
        g = out.groupby("stock_code")
        ma20_value = g["value"].rolling(20, min_periods=10).mean().droplevel(0)
        ma252_value = g["value"].rolling(252, min_periods=60).mean().droplevel(0)
        out["liquidity_decay_score"] = (ma20_value / ma252_value.replace(0, np.nan)).clip(0, 5)
        out["is_liquidity_distressed"] = (out["liquidity_decay_score"] <= 0.2).astype(float)

        sector_avg_price = out.groupby(["date", "sector"])["closing_price"].transform("mean")
        out["low_price_trap"] = np.log(
            out["closing_price"].clip(lower=1) / sector_avg_price.clip(lower=1)
        ).replace([np.inf, -np.inf], np.nan)
        out["is_low_price_trap"] = (
            (out["closing_price"] < 1000) | (out["low_price_trap"] < -1.0)
        ).astype(float)

        liq_risk = (0.2 - out["liquidity_decay_score"]).clip(lower=0, upper=0.2) / 0.2
        low_price_risk = (-out["low_price_trap"]).clip(lower=0, upper=2) / 2
        out["distress_composite_score"] = (
            0.5 * liq_risk.fillna(0.0)
            + 0.35 * low_price_risk.fillna(0.0)
            + 0.15 * out["is_low_price_trap"]
        ).clip(0, 1)
        return out
