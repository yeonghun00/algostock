"""Macro × Stock interaction features (V5.5 Hybrid Layer).

These features solve the cross-sectional vs. time-series mismatch:
  - Raw macro signals (vkospi_level_pct, yield_curve_regime) are DATE-LEVEL
    constants with zero within-day variance → useless for ranking stocks.
  - Interaction features MULTIPLY macro regime with stock-specific signals,
    creating per-stock variation that LightGBM can actually split on.

Only 2 interactions are created (팀장 지시: 선별적으로, 핵심만):
  1. conditional_momentum  — mom_21d dampened by fear (VKOSPI)
  2. value_regime_boost    — quality factor (sector_zscore_roe) weighted by
                             yield-curve regime (steepening → reward quality)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class MacroInteractionFeatures(FeatureGroup):
    name = "macro_interaction"
    columns = [
        "conditional_momentum",
        "value_regime_boost",
    ]
    # mom_21d: produced by MomentumFeatures (phase 1)
    # sector_zscore_roe: produced by FundamentalFeatures (phase 2)
    # vkospi_level_pct, yield_curve_regime: produced by MacroFeatures (phase 2)
    dependencies = ["mom_21d", "sector_zscore_roe", "vkospi_level_pct", "yield_curve_regime"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        vkos = pd.to_numeric(df.get("vkospi_level_pct", 0.5), errors="coerce").fillna(0.5)
        mom  = pd.to_numeric(df.get("mom_21d", 0.0),          errors="coerce").fillna(0.0)
        yc   = pd.to_numeric(df.get("yield_curve_regime", 0.5), errors="coerce").fillna(0.5)
        roe_z = pd.to_numeric(df.get("sector_zscore_roe", 0.0), errors="coerce").fillna(0.0)

        # --- Feature 1: Conditional Momentum ---
        # Trust momentum when fear is low; discount it when fear spikes.
        # vkospi_level_pct ∈ [0,1]: 0 = calm, 1 = extreme fear
        # When vkos=0.2 → retain 80% of momentum signal
        # When vkos=0.9 → retain only 10% of momentum signal
        df["conditional_momentum"] = mom * (1.0 - vkos)

        # --- Feature 2: Value × Yield Curve Regime ---
        # yield_sign = +1 if steepening (recovery), -1 if flattening (slowdown)
        # sector_zscore_roe: high z-score = high ROE (quality/growth)
        #                    low  z-score = low  ROE (value/cheap)
        # Steepening → reward quality (high ROE), sign = +1
        # Flattening → penalise quality, reward value (low ROE), sign = -1
        # i.e., in a slowdown, prefer cheap over quality.
        yield_sign = (yc > 0.5).astype(float) * 2.0 - 1.0  # +1 or -1
        df["value_regime_boost"] = roe_z * yield_sign

        return df
