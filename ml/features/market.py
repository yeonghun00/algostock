"""Market regime, index membership, and macro index features."""

from __future__ import annotations

import pandas as pd

from .registry import FeatureGroup, register


@register
class MarketFeatures(FeatureGroup):
    name = "market"
    columns = ["market_regime_120d", "constituent_index_count"]
    dependencies = []  # Merged externally by pipeline

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # These columns are merged from external data sources in the pipeline.
        # This compute is a no-op; they're already present.
        if "market_regime_120d" not in df.columns:
            df["market_regime_120d"] = 0.0
        if "constituent_index_count" not in df.columns:
            df["constituent_index_count"] = 0.0
        return df


@register
class MacroFeatures(FeatureGroup):
    """Macro index features from deriv_index_daily (V5.5 Macro Layer).

    These are DATE-LEVEL time-series signals (same value for every stock on a
    given date).  Adding them directly as LightGBM features hurts cross-
    sectional ranking because they have zero within-day variance — the tree
    wastes splits distinguishing "2020 vs 2023" instead of "stock A vs B".

    Correct usage:
      • Alpha layer  → NOT in columns (not fed to LightGBM)
      • Interaction  → MacroInteractionFeatures combines them with stock features
      • Risk layer   → run_backtest.py uses vkospi_level_pct for cash-out sizing
    """

    name = "macro"
    # Empty: macro signals are NOT direct model features.
    columns = []
    # produced_cols: still computed by the pipeline and available in the
    # DataFrame for interaction features and the backtester cash-out logic.
    produced_cols = [
        "yield_curve_regime",
        "yield_curve_momentum_5d",
        "usd_pressure_5d",
        "usd_ma_ratio_20d",
        "fx_pressure_index",
        "vkospi_level_pct",
        "vkospi_change_5d",
        "market_vol_efficiency",
        "kosdaq_kospi_rotation",
        "rotation_momentum_5d",
        "sector_semicon_rel_21d",
        "sector_battery_rel_21d",
        "sector_bbig_rel_21d",
    ]
    dependencies = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Macro columns are merged from deriv_index_daily in the pipeline.
        # Fallback: fill any missing macro columns with 0.5 (neutral percentile).
        for col in self.produced_cols:
            if col not in df.columns:
                df[col] = 0.5
        return df
