"""Academic momentum features.

Based on:
- Jegadeesh & Titman (1993): 3-12 month price momentum
- George & Hwang (2004): 52-week high proximity
- Moving Average Ratio as momentum predictor
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class AcademicMomentumFeatures(FeatureGroup):
    name = "momentum_academic"
    columns = [
        "high_52w_proximity",
        "ma_ratio_20_120",
        "ma_ratio_5_60",
        "momentum_quality",
    ]
    dependencies = ["closing_price", "mom_21d", "mom_63d", "mom_126d"]
    phase = 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use split-adjusted prices so corporate actions within the window
        # (splits, rights issues) don't create fake momentum signals.
        p = "adj_closing_price" if "adj_closing_price" in df.columns else "closing_price"
        g = df.groupby("stock_code")

        # ── 52-Week High Proximity (George & Hwang, 2004) ──
        # current_price / 252-day rolling max
        # Closer to 1.0 = near 52-week high = stronger forward returns
        high_252 = g[p].rolling(252, min_periods=60).max().droplevel(0)
        df["high_52w_proximity"] = df[p] / high_252.replace(0, np.nan)

        # ── Moving Average Ratios ──
        # Short MA / Long MA: >1 means short-term trend is above long-term
        # More predictive than simple "price above MA" (binary)
        ma_20 = g[p].rolling(20, min_periods=10).mean().droplevel(0)
        ma_60 = g[p].rolling(60, min_periods=20).mean().droplevel(0)
        ma_120 = g[p].rolling(120, min_periods=40).mean().droplevel(0)
        df["ma_ratio_20_120"] = ma_20 / ma_120.replace(0, np.nan)
        df["ma_ratio_5_60"] = (
            g[p].rolling(5, min_periods=3).mean().droplevel(0)
            / ma_60.replace(0, np.nan)
        )

        # ── Momentum Quality (consistency) ──
        # Fraction of the last 126 days with positive returns.
        # High momentum + high consistency = less reversal risk.
        # Jegadeesh & Titman observed that "smooth" momentum
        # (steady climbers) outperforms "jumpy" momentum.
        df["_pos_day"] = (g[p].pct_change() > 0).astype(float)
        df["momentum_quality"] = (
            df.groupby("stock_code")["_pos_day"]
            .rolling(126, min_periods=40)
            .mean()
            .droplevel(0)
        )
        df = df.drop(columns=["_pos_day"])

        return df
