"""Sector momentum, breadth, and relative momentum features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class SectorFeatures(FeatureGroup):
    name = "sector"
    columns = [
        "sector_momentum_21d",
        "sector_momentum_63d",
        "sector_relative_momentum_21d",
        "sector_relative_momentum_63d",
        "sector_breadth_21d",
        "sector_constituent_share",
    ]
    dependencies = ["ret_1d", "mom_21d", "mom_63d", "constituent_index_count"]

    MIN_STOCKS = 3  # minimum stocks per sector per date to trust mcap-weighted momentum

    def _compute_mcap_sector_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build synthetic sector indices from market-cap weighted daily returns.

        For each (date, sector) group with >= MIN_STOCKS stocks, computes a
        cumulative price series and derives 21d / 63d momentum.  Relative
        momentum is expressed vs. the market-cap weighted market return.

        Returns DataFrame[date, sector, sector_momentum_21d, sector_momentum_63d,
        sector_relative_momentum_21d, sector_relative_momentum_63d].
        Sectors below MIN_STOCKS on a given date
        are absent → caller fills NaN via cross-sectional fallback.
        """
        required = {"date", "sector", "stock_code", "market_cap", "ret_1d"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        tmp = df[["date", "sector", "stock_code", "market_cap", "ret_1d"]].copy()
        tmp["market_cap"] = pd.to_numeric(tmp["market_cap"], errors="coerce")
        tmp["ret_1d"] = pd.to_numeric(tmp["ret_1d"], errors="coerce")
        tmp = tmp.dropna(subset=["market_cap", "ret_1d"])
        tmp = tmp[tmp["market_cap"] > 0]

        # Drop (date, sector) pairs that are too small to trust
        sector_n = tmp.groupby(["date", "sector"])["stock_code"].transform("count")
        tmp = tmp[sector_n >= self.MIN_STOCKS]
        if tmp.empty:
            return pd.DataFrame()

        # --- Market-cap weighted sector daily return ---
        mcap_sum = tmp.groupby(["date", "sector"])["market_cap"].transform("sum")
        tmp["_wr"] = tmp["ret_1d"] * tmp["market_cap"] / mcap_sum
        sector_daily = (
            tmp.groupby(["date", "sector"])["_wr"]
            .sum()
            .reset_index()
            .rename(columns={"_wr": "_sr"})
        )

        # --- Market-cap weighted market daily return (for relative momentum) ---
        mkt_sum = tmp.groupby("date")["market_cap"].transform("sum")
        tmp["_mwr"] = tmp["ret_1d"] * tmp["market_cap"] / mkt_sum
        market_daily = (
            tmp.groupby("date")["_mwr"]
            .sum()
            .reset_index()
            .rename(columns={"_mwr": "_mr"})
            .sort_values("date")
        )

        # --- Cumulative price series → pct_change for momentum ---
        sector_daily = sector_daily.sort_values(["sector", "date"])
        sector_daily["_cum"] = sector_daily.groupby("sector")["_sr"].transform(
            lambda x: (1 + x).cumprod()
        )
        g = sector_daily.groupby("sector")["_cum"]
        sector_daily["sector_momentum_21d"] = g.pct_change(21, fill_method=None)
        sector_daily["sector_momentum_63d"] = g.pct_change(63, fill_method=None)

        market_daily["_mcum"] = (1 + market_daily["_mr"]).cumprod()
        market_daily["_mkt_21d"] = market_daily["_mcum"].pct_change(21, fill_method=None)
        market_daily["_mkt_63d"] = market_daily["_mcum"].pct_change(63, fill_method=None)

        sector_daily = sector_daily.merge(
            market_daily[["date", "_mkt_21d", "_mkt_63d"]], on="date", how="left"
        )
        sector_daily["sector_relative_momentum_21d"] = (
            sector_daily["sector_momentum_21d"] - sector_daily["_mkt_21d"]
        )
        sector_daily["sector_relative_momentum_63d"] = (
            sector_daily["sector_momentum_63d"] - sector_daily["_mkt_63d"]
        )

        return sector_daily[
            [
                "date", "sector",
                "sector_momentum_21d", "sector_momentum_63d",
                "sector_relative_momentum_21d",
                "sector_relative_momentum_63d",
            ]
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Synthetic mcap-weighted sector momentum ---
        sec_ret = self._compute_mcap_sector_momentum(df)
        if not sec_ret.empty:
            df = df.merge(sec_ret, on=["date", "sector"], how="left")

        # sector_momentum_{21,63}d: 0.0 if no mcap data (e.g. market_cap missing)
        for col in ["sector_momentum_21d", "sector_momentum_63d"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # sector_relative_momentum_*: NaN so cross-sectional fallback activates
        # for sectors that were below MIN_STOCKS on a given date.
        for col in ["sector_relative_momentum_21d", "sector_relative_momentum_63d"]:
            if col not in df.columns:
                df[col] = np.nan

        sec21_xs = df.groupby(["date", "sector"])["mom_21d"].transform("mean")
        sec63_xs = df.groupby(["date", "sector"])["mom_63d"].transform("mean")
        stock_vs_21 = (df["mom_21d"] - sec21_xs).fillna(0.0)
        stock_vs_63 = (df["mom_63d"] - sec63_xs).fillna(0.0)
        df["sector_relative_momentum_21d"] = df["sector_relative_momentum_21d"].fillna(stock_vs_21)
        df["sector_relative_momentum_63d"] = df["sector_relative_momentum_63d"].fillna(stock_vs_63)

        # --- Breadth and constituent share ---
        has_constituent = df["constituent_index_count"] > 0
        if not has_constituent.any():
            df["sector_breadth_21d"] = 0.0
            df["sector_constituent_share"] = 0.0
            return df

        constituent = df.loc[has_constituent]
        breadth = (
            constituent.assign(pos=(constituent["mom_21d"] > 0).astype(float))
            .groupby(["date", "sector"], as_index=False)
            .agg(
                sector_breadth_21d=("pos", "mean"),
                sector_constituent_share=("stock_code", "count"),
            )
        )
        daily_max = breadth.groupby("date")["sector_constituent_share"].transform("max").replace(0, np.nan)
        breadth["sector_constituent_share"] = breadth["sector_constituent_share"] / daily_max

        df = df.merge(breadth, on=["date", "sector"], how="left")
        df["sector_breadth_21d"] = df["sector_breadth_21d"].fillna(0.0)
        df["sector_constituent_share"] = df["sector_constituent_share"].fillna(0.0)
        return df
