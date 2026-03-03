"""Feature engineering pipeline — data loading, merging, and orchestration.

This module contains all the database-level logic (loading prices, financials,
index membership, etc.) and orchestrates feature groups via the registry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .registry import get_all_feature_columns, get_all_groups, resolve_order


class _FeatureColumnsDescriptor:
    """Descriptor that works on both class and instance access."""

    def __get__(self, obj, objtype=None):
        return get_all_feature_columns()


class FeatureEngineer:
    """Build a unified feature set for model training.

    All data loading and merging happens here. Feature computation is delegated
    to registered FeatureGroup classes.
    """

    CACHE_VERSION = "unified_v55_interact_20260228"
    BS_ITEM_CODES = {
        "equity": "ifrs-full_Equity",
        "assets": "ifrs-full_Assets",
        "operating_cf": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
    }
    PL_ITEM_CODES = {
        "net_income": "ifrs-full_ProfitLoss",
        "gross_profit": "ifrs-full_GrossProfit",
    }
    BROAD_INDEX_CODES = [
        "KOSPI_코스피",
        "KOSPI_코스피_(외국주포함)",
        "KOSDAQ_코스닥",
        "KOSDAQ_코스닥_(외국주포함)",
    ]

    FEATURE_COLUMNS = _FeatureColumnsDescriptor()

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-256000")
        try:
            conn.execute("PRAGMA mmap_size=2147483648")
        except Exception:
            pass
        self._conn = conn
        return conn

    @staticmethod
    def _to_iso(date_yyyymmdd: str) -> str:
        if len(date_yyyymmdd) == 8 and "-" not in date_yyyymmdd:
            return f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:]}"
        return date_yyyymmdd

    def _ensure_indexes(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_stock_date ON daily_prices(stock_code, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_date_mcap ON daily_prices(date, market_cap)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ic_date_stock ON index_constituents(date, stock_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ic_date_index ON index_constituents(date, index_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_stock_avail_consol ON financial_periods(stock_code, available_date, consolidation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bs_item_period ON financial_items_bs_cf(item_code_normalized, period_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pl_item_period ON financial_items_pl(item_code_normalized, period_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_code_date ON delisted_stocks(stock_code, delisting_date)")
            conn.commit()

    def _cache_path(self, cache_key: str) -> str:
        os.makedirs(".cache", exist_ok=True)
        digest = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return f".cache/features_unified_{digest}.parquet"

    def _normalize_delist_date(self, s: pd.Series) -> pd.Series:
        text = s.astype(str).str.strip()
        text = text.str.replace("-", "", regex=False)
        return text.where(text.str.len() == 8, None)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_prices(
        self,
        start_date: str,
        end_date: str,
        min_market_cap: int,
        markets: List[str],
        universe_end_date: Optional[str] = None,
        max_market_cap: Optional[int] = None,
    ) -> pd.DataFrame:
        _ = universe_end_date or end_date
        placeholders = ",".join(["?" for _ in markets])
        max_cap_clause = "AND dp.market_cap <= ?" if max_market_cap else ""
        with self._connect() as conn:
            # LEFT JOIN adj_daily_prices to get split-adjusted prices.
            # COALESCE fallback: stocks not yet in adj_daily_prices use raw price.
            # Raw closing_price is kept for level-based filters (min_price, low_price_trap).
            price_q = f"""
            SELECT dp.stock_code, dp.date, dp.market_type,
                   dp.closing_price, dp.opening_price,
                   dp.high_price, dp.low_price, dp.volume, dp.value, dp.market_cap,
                   COALESCE(adj.adj_closing_price, dp.closing_price) AS adj_closing_price,
                   COALESCE(adj.adj_opening_price, dp.opening_price) AS adj_opening_price
            FROM daily_prices dp
            LEFT JOIN adj_daily_prices adj
                   ON adj.stock_code = dp.stock_code AND adj.date = dp.date
            WHERE dp.date >= ? AND dp.date <= ?
              AND dp.market_type IN ({placeholders})
              AND dp.closing_price > 0
              AND dp.volume > 0
              AND dp.market_cap >= ?
              {max_cap_clause}
            ORDER BY dp.stock_code, dp.date
            """
            params = [start_date, end_date] + markets + [min_market_cap]
            if max_market_cap:
                params.append(max_market_cap)
            prices = pd.read_sql_query(price_q, conn, params=params)
            stocks = pd.read_sql_query(
                """
                SELECT stock_code,
                       current_name AS name,
                       current_market_type AS current_market_type
                FROM stocks
                """,
                conn,
            )
        return prices.merge(stocks, on="stock_code", how="left")

    def _exclude_delisted(self, df: pd.DataFrame) -> pd.DataFrame:
        with self._connect() as conn:
            delisted = pd.read_sql_query(
                "SELECT stock_code, delisting_date FROM delisted_stocks WHERE delisting_date IS NOT NULL",
                conn,
            )
        if delisted.empty:
            return df
        delisted["delisting_date"] = self._normalize_delist_date(delisted["delisting_date"])
        delisted = delisted.dropna(subset=["delisting_date"]).drop_duplicates("stock_code", keep="last")
        merged = df.merge(delisted, on="stock_code", how="left")
        keep = merged["delisting_date"].isna() | (merged["date"] < merged["delisting_date"])
        return merged.loc[keep].drop(columns=["delisting_date"])

    def _load_index_membership(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load index constituent snapshots for a date range.

        Loads a small warmup window before start_date so that merge_asof can
        find the most recent membership entry even for the first price rows.
        """
        # Load from 400 days before start to capture the most recent rebalance
        # snapshot that precedes the window, so merge_asof never misses a stock.
        iso_start_with_warmup = self._to_iso(
            (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
        )
        iso_end = self._to_iso(end_date)
        conn = self._connect()
        members = pd.read_sql_query(
            """
            SELECT date AS membership_date, stock_code, COUNT(DISTINCT index_code) AS constituent_index_count
            FROM index_constituents
            WHERE date >= ? AND date <= ?
            GROUP BY membership_date, stock_code
            """,
            conn,
            params=[iso_start_with_warmup, iso_end],
        )
        if not members.empty:
            members["membership_date"] = members["membership_date"].astype(str).str.replace("-", "", regex=False)
        return members

    def _merge_index_membership(self, data: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
        """PIT-join index membership onto price data using merge_asof.

        For each (stock_code, date) in data, picks the most recent membership
        snapshot whose membership_date <= date.  This correctly handles index
        rebalances that occur mid-month (the old first-of-month approximation
        could miss up to 15 trading days of membership changes).
        """
        if members.empty:
            data["constituent_index_count"] = 0.0
            return data
        data["_date_dt"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="coerce")
        right = members.copy()
        right["_mem_dt"] = pd.to_datetime(right["membership_date"], format="%Y%m%d", errors="coerce")
        right = right.dropna(subset=["_mem_dt"]).sort_values(["_mem_dt", "stock_code"])
        merged = pd.merge_asof(
            data.sort_values(["_date_dt", "stock_code"]),
            right[["stock_code", "_mem_dt", "constituent_index_count"]],
            left_on="_date_dt",
            right_on="_mem_dt",
            by="stock_code",
            direction="backward",
        )
        merged["constituent_index_count"] = pd.to_numeric(
            merged["constituent_index_count"], errors="coerce"
        ).fillna(0.0)
        merged = merged.drop(columns=["_date_dt", "_mem_dt"], errors="ignore")
        return merged.sort_values(["stock_code", "date"])

    def _load_sector_membership(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load sector labels from financial_periods.industry_name (point-in-time).

        Returns one row per (stock_code, available_date) with the sector label.
        The caller should use _merge_sector_pit() to join this onto the price data.
        start_date/end_date are unused but kept for API compatibility.
        """
        del start_date, end_date  # PIT join uses the full history up to end_date at merge time
        conn = self._connect()
        df = pd.read_sql_query(
            """
            SELECT DISTINCT
                stock_code,
                REPLACE(available_date, '-', '') AS available_date,
                industry_name AS sector
            FROM financial_periods
            WHERE industry_name IS NOT NULL
              AND TRIM(industry_name) != ''
            ORDER BY stock_code, available_date
            """,
            conn,
        )
        if df.empty:
            return pd.DataFrame(columns=["stock_code", "available_date", "sector"])
        # Normalize sector names: remove all whitespace to merge duplicates caused by
        # inconsistent spacing in KSIC labels (e.g. "전자부품 제조업" vs "전자 부품 제조업")
        df["sector"] = df["sector"].str.replace(r"\s+", "", regex=True)
        df = df.drop_duplicates(["stock_code", "available_date"], keep="last")
        return df

    def _merge_sector_pit(self, data: pd.DataFrame, sector_pit: pd.DataFrame) -> pd.DataFrame:
        """PIT-join industry_name from financial_periods onto the price data.

        For each (stock_code, date) row in *data*, picks the most recent
        industry_name whose available_date <= date.  Sets data["sector"].
        """
        if sector_pit.empty:
            data["sector"] = "UNMAPPED_SECTOR"
            return data
        data["_date_dt"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="coerce")
        right = sector_pit.copy()
        right["_avail_dt"] = pd.to_datetime(right["available_date"], format="%Y%m%d", errors="coerce")
        right = right.dropna(subset=["_avail_dt"]).sort_values(["_avail_dt", "stock_code"])
        merged = pd.merge_asof(
            data.sort_values(["_date_dt", "stock_code"]),
            right[["stock_code", "_avail_dt", "sector"]],
            left_on="_date_dt",
            right_on="_avail_dt",
            by="stock_code",
            direction="backward",
        )
        merged["sector"] = merged["sector"].fillna("UNMAPPED_SECTOR")
        merged = merged.drop(columns=["_date_dt", "_avail_dt"], errors="ignore")
        return merged.sort_values(["stock_code", "date"])

    def _load_market_regime(self, start_date: str, end_date: str, target_horizon: int) -> pd.DataFrame:
        with self._connect() as conn:
            idx = pd.read_sql_query(
                """
                SELECT date, closing_index
                FROM index_daily_prices
                WHERE index_code = 'KOSPI_코스피_200'
                  AND date >= ? AND date <= ?
                ORDER BY date
                """,
                conn,
                params=[start_date, end_date],
            )
        if idx.empty:
            print(
                "[Pipeline] WARNING: KOSPI_코스피_200 has NO data in index_daily_prices for "
                f"{start_date}–{end_date}. All market-regime features will be 0/NaN. "
                "Run: algostock index backfill -s <start> -e <end> -t kospi_index --force",
                flush=True,
            )
            return pd.DataFrame(columns=["date", "market_regime_120d", "market_regime_20d", "market_ret_1d", f"market_forward_return_{target_horizon}d"])
        # Warn if coverage is sparse (< 60% of expected trading days ≈ 252/year).
        n_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        expected_trading_days = max(1, int(n_days * 5 / 7 * 0.98))  # rough weekday estimate
        coverage = len(idx) / expected_trading_days
        if coverage < 0.6:
            print(
                f"[Pipeline] WARNING: KOSPI_코스피_200 coverage is only {coverage:.1%} "
                f"({len(idx)} rows vs ~{expected_trading_days} expected) for {start_date}–{end_date}. "
                "Market-regime features will be unreliable. "
                "Run: algostock index backfill -s <start> -e <end> -t kospi_index --force",
                flush=True,
            )
        idx["market_regime_120d"] = idx["closing_index"] / idx["closing_index"].rolling(120, min_periods=60).mean() - 1
        idx["market_regime_20d"] = idx["closing_index"] / idx["closing_index"].rolling(20, min_periods=10).mean() - 1
        idx["market_ret_1d"] = idx["closing_index"].pct_change()
        idx[f"market_forward_return_{target_horizon}d"] = idx["closing_index"].shift(-target_horizon) / idx["closing_index"] - 1
        return idx[["date", "market_regime_120d", "market_regime_20d", "market_ret_1d", f"market_forward_return_{target_horizon}d"]]

    def _load_macro_indices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load deriv_index_daily data and compute 13 macro features (V5.5 Macro Layer).

        All raw signals are normalized via a 252-day rolling percentile so that
        features with different units land on the same [0, 1] scale as stock
        features.  NaN values (pre-history indices or warmup period) are filled
        with 0.5 (neutral percentile).

        Returns a date-indexed DataFrame (date in YYYYMMDD string format) with
        13 macro feature columns ready to merge onto the stock-level DataFrame.
        """
        MACRO_CODES = [
            "DERIV_전략지수_국채선물_3년_10년_일드커브_스티프닝_지수",
            "DERIV_전략지수_국채선물_3년_10년_일드커브_플래트닝_지수",
            "DERIV_선물지수_미국달러선물지수",
            "DERIV_선물지수_엔선물지수",
            "DERIV_옵션지수_코스피_200_변동성지수",
            "DERIV_전략지수_코스피_200_TR",
            "DERIV_옵션지수_코스피_200_커버드콜_5%_OTM_지수",
            "DERIV_전략지수_코스닥150_롱_100%_코스피200_숏_100%_선물지수",
            "DERIV_전략지수_KRX_반도체_TR_지수",
            "DERIV_전략지수_KRX_2차전지_TOP_10_TR_지수",
            "DERIV_전략지수_KRX_BBIG_리스크컨트롤_12%_지수",
            "DERIV_전략지수_KRX_300_TR",
        ]
        iso_start = self._to_iso(start_date)
        iso_end = self._to_iso(end_date)
        placeholders = ",".join(["?" for _ in MACRO_CODES])
        try:
            with self._connect() as conn:
                raw = pd.read_sql_query(
                    f"""
                    SELECT REPLACE(date, '-', '') AS date, index_code, closing_index
                    FROM deriv_index_daily
                    WHERE date >= ? AND date <= ?
                      AND index_code IN ({placeholders})
                    ORDER BY date
                    """,
                    conn,
                    params=[iso_start, iso_end] + MACRO_CODES,
                )
        except Exception as exc:
            self.logger.warning("[Pipeline] _load_macro_indices: deriv_index_daily unavailable: %s", exc)
            return pd.DataFrame(columns=["date"])

        if raw.empty:
            self.logger.warning(
                "[Pipeline] _load_macro_indices: no data for %s–%s. "
                "Run: algostock index backfill -t derivatives --force",
                start_date, end_date,
            )
            return pd.DataFrame(columns=["date"])

        raw["closing_index"] = pd.to_numeric(raw["closing_index"], errors="coerce")
        wide = raw.pivot_table(
            index="date", columns="index_code", values="closing_index", aggfunc="last"
        )
        wide.columns.name = None
        wide = wide.sort_index()

        def pct_norm(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
            """252-day rolling percentile normalization → [0, 1]. Robust to outliers."""
            return s.rolling(window, min_periods=min_periods).rank(pct=True)

        macro = pd.DataFrame(index=wide.index)

        # --- Group A: Interest Rate — Yield Curve ---
        steep = wide.get("DERIV_전략지수_국채선물_3년_10년_일드커브_스티프닝_지수")
        flat  = wide.get("DERIV_전략지수_국채선물_3년_10년_일드커브_플래트닝_지수")
        if steep is not None and flat is not None:
            yc_regime = steep.pct_change(20, fill_method=None) - flat.pct_change(20, fill_method=None)
            macro["yield_curve_regime"] = pct_norm(yc_regime)
            macro["yield_curve_momentum_5d"] = pct_norm(yc_regime.pct_change(5, fill_method=None))
        else:
            macro["yield_curve_regime"] = np.nan
            macro["yield_curve_momentum_5d"] = np.nan

        # --- Group B: Currency — FX Pressure ---
        usd = wide.get("DERIV_선물지수_미국달러선물지수")
        jpy = wide.get("DERIV_선물지수_엔선물지수")
        if usd is not None:
            macro["usd_pressure_5d"] = pct_norm(usd.pct_change(5, fill_method=None))
            macro["usd_ma_ratio_20d"] = pct_norm(
                usd / usd.rolling(20, min_periods=10).mean() - 1
            )
        else:
            macro["usd_pressure_5d"] = np.nan
            macro["usd_ma_ratio_20d"] = np.nan
        if usd is not None and jpy is not None:
            usd_r20 = usd.pct_change(20, fill_method=None)
            jpy_r20 = jpy.pct_change(20, fill_method=None)
            fx_idx = usd_r20 / jpy_r20.replace(0, np.nan)
            macro["fx_pressure_index"] = pct_norm(fx_idx)
        else:
            macro["fx_pressure_index"] = np.nan

        # --- Group C: Sentiment — Volatility ---
        vkos   = wide.get("DERIV_옵션지수_코스피_200_변동성지수")
        k200tr = wide.get("DERIV_전략지수_코스피_200_TR")
        cc5otm = wide.get("DERIV_옵션지수_코스피_200_커버드콜_5%_OTM_지수")
        if vkos is not None:
            macro["vkospi_level_pct"] = pct_norm(vkos)
            macro["vkospi_change_5d"] = pct_norm(vkos.pct_change(5, fill_method=None))
        else:
            macro["vkospi_level_pct"] = np.nan
            macro["vkospi_change_5d"] = np.nan
        if k200tr is not None and cc5otm is not None:
            vol_eff = k200tr.pct_change(20, fill_method=None) - cc5otm.pct_change(20, fill_method=None)
            macro["market_vol_efficiency"] = pct_norm(vol_eff)
        else:
            macro["market_vol_efficiency"] = np.nan

        # --- Group D: Relative Strength — KOSDAQ/KOSPI Rotation ---
        rot = wide.get("DERIV_전략지수_코스닥150_롱_100%_코스피200_숏_100%_선물지수")
        if rot is not None:
            ma5  = rot.rolling(5, min_periods=3).mean()
            ma20 = rot.rolling(20, min_periods=10).mean()
            rot_signal = ma5 / ma20.replace(0, np.nan) - 1
            macro["kosdaq_kospi_rotation"] = pct_norm(rot_signal)
            macro["rotation_momentum_5d"] = pct_norm(rot.pct_change(5, fill_method=None))
        else:
            macro["kosdaq_kospi_rotation"] = np.nan
            macro["rotation_momentum_5d"] = np.nan

        # --- Group E: Sector Leadership vs KRX 300 TR ---
        semi   = wide.get("DERIV_전략지수_KRX_반도체_TR_지수")
        batt   = wide.get("DERIV_전략지수_KRX_2차전지_TOP_10_TR_지수")
        bbig   = wide.get("DERIV_전략지수_KRX_BBIG_리스크컨트롤_12%_지수")
        krx300 = wide.get("DERIV_전략지수_KRX_300_TR")
        if krx300 is not None:
            bm_r21 = krx300.pct_change(21, fill_method=None)
            macro["sector_semicon_rel_21d"] = (
                pct_norm(semi.pct_change(21, fill_method=None) - bm_r21)
                if semi is not None else np.nan
            )
            macro["sector_battery_rel_21d"] = (
                pct_norm(batt.pct_change(21, fill_method=None) - bm_r21)
                if batt is not None else np.nan
            )
            macro["sector_bbig_rel_21d"] = (
                pct_norm(bbig.pct_change(21, fill_method=None) - bm_r21)
                if bbig is not None else np.nan
            )
        else:
            macro["sector_semicon_rel_21d"] = np.nan
            macro["sector_battery_rel_21d"] = np.nan
            macro["sector_bbig_rel_21d"] = np.nan

        # Fill NaN (warmup period, short-history indices pre-2015/2016) → 0.5 neutral
        macro = macro.fillna(0.5)
        macro = macro.reset_index()  # 'date' becomes a column
        return macro

    def _load_financial_ratios_pit(self, stock_codes: List[str], end_date: str) -> pd.DataFrame:
        if not stock_codes:
            return pd.DataFrame(columns=["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"])

        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS _elig_fin")
            conn.execute("CREATE TEMP TABLE _elig_fin (stock_code TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO _elig_fin(stock_code) VALUES (?)", [(c,) for c in stock_codes])

            params_common = [
                self.BS_ITEM_CODES["equity"], self.BS_ITEM_CODES["assets"],
                self.BS_ITEM_CODES["operating_cf"], "연결", end_date,
            ]
            bs_df = pd.read_sql_query(
                """
                SELECT
                    fp.id AS period_id, fp.stock_code,
                    REPLACE(fp.available_date, '-', '') AS available_date,
                    fp.fiscal_date, fp.fiscal_month,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS equity,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS assets,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS operating_cf
                FROM financial_periods fp
                JOIN financial_items_bs_cf bs ON bs.period_id = fp.id
                JOIN _elig_fin e ON e.stock_code = fp.stock_code
                WHERE fp.consolidation_type = ?
                  AND REPLACE(fp.available_date, '-', '') <= ?
                  AND bs.item_code_normalized IN (?, ?, ?)
                GROUP BY fp.id, fp.stock_code, fp.available_date
                """,
                conn,
                params=params_common + [
                    self.BS_ITEM_CODES["equity"], self.BS_ITEM_CODES["assets"],
                    self.BS_ITEM_CODES["operating_cf"],
                ],
            )

            params_common_pl = [self.PL_ITEM_CODES["net_income"], self.PL_ITEM_CODES["gross_profit"], "연결", end_date]
            pl_df = pd.read_sql_query(
                """
                SELECT
                    fp.id AS period_id, fp.stock_code,
                    REPLACE(fp.available_date, '-', '') AS available_date,
                    fp.fiscal_date, fp.fiscal_month,
                    MAX(CASE WHEN pl.item_code_normalized = ? THEN pl.amount_current_ytd END) AS net_income,
                    MAX(CASE WHEN pl.item_code_normalized = ? THEN pl.amount_current_ytd END) AS gross_profit
                FROM financial_periods fp
                JOIN financial_items_pl pl ON pl.period_id = fp.id
                JOIN _elig_fin e ON e.stock_code = fp.stock_code
                WHERE fp.consolidation_type = ?
                  AND REPLACE(fp.available_date, '-', '') <= ?
                  AND pl.item_code_normalized IN (?, ?)
                GROUP BY fp.id, fp.stock_code, fp.available_date
                """,
                conn,
                params=params_common_pl + [self.PL_ITEM_CODES["net_income"], self.PL_ITEM_CODES["gross_profit"]],
            )

        if bs_df.empty and pl_df.empty:
            return pd.DataFrame(columns=["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"])

        fin = bs_df.merge(pl_df, on=["period_id", "stock_code", "available_date", "fiscal_date", "fiscal_month"], how="outer")
        fin["equity"] = pd.to_numeric(fin["equity"], errors="coerce")
        fin["assets"] = pd.to_numeric(fin["assets"], errors="coerce")
        fin["operating_cf"] = pd.to_numeric(fin["operating_cf"], errors="coerce")
        fin["net_income"] = pd.to_numeric(fin["net_income"], errors="coerce")
        fin["gross_profit"] = pd.to_numeric(fin["gross_profit"], errors="coerce")
        fin["fiscal_month"] = pd.to_numeric(fin["fiscal_month"], errors="coerce").fillna(12).astype(int)

        # --- Annualize YTD P&L figures ---
        # amount_current_ytd is cumulative: Q1=3mo, Q2=6mo, Q3=9mo, Annual=12mo.
        # Without annualization, Q1 ROE looks 4x lower than annual ROE.
        fiscal_date_month = pd.to_datetime(fin["fiscal_date"], errors="coerce").dt.month
        months_ytd = ((fiscal_date_month - fin["fiscal_month"]) % 12).replace(0, 12)
        annualization_factor = 12.0 / months_ytd.clip(lower=3)
        fin["net_income"] = fin["net_income"] * annualization_factor
        fin["gross_profit"] = fin["gross_profit"] * annualization_factor
        # operating_cf is also YTD on BS/CF statements — annualize for consistency
        fin["operating_cf"] = fin["operating_cf"] * annualization_factor

        # --- Compute ratios from annualized figures ---
        # Reject negative equity (insolvent companies)
        valid_equity = fin["equity"].where(fin["equity"] > 0, np.nan)
        fin["roe"] = fin["net_income"] / valid_equity
        fin["gpa"] = fin["gross_profit"] / fin["assets"].replace(0, np.nan)

        # --- Dedup: prefer reports with more months (annual > Q3 > Q2 > Q1) ---
        fin["_months_ytd"] = months_ytd
        fin = fin.sort_values(["stock_code", "available_date", "_months_ytd"])
        fin = fin.drop_duplicates(["stock_code", "available_date"], keep="last")

        fin = fin[["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"]].copy()
        fin = fin.sort_values(["stock_code", "available_date"])
        return fin

    def _merge_financial_features(self, df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
        if fin_df.empty:
            df["roe"] = np.nan
            df["gpa"] = np.nan
            df["net_income"] = np.nan
            df["operating_cf"] = np.nan
            return df

        df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        right = fin_df.copy()
        right["available_dt"] = pd.to_datetime(right["available_date"], format="%Y%m%d", errors="coerce")
        right = right.dropna(subset=["available_dt"]).sort_values(["stock_code", "available_dt"])

        # merge_asof requires left_on to be globally sorted
        df = df.sort_values(["date_dt", "stock_code"])
        merged = pd.merge_asof(
            df,
            right.sort_values(["available_dt", "stock_code"]),
            left_on="date_dt",
            right_on="available_dt",
            by="stock_code",
            direction="backward",
            suffixes=("", "_fin"),
        )

        # --- Staleness guard: if financial data is >15 months old, treat as missing ---
        staleness = (merged["date_dt"] - merged["available_dt"]).dt.days
        is_stale = staleness > 450  # ~15 months
        for col in ["roe", "gpa", "net_income", "operating_cf"]:
            if col in merged.columns:
                merged.loc[is_stale, col] = np.nan

        merged = merged.drop(columns=["date_dt", "available_dt", "available_date"], errors="ignore")

        for col in ["roe", "gpa"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            sector_med = merged.groupby(["date", "sector"])[col].transform("median")
            market_med = merged.groupby("date")[col].transform("median")
            merged[col] = merged[col].fillna(sector_med).fillna(market_med).fillna(0.0)
            merged[col] = merged[col].clip(-5.0, 5.0)
        merged["net_income"] = pd.to_numeric(merged["net_income"], errors="coerce")
        merged["operating_cf"] = pd.to_numeric(merged["operating_cf"], errors="coerce")
        return merged

    # ------------------------------------------------------------------
    # Universe filters
    # ------------------------------------------------------------------

    def _apply_hard_universe_filters(
        self,
        df: pd.DataFrame,
        min_price: int = 2000,
        liquidity_drop_pct: float = 0.20,
    ) -> pd.DataFrame:
        mask = df["closing_price"] >= min_price
        if "avg_value_20d" in df.columns and mask.any():
            liq_cut = df.groupby("date")["avg_value_20d"].transform(
                lambda s: s.quantile(liquidity_drop_pct)
            )
            mask &= df["avg_value_20d"] >= liq_cut
        if "net_income" in df.columns and "operating_cf" in df.columns:
            bad_accrual = (df["net_income"] > 0) & (df["operating_cf"] < 0)
            mask &= ~bad_accrual
        # Exclude stocks with extreme ROE (likely negative/micro equity)
        if "roe" in df.columns:
            mask &= df["roe"].abs() <= 3.0  # |ROE| > 300% = distressed or data issue
        return df.loc[mask]

    # ------------------------------------------------------------------
    # Rolling beta (needs market_ret_1d which comes from regime merge)
    # ------------------------------------------------------------------

    def _compute_rolling_beta(self, df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
        out = df.sort_values(["stock_code", "date"])
        if "market_ret_1d" not in out.columns:
            out["rolling_beta_60d"] = 1.0
            return out
        out["_xy"] = out["ret_1d"] * out["market_ret_1d"]
        out["_y2"] = out["market_ret_1d"] ** 2
        g = out.groupby("stock_code", sort=False)
        roll_xy = g["_xy"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_x = g["ret_1d"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_y = g["market_ret_1d"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_y2 = g["_y2"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        cov = roll_xy - roll_x * roll_y
        var = roll_y2 - roll_y ** 2
        beta = cov / var.replace(0, np.nan)
        out = out.drop(columns=["_xy", "_y2"])
        out["rolling_beta_60d"] = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-3.0, 3.0)
        return out

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _add_targets(self, df: pd.DataFrame, target_horizon: int) -> pd.DataFrame:
        out = df.sort_values(["stock_code", "date"])
        g = out.groupby("stock_code")
        fwd_col = f"forward_return_{target_horizon}d"
        rank_col = f"target_rank_{target_horizon}d"
        risk_adj_col = f"target_riskadj_{target_horizon}d"
        risk_adj_rank_col = f"target_riskadj_rank_{target_horizon}d"
        residual_col = f"target_residual_{target_horizon}d"
        residual_rank_col = f"target_residual_rank_{target_horizon}d"

        # Use adj_closing_price so splits within the horizon don't create
        # fake large returns (e.g. a 50:1 split mid-period looks like −98%).
        price_col = "adj_closing_price" if "adj_closing_price" in out.columns else "closing_price"

        # fwd_col is pre-computed on the UNFILTERED series in _prepare_range_core
        # (before _apply_hard_universe_filters) so that row gaps from filters like
        # bad_accrual do not distort shift(-N).  Only recompute here as a fallback
        # when called outside of _prepare_range_core (e.g. unit tests).
        if fwd_col not in out.columns:
            out[fwd_col] = g[price_col].shift(-target_horizon) / out[price_col] - 1

            # --- Survivorship-bias fix for suspended / delisted stocks ---
            # last_price: last available adj price for each stock in the dataset.
            # For delisted stocks this is the final crash/M&A price before delisting.
            last_price = g[price_col].transform("last")

            # Fix A: NaN forward return → stock was delisted before T+horizon.
            nan_mask = out[fwd_col].isna() & out[price_col].gt(0)
            out.loc[nan_mask, fwd_col] = (
                last_price[nan_mask] / out.loc[nan_mask, price_col] - 1
            )

            # Fix B: Frozen price at T+horizon → stock was suspended (거래정지).
            if "value" in out.columns:
                future_value = g["value"].shift(-target_horizon)
                frozen_mask = (
                    out[fwd_col].notna()
                    & out[price_col].gt(0)
                    & (future_value == 0)
                )
                out.loc[frozen_mask, fwd_col] = (
                    last_price[frozen_mask] / out.loc[frozen_mask, price_col] - 1
                )

        out[rank_col] = out.groupby("date")[fwd_col].rank(method="average", pct=True).fillna(0.5)
        vol = out["volatility_21d"] if "volatility_21d" in out.columns else np.nan
        out[risk_adj_col] = out[fwd_col] / pd.to_numeric(vol, errors="coerce").replace(0, np.nan)
        out[risk_adj_rank_col] = out.groupby("date")[risk_adj_col].rank(method="average", pct=True).fillna(0.5)
        market_fwd_col = f"market_forward_return_{target_horizon}d"
        if market_fwd_col in out.columns and "rolling_beta_60d" in out.columns:
            out[residual_col] = out[fwd_col] - (out["rolling_beta_60d"] * out[market_fwd_col])
            out[residual_rank_col] = out.groupby("date")[residual_col].rank(method="average", pct=True).fillna(0.5)
        else:
            out[residual_col] = out[fwd_col]
            out[residual_rank_col] = out[rank_col]
        return out

    # ------------------------------------------------------------------
    # Year-chunk batching
    # ------------------------------------------------------------------

    def _build_year_chunks(self, start_date: str, end_date: str, target_horizon: int) -> List[dict]:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        warmup_days = 420
        # Convert trading-day horizon to calendar days with a safety buffer.
        # 42 trading days ≈ 63 calendar days (× 7/5).  Add 15-day buffer for
        # holidays and month-end signals so shift(-N) always has enough lookahead.
        lookahead_days = max(int(target_horizon * 7 / 5) + 15, 42)
        chunks: List[dict] = []
        for year in range(start_dt.year, end_dt.year + 1):
            trim_start = max(start_dt, datetime(year, 1, 1))
            trim_end = min(end_dt, datetime(year, 12, 31))
            if trim_start > trim_end:
                continue
            core_start = trim_start - timedelta(days=warmup_days)
            core_end = trim_end + timedelta(days=lookahead_days)
            chunks.append({
                "year": year,
                "core_start": core_start.strftime("%Y%m%d"),
                "core_end": core_end.strftime("%Y%m%d"),
                "trim_start": trim_start.strftime("%Y%m%d"),
                "trim_end": trim_end.strftime("%Y%m%d"),
            })
        return chunks

    # ------------------------------------------------------------------
    # Core range pipeline (loads data, merges, runs feature groups)
    # ------------------------------------------------------------------

    def _prepare_range_core(
        self,
        start_date: str,
        end_date: str,
        target_horizon: int,
        min_market_cap: int,
        markets: List[str],
        universe_end_date: Optional[str] = None,
        max_market_cap: Optional[int] = None,
    ) -> pd.DataFrame:
        raw = self._load_prices(
            start_date, end_date, min_market_cap, markets,
            universe_end_date=universe_end_date or end_date,
            max_market_cap=max_market_cap,
        )
        if raw.empty:
            return raw

        data = self._exclude_delisted(raw)

        # --- External data merges ---
        members = self._load_index_membership(start_date, end_date)
        sector_pit = self._load_sector_membership(start_date, end_date)
        regime = self._load_market_regime(start_date, end_date, target_horizon)
        macro_regime = self._load_macro_indices(start_date, end_date)

        data = self._merge_index_membership(data, members)
        data = self._merge_sector_pit(data, sector_pit)

        fin_pit = self._load_financial_ratios_pit(data["stock_code"].unique().tolist(), end_date)
        data = self._merge_financial_features(data, fin_pit)

        # Sort before feature computation
        data = data.sort_values(["stock_code", "date"])

        # --- Phase 1: price-based feature groups (before filters and external merges) ---
        all_groups = resolve_order(get_all_groups())
        phase1_groups = [g for g in all_groups if g.phase == 1]
        phase2_groups = [g for g in all_groups if g.phase == 2]

        for group_cls in phase1_groups:
            group = group_cls()
            data = group.compute(data)

        # --- Pre-compute forward return on COMPLETE (unfiltered) per-stock time series ---
        # MUST run before _apply_hard_universe_filters.  Filters like bad_accrual and
        # avg_value_20d remove rows from the DataFrame, creating calendar gaps in each
        # stock's series.  shift(-N) on the gapped series lands at the wrong calendar date
        # (e.g. shift(-42) from 2020-03-04 jumps to 2020-09-17 instead of 2020-05-07 when
        # the bad_accrual filter removes all rows between 2020-04-01 and 2020-08-15).
        _fwd_col = f"forward_return_{target_horizon}d"
        _pc = "adj_closing_price" if "adj_closing_price" in data.columns else "closing_price"
        data = data.sort_values(["stock_code", "date"])
        _g = data.groupby("stock_code")
        data[_fwd_col] = _g[_pc].shift(-target_horizon) / data[_pc] - 1
        _last_px = _g[_pc].transform("last")
        # Fix A: NaN (delisted / data tail before T+horizon)
        _nm = data[_fwd_col].isna() & data[_pc].gt(0)
        data.loc[_nm, _fwd_col] = _last_px[_nm] / data.loc[_nm, _pc] - 1
        # Fix B: frozen price (거래정지) at T+horizon
        if "value" in data.columns:
            _fv = _g["value"].shift(-target_horizon)
            _fm = data[_fwd_col].notna() & data[_pc].gt(0) & (_fv == 0)
            data.loc[_fm, _fwd_col] = _last_px[_fm] / data.loc[_fm, _pc] - 1

        # --- Universe filters (after volume/price features are computed) ---
        data = self._apply_hard_universe_filters(data, min_price=2000, liquidity_drop_pct=0.20)

        # --- Merge market regime ---
        data = data.merge(regime, on="date", how="left")
        data["market_regime_120d"] = data["market_regime_120d"].fillna(0.0)
        # market_regime_20d must also be filled: NaN leaves the cash-out rule permanently disabled
        # for any date where index data is missing (most years in the current DB).
        data["market_regime_20d"] = data["market_regime_20d"].fillna(0.0)
        data["market_ret_1d"] = data["market_ret_1d"].fillna(0.0)
        data[f"market_forward_return_{target_horizon}d"] = data[f"market_forward_return_{target_horizon}d"].fillna(0.0)

        # --- Merge macro features (deriv_index_daily) ---
        _macro_cols = [c for c in macro_regime.columns if c != "date"]
        if _macro_cols:
            data = data.merge(macro_regime, on="date", how="left")
            for col in _macro_cols:
                data[col] = data[col].fillna(0.5)

        # --- Rolling beta (needs market_ret_1d from regime) ---
        data = self._compute_rolling_beta(data)

        # --- Phase 2: feature groups needing sector/market data ---
        for group_cls in phase2_groups:
            group = group_cls()
            data = group.compute(data)

        # --- Targets ---
        data = self._add_targets(data, target_horizon)
        data = data.drop(columns=["membership_date"], errors="ignore")
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_ml_data(
        self,
        start_date: str,
        end_date: str,
        target_horizon: int = 21,
        min_market_cap: int = 500_000_000_000,
        max_market_cap: Optional[int] = None,
        markets: Optional[List[str]] = None,
        include_fundamental: bool = True,
        include_macro: bool = True,
        use_cache: bool = True,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        del include_fundamental, include_macro

        markets = markets or ["kospi", "kosdaq"]
        self._ensure_indexes()
        workers = max(1, n_workers or 4)

        feature_columns = self.FEATURE_COLUMNS
        max_cap_key = f"_max{max_market_cap}" if max_market_cap else ""
        cache_key = (
            f"{self.CACHE_VERSION}_{start_date}_{end_date}_{target_horizon}_{min_market_cap}{max_cap_key}_"
            f"{'_'.join(sorted(markets))}"
        )
        cache_path = self._cache_path(cache_key)
        if use_cache and os.path.exists(cache_path):
            db_mtime = os.path.getmtime(self.db_path)
            if os.path.getmtime(cache_path) > db_mtime:
                cached = pd.read_parquet(cache_path)
                self.logger.info("Loaded features from cache: %s rows", len(cached))
                return cached

        chunks = self._build_year_chunks(start_date, end_date, target_horizon)
        if not chunks:
            return pd.DataFrame()

        years = [c["year"] for c in chunks]
        print(
            f"[Features] year-batch mode years={years} workers={workers} cache={'on' if use_cache else 'off'}",
            flush=True,
        )

        frames: List[pd.DataFrame] = []
        if workers == 1 or len(chunks) == 1:
            for chunk in chunks:
                print(
                    f"[Features] year={chunk['year']} load core={chunk['core_start']}~{chunk['core_end']} "
                    f"trim={chunk['trim_start']}~{chunk['trim_end']}",
                    flush=True,
                )
                chunk_df = self._prepare_range_core(
                    start_date=chunk["core_start"],
                    end_date=chunk["core_end"],
                    target_horizon=target_horizon,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    markets=markets,
                    universe_end_date=chunk["trim_end"],
                )
                if chunk_df.empty:
                    print(f"[Features] year={chunk['year']} produced 0 rows", flush=True)
                    continue
                chunk_df = chunk_df[
                    (chunk_df["date"] >= chunk["trim_start"]) & (chunk_df["date"] <= chunk["trim_end"])
                ]
                print(f"[Features] year={chunk['year']} rows={len(chunk_df):,}", flush=True)
                frames.append(chunk_df)
        else:
            payloads = [
                {
                    "db_path": self.db_path,
                    "core_start": c["core_start"],
                    "core_end": c["core_end"],
                    "trim_start": c["trim_start"],
                    "trim_end": c["trim_end"],
                    "target_horizon": target_horizon,
                    "min_market_cap": min_market_cap,
                    "max_market_cap": max_market_cap,
                    "markets": markets,
                    "year": c["year"],
                    "universe_end_date": c["trim_end"],
                }
                for c in chunks
            ]
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_prepare_year_chunk_worker, p): p["year"] for p in payloads}
                    for fut in as_completed(futures):
                        year = futures[fut]
                        chunk_df = fut.result()
                        if not chunk_df.empty:
                            frames.append(chunk_df)
                            print(f"[Features] year={year} rows={len(chunk_df):,}", flush=True)
                        else:
                            print(f"[Features] year={year} produced 0 rows", flush=True)
            except (PermissionError, OSError) as exc:
                self.logger.warning(
                    "Multiprocessing unavailable (%s). Falling back to sequential.", exc,
                )
                print(f"[Features] multiprocessing unavailable ({exc}); fallback to sequential", flush=True)
                for p in payloads:
                    print(
                        f"[Features] year={p['year']} load core={p['core_start']}~{p['core_end']} "
                        f"trim={p['trim_start']}~{p['trim_end']}",
                        flush=True,
                    )
                    chunk_df = _prepare_year_chunk_worker(p)
                    if not chunk_df.empty:
                        frames.append(chunk_df)
                        print(f"[Features] year={p['year']} rows={len(chunk_df):,}", flush=True)
                    else:
                        print(f"[Features] year={p['year']} produced 0 rows", flush=True)

        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values(["date", "stock_code"]).drop_duplicates(["date", "stock_code"], keep="last")

        fwd_col = f"forward_return_{target_horizon}d"
        required = [c for c in feature_columns if c in data.columns] + [fwd_col]
        data = data.dropna(subset=required)

        feature_cols = [c for c in feature_columns if c in data.columns]
        for col in feature_cols:
            lo = data.groupby("date")[col].transform(lambda s: s.quantile(0.01))
            hi = data.groupby("date")[col].transform(lambda s: s.quantile(0.99))
            data[col] = data[col].clip(lower=lo, upper=hi)

        data = data.sort_values(["date", "stock_code"]).reset_index(drop=True)
        print(f"[Features] merged rows={len(data):,}", flush=True)

        if use_cache:
            data.to_parquet(cache_path, index=False)

        self.logger.info("Prepared unified ML dataset: %s rows", len(data))
        return data

    def prepare_prediction_data(
        self,
        end_date: str,
        target_horizon: int = 21,
        min_market_cap: int = 500_000_000_000,
        max_market_cap: Optional[int] = None,
        markets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        markets = markets or ["kospi", "kosdaq"]
        self._ensure_indexes()

        warmup_days = 420
        start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=warmup_days)
        start_date = start_dt.strftime("%Y%m%d")

        print(f"[Prediction] building features {start_date}~{end_date}", flush=True)

        data = self._prepare_range_core(
            start_date=start_date,
            end_date=end_date,
            target_horizon=target_horizon,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            markets=markets,
        )

        if data.empty:
            return data

        feature_columns = self.FEATURE_COLUMNS
        feature_required = [c for c in feature_columns if c in data.columns]
        data = data.dropna(subset=feature_required)

        for col in feature_required:
            data[col] = data[col].clip(
                lower=np.nanpercentile(data[col], 1),
                upper=np.nanpercentile(data[col], 99),
            )

        latest_date = data["date"].max()
        pred = data[data["date"] == latest_date].copy()
        print(f"[Prediction] date={latest_date}, universe={len(pred)}", flush=True)
        return pred


def _prepare_year_chunk_worker(payload: dict) -> pd.DataFrame:
    fe = FeatureEngineer(payload["db_path"])
    chunk_df = fe._prepare_range_core(
        start_date=payload["core_start"],
        end_date=payload["core_end"],
        target_horizon=payload["target_horizon"],
        min_market_cap=payload["min_market_cap"],
        max_market_cap=payload.get("max_market_cap"),
        markets=payload["markets"],
        universe_end_date=payload.get("universe_end_date", payload["core_end"]),
    )
    if chunk_df.empty:
        return chunk_df
    return chunk_df[
        (chunk_df["date"] >= payload["trim_start"]) & (chunk_df["date"] <= payload["trim_end"])
    ].copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build unified feature dataset")
    parser.add_argument("--start", default="20110101")
    parser.add_argument("--end", default="20260213")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    fe = FeatureEngineer("krx_stock_data.db")
    out = fe.prepare_ml_data(
        start_date=args.start,
        end_date=args.end,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
        use_cache=not args.no_cache,
    )
    print(f"rows={len(out):,}, cols={len(out.columns)}")
    print("features:")
    for feature in fe.FEATURE_COLUMNS:
        print(f"- {feature}")
