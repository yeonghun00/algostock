#!/usr/bin/env python3
"""Unified backtest runner (single model, single feature pipeline)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Friendly name → DB index_code mapping for benchmark selection
BENCHMARK_INDEX_MAP: dict[str, str | None] = {
    "kospi200":  "KOSPI_코스피_200",
    "kospi":     "KOSPI_코스피",
    "kosdaq":    "KOSDAQ_코스닥",
    "kosdaq150": "KOSDAQ_코스닥_150",
    "universe":  None,   # equal-weight average of the stock universe
}


def _load_benchmark_returns(db_path: str, index_code: str, horizon: int) -> dict[str, float]:
    """Load N-day forward returns for a given index from the DB.

    Returns a dict mapping YYYYMMDD date string → forward return (float).
    Dates near the tail where T+horizon doesn't exist will have NaN and are excluded.
    """
    import sqlite3 as _sqlite3
    try:
        with _sqlite3.connect(db_path) as conn:
            idx = pd.read_sql_query(
                "SELECT date, closing_index FROM index_daily_prices "
                "WHERE index_code = ? ORDER BY date",
                conn,
                params=(index_code,),
            )
    except Exception as exc:
        print(f"[Benchmark] WARNING: failed to load {index_code}: {exc}", flush=True)
        return {}
    if idx.empty:
        print(f"[Benchmark] WARNING: no data for {index_code}", flush=True)
        return {}
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["fwd"] = idx["closing_index"].shift(-horizon) / idx["closing_index"] - 1
    return dict(zip(idx["date"], idx["fwd"].where(idx["fwd"].notna())))


def _compute_core_stats(results: pd.DataFrame) -> dict:
    """Compute all backtest statistics from results DataFrame."""
    s = {}
    s["n_rebalances"] = len(results)
    s["n_years"] = max(results["year"].nunique(), 1)

    # --- Cumulative ---
    s["total_return"] = (1.0 + results["portfolio_return"]).prod() - 1.0
    s["benchmark_return"] = (1.0 + results["benchmark_return"]).prod() - 1.0
    s["alpha"] = s["total_return"] - s["benchmark_return"]

    # --- Drawdown ---
    cum = (1.0 + results["portfolio_return"]).cumprod()
    drawdown = cum / cum.cummax() - 1.0
    s["max_dd"] = float(drawdown.min())
    s["cum_portfolio"] = cum
    s["cum_benchmark"] = (1.0 + results["benchmark_return"]).cumprod()
    s["drawdown"] = drawdown

    # Underwater duration
    uw = 0
    max_uw = 0
    for flag in (drawdown < 0).tolist():
        uw = uw + 1 if flag else 0
        max_uw = max(max_uw, uw)
    s["max_underwater"] = max_uw

    # --- Annualized ---
    rebals_per_year = max(len(results) / s["n_years"], 1)
    s["ann_vol"] = float(results["portfolio_return"].std() * np.sqrt(rebals_per_year))
    s["ann_return"] = (1.0 + s["total_return"]) ** (1.0 / s["n_years"]) - 1.0
    s["sharpe"] = s["ann_return"] / s["ann_vol"] if s["ann_vol"] > 0 else 0.0
    s["calmar"] = s["ann_return"] / abs(s["max_dd"]) if s["max_dd"] < 0 else np.nan

    # --- Trade statistics ---
    wins = results[results["alpha"] > 0]["alpha"]
    losses = results[results["alpha"] <= 0]["alpha"]
    s["hit_rate"] = float((results["alpha"] > 0).mean())
    s["avg_win"] = float(wins.mean()) if len(wins) > 0 else 0.0
    s["avg_loss"] = float(losses.mean()) if len(losses) > 0 else 0.0
    total_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    total_loss = float(losses.abs().sum()) if len(losses) > 0 else 0.0
    s["profit_factor"] = total_profit / total_loss if total_loss > 0 else np.inf
    s["win_loss_ratio"] = abs(s["avg_win"] / s["avg_loss"]) if s["avg_loss"] != 0 else np.inf

    # --- IC ---
    if "ic_spearman" in results.columns:
        s["ic_mean"] = float(results["ic_spearman"].mean())
        ic_std = float(results["ic_spearman"].std())
        s["ic_ir"] = s["ic_mean"] / ic_std if ic_std > 0 else np.nan
    else:
        s["ic_mean"] = np.nan
        s["ic_ir"] = np.nan

    # --- Market capture ---
    up_mask = results["benchmark_return"] > 0
    down_mask = results["benchmark_return"] < 0
    if up_mask.sum() > 0:
        s["up_capture"] = float(results.loc[up_mask, "portfolio_return"].mean() / results.loc[up_mask, "benchmark_return"].mean())
    else:
        s["up_capture"] = np.nan
    if down_mask.sum() > 0:
        s["down_capture"] = float(results.loc[down_mask, "portfolio_return"].mean() / results.loc[down_mask, "benchmark_return"].mean())
    else:
        s["down_capture"] = np.nan

    # --- Rolling Sharpe ---
    s["rolling_sharpe_12"] = (
        results["portfolio_return"].rolling(12).mean()
        / results["portfolio_return"].rolling(12).std().replace(0, np.nan)
        * np.sqrt(12)
    )

    # --- Rolling Beta (portfolio vs benchmark) ---
    cov_window = 12
    port_r = results["portfolio_return"]
    bench_r = results["benchmark_return"]
    roll_cov = port_r.rolling(cov_window).cov(bench_r)
    roll_var = bench_r.rolling(cov_window).var()
    s["rolling_beta"] = (roll_cov / roll_var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    s["overall_beta"] = float(port_r.cov(bench_r) / bench_r.var()) if bench_r.var() > 0 else np.nan

    # --- Turnover ---
    s["avg_turnover"] = float(results["turnover"].mean()) if "turnover" in results.columns else np.nan
    s["total_tx_cost"] = float(results["transaction_cost"].sum()) if "transaction_cost" in results.columns else np.nan
    s["avg_cash_drag"] = float(results["cash_drag_pct"].mean()) if "cash_drag_pct" in results.columns else np.nan
    s["avg_sl_triggered"] = float(results["sl_triggered_rate"].mean()) if "sl_triggered_rate" in results.columns and results["sl_triggered_rate"].gt(0).any() else 0.0

    # --- Annual stats ---
    annual = results.groupby("year").agg(
        ann_port=("portfolio_return", lambda x: (1 + x).prod() - 1),
        ann_bench=("benchmark_return", lambda x: (1 + x).prod() - 1),
        ann_vol=("portfolio_return", lambda x: x.std() * np.sqrt(max(len(x), 1))),
    )
    annual["ann_alpha"] = annual["ann_port"] - annual["ann_bench"]
    annual["ann_sharpe"] = annual["ann_port"] / annual["ann_vol"].replace(0, np.nan)
    s["annual"] = annual

    # --- Quintile ---
    if {"q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"}.issubset(results.columns):
        s["q_means"] = results[["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"]].mean()
        s["q_mono"] = bool(
            s["q_means"]["q5_ret"] > s["q_means"]["q4_ret"] > s["q_means"]["q3_ret"]
            > s["q_means"]["q2_ret"] > s["q_means"]["q1_ret"]
        )
    else:
        s["q_means"] = None
        s["q_mono"] = False

    # --- Statistical Significance ---
    s["sig"] = _compute_stat_significance(s, results)

    return s


def _compute_stat_significance(s: dict, results: pd.DataFrame) -> dict:
    """Compute statistical significance metrics for the backtest.

    Returns a dict with t-stats, p-values, CIs, and an overall verdict.
    All tests are two-tailed unless noted.

    Tests performed
    ---------------
    1. Portfolio return t-stat (OLS):  H₀ = E[r] = 0
    2. Newey-West HAC t-stat:          Same H₀, corrected for autocorrelation
    3. Sharpe t-stat (Lo 2002):        t ≈ SR_period × √T
    4. IC t-stat:                      IC_IR × √n
    5. Bootstrap Sharpe 95% CI:        Percentile bootstrap (2000 draws)
    6. Binomial hit-rate test:         H₀ = P(alpha > 0) = 0.5  (one-tailed)
    """
    from scipy import stats as _stats

    sig: dict = {}
    port_r = results["portfolio_return"].dropna()
    alpha_r = results["alpha"].dropna() if "alpha" in results.columns else port_r
    n = len(port_r)
    rebals_per_year = max(n / max(s["n_years"], 1), 1)

    # ── 1. OLS t-stat on portfolio return ─────────────────────────────────
    r_mean = float(port_r.mean())
    r_std  = float(port_r.std(ddof=1))
    if r_std > 0 and n > 1:
        ols_tstat = r_mean / (r_std / np.sqrt(n))
        ols_pval  = 2.0 * float(_stats.t.sf(abs(ols_tstat), df=n - 1))
    else:
        ols_tstat = np.nan
        ols_pval  = np.nan
    sig["ols_tstat"] = ols_tstat
    sig["ols_pval"]  = ols_pval

    # ── 2. Newey-West HAC t-stat ──────────────────────────────────────────
    # Bartlett kernel with lag = ceil(4 × (n/100)^(2/9))  (Andrews 1991 rule)
    nw_lags = max(1, int(np.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    r_dm = (port_r - r_mean).values
    nw_var = float(np.mean(r_dm ** 2))
    for lag in range(1, nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)          # Bartlett weight
        gamma = float(np.mean(r_dm[lag:] * r_dm[:-lag]))
        nw_var += 2.0 * w * gamma
    nw_se = np.sqrt(max(nw_var, 0.0) / n)
    if nw_se > 0:
        nw_tstat = r_mean / nw_se
        nw_pval  = 2.0 * float(_stats.t.sf(abs(nw_tstat), df=n - 1))
    else:
        nw_tstat = np.nan
        nw_pval  = np.nan
    sig["nw_tstat"]  = nw_tstat
    sig["nw_pval"]   = nw_pval
    sig["nw_lags"]   = nw_lags

    # ── 3. Sharpe t-stat (Lo 2002 IID approximation) ──────────────────────
    # t ≈ SR_period × √T  where SR_period = mean/std per rebalance period
    sr_period = r_mean / r_std if r_std > 0 else np.nan
    if pd.notna(sr_period):
        sharpe_tstat = sr_period * np.sqrt(n)
        sharpe_pval  = 2.0 * float(_stats.t.sf(abs(sharpe_tstat), df=n - 1))
    else:
        sharpe_tstat = np.nan
        sharpe_pval  = np.nan
    sig["sharpe_tstat"] = sharpe_tstat
    sig["sharpe_pval"]  = sharpe_pval

    # ── 4. IC t-stat ──────────────────────────────────────────────────────
    if "ic_spearman" in results.columns:
        ic_series = results["ic_spearman"].dropna()
        ic_n = len(ic_series)
        ic_ir = s.get("ic_ir", np.nan)
        if pd.notna(ic_ir) and ic_n > 1:
            ic_tstat = float(ic_ir) * np.sqrt(ic_n)
            ic_pval  = 2.0 * float(_stats.t.sf(abs(ic_tstat), df=ic_n - 1))
        else:
            ic_tstat = np.nan
            ic_pval  = np.nan
        sig["ic_tstat"] = ic_tstat
        sig["ic_pval"]  = ic_pval
        sig["ic_n"]     = ic_n
    else:
        sig["ic_tstat"] = np.nan
        sig["ic_pval"]  = np.nan
        sig["ic_n"]     = 0

    # ── 5. Bootstrap Sharpe 95% CI (percentile bootstrap) ─────────────────
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_sharpes = np.empty(n_boot)
    r_arr = port_r.values
    for b in range(n_boot):
        sample = rng.choice(r_arr, size=n, replace=True)
        b_mean = sample.mean()
        b_std  = sample.std(ddof=1)
        if b_std > 0:
            b_ann_r = (1.0 + b_mean) ** rebals_per_year - 1.0
            b_ann_v = b_std * np.sqrt(rebals_per_year)
            boot_sharpes[b] = b_ann_r / b_ann_v
        else:
            boot_sharpes[b] = 0.0
    sig["sharpe_ci_lo"] = float(np.percentile(boot_sharpes, 2.5))
    sig["sharpe_ci_hi"] = float(np.percentile(boot_sharpes, 97.5))
    sig["sharpe_ci_pos"] = bool(sig["sharpe_ci_lo"] > 0)

    # ── 6. Binomial test: hit rate > 50% (one-tailed) ─────────────────────
    n_pos = int((alpha_r > 0).sum())
    n_tot = len(alpha_r)
    if n_tot > 0:
        binom_pval = float(_stats.binomtest(n_pos, n_tot, p=0.5, alternative="greater").pvalue)
    else:
        binom_pval = np.nan
    sig["binom_n_pos"]  = n_pos
    sig["binom_n_tot"]  = n_tot
    sig["binom_pval"]   = binom_pval

    # ── Verdict ──────────────────────────────────────────────────────────
    # Count how many core tests are significant at 5% (two-tailed)
    core_tstats = [v for k, v in sig.items() if k.endswith("_tstat") and pd.notna(v)]
    n_sig_5pct  = sum(abs(t) >= 1.96 for t in core_tstats)
    n_sig_1pct  = sum(abs(t) >= 2.576 for t in core_tstats)

    nw_ok     = pd.notna(nw_pval) and nw_pval < 0.05
    sharpe_ok = sig["sharpe_ci_pos"]
    ic_ok     = pd.notna(sig["ic_pval"]) and sig["ic_pval"] < 0.05
    binom_ok  = pd.notna(binom_pval) and binom_pval < 0.05

    checks_passed = sum([nw_ok, sharpe_ok, ic_ok, binom_ok])

    if n_sig_1pct >= 3 and checks_passed >= 3:
        verdict = "STRONG  ✅"
        verdict_note = "All major tests significant at 1%. Alpha is likely real."
    elif n_sig_5pct >= 2 and checks_passed >= 2:
        verdict = "MODERATE ⚠️"
        verdict_note = "Most tests significant at 5%. Promising but validate further."
    else:
        verdict = "WEAK  ❌"
        verdict_note = "Insufficient evidence. Results may be noise or overfit."

    sig["verdict"]      = verdict
    sig["verdict_note"] = verdict_note
    sig["n_sig_5pct"]   = n_sig_5pct
    sig["n_sig_1pct"]   = n_sig_1pct

    return sig


def _compute_performance(returns: pd.Series, years: pd.Series | None = None) -> dict:
    """Compute annualized return/vol/sharpe for a return series."""
    r = returns.dropna()
    if r.empty:
        return {"total_return": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}

    if years is not None:
        y = years.loc[r.index]
        n_years = max(int(y.nunique()), 1)
    else:
        n_years = 1
    rebals_per_year = max(len(r) / n_years, 1)

    total_return = float((1.0 + r).prod() - 1.0)
    ann_return = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)
    ann_vol = float(r.std() * np.sqrt(rebals_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


def _parse_exclude_years(raw: str) -> set[str]:
    years: set[str] = set()
    if not raw:
        return years
    for token in raw.split(","):
        t = token.strip()
        if len(t) == 4 and t.isdigit():
            years.add(t)
    return years


def _format_sector_names(names) -> dict:
    """Map sector names to display names.

    Sectors are now industry_name strings from financial_periods and are already
    human-readable. Handle the legacy unmapped sentinel just in case.
    """
    def _strip(name: str) -> str:
        if name in ("UNMAPPED_SECTOR_INDEX", "UNMAPPED_SECTOR"):
            return "UNMAPPED"
        # Strip legacy index-code prefixes if present (backwards compatibility)
        for prefix in ["KOSPI_코스피_200_", "KOSPI_코스피_", "KOSDAQ_코스닥_", "KOSDAQ_", "KOSPI_"]:
            if name.startswith(prefix):
                return name[len(prefix):].replace("_", " ")
        return name

    return {n: _strip(n) for n in names}


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Pretty-print table-like outputs with prettytable."""
    print(f"\n{title:^70}")
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Fallback if dependency is missing.
        print("  (prettytable not installed)")
        print("  " + " | ".join(headers))
        for row in rows:
            print("  " + " | ".join(row))
        return

    table = PrettyTable()
    table.field_names = headers
    table.align = "l"
    for row in rows:
        table.add_row(row)
    print(table)


def _print_requested_tests(results: pd.DataFrame) -> None:
    """Run and print the 4 requested follow-up tests."""
    if results.empty:
        return

    print("\n" + "=" * 70)
    print("  REQUESTED TESTS")
    print("=" * 70)

    years = results["year"] if "year" in results.columns else None

    test_rows: list[list[str]] = []

    # 1) Long-short decile spread
    if "long_short_return" in results.columns:
        ls = _compute_performance(results["long_short_return"], years)
        test_rows.append([
            "1",
            "Long-Short (Top 10% - Bottom 10%)",
            f"{ls['ann_return']:.2%}",
            f"{ls['sharpe']:.2f}",
            "OK",
        ])
    else:
        test_rows.append(["1", "Long-Short (Top 10% - Bottom 10%)", "N/A", "N/A", "Unavailable"])

    # 2) Beta hedge test (market-neutralized by realized beta)
    if {"portfolio_return", "benchmark_return"}.issubset(results.columns):
        bench = results["benchmark_return"]
        port = results["portfolio_return"]
        beta = float(port.cov(bench) / bench.var()) if bench.var() > 0 else np.nan
        hedged = port - beta * bench if pd.notna(beta) else port - bench
        hedged_stats = _compute_performance(hedged, years)
        beta_str = f"{beta:.2f}" if pd.notna(beta) else "N/A"
        test_rows.append([
            "2",
            f"Beta-Hedged (beta={beta_str})",
            f"{hedged_stats['ann_return']:.2%}",
            f"{hedged_stats['sharpe']:.2f}",
            "OK",
        ])
    else:
        test_rows.append(["2", "Beta-Hedged", "N/A", "N/A", "Unavailable"])

    # 3) Remove 2023 and re-check sharpe on remaining periods
    if "year" in results.columns:
        ex_2023 = results[results["year"] != 2023].copy()
        if not ex_2023.empty:
            ex_stats = _compute_performance(ex_2023["portfolio_return"], ex_2023["year"])
            verdict = "PASS" if pd.notna(ex_stats["sharpe"]) and ex_stats["sharpe"] >= 0.7 else "FAIL"
            test_rows.append([
                "3",
                "Ex-2023 robustness",
                f"{ex_stats['ann_return']:.2%}",
                f"{ex_stats['sharpe']:.2f}",
                f"{verdict} (>=0.70)",
            ])
        else:
            test_rows.append(["3", "Ex-2023 robustness", "N/A", "N/A", "Unavailable"])
    else:
        test_rows.append(["3", "Ex-2023 robustness", "N/A", "N/A", "Unavailable"])

    # 4) Turnover reduction variant (relaxed hold threshold + score smoothing)
    required_cols = {"turnover_tuned", "transaction_cost_tuned", "portfolio_return_tuned"}
    if required_cols.issubset(results.columns):
        base_stats = _compute_performance(results["portfolio_return"], years)
        tuned_stats = _compute_performance(results["portfolio_return_tuned"], years)
        base_turnover = float(results["turnover"].mean()) if "turnover" in results.columns else np.nan
        tuned_turnover = float(results["turnover_tuned"].mean())
        base_cost = float(results["transaction_cost"].sum()) if "transaction_cost" in results.columns else np.nan
        tuned_cost = float(results["transaction_cost_tuned"].sum())
        test_rows.append([
            "4",
            f"Turnover reduction ({base_turnover:.2%}->{tuned_turnover:.2%})",
            f"{base_cost:.2%}->{tuned_cost:.2%}",
            f"{base_stats['sharpe']:.2f}->{tuned_stats['sharpe']:.2f}",
            "OK",
        ])
    else:
        test_rows.append(["4", "Turnover reduction", "N/A", "N/A", "Unavailable"])

    _print_table(
        "--- Requested Tests ---",
        ["#", "Test", "Return / Cost", "Sharpe", "Status"],
        test_rows,
    )


def summarize(results: pd.DataFrame, sector_rows: list, output_path: str = "backtest_report.png", model=None) -> None:
    """Print enhanced summary + generate visual report."""
    if results.empty:
        print("No backtest results were generated.")
        return

    s = _compute_core_stats(results)
    sector_df = pd.DataFrame(sector_rows) if sector_rows else pd.DataFrame()

    # ======================================================================
    # TEXT SUMMARY
    # ======================================================================
    print("\n" + "=" * 70)
    print("  BACKTEST REPORT")
    print("=" * 70)

    print(f"\n{'--- Overview ---':^70}")
    print(f"  Rebalances: {s['n_rebalances']}  |  Years: {s['n_years']}")
    print(f"  Total Return:     {s['total_return']:>8.2%}   Benchmark: {s['benchmark_return']:>8.2%}")
    print(f"  Alpha:            {s['alpha']:>8.2%}   Hit Rate:  {s['hit_rate']:>8.2%}")
    print(f"  Ann. Return:      {s['ann_return']:>8.2%}   Ann. Vol:  {s['ann_vol']:>8.2%}")
    print(f"  Sharpe:           {s['sharpe']:>8.2f}   Calmar:    {s['calmar']:>8.2f}" if pd.notna(s['calmar']) else f"  Sharpe:           {s['sharpe']:>8.2f}   Calmar:        N/A")
    print(f"  Max Drawdown:     {s['max_dd']:>8.2%}   Max Underwater: {s['max_underwater']} rebals")

    print(f"\n{'--- Trade Statistics (매매 세부 지표) ---':^70}")
    print(f"  Win Rate:         {s['hit_rate']:>8.2%}   (벤치마크 대비 이긴 비율)")
    pf_str = f"{s['profit_factor']:.2f}" if np.isfinite(s['profit_factor']) else "INF"
    print(f"  Profit Factor:    {pf_str:>8s}   (총이익/총손실, 1.5+ 우수, 2.0+ 성배)")
    wl_str = f"{s['win_loss_ratio']:.2f}" if np.isfinite(s['win_loss_ratio']) else "INF"
    print(f"  Win/Loss Ratio:   {wl_str:>8s}   (평균이익/평균손실)")
    print(f"  Avg Win:          {s['avg_win']:>8.2%}   Avg Loss:  {s['avg_loss']:>8.2%}")
    if pd.notna(s['avg_turnover']):
        print(f"  Avg Turnover:     {s['avg_turnover']:>8.2%}   Total Tx Cost: {s['total_tx_cost']:>8.2%}")
    if pd.notna(s.get("avg_cash_drag")):
        print(f"  Avg Cash Drag:    {s['avg_cash_drag']:>8.2%}   (uninvested cash incl. rounding + regime)")
    if s.get("avg_sl_triggered", 0) > 0:
        print(f"  Avg SL Triggered: {s['avg_sl_triggered']:>8.2%}   (% of picks hitting stop-loss per rebalance)")

    print(f"\n{'--- Market Regime Analysis (하락장 방어력) ---':^70}")
    uc_str = f"{s['up_capture']:.2f}" if pd.notna(s['up_capture']) else "N/A"
    dc_str = f"{s['down_capture']:.2f}" if pd.notna(s['down_capture']) else "N/A"
    print(f"  Up Capture:       {uc_str:>8s}   (시장 1% 상승시 포트폴리오 변동)")
    print(f"  Down Capture:     {dc_str:>8s}   (0.7 이하면 우수한 방어력)")
    beta_str = f"{s['overall_beta']:.2f}" if pd.notna(s['overall_beta']) else "N/A"
    print(f"  Overall Beta:     {beta_str:>8s}   (0.5 이하면 독자적 알파)")

    print(f"\n{'--- IC & Quintile ---':^70}")
    if pd.notna(s['ic_mean']):
        ic_ir_str = f"{s['ic_ir']:.2f}" if pd.notna(s['ic_ir']) else "N/A"
        print(f"  Mean IC:          {s['ic_mean']:>8.4f}   IC IR:     {ic_ir_str:>8s}")
    if s['q_means'] is not None:
        q_str = "  ".join([f"Q{i}={s['q_means'][f'q{i}_ret']:.2%}" for i in range(1, 6)])
        print(f"  Quintile: {q_str}")
        print(f"  Monotonic: {'PASS' if s['q_mono'] else 'FAIL'}")

    annual_rows: list[list[str]] = []
    for yr, row in s["annual"].iterrows():
        sh = f"{row['ann_sharpe']:.2f}" if pd.notna(row["ann_sharpe"]) else "N/A"
        annual_rows.append([str(int(yr)), f"{row['ann_port']:.2%}", f"{row['ann_alpha']:.2%}", sh])
    _print_table(
        "--- Annual Sharpe (연도별 안정성) ---",
        ["Year", "Return", "Alpha", "Sharpe"],
        annual_rows,
    )

    # Sector attribution text
    if not sector_df.empty:
        sec_agg = sector_df.groupby("sector").agg(
            total_contribution=("contribution", "sum"),
            avg_weight=("weight", "mean"),
            appearances=("date", "count"),
        ).sort_values("total_contribution", ascending=False)
        total_contrib = sec_agg["total_contribution"].sum()
        display_names = _format_sector_names(sec_agg.head(10).index.tolist())
        sec_rows: list[list[str]] = []
        for sec_name, row in sec_agg.head(10).iterrows():
            pct = row["total_contribution"] / total_contrib * 100 if total_contrib != 0 else 0
            short_name = display_names[sec_name]
            sec_rows.append([
                str(short_name),
                f"{row['total_contribution']:.2%}",
                f"{pct:.1f}%",
                f"{row['avg_weight']:.1%}",
            ])
        _print_table(
            "--- Sector Attribution (섹터 귀인 분석) ---",
            ["Sector", "Contribution", "Share", "Avg Weight"],
            sec_rows,
        )

    # Feature group importance
    if model is not None:
        try:
            from ml.features.registry import get_feature_group_map
            imp_df = model.feature_importance()
            group_map = get_feature_group_map()
            imp_df["group"] = imp_df["feature"].map(group_map).fillna("other")
            group_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)
            total = group_imp.sum()
            top_features = (
                imp_df.sort_values("importance", ascending=False)
                .groupby("group")
                .head(3)
                .groupby("group")["feature"]
                .apply(list)
            )
            fg_rows: list[list[str]] = []
            for grp, imp in group_imp.items():
                pct = imp / total * 100 if total > 0 else 0.0
                bar = "█" * int(pct / 100 * 30)
                feats = ", ".join(top_features.get(grp, [])[:3])
                fg_rows.append([grp, f"{pct:.1f}%", bar, feats])
            _print_table(
                "--- Feature Group Importance (Gain) ---",
                ["Group", "Share", "Bar", "Top Features"],
                fg_rows,
            )
        except Exception as _e:
            print(f"  [Feature Importance] skipped: {_e}")

    # --- Statistical Significance ---
    sig = s.get("sig", {})
    if sig:
        def _fmt_t(t, p) -> str:
            if not pd.notna(t):
                return "N/A"
            stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            return f"{t:+.2f}  (p={p:.3f}) {stars}"

        print(f"\n{'--- Statistical Significance (통계적 유의성) ---':^70}")
        print(f"  Observations (N):     {sig.get('ic_n', s['n_rebalances'])} IC periods  |  "
              f"{s['n_rebalances']} portfolio rebalances")
        print(f"")
        print(f"  ① OLS t-stat  (H₀: mean return = 0):")
        print(f"       t = {_fmt_t(sig.get('ols_tstat'), sig.get('ols_pval', 1))}")
        print(f"  ② Newey-West HAC t-stat  (autocorr-adjusted, lags={sig.get('nw_lags','?')}):")
        print(f"       t = {_fmt_t(sig.get('nw_tstat'), sig.get('nw_pval', 1))}")
        print(f"       → NW is the most reliable t-stat when returns are autocorrelated")
        print(f"  ③ Sharpe t-stat  (Lo 2002 IID approx, t = SR_period × √N):")
        print(f"       t = {_fmt_t(sig.get('sharpe_tstat'), sig.get('sharpe_pval', 1))}")
        ci_lo = sig.get("sharpe_ci_lo", np.nan)
        ci_hi = sig.get("sharpe_ci_hi", np.nan)
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if pd.notna(ci_lo) else "N/A"
        ci_flag = "✅ entirely positive" if sig.get("sharpe_ci_pos") else "⚠️  spans zero"
        print(f"  ④ Bootstrap Sharpe 95% CI (2000 draws):")
        print(f"       {ci_str}  {ci_flag}")
        print(f"  ⑤ IC t-stat  (IC_IR × √N):")
        print(f"       t = {_fmt_t(sig.get('ic_tstat'), sig.get('ic_pval', 1))}")
        n_pos = sig.get("binom_n_pos", 0)
        n_tot = sig.get("binom_n_tot", 1)
        hr_pct = n_pos / n_tot if n_tot > 0 else np.nan
        print(f"  ⑥ Binomial test  (H₀: hit rate = 50%, one-tailed):")
        bp = sig.get("binom_pval", 1.0)
        bstar = "***" if bp < 0.01 else ("**" if bp < 0.05 else ("*" if bp < 0.10 else ""))
        print(f"       {n_pos}/{n_tot} periods beat benchmark ({hr_pct:.1%}),  p={bp:.3f} {bstar}")
        print(f"")
        print(f"  Significance summary:  {sig.get('n_sig_5pct', 0)} / {len([k for k in sig if k.endswith('_tstat')])} t-tests pass at 5%  |  "
              f"{sig.get('n_sig_1pct', 0)} pass at 1%")
        print(f"  ★ VERDICT: {sig.get('verdict', 'N/A')}")
        print(f"    {sig.get('verdict_note', '')}")
        print(f"  Note: *** p<0.01  ** p<0.05  * p<0.10  |  "
              f"Critical t-values: 1.96 (5%) / 2.576 (1%)")

    print("\n" + "=" * 70)

    # ======================================================================
    # VISUAL REPORT
    # ======================================================================
    _generate_visual_report(results, s, sector_df, output_path)
    _print_requested_tests(results)


def _generate_visual_report(results: pd.DataFrame, s: dict, sector_df: pd.DataFrame, output_path: str) -> None:
    """Generate a multi-panel PNG report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("[Report] matplotlib not installed, skipping visual report.")
        return

    # Try Korean font
    try:
        import matplotlib.font_manager as fm
        korean_fonts = [f.name for f in fm.fontManager.ttflist if any(
            k in f.name for k in ["Nanum", "Malgun", "Apple SD", "NanumGothic", "AppleGothic"]
        )]
        if korean_fonts:
            plt.rcParams["font.family"] = korean_fonts[0]
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    # Color palette
    C_PORT = "#2563EB"   # blue
    C_BENCH = "#9CA3AF"  # gray
    C_ALPHA = "#10B981"  # green
    C_NEG = "#EF4444"    # red
    C_WARN = "#F59E0B"   # amber
    C_BG = "#F8FAFC"

    fig = plt.figure(figsize=(22, 28), facecolor="white", dpi=100)
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.28,
                  left=0.06, right=0.96, top=0.95, bottom=0.03)

    dates = pd.to_datetime(results["date"], format="%Y%m%d", errors="coerce")

    # ── Panel 1: Cumulative Returns + Drawdown ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(C_BG)
    ax1.plot(dates, s["cum_portfolio"], color=C_PORT, linewidth=2, label="Portfolio")
    ax1.plot(dates, s["cum_benchmark"], color=C_BENCH, linewidth=1.5, label="Benchmark", linestyle="--")
    ax1.fill_between(dates, s["cum_portfolio"], s["cum_benchmark"],
                     where=s["cum_portfolio"] >= s["cum_benchmark"],
                     alpha=0.15, color=C_ALPHA, interpolate=True)
    ax1.fill_between(dates, s["cum_portfolio"], s["cum_benchmark"],
                     where=s["cum_portfolio"] < s["cum_benchmark"],
                     alpha=0.15, color=C_NEG, interpolate=True)
    ax1.set_title("Cumulative Returns (Portfolio vs Benchmark)", fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax1.grid(True, alpha=0.3)
    # Drawdown on twin axis
    ax1b = ax1.twinx()
    ax1b.fill_between(dates, s["drawdown"], 0, alpha=0.25, color=C_NEG, label="Drawdown")
    ax1b.set_ylim(min(s["drawdown"].min() * 1.3, -0.05), 0.02)
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1b.set_ylabel("Drawdown", fontsize=10, color=C_NEG)
    ax1b.tick_params(axis="y", colors=C_NEG)

    # ── Panel 2: Annual Performance ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(C_BG)
    annual = s["annual"]
    yrs = annual.index.astype(str)
    x = np.arange(len(yrs))
    w = 0.35
    bars_port = ax2.bar(x - w / 2, annual["ann_port"] * 100, w, color=C_PORT, label="Portfolio", zorder=3)
    bars_bench = ax2.bar(x + w / 2, annual["ann_bench"] * 100, w, color=C_BENCH, label="Benchmark", zorder=3)
    # Alpha markers
    for i, (yr, row) in enumerate(annual.iterrows()):
        color = C_ALPHA if row["ann_alpha"] > 0 else C_NEG
        ax2.annotate(f"{row['ann_alpha']:+.1%}", (i, max(row['ann_port'], row['ann_bench']) * 100 + 1),
                     ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(yrs, rotation=45, fontsize=8)
    ax2.set_title("Annual Returns & Alpha", fontsize=13, fontweight="bold")
    ax2.set_ylabel("%", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Annual Sharpe ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C_BG)
    sharpe_vals = annual["ann_sharpe"].fillna(0)
    colors_sharpe = [C_ALPHA if v > 0.5 else (C_WARN if v > 0 else C_NEG) for v in sharpe_vals]
    ax3.bar(yrs, sharpe_vals, color=colors_sharpe, zorder=3)
    ax3.axhline(0.5, color=C_ALPHA, linewidth=1, linestyle="--", alpha=0.7, label="Sharpe=0.5")
    ax3.axhline(0, color="black", linewidth=0.5)
    for i, v in enumerate(sharpe_vals):
        ax3.annotate(f"{v:.2f}", (i, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax3.set_xticks(range(len(yrs)))
    ax3.set_xticklabels(yrs, rotation=45, fontsize=8)
    ax3.set_title("Annual Sharpe Ratio (연도별 안정성)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Trade Statistics ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(C_BG)
    ax4.axis("off")
    pf_val = s["profit_factor"]
    pf_color = C_ALPHA if pf_val >= 1.5 else (C_WARN if pf_val >= 1.0 else C_NEG)
    wr_color = C_ALPHA if s["hit_rate"] >= 0.5 else C_WARN
    dc_val = s["down_capture"]
    dc_color = C_ALPHA if (pd.notna(dc_val) and dc_val < 0.7) else C_WARN

    stats_lines = [
        ("Trade Statistics (매매 세부 지표)", "", 16, "bold", "black"),
        ("", "", 6, "normal", "black"),
        ("Win Rate (승률)", f"{s['hit_rate']:.1%}", 14, "bold", wr_color),
        ("Profit Factor (수익인자)", f"{pf_val:.2f}" if np.isfinite(pf_val) else "INF", 14, "bold", pf_color),
        ("Win/Loss Ratio (손익비)", f"{s['win_loss_ratio']:.2f}" if np.isfinite(s['win_loss_ratio']) else "INF", 14, "bold", C_PORT),
        ("Avg Win", f"{s['avg_win']:+.2%}", 12, "normal", C_ALPHA),
        ("Avg Loss", f"{s['avg_loss']:+.2%}", 12, "normal", C_NEG),
        ("", "", 10, "normal", "black"),
        ("Market Regime (하락장 방어력)", "", 16, "bold", "black"),
        ("", "", 6, "normal", "black"),
        ("Up Capture", f"{s['up_capture']:.2f}" if pd.notna(s['up_capture']) else "N/A", 14, "bold", C_PORT),
        ("Down Capture", f"{dc_val:.2f}" if pd.notna(dc_val) else "N/A", 14, "bold", dc_color),
        ("Overall Beta", f"{s['overall_beta']:.2f}" if pd.notna(s['overall_beta']) else "N/A", 14, "bold", C_PORT),
    ]
    y_pos = 0.97
    for label, value, fsize, fweight, color in stats_lines:
        if value:
            ax4.text(0.05, y_pos, label, fontsize=fsize, fontweight="normal",
                     transform=ax4.transAxes, va="top")
            ax4.text(0.95, y_pos, value, fontsize=fsize, fontweight=fweight, color=color,
                     transform=ax4.transAxes, va="top", ha="right")
        else:
            ax4.text(0.05, y_pos, label, fontsize=fsize, fontweight=fweight, color=color,
                     transform=ax4.transAxes, va="top")
        y_pos -= 0.075 if fsize >= 14 else 0.04

    # ── Panel 5: Quintile Returns ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(C_BG)
    if s["q_means"] is not None:
        q_vals = [s["q_means"][f"q{i}_ret"] * 100 for i in range(1, 6)]
        q_labels = ["Q1\n(Worst)", "Q2", "Q3", "Q4", "Q5\n(Best)"]
        q_colors = [C_NEG, "#F97316", C_WARN, "#34D399", C_ALPHA]
        bars = ax5.bar(q_labels, q_vals, color=q_colors, zorder=3, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, q_vals):
            ax5.annotate(f"{val:.2f}%", (bar.get_x() + bar.get_width() / 2, val),
                         ha="center", va="bottom" if val >= 0 else "top",
                         fontsize=11, fontweight="bold")
        ax5.axhline(0, color="black", linewidth=0.5)
        mono_text = "MONOTONIC" if s["q_mono"] else "NOT MONOTONIC"
        mono_color = C_ALPHA if s["q_mono"] else C_NEG
        ax5.set_title(f"Quintile Returns - {mono_text}", fontsize=13, fontweight="bold", color=mono_color)
        ax5.set_ylabel("Mean Forward Return (%)", fontsize=10)
    else:
        ax5.text(0.5, 0.5, "No quintile data", ha="center", va="center", fontsize=14)
        ax5.set_title("Quintile Returns", fontsize=13, fontweight="bold")
    ax5.grid(True, axis="y", alpha=0.3)

    # ── Panel 6: Rolling Sharpe + Rolling Beta ──
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor(C_BG)
    roll_sh = s["rolling_sharpe_12"]
    valid = roll_sh.notna()
    if valid.any():
        ax6.plot(dates[valid], roll_sh[valid], color=C_PORT, linewidth=1.5, label="Rolling Sharpe (12p)")
        ax6.axhline(0, color="black", linewidth=0.5)
        ax6.axhline(0.5, color=C_ALPHA, linewidth=1, linestyle="--", alpha=0.5)
        ax6.fill_between(dates[valid], roll_sh[valid], 0,
                         where=roll_sh[valid] > 0, alpha=0.15, color=C_ALPHA, interpolate=True)
        ax6.fill_between(dates[valid], roll_sh[valid], 0,
                         where=roll_sh[valid] <= 0, alpha=0.15, color=C_NEG, interpolate=True)
    # Rolling beta on twin
    roll_b = s["rolling_beta"]
    valid_b = roll_b.notna()
    if valid_b.any():
        ax6b = ax6.twinx()
        ax6b.plot(dates[valid_b], roll_b[valid_b], color=C_WARN, linewidth=1, alpha=0.7, label="Rolling Beta")
        ax6b.axhline(1.0, color=C_WARN, linewidth=0.8, linestyle=":", alpha=0.5)
        ax6b.set_ylabel("Beta", fontsize=10, color=C_WARN)
        ax6b.tick_params(axis="y", colors=C_WARN)
        ax6b.set_ylim(-0.5, 2.5)
    ax6.set_title("Rolling Sharpe & Beta (시장 상관 변화)", fontsize=13, fontweight="bold")
    ax6.set_ylabel("Sharpe", fontsize=10)
    ax6.legend(loc="upper left", fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── Panel 7: Sector Attribution ──
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor(C_BG)
    if not sector_df.empty:
        sec_agg = sector_df.groupby("sector")["contribution"].sum().sort_values(ascending=True)
        top_sec = sec_agg.tail(12)
        display_names_chart = _format_sector_names(top_sec.index.tolist())
        short_names = [display_names_chart[n] for n in top_sec.index]
        colors_sec = [C_ALPHA if v > 0 else C_NEG for v in top_sec.values]
        ax7.barh(short_names, top_sec.values * 100, color=colors_sec, zorder=3, height=0.7)
        ax7.axvline(0, color="black", linewidth=0.5)
        for i, v in enumerate(top_sec.values):
            ax7.annotate(f"{v:.2%}", (v * 100, i), ha="left" if v > 0 else "right",
                         va="center", fontsize=8, fontweight="bold")
        ax7.set_title("Sector Attribution (섹터별 기여도)", fontsize=13, fontweight="bold")
        ax7.set_xlabel("Cumulative Contribution (%)", fontsize=10)

        # HHI annotation
        if "sector_hhi" in results.columns:
            avg_hhi = results["sector_hhi"].mean()
            conc_label = "HIGH" if avg_hhi > 0.25 else ("MED" if avg_hhi > 0.15 else "LOW")
            conc_color = C_NEG if avg_hhi > 0.25 else (C_WARN if avg_hhi > 0.15 else C_ALPHA)
            ax7.annotate(f"Avg HHI={avg_hhi:.3f} ({conc_label})", (0.98, 0.02),
                         xycoords="axes fraction", ha="right", va="bottom",
                         fontsize=10, fontweight="bold", color=conc_color,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax7.text(0.5, 0.5, "No sector data", ha="center", va="center", fontsize=14)
        ax7.set_title("Sector Attribution", fontsize=13, fontweight="bold")
    ax7.grid(True, axis="x", alpha=0.3)

    # ── Panel 8: Up/Down Market Scatter ──
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.set_facecolor(C_BG)
    bench_pct = results["benchmark_return"] * 100
    port_pct = results["portfolio_return"] * 100
    up = results["benchmark_return"] > 0
    ax8.scatter(bench_pct[up], port_pct[up], color=C_ALPHA, alpha=0.5, s=30, label="Bull rebalance", zorder=3)
    ax8.scatter(bench_pct[~up], port_pct[~up], color=C_NEG, alpha=0.5, s=30, label="Bear rebalance", zorder=3)
    lims = [min(bench_pct.min(), port_pct.min()) - 1, max(bench_pct.max(), port_pct.max()) + 1]
    ax8.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="y=x")
    ax8.set_xlim(lims)
    ax8.set_ylim(lims)
    ax8.set_xlabel("Benchmark Return (%)", fontsize=10)
    ax8.set_ylabel("Portfolio Return (%)", fontsize=10)
    ax8.set_title("Up/Down Capture (시장 상승 vs 하락 구간)", fontsize=13, fontweight="bold")
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # ── Panel 9: IC time series ──
    ax9 = fig.add_subplot(gs[4, 1])
    ax9.set_facecolor(C_BG)
    if "ic_spearman" in results.columns:
        ic = results["ic_spearman"]
        ic_colors = [C_ALPHA if v > 0 else C_NEG for v in ic]
        ax9.bar(dates, ic, color=ic_colors, width=20, alpha=0.6, zorder=3)
        ic_rolling = ic.rolling(6).mean()
        valid_ic = ic_rolling.notna()
        if valid_ic.any():
            ax9.plot(dates[valid_ic], ic_rolling[valid_ic], color=C_PORT, linewidth=2, label="6-period MA")
        ax9.axhline(0, color="black", linewidth=0.5)
        ax9.axhline(s["ic_mean"], color=C_WARN, linewidth=1, linestyle="--", alpha=0.7,
                     label=f"Mean IC={s['ic_mean']:.4f}")
        ax9.set_title("Spearman IC Over Time (예측력 변화)", fontsize=13, fontweight="bold")
        ax9.set_ylabel("IC", fontsize=10)
        ax9.legend(fontsize=9)
    else:
        ax9.text(0.5, 0.5, "No IC data", ha="center", va="center", fontsize=14)
        ax9.set_title("Spearman IC", fontsize=13, fontweight="bold")
    ax9.grid(True, alpha=0.3)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved visual report to {out}")


def _generate_picks_chart(picks_df: pd.DataFrame, fwd_col: str, output_path: str) -> None:
    """Generate a multi-panel chart showing portfolio holdings over time.

    Panel 1 — Holdings Heatmap: top 30 most-selected stocks × rebalance dates,
               cell color = realized forward return (green=profit, red=loss, grey=not held).
    Panel 2 — Sector Allocation: stacked bar showing sector weight per rebalance date.
    Panel 3 — Top-10 Stock Contribution: cumulative realized returns for the most-held stocks.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Picks Chart] matplotlib not installed, skipping.")
        return

    # Korean font setup (mirrors _generate_visual_report)
    try:
        import matplotlib.font_manager as fm
        korean_fonts = [f.name for f in fm.fontManager.ttflist if any(
            k in f.name for k in ["Nanum", "Malgun", "Apple SD", "NanumGothic", "AppleGothic"]
        )]
        if korean_fonts:
            plt.rcParams["font.family"] = korean_fonts[0]
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    if picks_df.empty:
        print("[Picks Chart] No picks data. Skipping.")
        return

    picks_df = picks_df.copy()
    picks_df["date_dt"] = pd.to_datetime(picks_df["date"], format="%Y%m%d", errors="coerce")

    # ── Derive display labels ──────────────────────────────────────────────────
    if "name" in picks_df.columns:
        name_map = picks_df.groupby("stock_code")["name"].first().to_dict()
    else:
        name_map = {}

    # ── Sorted dates for x-axis ────────────────────────────────────────────────
    all_dates = sorted(picks_df["date"].unique())
    date_labels = [f"{d[:4]}\n{d[4:6]}/{d[6:8]}" for d in all_dates]

    # ── Panel 1: Holdings heatmap ──────────────────────────────────────────────
    has_return = fwd_col in picks_df.columns
    if has_return:
        pivot = picks_df.pivot_table(
            index="stock_code", columns="date", values=fwd_col, aggfunc="first"
        )
    else:
        pivot = picks_df.pivot_table(
            index="stock_code", columns="date", values="score_rank", aggfunc="first"
        )

    freq = (~pivot.isna()).sum(axis=1)
    n_show = min(30, len(pivot))
    top_stocks = freq.sort_values(ascending=False).head(n_show).index
    pivot = pivot.reindex(columns=all_dates).loc[top_stocks]

    y_labels = []
    for code in top_stocks:
        name = name_map.get(code, "")
        name_short = str(name)[:10] if name else ""
        y_labels.append(f"{code}  {name_short}")

    # ── Panel 2: Sector allocation ─────────────────────────────────────────────
    has_sector = "sector" in picks_df.columns
    if has_sector:
        sector_counts = (
            picks_df.groupby(["date", "sector"])
            .size()
            .reset_index(name="count")
        )
        sector_pivot = sector_counts.pivot_table(
            index="date", columns="sector", values="count", fill_value=0
        )
        # Normalise to weight
        sector_pivot = sector_pivot.div(sector_pivot.sum(axis=1), axis=0).fillna(0)
        # Show top-8 sectors; collapse rest into "Other"
        top_sec = sector_pivot.sum().sort_values(ascending=False).head(8).index
        other_cols = [c for c in sector_pivot.columns if c not in top_sec]
        if other_cols:
            sector_pivot["Other"] = sector_pivot[other_cols].sum(axis=1)
            sector_pivot = sector_pivot.drop(columns=other_cols)
        sector_pivot = sector_pivot.reindex(all_dates).fillna(0)
        sector_names = [c.split("_")[-1][:14] for c in sector_pivot.columns]

    # ── Panel 3: Per-stock cumulative return ───────────────────────────────────
    show_stock_perf = has_return and len(top_stocks) > 0
    if show_stock_perf:
        top10 = freq.sort_values(ascending=False).head(10).index
        stock_date_ret = picks_df[picks_df["stock_code"].isin(top10)].copy()
        stock_date_ret = stock_date_ret.sort_values("date_dt")

    # ── Figure layout ──────────────────────────────────────────────────────────
    n_panels = 2 + (1 if show_stock_perf else 0)
    heights = [max(n_show * 0.32, 5), 3.5]
    if show_stock_perf:
        heights.append(3.5)
    fig_height = sum(heights) + 1.5 * n_panels
    fig = plt.figure(figsize=(22, fig_height), facecolor="white", dpi=100)
    gs = GridSpec(
        n_panels, 1, figure=fig,
        height_ratios=heights,
        hspace=0.45,
        left=0.14, right=0.97, top=0.96, bottom=0.04,
    )

    C_BG = "#F8FAFC"
    PALETTE = [
        "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
    ]

    # ── Plot 1: Heatmap ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)

    heat_data = pivot.values.astype(float)
    if has_return:
        vmax = min(float(np.nanpercentile(np.abs(heat_data[~np.isnan(heat_data)]), 95)), 0.30) if not np.all(np.isnan(heat_data)) else 0.20
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = plt.cm.RdYlGn
    else:
        norm = None
        cmap = plt.cm.Blues

    # Grey background for "not held" cells
    not_held = np.full_like(heat_data, np.nan)
    held_mask = ~np.isnan(heat_data)
    not_held[~held_mask] = 0.0

    im_bg = ax1.imshow(
        np.zeros_like(heat_data),
        aspect="auto", cmap=plt.cm.Greys, vmin=0, vmax=1, alpha=0.07,
    )
    # Draw "not held" as light grey
    bg_arr = np.where(~held_mask, 0.5, np.nan)
    ax1.imshow(bg_arr, aspect="auto", cmap=plt.cm.Greys, vmin=0, vmax=1, alpha=0.25)

    if has_return and not np.all(np.isnan(heat_data)):
        im = ax1.imshow(heat_data, aspect="auto", cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.015)
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        cbar.set_label("Realized Return", fontsize=9)

    # Annotate cells with return %
    if has_return:
        for row_i in range(heat_data.shape[0]):
            for col_j in range(heat_data.shape[1]):
                val = heat_data[row_i, col_j]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > vmax * 0.55 else "black"
                    ax1.text(
                        col_j, row_i, f"{val:.1%}",
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color=txt_color,
                    )

    ax1.set_yticks(range(len(y_labels)))
    ax1.set_yticklabels(y_labels, fontsize=8.5)
    ax1.set_xticks(range(len(all_dates)))
    ax1.set_xticklabels(date_labels, fontsize=8, rotation=0)
    ax1.set_title(
        f"Holdings Heatmap — Top {n_show} Most-Selected Stocks  (green=profit, red=loss, grey=not held)",
        fontsize=13, fontweight="bold", pad=8,
    )
    ax1.set_xlabel("Rebalance Date", fontsize=10)

    # ── Plot 2: Sector allocation ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C_BG)
    if has_sector:
        bottoms = np.zeros(len(all_dates))
        x = np.arange(len(all_dates))
        for i, sec in enumerate(sector_pivot.columns):
            vals = sector_pivot.loc[all_dates].values[:, i] if all_dates[0] in sector_pivot.index else np.zeros(len(all_dates))
            ax2.bar(x, sector_pivot[sec].reindex(all_dates).fillna(0).values,
                    bottom=bottoms, color=PALETTE[i % len(PALETTE)], label=sector_names[i],
                    edgecolor="white", linewidth=0.5, zorder=3)
            bottoms += sector_pivot[sec].reindex(all_dates).fillna(0).values
        ax2.set_xticks(x)
        ax2.set_xticklabels(date_labels, fontsize=8)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.8)
        ax2.set_title("Sector Allocation Over Time", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Weight", fontsize=10)
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No sector data", ha="center", va="center", fontsize=13)
        ax2.set_title("Sector Allocation", fontsize=13, fontweight="bold")

    # ── Plot 3: Per-stock cumulative return ────────────────────────────────────
    if show_stock_perf:
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(C_BG)
        for i, code in enumerate(top10):
            stk = stock_date_ret[stock_date_ret["stock_code"] == code].sort_values("date_dt")
            if stk.empty:
                continue
            cum = (1 + stk[fwd_col]).cumprod() - 1
            name_lbl = name_map.get(code, code)
            label = f"{code} ({str(name_lbl)[:10]})" if name_lbl else code
            ax3.plot(stk["date_dt"], cum * 100, color=PALETTE[i % len(PALETTE)],
                     linewidth=1.8, marker="o", markersize=4, label=label, zorder=3)
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.set_title("Cumulative Realized Return — Top 10 Most-Selected Stocks", fontsize=13, fontweight="bold")
        ax3.set_ylabel("Cumulative Return (%)", fontsize=10)
        ax3.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.8)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax3.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved picks chart to {out}")


def _run_fold(payload: dict) -> dict:
    """Run one walk-forward fold in a worker process."""
    from ml.models import get_model_class

    train_df: pd.DataFrame = payload["train_df"]
    test_df: pd.DataFrame = payload["test_df"]
    info: dict = payload["info"]
    feature_cols: list[str] = payload["feature_cols"]
    target_col: str = payload["target_col"]
    fwd_col: str = payload["fwd_col"]
    eval_fwd_col: str = payload.get("eval_fwd_col", fwd_col)
    min_daily_value: int = payload.get("min_daily_value", 0)
    portfolio_size: int = payload.get("portfolio_size", 100_000_000)
    top_n: int = payload["top_n"]
    rebalance_days: int = payload["rebalance_days"]
    time_decay: float = payload["time_decay"]
    model_jobs: int = payload["model_jobs"]
    buy_fee_rate: float = payload["buy_fee_rate"]
    sell_fee_rate: float = payload["sell_fee_rate"]
    learning_rate: float = payload["learning_rate"]
    n_estimators: int = payload["n_estimators"]
    patience: int = payload["patience"]
    min_market_cap: int = payload["min_market_cap"]
    max_market_cap: int | None = payload.get("max_market_cap")
    stress_mode: bool = payload["stress_mode"]
    vol_exclude_pct: float = payload["vol_exclude_pct"]
    sector_neutral_score: bool = payload["sector_neutral_score"]
    buy_rank: int = payload["buy_rank"]
    hold_rank: int = payload["hold_rank"]
    embargo_days: int = payload["embargo_days"]
    cash_out_enabled: bool = payload.get("cash_out", False)
    bench_returns_by_date: dict = payload.get("bench_returns_by_date", {})
    model_class_name: str = payload.get("model_class", "lgbm")
    run_turnover_test: bool = payload.get("run_turnover_test", True)
    turnover_test_hold_rank: int = payload.get("turnover_test_hold_rank", hold_rank)
    turnover_test_smoothing_alpha: float = payload.get("turnover_test_smoothing_alpha", 1.0)
    stop_loss_pct: float = payload.get("stop_loss_pct", 0.0)
    print(
        f"[Fold {info['test_year']}] start "
        f"(train={info['train_period']}, train_rows={len(train_df):,}, test_rows={len(test_df):,})",
        flush=True,
    )

    train_years = sorted(train_df["date"].str[:4].unique())
    val_year = train_years[-1]
    sub_train = train_df[train_df["date"].str[:4] != val_year]
    val_df = train_df[train_df["date"].str[:4] == val_year]
    if sub_train.empty:
        sub_train, val_df = train_df, None
    # Purged training: enforce embargo gap before test period.
    all_dates = sorted(pd.concat([train_df["date"], test_df["date"]]).unique())
    test_start = min(test_df["date"])
    if test_start in all_dates:
        idx = all_dates.index(test_start)
        if idx > embargo_days:
            cutoff = all_dates[idx - embargo_days]
            sub_train = sub_train[sub_train["date"] < cutoff].copy()
            if val_df is not None:
                val_df = val_df[val_df["date"] < cutoff].copy()
    if sub_train.empty:
        if val_df is not None and not val_df.empty:
            print(
                f"[Fold {info['test_year']}] WARNING: sub_train empty after embargo; "
                f"using val_df as sole training set (no early-stopping).",
                flush=True,
            )
            sub_train = val_df.copy()
            val_df = None
        else:
            print(
                f"[Fold {info['test_year']}] ERROR: no training data after embargo; skipping fold.",
                flush=True,
            )
            return {
                "test_year": info["test_year"],
                "rows": [],
                "sector_rows": [],
                "pick_rows": [],
                "final_holdings": [],
                "final_holdings_tuned": [],
                "final_scores_tuned": {},
            }

    ModelClass = get_model_class(model_class_name)
    model = ModelClass(feature_cols=feature_cols, target_col=target_col, time_decay=time_decay)
    params = model.BEST_PARAMS.copy()
    params["n_jobs"] = model_jobs
    params["learning_rate"] = learning_rate
    params["n_estimators"] = n_estimators
    model.patience = patience
    model.train(sub_train, val_df, params=params)
    if val_df is not None and len(val_df) > 100 and fwd_col in val_df.columns:
        val_probe = val_df.copy()
        val_probe["score"] = model.predict(val_probe)
        score_rank_probe = val_probe["score"].rank(method="first", pct=True)
        val_probe["quintile"] = np.ceil(score_rank_probe * 5).clip(1, 5).astype(int)
        qv = val_probe.groupby("quintile")[fwd_col].mean()
        mono_ok = False
        if all(q in qv.index for q in [1, 2, 3, 4, 5]):
            mono_ok = bool(qv.loc[5] > qv.loc[4] > qv.loc[3] > qv.loc[2] > qv.loc[1])
        if not mono_ok:
            print(f"[Fold {info['test_year']}] quintiles not monotonic (diagnostic only, no retry)", flush=True)
    print(f"[Fold {info['test_year']}] model trained", flush=True)

    rows = []
    sector_rows = []
    pick_rows = []
    date_groups = {d: g.copy() for d, g in test_df.groupby("date", sort=True)}
    rebalance_dates = sorted(date_groups.keys())[::rebalance_days]
    prev_holdings: set[str] = set(payload.get("prev_holdings", []))
    prev_holdings_tuned: set[str] = set(payload.get("prev_holdings_tuned", []))
    prev_scores_tuned: dict[str, float] = dict(payload.get("prev_scores_tuned", {}))

    def _build_picks(
        frame: pd.DataFrame,
        rank_col: str,
        rank_pos_col: str,
        previous_holdings: set[str],
        hold_rank_limit: int,
        effective_top_n: int,
    ) -> tuple[pd.DataFrame, set[str], float, float]:
        keep_pool = frame[
            (frame["stock_code"].isin(previous_holdings)) & (frame[rank_pos_col] <= hold_rank_limit)
        ].copy()
        already_in = set(keep_pool["stock_code"])
        buy_candidates = frame[
            (~frame["stock_code"].isin(already_in)) & (frame[rank_pos_col] <= buy_rank)
        ].copy()

        if len(keep_pool) > 0 and len(buy_candidates) > 0:
            worst_keeper_score = keep_pool[rank_col].min()
            score_edge = buy_fee_rate + sell_fee_rate
            buy_candidates = buy_candidates[
                buy_candidates[rank_col] > worst_keeper_score + score_edge
            ].copy()

        picks = pd.concat([keep_pool, buy_candidates], ignore_index=True)
        picks = picks.sort_values(rank_col, ascending=False).drop_duplicates("stock_code")

        if len(picks) < effective_top_n:
            fill_pool = frame[
                (~frame["stock_code"].isin(set(picks["stock_code"])))
                & (frame[rank_pos_col] <= hold_rank_limit)
            ].copy()
            fill_pool = fill_pool.sort_values(rank_col, ascending=False)
            picks = pd.concat([picks, fill_pool.head(effective_top_n - len(picks))], ignore_index=True)

        picks = picks.sort_values(rank_col, ascending=False).drop_duplicates("stock_code")
        picks = picks.head(effective_top_n).copy()
        current_holdings = set(picks["stock_code"].tolist())
        if not previous_holdings:
            turnover = 1.0
            transaction_cost = buy_fee_rate
        else:
            overlap = len(previous_holdings & current_holdings)
            turnover = 1.0 - (overlap / max(effective_top_n, 1))
            transaction_cost = turnover * (buy_fee_rate + sell_fee_rate)
        return picks, current_holdings, turnover, transaction_cost

    for d in rebalance_dates:
        day_df = date_groups[d].copy()
        # PIT universe on rebalance date
        day_df = day_df[day_df["market_cap"] >= min_market_cap].copy()
        if max_market_cap:
            day_df = day_df[day_df["market_cap"] <= max_market_cap].copy()
        # Exclude suspended stocks (거래정지): zero daily trading value means
        # the stock is halted and cannot be traded at this rebalance date.
        if "value" in day_df.columns:
            day_df = day_df[day_df["value"] > 0].copy()
        # Liquidity filter: exclude stocks whose daily trading value is below threshold.
        # Prevents allocating to illiquid names that cannot be filled in practice.
        if min_daily_value > 0 and "value" in day_df.columns:
            day_df = day_df[day_df["value"] >= min_daily_value].copy()
        if stress_mode and 0 < vol_exclude_pct < 1 and "volatility_21d" in day_df.columns and len(day_df) > 10:
            vol_cut = day_df["volatility_21d"].quantile(1.0 - vol_exclude_pct)
            day_df = day_df[day_df["volatility_21d"] <= vol_cut].copy()
        if len(day_df) < top_n:
            continue
        # Cash-out: two-layer risk-off switch
        # Layer 1 (existing): KOSPI200 below 20-day MA → halve positions
        # Layer 2 (new):      VKOSPI fear index in top 20% → additional 50% cash
        effective_top_n = top_n
        cash_weight = 0.0
        if cash_out_enabled:
            # Layer 1: trend filter (KOSPI200 below 20d MA)
            if "market_regime_20d" in day_df.columns:
                regime_val = day_df["market_regime_20d"].iloc[0]
                if pd.notna(regime_val) and regime_val < 0:
                    effective_top_n = max(top_n // 2, 5)
                    cash_weight = 1.0 - (effective_top_n / top_n)
            # Layer 2: fear filter (VKOSPI in top-20% of 1-year distribution)
            # vkospi_level_pct is a 252d rolling percentile: >0.8 = extreme fear
            if "vkospi_level_pct" in day_df.columns:
                vkos_val = day_df["vkospi_level_pct"].iloc[0]
                if pd.notna(vkos_val) and vkos_val > 0.8:
                    # Additional 50% into cash on top of Layer 1
                    cash_weight = min(cash_weight + 0.5, 1.0)
                    effective_top_n = max(int(top_n * (1.0 - cash_weight)), 5)
        day_df["score"] = model.predict(day_df)
        if sector_neutral_score and "sector" in day_df.columns:
            sec_mean = day_df.groupby("sector")["score"].transform("mean")
            sec_std = day_df.groupby("sector")["score"].transform("std").replace(0, np.nan)
            day_df["score_rank"] = ((day_df["score"] - sec_mean) / sec_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            day_df["score_rank"] = day_df["score"]
        day_df["rank_pos"] = day_df["score_rank"].rank(ascending=False, method="first")
        score_rank = day_df["score_rank"].rank(method="first", pct=True)
        day_df["quintile"] = np.ceil(score_rank * 5).clip(1, 5).astype(int)
        # Use eval_fwd_col (the actually-traded return) for all signal-quality metrics.
        # When exec_lag > 0, this is open[T+lag]/open[T+H+lag] rather than close[T]/close[T+H].
        _ret_col = eval_fwd_col if eval_fwd_col in day_df.columns else fwd_col
        qret = day_df.groupby("quintile")[_ret_col].mean()
        q1 = float(qret.get(1, np.nan))
        q2 = float(qret.get(2, np.nan))
        q3 = float(qret.get(3, np.nan))
        q4 = float(qret.get(4, np.nan))
        q5 = float(qret.get(5, np.nan))
        q_mono = int(q5 > q4 > q3 > q2 > q1) if np.all(pd.notna([q1, q2, q3, q4, q5])) else 0
        ic = day_df[["score_rank", _ret_col]].corr(method="spearman").iloc[0, 1]
        ic = float(ic) if pd.notna(ic) else np.nan
        decile_n = max(int(len(day_df) * 0.10), 1)
        top_decile_return = float(day_df.nlargest(decile_n, "score_rank")[_ret_col].mean())
        bottom_decile_return = float(day_df.nsmallest(decile_n, "score_rank")[_ret_col].mean())
        long_short_return = top_decile_return - bottom_decile_return

        picks, current_holdings, turnover, transaction_cost = _build_picks(
            frame=day_df,
            rank_col="score_rank",
            rank_pos_col="rank_pos",
            previous_holdings=prev_holdings,
            hold_rank_limit=hold_rank,
            effective_top_n=effective_top_n,
        )
        if picks.empty:
            continue

        _ret_col_pick = eval_fwd_col if eval_fwd_col in picks.columns else fwd_col
        sl_triggered_rate = float(picks["_sl_triggered"].mean()) if stop_loss_pct > 0 and "_sl_triggered" in picks.columns else 0.0
        if portfolio_size > 0 and "closing_price" in picks.columns:
            # Discrete share sizing: equal-weight budget, floor to whole shares
            investable = portfolio_size * (1.0 - cash_weight)
            per_stock = investable / max(len(picks), 1)
            _prices = picks["closing_price"].clip(lower=1.0)
            _shares = np.floor(per_stock / _prices)
            _invested = _shares * _prices
            _total_invested = float(_invested.sum())
            cash_drag_pct = 1.0 - _total_invested / portfolio_size
            if _total_invested > 0:
                _w = _invested / _total_invested
                stock_ret = float((_w * picks[_ret_col_pick].fillna(0.0)).sum())
            else:
                stock_ret = 0.0
            port_ret = stock_ret * (_total_invested / portfolio_size)
            # Attach shares info to picks for CSV export
            picks = picks.copy()
            picks["shares"] = _shares.values
            picks["invested_krw"] = _invested.values
        else:
            stock_ret = float(picks[_ret_col_pick].mean())
            port_ret = stock_ret * (1.0 - cash_weight)
            cash_drag_pct = cash_weight
        if bench_returns_by_date and d in bench_returns_by_date and pd.notna(bench_returns_by_date[d]):
            bench_ret = float(bench_returns_by_date[d])
        else:
            bench_ret = float(day_df[eval_fwd_col].mean()) if eval_fwd_col in day_df.columns else float(day_df[fwd_col].mean())
        net_port_ret = (1.0 + port_ret) * (1.0 - transaction_cost) - 1.0
        prev_holdings = current_holdings

        # Turnover reduction test: relaxed hold threshold + score smoothing.
        net_port_ret_tuned = np.nan
        turnover_tuned = np.nan
        transaction_cost_tuned = np.nan
        if run_turnover_test:
            if 0.0 < turnover_test_smoothing_alpha < 1.0:
                prev_smoothed = day_df["stock_code"].map(prev_scores_tuned)
                day_df["score_tuned"] = day_df["score"]
                valid_prev = prev_smoothed.notna()
                day_df.loc[valid_prev, "score_tuned"] = (
                    turnover_test_smoothing_alpha * day_df.loc[valid_prev, "score"]
                    + (1.0 - turnover_test_smoothing_alpha) * prev_smoothed.loc[valid_prev].astype(float)
                )
            else:
                day_df["score_tuned"] = day_df["score"]

            prev_scores_tuned.update(
                {str(c): float(v) for c, v in zip(day_df["stock_code"], day_df["score_tuned"])}
            )

            if sector_neutral_score and "sector" in day_df.columns:
                sec_mean_t = day_df.groupby("sector")["score_tuned"].transform("mean")
                sec_std_t = day_df.groupby("sector")["score_tuned"].transform("std").replace(0, np.nan)
                day_df["score_rank_tuned"] = (
                    (day_df["score_tuned"] - sec_mean_t) / sec_std_t
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                day_df["score_rank_tuned"] = day_df["score_tuned"]

            day_df["rank_pos_tuned"] = day_df["score_rank_tuned"].rank(ascending=False, method="first")
            picks_tuned, current_holdings_tuned, turnover_tuned, transaction_cost_tuned = _build_picks(
                frame=day_df,
                rank_col="score_rank_tuned",
                rank_pos_col="rank_pos_tuned",
                previous_holdings=prev_holdings_tuned,
                hold_rank_limit=turnover_test_hold_rank,
                effective_top_n=effective_top_n,
            )
            if not picks_tuned.empty:
                _ret_col_tuned = eval_fwd_col if eval_fwd_col in picks_tuned.columns else fwd_col
                if portfolio_size > 0 and "closing_price" in picks_tuned.columns:
                    investable_t = portfolio_size * (1.0 - cash_weight)
                    per_stock_t = investable_t / max(len(picks_tuned), 1)
                    _prices_t = picks_tuned["closing_price"].clip(lower=1.0)
                    _invested_t = np.floor(per_stock_t / _prices_t) * _prices_t
                    _total_t = float(_invested_t.sum())
                    if _total_t > 0:
                        _w_t = _invested_t / _total_t
                        stock_ret_tuned = float((_w_t * picks_tuned[_ret_col_tuned].fillna(0.0)).sum())
                    else:
                        stock_ret_tuned = 0.0
                    port_ret_tuned = stock_ret_tuned * (_total_t / portfolio_size)
                else:
                    stock_ret_tuned = float(picks_tuned[_ret_col_tuned].mean())
                    port_ret_tuned = stock_ret_tuned * (1.0 - cash_weight)
                net_port_ret_tuned = (1.0 + port_ret_tuned) * (1.0 - transaction_cost_tuned) - 1.0
                prev_holdings_tuned = current_holdings_tuned

        _attr_col = eval_fwd_col if eval_fwd_col in picks.columns else fwd_col
        sec = (
            picks.groupby("sector", as_index=False)
            .agg(
                n=("stock_code", "count"),
                sector_forward_return=(_attr_col, "mean"),
            )
            .sort_values("n", ascending=False)
        )
        sec["weight"] = sec["n"] / max(effective_top_n, 1)
        sec["contribution"] = sec["weight"] * sec["sector_forward_return"]
        top_sector = str(sec.iloc[0]["sector"]) if len(sec) > 0 else "N/A"
        top_sector_weight = float(sec.iloc[0]["weight"]) if len(sec) > 0 else 0.0
        sector_hhi = float((sec["weight"] ** 2).sum()) if len(sec) > 0 else np.nan
        for _, srow in sec.iterrows():
            sector_rows.append(
                {
                    "date": d,
                    "test_year": info["test_year"],
                    "sector": srow["sector"],
                    "weight": float(srow["weight"]),
                    "sector_forward_return": float(srow["sector_forward_return"]),
                    "contribution": float(srow["contribution"]),
                }
            )

        # Collect per-pick details for optional CSV export
        pick_detail_cols = ["stock_code", "name", "sector", "closing_price", "shares", "invested_krw",
                            "매수가", "매도가", "매도날짜", "market_cap", "score", "score_rank", "rank_pos", eval_fwd_col]
        pick_detail_cols = [c for c in pick_detail_cols if c in picks.columns]
        for _, prow in picks.iterrows():
            pick_rows.append({
                "date": d,
                "test_year": info["test_year"],
                **{c: prow[c] for c in pick_detail_cols},
            })

        rows.append(
            {
                "date": d,
                "year": int(d[:4]),
                "portfolio_return": net_port_ret,
                "portfolio_return_gross": port_ret,
                "benchmark_return": bench_ret,
                "alpha": net_port_ret - bench_ret,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "cash_drag_pct": cash_drag_pct,
                "sl_triggered_rate": sl_triggered_rate,
                "ic_spearman": ic,
                "q1_ret": q1,
                "q2_ret": q2,
                "q3_ret": q3,
                "q4_ret": q4,
                "q5_ret": q5,
                "q_monotonic": q_mono,
                "top_decile_return": top_decile_return,
                "bottom_decile_return": bottom_decile_return,
                "long_short_return": long_short_return,
                "portfolio_return_tuned": net_port_ret_tuned,
                "turnover_tuned": turnover_tuned,
                "transaction_cost_tuned": transaction_cost_tuned,
                "top_sector": top_sector,
                "top_sector_weight": top_sector_weight,
                "sector_hhi": sector_hhi,
                "top_picks": " | ".join(
                    (
                        f"{row['stock_code']}({str(row.get('name', ''))[:10]}):{row[eval_fwd_col]:+.1%}"
                        if eval_fwd_col in picks.columns and pd.notna(row.get(eval_fwd_col))
                        else f"{row['stock_code']}({str(row.get('name', ''))[:10]})"
                    )
                    for _, row in picks.head(10).iterrows()
                ),
                "train_period": info["train_period"],
                "test_year": info["test_year"],
            }
        )

    print(
        f"[Fold {info['test_year']}] done "
        f"(rebalance_points={len(rebalance_dates)}, result_rows={len(rows)})",
        flush=True,
    )
    return {
        "test_year": info["test_year"],
        "rows": rows,
        "sector_rows": sector_rows,
        "pick_rows": pick_rows,
        "final_holdings": list(prev_holdings),
        "final_holdings_tuned": list(prev_holdings_tuned),
        "final_scores_tuned": prev_scores_tuned,
    }


def run(args: argparse.Namespace) -> None:
    from ml.features import FeatureEngineer
    from ml.model import walk_forward_split
    from ml.models import get_model_class

    effective_buy_fee = args.buy_fee
    effective_sell_fee = args.sell_fee
    if getattr(args, "no_sector_neutral", False):
        args.sector_neutral_score = False
    if getattr(args, "no_cash_out", False):
        args.cash_out = False
    effective_sector_neutral = args.sector_neutral_score or args.stress_mode
    if args.stress_mode:
        effective_buy_fee = 1.0
        effective_sell_fee = 1.0

    if args.horizon <= 0:
        raise ValueError("--horizon must be >= 1")

    exec_lag = int(getattr(args, "exec_lag", 1))
    if exec_lag < 0:
        raise ValueError("--exec-lag must be >= 0")
    args.exec_lag = exec_lag

    exec_price = str(getattr(args, "exec_price", "open")).lower()
    if exec_price not in {"open", "close"}:
        raise ValueError("--exec-price must be one of: open, close")
    args.exec_price = exec_price

    # Purge embargo must cover the full label horizon and execution lag.
    required_embargo = args.horizon + exec_lag
    if args.embargo_days < required_embargo:
        print(
            f"[Backtest] embargo auto-calc: {args.embargo_days}d -> {required_embargo}d "
            f"(>= horizon {args.horizon}d + exec_lag {exec_lag}d)",
            flush=True,
        )
        args.embargo_days = required_embargo

    print(f"[Backtest] loading data {args.start}~{args.end} ...", flush=True)
    fe = FeatureEngineer(args.db)
    df = fe.prepare_ml_data(
        start_date=args.start,
        end_date=args.end,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
        max_market_cap=getattr(args, "max_market_cap", None),
        use_cache=not args.no_cache,
        n_workers=args.workers,
    )

    if df.empty:
        print("No ML data available for the requested range.")
        return
    exclude_years = _parse_exclude_years(args.exclude_years)
    if exclude_years:
        before_rows = len(df)
        df = df[~df["date"].str[:4].isin(exclude_years)].copy()
        print(
            f"[Backtest] excluded years={sorted(exclude_years)} "
            f"(rows {before_rows:,} -> {len(df):,})",
            flush=True,
        )
        if df.empty:
            print("No rows left after applying --exclude-years filter.")
            return
    if args.stress_mode and 0 < args.vol_exclude_pct < 1 and "volatility_21d" in df.columns:
        vol_cut = df.groupby("date")["volatility_21d"].transform(
            lambda s: s.quantile(1.0 - args.vol_exclude_pct)
        )
        df = df[df["volatility_21d"] <= vol_cut].copy()
    print(f"[Backtest] feature rows={len(df):,}, cols={len(df.columns)}", flush=True)

    feature_cols = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in df.columns]
    fwd_col = f"forward_return_{args.horizon}d"

    # ── Test 1: Execution Lag ─────────────────────────────────────────────
    # Trade after signal with explicit execution lag and price basis.
    # Model is still trained on spot fwd_col; only portfolio evaluation uses lag.
    eval_fwd_col = fwd_col
    # Prefer adjusted prices so splits within the holding period don't distort returns.
    # Fall back to raw prices if adj_daily_prices table hasn't been built yet.
    if exec_price == "open":
        if "adj_opening_price" in df.columns:
            trade_price_col = "adj_opening_price"
        elif "opening_price" in df.columns:
            print("[Backtest] WARNING: adj_opening_price missing, falling back to raw opening_price.", flush=True)
            trade_price_col = "opening_price"
        elif "closing_price" in df.columns:
            print("[Backtest] WARNING: opening_price missing, falling back to closing_price execution.", flush=True)
            exec_price = "close"
            trade_price_col = "closing_price"
        else:
            raise ValueError("No execution price column found.")
    else:
        trade_price_col = "adj_closing_price" if "adj_closing_price" in df.columns else "closing_price"
    if trade_price_col not in df.columns:
        raise ValueError(f"Required execution price column not found: {trade_price_col}")

    if exec_lag > 0:
        lag_col = f"forward_return_{args.horizon}d_lag{exec_lag}_{exec_price}"
        _df_sorted = df.sort_values(["stock_code", "date"]).copy()
        # Replace 0 prices with NaN: opening_price=0 occurs on circuit-breaker / upper-lock
        # days in KRX data. Dividing by zero would produce inf returns silently.
        _df_sorted[trade_price_col] = _df_sorted[trade_price_col].replace(0, float("nan"))
        _grp = _df_sorted.groupby("stock_code")[trade_price_col]
        _entry_px = _grp.shift(-exec_lag)
        _exit_px = _grp.shift(-(args.horizon + exec_lag))
        _df_sorted[lag_col] = _exit_px / _entry_px - 1
        # Tail fallback: when exit price is unavailable (NaN), fall back to the
        # pipeline's adj-closing forward return which is pre-computed on the
        # UNFILTERED per-stock series and is immune to universe-filter row gaps.
        # The old mark-to-last (_last_px / _entry_px - 1) could produce extreme
        # returns (e.g. 191%) when hard filters like bad_accrual create mid-series
        # gaps and _last_px is a distant peak price from a later year.
        _nan_mask = _df_sorted[lag_col].isna() & _entry_px.gt(0)
        _base_fwd_col = f"forward_return_{args.horizon}d"
        if _base_fwd_col in _df_sorted.columns:
            _df_sorted.loc[_nan_mask, lag_col] = _df_sorted.loc[_nan_mask, _base_fwd_col]
        else:
            _last_px = _df_sorted.groupby("stock_code")[trade_price_col].transform("last")
            _df_sorted.loc[_nan_mask, lag_col] = _last_px[_nan_mask] / _entry_px[_nan_mask] - 1
        df = _df_sorted
        eval_fwd_col = lag_col
        print(
            f"[Backtest] exec_lag={exec_lag}d — execution: T+{exec_lag} {exec_price} ({lag_col})",
            flush=True,
        )

    # ── TWAP Execution Mode ───────────────────────────────────────────────
    # Simulates spreading execution over N trading days at each rebalance.
    #
    # Bias-free design:
    #   • Model is trained on spot fwd_col (signal quality is separate from execution)
    #   • entry_avg  = equal-weight mean of close[T+1 .. T+N]   (buy leg)
    #   • exit_avg   = equal-weight mean of close[T+H-N+1 .. T+H] (sell leg)
    #   • Suspended days (value==0) are excluded from each average
    #   • N is capped at H//3 so entry and exit windows never overlap
    #   • NaN (data tail / full suspension) → last observed price fallback
    #
    # This overrides exec_lag if both flags are set (TWAP already implies T+1 start).
    twap_days = getattr(args, "twap_days", 0)
    if twap_days > 0 and "closing_price" in df.columns:
        H = args.horizon
        max_twap = H // 3  # entry [T+1..T+N] and exit [T+H-N+1..T+H] must not overlap
        if twap_days > max_twap:
            print(f"[Backtest] Warning: --twap-days={twap_days} > horizon//3={max_twap}. Capped.", flush=True)
            twap_days = max_twap

        twap_col = f"forward_return_{H}d_twap{twap_days}"
        _df_tw = df.sort_values(["stock_code", "date"]).copy()
        has_value = "value" in _df_tw.columns

        # ── Entry window: average close over days T+1 .. T+twap_days ──────
        _entry_list = []
        for k in range(1, twap_days + 1):
            px = _df_tw.groupby("stock_code")["closing_price"].shift(-k)
            if has_value:
                vl = _df_tw.groupby("stock_code")["value"].shift(-k)
                px = px.where(vl > 0)   # exclude suspended days from average
            _entry_list.append(px)
        entry_avg = pd.concat(_entry_list, axis=1).mean(axis=1, skipna=True)

        # ── Exit window: average close over days T+H-twap_days+1 .. T+H ──
        _exit_list = []
        for k in range(H - twap_days + 1, H + 1):
            px = _df_tw.groupby("stock_code")["closing_price"].shift(-k)
            if has_value:
                vl = _df_tw.groupby("stock_code")["value"].shift(-k)
                px = px.where(vl > 0)   # exclude suspended days from average
            _exit_list.append(px)
        exit_avg = pd.concat(_exit_list, axis=1).mean(axis=1, skipna=True)

        _df_tw[twap_col] = exit_avg / entry_avg - 1

        # Fix NaN: data tail or fully suspended window → last observed price
        _last_px = _df_tw.groupby("stock_code")["closing_price"].transform("last")
        _nan_mask = _df_tw[twap_col].isna() & _df_tw["closing_price"].gt(0)
        _df_tw.loc[_nan_mask, twap_col] = (
            _last_px[_nan_mask] / _df_tw.loc[_nan_mask, "closing_price"] - 1
        )

        # Store exact entry/exit prices for picks.csv
        _df_tw["매수가"] = entry_avg.round(0)
        _df_tw["매도가"] = exit_avg.round(0)

        df = _df_tw
        eval_fwd_col = twap_col   # overrides exec_lag if both set
        print(
            f"[Backtest] twap_days={twap_days}: "
            f"entry=avg(close T+1~T+{twap_days}), "
            f"exit=avg(close T+{H - twap_days + 1}~T+{H}), "
            f"suspended days excluded  [{twap_col}]",
            flush=True,
        )
    else:
        # Non-TWAP execution prices for picks.csv
        _df_base = df.sort_values(["stock_code", "date"]).copy()
        _grp_px_close = _df_base.groupby("stock_code")["closing_price"]
        _grp_px_exec = _df_base.groupby("stock_code")[trade_price_col]
        _grp_date = _df_base.groupby("stock_code")["date"]
        if exec_lag > 0:
            _df_base["매수가"] = _grp_px_exec.shift(-exec_lag).round(0)
            _df_base["매도가"] = _grp_px_exec.shift(-(args.horizon + exec_lag)).round(0)
            _df_base["매도날짜"] = _grp_date.shift(-(args.horizon + exec_lag))
        else:
            _df_base["매수가"] = _df_base["closing_price"].round(0)
            _df_base["매도가"] = _grp_px_close.shift(-args.horizon).round(0)
            _df_base["매도날짜"] = _grp_date.shift(-args.horizon)
        df = _df_base

    # ── Stop-Loss Pre-computation ─────────────────────────────────────────
    # For each stock on date T, find the minimum intraperiod close price in the
    # holding window [T+exec_lag+1 .. T+exec_lag+horizon].  If that minimum is
    # more than stop_loss_pct below the entry price (close[T+exec_lag]), the
    # position is assumed to have been exited at the stop level.
    # Result: eval_fwd_col returns are capped at -stop_loss_pct for those rows.
    # Only supported when exec_lag >= 1 (entry price is unambiguous).
    stop_loss_pct: float = getattr(args, "stop_loss", 0.0)
    if stop_loss_pct > 0 and exec_lag >= 1 and twap_days == 0:
        print(
            f"[Backtest] stop_loss={stop_loss_pct:.0%} — "
            f"pre-computing intraperiod minimums over {args.horizon} days ...",
            flush=True,
        )
        _df_sl = df.sort_values(["stock_code", "date"]).copy()
        _grp_sl = _df_sl.groupby("stock_code")[trade_price_col]
        _entry_px_sl = _grp_sl.shift(-exec_lag).replace(0, float("nan"))
        # Build list of prices at each day inside the holding window
        _min_prices = []
        for _k in range(exec_lag + 1, exec_lag + args.horizon + 1):
            _min_prices.append(_grp_sl.shift(-_k))
        _intraperiod_min = pd.concat(_min_prices, axis=1).min(axis=1, skipna=True)
        _min_return = _intraperiod_min / _entry_px_sl - 1
        _sl_triggered = (_min_return < -stop_loss_pct) & _entry_px_sl.notna()
        sl_col = f"{eval_fwd_col}_sl{int(stop_loss_pct * 100)}"
        _df_sl[sl_col] = np.where(_sl_triggered, -stop_loss_pct, _df_sl[eval_fwd_col])
        _df_sl["_sl_triggered"] = _sl_triggered.astype(float)
        df = _df_sl
        eval_fwd_col = sl_col
        _valid = _entry_px_sl.notna().sum()
        _hit = int(_sl_triggered.sum())
        print(
            f"[Backtest] stop_loss applied: {_hit:,}/{_valid:,} rows triggered "
            f"({_hit / max(_valid, 1):.1%})",
            flush=True,
        )
    elif stop_loss_pct > 0:
        print("[Backtest] WARNING: --stop-loss requires --exec-lag >= 1 and no --twap-days. Skipped.", flush=True)
        stop_loss_pct = 0.0

    # ── Test 4: Feature Permutation ───────────────────────────────────────
    # Shuffle one or all feature columns across ALL rows (stocks × dates).
    # This completely destroys both temporal and cross-sectional signal in that feature.
    #
    # Interpretation:
    #   If IC / Sharpe drops significantly after permutation → feature has real signal.
    #   If IC / Sharpe is maintained after permuting ALL features → look-ahead leakage.
    #
    # Bias-free design:
    #   • Permutation is applied BEFORE walk-forward splits (same shuffled data seen in
    #     all folds → consistent apples-to-apples comparison within this run).
    #   • random_state is fixed for reproducibility.
    #   • Model is retrained on the permuted dataset (full walk-forward preserved).
    #   • eval_fwd_col (portfolio returns) is NOT shuffled — only features are.
    permute_feature = getattr(args, "permute_feature", "")
    if permute_feature:
        rng = np.random.default_rng(seed=42)
        if permute_feature.lower() == "all":
            targets = [c for c in feature_cols if c in df.columns]
        else:
            targets = [f for f in permute_feature.split(",") if f.strip() in df.columns]
            unknown = [f for f in permute_feature.split(",") if f.strip() not in df.columns]
            if unknown:
                print(f"[Permutation] WARNING: unknown feature(s) skipped: {unknown}", flush=True)
        for feat in targets:
            df[feat] = rng.permutation(df[feat].values)
        print(
            f"[Permutation] Test 4: shuffled {len(targets)} feature(s) → "
            f"{targets if len(targets) <= 5 else str(targets[:5]) + '...'}\n"
            f"  Expected: IC ≈ 0, Sharpe ≈ 0 if these features drive signal (no leakage).",
            flush=True,
        )

    residual_rank_col = f"target_residual_rank_{args.horizon}d"
    rank_label_col = f"target_rank_label_{args.horizon}d"
    base_col = residual_rank_col if residual_rank_col in df.columns else f"target_rank_{args.horizon}d"

    # Ranking objectives (lambdarank) require integer labels [0-4].
    # Regression objectives (huber, rmse, etc.) use the continuous rank [0,1] directly.
    _model_objective = get_model_class(args.model).BEST_PARAMS.get("objective", "")
    _is_ranking = _model_objective in ("lambdarank", "rank_xendcg")
    if _is_ranking and base_col in df.columns:
        df[rank_label_col] = np.clip((df[base_col] * 5).astype(int), 0, 4)
        target_col = rank_label_col
    else:
        target_col = base_col  # continuous residual rank [0,1]

    # Load benchmark index returns
    _bench_label = getattr(args, "benchmark", "kospi200")
    _bench_index_code = BENCHMARK_INDEX_MAP.get(_bench_label)
    if _bench_index_code:
        bench_returns_by_date = _load_benchmark_returns(args.db, _bench_index_code, args.horizon)
        if not bench_returns_by_date:
            print(f"[Benchmark] WARNING: falling back to universe average (no data for {_bench_index_code})", flush=True)
    else:
        bench_returns_by_date = {}

    splits = walk_forward_split(df, train_years=args.train_years)
    if not splits:
        print("No walk-forward splits available. Widen date range or reduce train years.")
        return
    cpu_count = os.cpu_count() or 4
    workers = max(1, args.workers)
    split_years = [int(s[2]["test_year"]) for s in splits]

    if args.model_jobs > 0:
        model_jobs = args.model_jobs
    else:
        model_jobs = max(1, cpu_count // workers) if workers > 1 else -1

    # Resolve model params for summary
    ModelClassInfo = get_model_class(args.model)
    model_params = ModelClassInfo.BEST_PARAMS.copy()
    model_params["learning_rate"] = args.learning_rate
    model_params["n_estimators"] = args.n_estimators

    # ── Config Summary ──
    print("\n" + "=" * 70)
    print("  BACKTEST CONFIG")
    print("=" * 70)
    print(f"\n{'--- Data ---':^70}")
    print(f"  Period:           {args.start} ~ {args.end}")
    _max_cap = getattr(args, "max_market_cap", None)
    _cap_str = f"{args.min_market_cap:,} ~ {_max_cap:,}" if _max_cap else f">= {args.min_market_cap:,}"
    print(f"  Universe:         market_cap {_cap_str}")
    print(f"  Rows:             {len(df):,}   Features: {len(feature_cols)}")
    print(f"\n{'--- Model ---':^70}")
    print(f"  Type:             {args.model}")
    obj = model_params.get('objective', 'N/A')
    obj_detail = ""
    if obj == "lambdarank":
        trunc = model_params.get('lambdarank_truncation_level', 'N/A')
        obj_detail = f" (truncation={trunc}, eval_at={model_params.get('eval_at', 'N/A')})"
    elif "huber" in str(obj):
        obj_detail = f" (huber_delta={model_params.get('huber_delta', 'N/A')})"
    print(f"  Objective:        {obj}{obj_detail}")
    print(f"  Target:           {target_col}")
    print(f"  Target Source:    {base_col}")
    print(f"  LR / Estimators:  {args.learning_rate} / {args.n_estimators}")
    print(f"  Early Stop:       patience={args.patience}")
    depth = model_params.get('max_depth', 'N/A')
    print(f"  Leaves / Depth:   {model_params.get('num_leaves', 'N/A')} / max_depth={depth}")
    print(f"  Min Data/Leaf:    {model_params.get('min_data_in_leaf', 'N/A')}")
    print(f"  Feature Frac:     {model_params.get('feature_fraction', 'N/A')}")
    print(f"  Time Decay:       {args.time_decay}")
    print(f"\n{'--- Walk-Forward ---':^70}")
    print(f"  Train Window:     {args.train_years} years (rolling)")
    print(f"  Folds:            {len(splits)}   Test Years: {split_years}")
    print(f"  Embargo:          {args.embargo_days} days")
    if exclude_years:
        print(f"  Excluded Years:   {sorted(exclude_years)}")
    print(f"\n{'--- Portfolio ---':^70}")
    print(f"  Top N:            {args.top_n}")
    print(f"  Portfolio Size:   {args.portfolio_size:,} KRW  (discrete share rounding)")
    print(f"  Rebalance/Horizon: every {args.horizon} trading days")
    _bench_display = _bench_index_code if _bench_index_code else "universe (equal-weight)"
    print(f"  Benchmark:        {_bench_display}  [{_bench_label}]")
    print(f"  Buy Rank:         <= {args.buy_rank}   Hold Rank: <= {args.hold_rank}")
    print(f"  Fees:             buy={effective_buy_fee:.2f}%  sell={effective_sell_fee:.2f}%")
    print(f"  Sector Neutral:   {effective_sector_neutral}")
    cash_out_flag = getattr(args, "cash_out", False)
    print(f"  Cash-Out (20d):   {cash_out_flag}")
    print(
        "  Turnover Test:    "
        f"hold_rank<={args.turnover_test_hold_rank}, "
        f"smoothing_alpha={args.turnover_test_smoothing_alpha:.2f}"
    )
    if args.stress_mode:
        print(f"  Stress Mode:      ON (vol_exclude={args.vol_exclude_pct:.0%})")
    if exec_lag > 0 and twap_days == 0:
        print(f"  Exec Lag:         T+{exec_lag} {exec_price}  [{eval_fwd_col}]  ← Test 1")
    if stop_loss_pct > 0:
        print(f"  Stop-Loss:        {stop_loss_pct:.0%}  (intraperiod cap, cash for remainder)")
    if twap_days > 0:
        H = args.horizon
        print(f"  TWAP:             {twap_days}d — entry avg(T+1~T+{twap_days}), exit avg(T+{H-twap_days+1}~T+{H})  [{eval_fwd_col}]")
    if getattr(args, "min_daily_value", 0) > 0:
        print(f"  Liquidity Floor:  daily_value >= {args.min_daily_value:,} KRW  ← Test 5")
    if permute_feature:
        print(f"  Permutation:      feature='{permute_feature}' (ALL rows shuffled)  ← Test 4")
    print("=" * 70 + "\n", flush=True)

    rows = []
    sector_rows = []
    fold_payloads = [
        {
            "train_df": train_df,
            "test_df": test_df,
            "info": info,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "fwd_col": fwd_col,
            "eval_fwd_col": eval_fwd_col,
            "min_daily_value": getattr(args, "min_daily_value", 0),
            "top_n": args.top_n,
            "rebalance_days": args.horizon,
            "time_decay": args.time_decay,
            "model_jobs": model_jobs,
            "buy_fee_rate": effective_buy_fee / 100.0,
            "sell_fee_rate": effective_sell_fee / 100.0,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "patience": args.patience,
            "min_market_cap": args.min_market_cap,
            "max_market_cap": getattr(args, "max_market_cap", None),
            "stress_mode": args.stress_mode,
            "vol_exclude_pct": args.vol_exclude_pct,
            "sector_neutral_score": effective_sector_neutral,
            "buy_rank": args.buy_rank,
            "hold_rank": args.hold_rank,
            "embargo_days": args.embargo_days,
            "cash_out": args.cash_out,
            "bench_returns_by_date": bench_returns_by_date,
            "model_class": args.model,
            "run_turnover_test": not args.disable_turnover_test,
            "turnover_test_hold_rank": args.turnover_test_hold_rank,
            "turnover_test_smoothing_alpha": args.turnover_test_smoothing_alpha,
            "portfolio_size": args.portfolio_size,
            "stop_loss_pct": stop_loss_pct,
        }
        for train_df, test_df, info in splits
    ]

    if workers == 1 or len(fold_payloads) == 1:
        # Sequential: carry holdings across folds to avoid 100% turnover at fold boundaries
        fold_results = []
        carry_holdings: list[str] = []
        carry_holdings_tuned: list[str] = []
        carry_scores_tuned: dict[str, float] = {}
        for p in fold_payloads:
            p["prev_holdings"] = carry_holdings
            p["prev_holdings_tuned"] = carry_holdings_tuned
            p["prev_scores_tuned"] = carry_scores_tuned
            res = _run_fold(p)
            fold_results.append(res)
            carry_holdings = res.get("final_holdings", [])
            carry_holdings_tuned = res.get("final_holdings_tuned", [])
            carry_scores_tuned = res.get("final_scores_tuned", {})
    else:
        fold_results = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_run_fold, p) for p in fold_payloads]
                for fut in as_completed(futures):
                    fold_results.append(fut.result())
                    done_years = sorted([int(r["test_year"]) for r in fold_results])
                    print(f"[Backtest] completed folds so far: {done_years}", flush=True)
        except (PermissionError, OSError) as exc:
            print(f"[Backtest] multiprocessing unavailable ({exc}); fallback to sequential", flush=True)
            carry_holdings = []
            carry_holdings_tuned = []
            carry_scores_tuned = {}
            fold_results = []
            for p in fold_payloads:
                p["prev_holdings"] = carry_holdings
                p["prev_holdings_tuned"] = carry_holdings_tuned
                p["prev_scores_tuned"] = carry_scores_tuned
                res = _run_fold(p)
                fold_results.append(res)
                carry_holdings = res.get("final_holdings", [])
                carry_holdings_tuned = res.get("final_holdings_tuned", [])
                carry_scores_tuned = res.get("final_scores_tuned", {})

    fold_results.sort(key=lambda x: x["test_year"])
    pick_rows = []
    for res in fold_results:
        rows.extend(res["rows"])
        sector_rows.extend(res.get("sector_rows", []))
        pick_rows.extend(res.get("pick_rows", []))

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["date", "test_year"]).reset_index(drop=True)
    if not results.empty:
        # ── Output folder: runs/<name>/ ────────────────────────────────────
        run_name = Path(args.output).stem   # strip any accidental .csv suffix
        run_dir  = Path("runs") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        out_csv = run_dir / "results.csv"
        results.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved detailed results to {out_csv}")

        rolling = results[["date", "portfolio_return"]].copy()
        rolling["rolling_12_sharpe"] = (
            rolling["portfolio_return"].rolling(12).mean()
            / rolling["portfolio_return"].rolling(12).std().replace(0, np.nan)
            * np.sqrt(12)
        )
        rolling.to_csv(run_dir / "rolling_sharpe.csv", index=False, encoding="utf-8-sig")
        print(f"Saved rolling Sharpe to {run_dir / 'rolling_sharpe.csv'}")

        quintile_summary = results[["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"]].mean().to_frame("mean_return")
        quintile_summary.to_csv(run_dir / "quintiles.csv", encoding="utf-8-sig")
        print(f"Saved quintile summary to {run_dir / 'quintiles.csv'}")

        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            sector_df.to_csv(run_dir / "sector_attribution.csv", index=False, encoding="utf-8-sig")
            print(f"Saved sector attribution to {run_dir / 'sector_attribution.csv'}")

        # Save statistical significance report
        s_save = _compute_core_stats(results)
        sig = s_save.get("sig", {})
        if sig:
            sig_rows = [
                {"metric": "OLS t-stat",          "value": sig.get("ols_tstat"),    "p_value": sig.get("ols_pval")},
                {"metric": "Newey-West t-stat",    "value": sig.get("nw_tstat"),     "p_value": sig.get("nw_pval"),
                 "note": f"lags={sig.get('nw_lags')}"},
                {"metric": "Sharpe t-stat (Lo02)", "value": sig.get("sharpe_tstat"), "p_value": sig.get("sharpe_pval")},
                {"metric": "IC t-stat",            "value": sig.get("ic_tstat"),     "p_value": sig.get("ic_pval"),
                 "note": f"n={sig.get('ic_n')}"},
                {"metric": "Bootstrap Sharpe CI lo","value": sig.get("sharpe_ci_lo"), "p_value": np.nan},
                {"metric": "Bootstrap Sharpe CI hi","value": sig.get("sharpe_ci_hi"), "p_value": np.nan},
                {"metric": "Binomial hit-rate p",  "value": sig.get("binom_n_pos"),  "p_value": sig.get("binom_pval"),
                 "note": f"{sig.get('binom_n_pos')}/{sig.get('binom_n_tot')}"},
                {"metric": "VERDICT",              "value": sig.get("verdict"),       "p_value": np.nan,
                 "note": sig.get("verdict_note")},
            ]
            pd.DataFrame(sig_rows).to_csv(run_dir / "stat_significance.csv", index=False, encoding="utf-8-sig")
            print(f"Saved stat significance to {run_dir / 'stat_significance.csv'}")

        if args.save_picks and pick_rows:
            picks_df = pd.DataFrame(pick_rows).sort_values(["date", "rank_pos"])
            picks_df.to_csv(run_dir / "picks.csv", index=False, encoding="utf-8-sig")
            print(f"Saved picks to {run_dir / 'picks.csv'} ({len(picks_df)} rows)")
            _generate_picks_chart(picks_df, fwd_col=fwd_col, output_path=str(run_dir / "picks.png"))

    latest_model = None
    if splits:
        latest_split = max(splits, key=lambda x: x[2]["test_year"])
        latest_train_df = latest_split[0]
        train_years = sorted(latest_train_df["date"].str[:4].unique())
        val_year = train_years[-1]
        sub_train = latest_train_df[latest_train_df["date"].str[:4] != val_year]
        val_df = latest_train_df[latest_train_df["date"].str[:4] == val_year]
        if sub_train.empty:
            sub_train, val_df = latest_train_df, None
        FinalModelClass = get_model_class(args.model)
        latest_model = FinalModelClass(feature_cols=feature_cols, target_col=target_col, time_decay=args.time_decay)
        params = latest_model.BEST_PARAMS.copy()
        params["n_jobs"] = max(1, cpu_count // 2)
        params["learning_rate"] = args.learning_rate
        params["n_estimators"] = args.n_estimators
        latest_model.patience = args.patience
        latest_model.train(sub_train, val_df, params=params)
        latest_model.metadata = {
            "min_market_cap": args.min_market_cap,
            "max_market_cap": getattr(args, "max_market_cap", None),
            "horizon": args.horizon,
            "top_n": args.top_n,
            "sector_neutral_score": effective_sector_neutral,
            "min_daily_value": getattr(args, "min_daily_value", 0),
            "backtest_end": args.end,
        }
        run_name = Path(args.output).stem
        run_dir  = Path("runs") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pkl"
        latest_model.save(str(model_path))
        print(f"Saved unified model to {model_path}")

    run_name = Path(args.output).stem
    run_dir  = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summarize(results, sector_rows, output_path=str(run_dir / "report.png"), model=latest_model)

    # ── Auto-generate interactive dashboard ───────────────────────────────
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import dashboard as _dash
        print("\n[Dashboard] Generating interactive HTML dashboard ...")
        universe_df = _dash.query_universe(
            args.db, results["date"].str[:8].tolist() if results["date"].dtype == object
            else pd.to_datetime(results["date"]).dt.strftime("%Y%m%d").tolist(),
            args.min_market_cap,
        )
        picks_df = _dash.parse_top_picks(
            results if not isinstance(results["date"].iloc[0], str)
            else results.assign(date=pd.to_datetime(results["date"]))
        )
        sector_df_dash = pd.read_csv(run_dir / "sector_attribution.csv") \
            if (run_dir / "sector_attribution.csv").exists() else pd.DataFrame()

        results_dt = results.copy()
        if results_dt["date"].dtype == object:
            results_dt["date"] = pd.to_datetime(results_dt["date"])

        figs = {
            "cumret":        _dash.fig_cumret(results_dt),
            "3d_picks":      _dash.fig_3d_picks(picks_df),
            "3d_quintile":   _dash.fig_3d_quintile(results_dt),
            "3d_alpha":      _dash.fig_3d_risk_return(results_dt, universe_df),
            "3d_mc":         _dash.fig_marketcap_3d(universe_df),
            "return_dist":   _dash.fig_return_dist(results_dt),
            "ic":            _dash.fig_ic_bar(results_dt),
            "annual_sharpe": _dash.fig_annual_sharpe(results_dt),
            "drawdown":      _dash.fig_drawdown(results_dt),
            "turnover":      _dash.fig_turnover(results_dt),
            "sector":        _dash.fig_sector_bar(sector_df_dash),
            "mc_box":        _dash.fig_marketcap_box(universe_df),
            "vol_box":       _dash.fig_volume_box(universe_df),
        }
        html = _dash.build_html(figs, title=run_name)
        dash_path = run_dir / "dashboard.html"
        dash_path.write_text(html, encoding="utf-8")
        size_mb = dash_path.stat().st_size / 1_048_576
        print(f"[Dashboard] ✅ Saved → {dash_path}  ({size_mb:.1f} MB)")
        print(f"  open {dash_path}")
    except Exception as _e:
        print(f"[Dashboard] Warning: dashboard generation failed ({_e}). Run manually: python3 scripts/dashboard.py {run_name}")


def main() -> None:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
        """Show defaults and type-based metavars in --help output."""

    parser = argparse.ArgumentParser(
        description="Run unified model backtest",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["lgbm", "xgboost", "catboost"],
        help="Model family",
    )
    parser.add_argument("--db", type=str, default="krx_stock_data.db", help="SQLite DB path")
    parser.add_argument("--start", type=str, default="20100101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20260213", help="End date (YYYYMMDD)")
    parser.add_argument("--horizon", type=int, default=63, help="Forward return horizon (trading days)")
    parser.add_argument(
        "--benchmark", type=str, default="kospi200",
        choices=list(BENCHMARK_INDEX_MAP.keys()),
        help="Benchmark index for performance comparison "
             "(kospi200, kospi, kosdaq, kosdaq150, universe). Default: kospi200",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Portfolio size at each rebalance")
    parser.add_argument("--portfolio-size", type=int, default=100_000_000,
                        help="Portfolio size in KRW for discrete share rounding (default: 100,000,000 = 1억)")
    parser.add_argument("--train-years", type=int, default=5, help="Walk-forward training window in years")
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000, help="Minimum market cap filter")
    parser.add_argument("--max-market-cap", type=int, default=None, help="Maximum market cap filter (e.g. 5000000000000 = 5조, targets SMID-cap universe)")
    parser.add_argument("--time-decay", type=float, default=0.2, help="Sample time-decay strength")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Model learning rate")
    parser.add_argument("--n-estimators", type=int, default=3000, help="Max boosting rounds")
    parser.add_argument("--patience", type=int, default=300, help="Early-stopping rounds")
    parser.add_argument("--output", type=str, default="run",
                        help="Run name — all outputs saved to runs/<name>/ (results.csv, report.png, model.pkl, ...)")
    parser.add_argument("--model-out", type=str, default="",
                        help="(ignored) model saved to runs/<name>/model.pkl automatically")
    parser.add_argument("--workers", type=int, default=4, help="Parallel walk-forward workers")
    parser.add_argument("--model-jobs", type=int, default=0, help="Model threads per worker (0=auto)")
    parser.add_argument("--buy-fee", type=float, default=0.5, help="Buy fee percent per trade")
    parser.add_argument("--sell-fee", type=float, default=0.5, help="Sell fee percent per trade")
    parser.add_argument("--stress-mode", action="store_true", help="Enable realism stress tests")
    parser.add_argument("--vol-exclude-pct", type=float, default=0.10, help="Exclude top N%% volatility names")
    parser.add_argument("--sector-neutral-score", action="store_true", default=True, help="Enable sector-neutral ranking")
    parser.add_argument("--no-sector-neutral", action="store_true", help="Disable sector-neutral ranking")
    parser.add_argument("--buy-rank", type=int, default=10, help="Buy only if rank <= threshold")
    parser.add_argument("--hold-rank", type=int, default=90, help="Hold while rank <= threshold")
    parser.add_argument("--embargo-days", type=int, default=21, help="Purged embargo gap in trading days")
    parser.add_argument("--cash-out", action="store_true", default=True, help="Enable 20d regime cash-out rule")
    parser.add_argument("--no-cash-out", action="store_true", help="Disable cash-out rule")
    parser.add_argument("--exclude-years", type=str, default="", help="Comma-separated years to remove (e.g. 2023,2024)")
    parser.add_argument("--turnover-test-hold-rank", type=int, default=120, help="Hold-rank in turnover test variant")
    parser.add_argument("--turnover-test-smoothing-alpha", type=float, default=0.70, help="EMA alpha for turnover test")
    parser.add_argument("--disable-turnover-test", action="store_true", help="Disable turnover test variant")
    parser.add_argument("--save-picks", action="store_true", help="Save picked stocks per rebalance date to CSV")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache")
    parser.add_argument("--log-level", type=str, default="WARNING", help="Python logging level")
    # ── Stress Tests ──────────────────────────────────────────────────────
    parser.add_argument("--exec-lag", type=int, default=1,
                        help="Test 1 (Execution Lag): execute at T+N (default: 1 = next session)")
    parser.add_argument("--exec-price", type=str, default="close", choices=["open", "close"],
                        help="Execution price basis when --exec-lag > 0 (default: close)")
    parser.add_argument("--min-daily-value", type=int, default=0,
                        help="Test 5 (Liquidity): exclude stocks with daily trading value < N KRW (0=off, e.g. 10000000000 for 100억)")
    parser.add_argument("--twap-days", type=int, default=0,
                        help="TWAP execution: spread buy/sell over N trading days. "
                             "entry=avg(close T+1..T+N), exit=avg(close T+H-N+1..T+H). "
                             "Suspended days (value=0) excluded. Capped at horizon//3. "
                             "Overrides --exec-lag. (0=off, e.g. 5)")
    parser.add_argument("--permute-feature", type=str, default="",
                        help="Test 4 (Feature Permutation): shuffle specified feature(s) across ALL rows. "
                             "Use 'all' to permute every feature. Comma-separated for multiple. "
                             "If IC/Sharpe collapses → feature has real signal. "
                             "If performance maintained after 'all' → look-ahead leakage. (''=off)")
    parser.add_argument("--stop-loss", type=float, default=0.0,
                        help="Intraperiod stop-loss threshold (0=off, e.g. 0.10 = 10%%). "
                             "If a holding drops >N%% from entry during the period, "
                             "return is capped at -N%% and remainder is held as cash. "
                             "Requires --exec-lag >= 1.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    run(args)


if __name__ == "__main__":
    main()
