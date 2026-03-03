#!/usr/bin/env python3
"""
Backtest Verification Tool — cross-checks picks.csv against FinanceDataReader.

For every trade in picks.csv, fetches an independent adjusted-close price
series from Naver Finance (수정주가) and recomputes the holding-period return.
Naver Finance prices are retroactively adjusted for splits/rights offerings,
matching pykrx's methodology used in our ETL.

Delisted / unavailable stocks (fdr returns empty data) are skipped from
the accuracy calculation but included in a separate delisted_report.csv.

Usage
-----
  python3 verification/verify_backtest.py --run myrun
  python3 verification/verify_backtest.py --picks runs/myrun/picks.csv
  python3 verification/verify_backtest.py --run myrun --tolerance 0.03
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import FinanceDataReader as fdr
except ImportError:
    print("ERROR: FinanceDataReader not installed.")
    print("  pip install finance-datareader")
    sys.exit(1)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_iso(date_str: str) -> str:
    """YYYYMMDD → YYYY-MM-DD.  ISO strings pass through unchanged."""
    s = str(date_str).strip().replace("-", "")
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return date_str


def _to_yyyymmdd(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD."""
    return str(date_str).replace("-", "")


def _add_buffer(iso_date: str, days: int = 10) -> str:
    """Return iso_date + N calendar days (for holiday buffer at tail)."""
    dt = datetime.strptime(iso_date, "%Y-%m-%d") + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


def _fetch(code: str, start_iso: str, end_iso: str, retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLC prices from FinanceDataReader (Naver Finance).

    Naver Finance returns 수정주가 (retroactively adjusted prices) that match
    pykrx's adjustment methodology, unlike raw KRX prices which are unadjusted.

    Source priority: NAVER:{code} → bare code (fallback for delisted).
    Returns a DataFrame with a DatetimeIndex and 'open' + 'close' columns.
    Returns an empty DataFrame on failure (e.g. delisted stock).
    """
    sources = [f"NAVER:{code}", code]
    for src in sources:
        for attempt in range(retries):
            try:
                df = fdr.DataReader(src, start_iso, end_iso)
                if df is None or df.empty:
                    break
                df.index = pd.to_datetime(df.index)
                # Naver returns: Open High Low Close Volume Change
                open_col  = next((c for c in df.columns if c.lower() == "open"),  None)
                close_col = next((c for c in df.columns if c.lower() in ("close", "adj close")), None)
                if close_col is None:
                    close_col = df.columns[0]
                cols = {}
                if open_col:
                    cols[open_col] = "open"
                cols[close_col] = "close"
                return df[list(cols.keys())].rename(columns=cols)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(1)
    return pd.DataFrame(columns=["open", "close"])


def _exec_price_from_col(fwd_col: str) -> str:
    """Infer execution price type ('close' or 'open') from the forward return column name."""
    if "open" in fwd_col.lower():
        return "open"
    return "close"  # default: close


def _next_price(price_df: pd.DataFrame, signal_date_iso: str, use: str = "close") -> float:
    """
    Return the price on the first trading day STRICTLY AFTER signal_date_iso.

    use='close' → closing price (matches forward_return_*_lag*_close columns)
    use='open'  → opening price (matches forward_return_*_lag*_open columns)
    Falls back to close if the requested column is unavailable.
    Returns NaN if no data found within 10 calendar days.
    """
    if price_df.empty:
        return np.nan
    signal = pd.Timestamp(signal_date_iso)
    limit = signal + timedelta(days=10)
    candidates = price_df.index[(price_df.index > signal) & (price_df.index <= limit)]
    if not len(candidates):
        return np.nan
    row = price_df.loc[candidates[0]]
    if use == "open" and "open" in price_df.columns and pd.notna(row.get("open")):
        return float(row["open"])
    return float(row["close"])


def _price_on(price_df: pd.DataFrame, iso_date: str, use: str = "close") -> float:
    """
    Return the price on iso_date (or the next available trading day within 10 days).

    use='close' → closing price  (matches sell date stored in picks.csv 매도날짜)
    use='open'  → opening price
    Falls back to close if open is unavailable.
    Returns NaN if no data found.
    """
    if price_df.empty:
        return np.nan
    target = pd.Timestamp(iso_date)
    limit = target + timedelta(days=10)
    candidates = price_df.index[(price_df.index >= target) & (price_df.index <= limit)]
    if not len(candidates):
        return np.nan
    row = price_df.loc[candidates[0]]
    if use == "open" and "open" in price_df.columns and pd.notna(row.get("open")):
        return float(row["open"])
    return float(row["close"])


# Keep old names as aliases so any external callers don't break
def _next_open(price_df: pd.DataFrame, signal_date_iso: str) -> float:
    return _next_price(price_df, signal_date_iso, use="open")


def _open_on(price_df: pd.DataFrame, iso_date: str) -> float:
    return _price_on(price_df, iso_date, use="open")


def _detect_fwd_col(picks_df: pd.DataFrame) -> str:
    """Auto-detect the forward return column in picks.csv."""
    for c in picks_df.columns:
        if "forward_return" in c.lower() or "fwd_return" in c.lower():
            return c
    return ""


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def verify_picks(
    picks_df: pd.DataFrame,
    fwd_col: str,
    tolerance: float,
    request_delay: float = 0.3,
    exec_price: str = "",
) -> pd.DataFrame:
    """
    For each trade in picks_df, fetch external prices and compute return.

    Parameters
    ----------
    picks_df      : DataFrame from picks.csv
    fwd_col       : column holding the backtest's reported holding-period return
    tolerance     : max |return diff| to classify as 'match'
    request_delay : seconds between fdr requests (rate limiting)

    Returns
    -------
    DataFrame with one row per trade and verification columns appended.
    """
    # Auto-detect execution price type from column name if not explicitly given
    price_type = exec_price if exec_price in ("open", "close") else _exec_price_from_col(fwd_col)
    print(f"  Exec price: {price_type}  (inferred from fwd_col: {fwd_col})", flush=True)

    records = []
    stocks = picks_df["stock_code"].unique()
    n_total = len(stocks)

    for idx, code in enumerate(stocks, 1):
        print(f"\r  [{idx:>4}/{n_total}] {code:<10}", end="", flush=True)
        time.sleep(request_delay)

        group = picks_df[picks_df["stock_code"] == code].copy()

        # Date range for this stock across all its rebalances
        buy_dates  = group["date"].astype(str).tolist()
        sell_dates = group["매도날짜"].dropna().astype(str).tolist() if "매도날짜" in group.columns else []
        all_dates  = [_to_iso(d) for d in buy_dates + sell_dates if str(d) not in ("nan", "NaT", "")]
        if not all_dates:
            continue

        start_iso = min(all_dates)
        end_iso   = _add_buffer(max(all_dates), days=15)

        price_df = _fetch(code, start_iso, end_iso)

        # Detect delisted: fdr returned no data at all
        is_unavailable = price_df.empty

        for _, row in group.iterrows():
            buy_date_raw  = str(row["date"])
            sell_date_raw = str(row.get("매도날짜", ""))
            bt_return     = float(row[fwd_col]) if fwd_col and pd.notna(row.get(fwd_col)) else np.nan

            rec = {
                "stock_code":     code,
                "name":           row.get("name", ""),
                "sector":         row.get("sector", ""),
                "buy_date":       buy_date_raw,
                "sell_date":      sell_date_raw if sell_date_raw not in ("nan", "NaT", "") else "",
                "bt_buy_price":   row.get("매수가", np.nan),
                "bt_sell_price":  row.get("매도가", np.nan),
                "bt_return":      bt_return,
                "fdr_buy_price":  np.nan,
                "fdr_sell_price": np.nan,
                "fdr_return":     np.nan,
                "return_diff":    np.nan,
                "abs_diff":       np.nan,
                "status":         "unknown",
            }

            if is_unavailable:
                rec["status"] = "delisted_or_unavailable"
                records.append(rec)
                continue

            sell_date_iso = _to_iso(sell_date_raw) if sell_date_raw not in ("nan", "NaT", "") else ""
            if not sell_date_iso:
                rec["status"] = "no_sell_date"
                records.append(rec)
                continue

            # Detect mid-holding delisting: sell_date past last available fdr date
            last_fdr_date = price_df.index.max()
            sell_ts = pd.Timestamp(sell_date_iso)
            if sell_ts > last_fdr_date + timedelta(days=30):
                rec["status"] = "delisted_or_unavailable"
                records.append(rec)
                continue

            # Buy: T+1 price using exec price type (close or open)
            # Sell: price on 매도날짜 (already T+horizon+exec_lag in picks.csv)
            fdr_buy  = _next_price(price_df, _to_iso(buy_date_raw), use=price_type)
            fdr_sell = _price_on(price_df, sell_date_iso, use=price_type)

            rec["fdr_buy_price"]  = round(fdr_buy,  0) if pd.notna(fdr_buy)  else np.nan
            rec["fdr_sell_price"] = round(fdr_sell, 0) if pd.notna(fdr_sell) else np.nan

            if pd.notna(fdr_buy) and pd.notna(fdr_sell) and fdr_buy > 0:
                fdr_ret = fdr_sell / fdr_buy - 1.0
                rec["fdr_return"] = fdr_ret

                if pd.notna(bt_return):
                    diff = bt_return - fdr_ret
                    rec["return_diff"] = diff
                    rec["abs_diff"]    = abs(diff)
                    rec["status"]      = "match" if abs(diff) <= tolerance else "discrepancy"
                else:
                    rec["status"] = "bt_return_missing"
            else:
                rec["status"] = "fdr_price_missing"

            records.append(rec)

    print()  # newline after progress line
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_and_save_report(
    verified: pd.DataFrame,
    tolerance: float,
    out_dir: Path,
) -> None:
    total       = len(verified)
    matched     = int((verified["status"] == "match").sum())
    discrepant  = int((verified["status"] == "discrepancy").sum())
    delisted    = int((verified["status"] == "delisted_or_unavailable").sum())
    no_sell     = int((verified["status"] == "no_sell_date").sum())
    price_miss  = int((verified["status"] == "fdr_price_missing").sum())
    bt_miss     = int((verified["status"] == "bt_return_missing").sum())
    verified_n  = matched + discrepant          # trades with full comparison

    match_rate  = matched / max(verified_n, 1)

    valid        = verified[verified["abs_diff"].notna()]
    mean_abs_err = float(valid["abs_diff"].mean()) if len(valid) else np.nan
    median_abs   = float(valid["abs_diff"].median()) if len(valid) else np.nan
    max_abs_err  = float(valid["abs_diff"].max()) if len(valid) else np.nan

    lines = []
    lines.append("=" * 72)
    lines.append("BACKTEST VERIFICATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Total trade-records             : {total:,}")
    lines.append(f"  Fully verified (both returns)  : {verified_n:,}")
    lines.append(f"    ✅  Match  (|Δ| ≤ {tolerance:.0%})        : {matched:,}  ({match_rate:.1%})")
    lines.append(f"    ⚠️   Discrepancy (|Δ| > {tolerance:.0%})   : {discrepant:,}")
    lines.append(f"  🔴  Delisted / unavailable      : {delisted:,}  (skipped from accuracy)")
    lines.append(f"  ❓  No sell date                : {no_sell:,}")
    lines.append(f"  ❓  fdr price missing           : {price_miss:,}")
    lines.append(f"  ❓  BT return missing           : {bt_miss:,}")
    lines.append("")
    lines.append("Return accuracy (fully verified trades):")
    lines.append(f"  Mean   |Δreturn|  : {mean_abs_err:.3%}" if pd.notna(mean_abs_err) else "  Mean |Δreturn|  : N/A")
    lines.append(f"  Median |Δreturn|  : {median_abs:.3%}"   if pd.notna(median_abs)   else "  Median |Δreturn|: N/A")
    lines.append(f"  Max    |Δreturn|  : {max_abs_err:.3%}"  if pd.notna(max_abs_err)  else "  Max  |Δreturn|  : N/A")

    # Top discrepancies
    disc = verified[verified["status"] == "discrepancy"].sort_values("abs_diff", ascending=False)
    if len(disc):
        lines.append("")
        lines.append("─" * 72)
        lines.append(f"TOP DISCREPANCIES  (>{tolerance:.0%} return diff)  — top 30 shown")
        lines.append(
            f"{'Code':<10} {'Name':<14} {'BuyDate':<11} {'SellDate':<11}"
            f" {'BT_Ret':>8} {'FDR_Ret':>8} {'Diff':>8}"
        )
        lines.append("─" * 72)
        for _, r in disc.head(30).iterrows():
            lines.append(
                f"{r['stock_code']:<10} {str(r['name'])[:13]:<14} "
                f"{str(r['buy_date']):<11} {str(r['sell_date']):<11} "
                f"{r['bt_return']:>8.2%} {r['fdr_return']:>8.2%} {r['return_diff']:>+8.2%}"
            )

    # Delisted stocks
    delist = verified[verified["status"] == "delisted_or_unavailable"]
    if len(delist):
        lines.append("")
        lines.append("─" * 72)
        lines.append(f"DELISTED / UNAVAILABLE STOCKS  ({delist['stock_code'].nunique()} unique codes)")
        lines.append("  These stocks could not be fetched from FinanceDataReader.")
        lines.append("  Excluded from return accuracy — details in delisted_report.csv")
        lines.append("")
        summary = (
            delist.groupby(["stock_code", "name"])
            .agg(n_trades=("buy_date", "count"),
                 first_buy=("buy_date", "min"),
                 last_buy=("buy_date", "max"))
            .reset_index()
            .sort_values("n_trades", ascending=False)
        )
        lines.append(f"  {'Code':<10} {'Name':<20} {'Trades':>6}  {'FirstBuy':<11} {'LastBuy':<11}")
        for _, r in summary.iterrows():
            lines.append(
                f"  {r['stock_code']:<10} {str(r['name'])[:19]:<20} "
                f"{r['n_trades']:>6}  {str(r['first_buy']):<11} {str(r['last_buy']):<11}"
            )

    lines.append("=" * 72)
    report_text = "\n".join(lines)
    print(report_text)

    # ── Save files ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full detail CSV
    detail_path = out_dir / "verification_detail.csv"
    verified.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed CSV  → {detail_path}")

    # Delisted-only CSV
    if len(delist):
        delist_path = out_dir / "delisted_report.csv"
        delist[["stock_code", "name", "sector", "buy_date", "sell_date",
                "bt_buy_price", "bt_sell_price", "bt_return"]].to_csv(
            delist_path, index=False, encoding="utf-8-sig"
        )
        print(f"Delisted CSV  → {delist_path}")

    # Discrepancy-only CSV
    if len(disc):
        disc_path = out_dir / "discrepancy_report.csv"
        disc.to_csv(disc_path, index=False, encoding="utf-8-sig")
        print(f"Discrepancy CSV → {disc_path}")

    # Summary text
    summary_path = out_dir / "verification_summary.txt"
    summary_path.write_text(report_text, encoding="utf-8")
    print(f"Summary TXT   → {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify backtest picks against FinanceDataReader (external API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Run verification for a named backtest run
  python3 verification/verify_backtest.py --run myrun

  # Direct path to picks.csv
  python3 verification/verify_backtest.py --picks runs/myrun/picks.csv

  # Stricter tolerance (1%%)
  python3 verification/verify_backtest.py --run myrun --tolerance 0.01

  # Save output to a separate verification folder
  python3 verification/verify_backtest.py --run myrun --out verification/myrun_check

Notes
-----
  • Returns are compared using Naver Finance 수정주가 (adjusted close),
    which matches our pykrx ETL's retroactive adjustment methodology.
    Small differences (< 2%%) are still expected due to open-vs-close timing
    (backtest buys at lag1_open; verification uses Naver close).
  • Delisted stocks (fdr returns empty data) are skipped from accuracy
    but listed in delisted_report.csv.
  • Rate-limited to ~3 requests/second by default (--delay 0.3).
        """,
    )
    parser.add_argument("--run",       type=str,   default="",
                        help="Run name — loads runs/<name>/picks.csv")
    parser.add_argument("--picks",     type=str,   default="",
                        help="Direct path to picks.csv")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Max |return diff| to classify as a match (default 0.02 = 2%%)")
    parser.add_argument("--out",       type=str,   default="",
                        help="Output directory (default: same dir as picks.csv)")
    parser.add_argument("--fwd-col",   type=str,   default="",
                        help="Forward return column in picks.csv (auto-detected if empty)")
    parser.add_argument("--delay",      type=float, default=0.3,
                        help="Seconds between fdr API requests (default 0.3)")
    parser.add_argument("--exec-price", type=str,   default="",
                        choices=["", "open", "close"],
                        help="Execution price type: 'open' or 'close' (auto-detected from fwd_col if empty)")
    args = parser.parse_args()

    # ── Resolve picks path ──────────────────────────────────────────────────
    if args.picks:
        picks_path = Path(args.picks)
    elif args.run:
        picks_path = ROOT / "runs" / args.run / "picks.csv"
    else:
        parser.error("Provide --run <name>  or  --picks <path>")

    if not picks_path.exists():
        print(f"ERROR: picks.csv not found: {picks_path}")
        sys.exit(1)

    # ── Output directory ────────────────────────────────────────────────────
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = picks_path.parent / "verification"

    # ── Load picks ──────────────────────────────────────────────────────────
    print(f"Loading  : {picks_path}")
    picks_df = pd.read_csv(
        picks_path,
        dtype={"stock_code": str, "date": str, "매도날짜": str},
        low_memory=False,
    )
    print(f"  Rows   : {len(picks_df):,}")
    print(f"  Stocks : {picks_df['stock_code'].nunique():,} unique codes")

    # ── Forward return column ───────────────────────────────────────────────
    fwd_col = args.fwd_col or _detect_fwd_col(picks_df)
    if fwd_col and fwd_col in picks_df.columns:
        print(f"  Fwd col: {fwd_col}")
    else:
        print(f"  WARNING: forward return column not found — return accuracy will be N/A")
        print(f"  Available columns: {list(picks_df.columns)}")
        picks_df["_placeholder"] = np.nan
        fwd_col = "_placeholder"

    print(f"  Tolerance: ±{args.tolerance:.1%}")
    print(f"\nFetching prices from FinanceDataReader...")
    print(f"  (~{picks_df['stock_code'].nunique()} API calls, ETA "
          f"~{picks_df['stock_code'].nunique() * args.delay / 60:.1f} min)\n")

    # ── Verify ──────────────────────────────────────────────────────────────
    verified = verify_picks(
        picks_df,
        fwd_col=fwd_col,
        tolerance=args.tolerance,
        request_delay=args.delay,
        exec_price=getattr(args, "exec_price", ""),
    )

    # ── Report ──────────────────────────────────────────────────────────────
    print_and_save_report(verified, tolerance=args.tolerance, out_dir=out_dir)


if __name__ == "__main__":
    main()
