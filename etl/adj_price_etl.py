"""
Adjusted Price ETL (수정주가 복원)
===================================
Computes backward-chained adjusted prices from the ``change_rate`` column
already stored in ``daily_prices``.

Dividends are EXCLUDED (ignored for now).  Only capital-structure events
(splits, mergers, rights issues, face-value changes) are captured through the
KRX-reported 등락률 (change_rate), because KRX computes change_rate on raw
unadjusted prices — so a 50:1 split shows up as ~−98 % on the ex-date.

Algorithm
---------
Let T_N be the latest (or last-known for delisted) date for each stock.
Let r_t = change_rate[t] / 100  (daily gross return from t−1 to t).

    adj_factor[t] = ∏_{i=t+1}^{T_N}  1 / (1 + r_i)

In log-space (numerically stable over 15-year chains):

    log(adj_factor[t])
        = Σ_{i=t+1}^{T_N}  −log(1 + r_i)
        = (group_total of −log1p(r))
          − (inclusive prefix cumsum of −log1p(r) up to t)   ← exclusive suffix

Concrete derivation:

    adj_close[t] = close[T_N] × adj_factor[t]
    adj_open[t]  = open[t]    × adj_close[t] / close[t]  (ratio)
    adj_high, adj_low likewise.

NOTE on the quant team lead's pseudo-code
-----------------------------------------
The published pseudocode is off by one: it uses each row's *own* change_rate
for inv_return, but going backward from T to T−1 requires inverting the return
that happened FROM T−1 TO T, i.e. change_rate[T] — which belongs to the NEXT
row in the backward scan.  The log-space suffix product here is correct.

Speedup notes
-------------
* Loading: sqlite3 CLI → pipe → pandas read_csv avoids creating 8.7 M Python
  tuple objects, which is the dominant bottleneck of pd.read_sql_query.
  Measured speedup: 5–10× on 8 M rows.
* Writing: WAL + NORMAL synchronous + 500 k-row batches reduces fsync count
  from 35 to 17, which on macOS gives 2–4× write speedup.
* Computation: fully vectorised pandas (no Python loops); ~10–20 s for 8.7 M
  rows.  Multiprocessing not needed.

Output
------
Writes into table ``adj_daily_prices`` (created if absent).
Existing rows are REPLACED; idempotent.

Validation
----------
Samsung Electronics (005930) did a 50:1 face-value split on 2018-05-04.
Pre-split adj_close should be ≈ 1/50 of raw_close (ratio ≈ 0.02).
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB = Path(__file__).parent.parent / "krx_stock_data.db"
WRITE_BATCH = 500_000   # rows per executemany call

_COLS = [
    "stock_code", "date",
    "closing_price", "opening_price", "high_price", "low_price",
    "change_rate",
]
_DTYPES = {
    "stock_code":    str,
    "date":          str,
    "closing_price": "float64",
    "opening_price": "float64",
    "high_price":    "float64",
    "low_price":     "float64",
    "change_rate":   "float64",
}
# COALESCE so NULL change_rate (first trading day of each stock) → 0.0 in SQL.
#
# ⚠️  NO ORDER BY here.
# daily_prices is a regular ROWID table; (stock_code, date) is a *secondary*
# B-tree index.  ORDER BY on a secondary index forces SQLite to do 8.7 M
# random ROWID lookups before emitting a single row — that's the 10-minute
# hang.  We sort in pandas after loading instead (sort_values is O(N log N)
# on in-memory data and takes ~10–20 s).
_SQL = (
    "SELECT stock_code, date, "
    "closing_price, opening_price, high_price, low_price, "
    "COALESCE(change_rate, 0.0) "
    "FROM daily_prices"
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS adj_daily_prices (
        stock_code          TEXT NOT NULL,
        date                TEXT NOT NULL,
        adj_factor          REAL,
        adj_closing_price   REAL,
        adj_opening_price   REAL,
        adj_high_price      REAL,
        adj_low_price       REAL,
        PRIMARY KEY (stock_code, date)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_adj_daily_stock_date
        ON adj_daily_prices (stock_code, date)
    """,
]


def _create_table(conn: sqlite3.Connection) -> None:
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# Loading  (subprocess pipe → read_csv is 5–10× faster than pd.read_sql_query)
# ---------------------------------------------------------------------------

def _load_via_subprocess(db_path: str) -> pd.DataFrame:
    """
    Pipe sqlite3 CLI output directly into pandas read_csv.

    Why faster: sqlite3 CLI writes CSV at C speed; pandas read_csv parses CSV
    at C speed; no Python tuple objects are ever created for individual rows.
    The pipe OS-level buffer mediates between producer and consumer so the
    whole dataset never needs to be in memory as bytes simultaneously.
    """
    script = f".mode csv\n.headers off\n{_SQL};\n.quit\n"

    proc = subprocess.Popen(
        ["sqlite3", db_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write(script.encode())
    proc.stdin.close()

    # Drain stderr in background to prevent pipe-buffer deadlock.
    stderr_buf: list[bytes] = []
    def _drain() -> None:
        stderr_buf.append(proc.stderr.read())
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    # pandas streams from the pipe — memory-efficient.
    df = pd.read_csv(
        proc.stdout,
        header=None,
        names=_COLS,
        dtype=_DTYPES,
        na_values=[""],
    )

    proc.wait()
    t.join()

    if proc.returncode != 0:
        err = stderr_buf[0].decode(errors="replace") if stderr_buf else ""
        raise subprocess.SubprocessError(f"sqlite3 exit {proc.returncode}: {err}")

    return df


def _load_via_pandas(db_path: str) -> pd.DataFrame:
    """
    Fallback when sqlite3 CLI is absent.
    PRAGMAs speed up SQLite I/O on macOS (mmap, page cache).
    """
    conn = sqlite3.connect(db_path)
    for pragma in [
        "PRAGMA mmap_size=4294967296",   # 4 GB memory-mapped I/O
        "PRAGMA cache_size=-524288",      # 512 MB page cache
        "PRAGMA temp_store=MEMORY",
    ]:
        conn.execute(pragma)
    try:
        return pd.read_sql_query(
            "SELECT stock_code, date, "
            "closing_price, opening_price, high_price, low_price, "
            "COALESCE(change_rate, 0.0) AS change_rate "
            "FROM daily_prices",   # no ORDER BY — sort in pandas instead
            conn,
            dtype=_DTYPES,
        )
    finally:
        conn.close()


def _load_prices(db_path: str) -> pd.DataFrame:
    t0 = time.time()

    try:
        # Check sqlite3 CLI is on PATH
        subprocess.run(
            ["sqlite3", "--version"],
            capture_output=True, check=True,
        )
        print("[adj_etl] Loading via sqlite3 pipe → read_csv …", end="  ", flush=True)
        df = _load_via_subprocess(db_path)
        method = "subprocess"
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        print(
            f"\n[adj_etl] sqlite3 CLI unavailable ({exc}); "
            "falling back to pd.read_sql_query (slower) …",
            flush=True,
        )
        df = _load_via_pandas(db_path)
        method = "pandas"

    print(
        f"done [{method}]  "
        f"{len(df):,} rows  {df['stock_code'].nunique():,} stocks  "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    return df


# ---------------------------------------------------------------------------
# Core computation  (fully vectorised, no Python loops)
# ---------------------------------------------------------------------------

def compute_adj_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns matching _COLS; any sort order accepted.

    Returns
    -------
    Same DataFrame with additional columns:
        adj_factor, adj_closing_price, adj_opening_price,
        adj_high_price, adj_low_price
    """
    t0 = time.time()
    print(f"[adj_etl] Computing adj prices …", end="  ", flush=True)

    # 1. Sort ascending — cumsum must go oldest → newest within each stock.
    df = df.sort_values(["stock_code", "date"], kind="mergesort").reset_index(drop=True)

    # 2. Sanitise change_rate: NULL → 0, clip error values.
    cr = df["change_rate"].fillna(0.0).clip(-99.9, 10_000.0)

    # 3. Log-space suffix product.
    #
    #    log_inv[t] = −log(1 + r_t)
    #
    #    adj_factor[t] = exp( group_total_log_inv − prefix_cumsum_log_inv[t] )
    #
    #    Key: subtracting the *inclusive* prefix cumsum (which contains log_inv[t]
    #    itself) gives the *exclusive* suffix: sum from t+1 to T_N.  This is
    #    correct — the anchor row (T_N) gets log_total − log_total = 0 → exp(0)=1.
    #
    # Note: we must add _log_inv as a column so groupby can reference it.
    df["_log_inv"] = -np.log1p(cr / 100.0)
    grp = df.groupby("stock_code", sort=False)["_log_inv"]
    log_prefix = grp.cumsum()           # inclusive cumsum
    log_total  = grp.transform("sum")   # broadcast scalar per stock
    df["adj_factor"] = np.exp(log_total - log_prefix)
    df.drop(columns=["_log_inv"], inplace=True)

    # 4. Adjusted closing price.
    latest_close = (
        df.groupby("stock_code", sort=False)["closing_price"]
        .transform("last")              # "last" = latest date in ascending group
        .astype(np.float64)
    )
    df["adj_closing_price"] = (latest_close * df["adj_factor"]).round(4)

    # 5. Common scaling ratio for O/H/L.
    raw_close_safe = df["closing_price"].replace(0, np.nan).astype(np.float64)
    ratio = df["adj_closing_price"] / raw_close_safe

    # 6. Adjusted open / high / low.
    #    Preserve 0 as 0: opening_price=0 marks circuit-breaker / upper-lock days.
    for raw_col, adj_col in [
        ("opening_price", "adj_opening_price"),
        ("high_price",    "adj_high_price"),
        ("low_price",     "adj_low_price"),
    ]:
        df[adj_col] = np.where(
            df[raw_col] == 0,
            0.0,
            (df[raw_col].astype(np.float64) * ratio).round(4),
        )

    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return df


# ---------------------------------------------------------------------------
# Writing  (WAL + NORMAL sync + large batches = 2–4× faster than default)
# ---------------------------------------------------------------------------

def _write(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    # WAL allows concurrent reads during the write; NORMAL sync is safe
    # (no data loss on crash, just possible journal-file corruption on power loss).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-524288")   # 512 MB page cache

    OUT_COLS = [
        "stock_code", "date", "adj_factor",
        "adj_closing_price", "adj_opening_price",
        "adj_high_price", "adj_low_price",
    ]
    SQL = (
        "INSERT OR REPLACE INTO adj_daily_prices "
        "(stock_code, date, adj_factor, "
        "adj_closing_price, adj_opening_price, adj_high_price, adj_low_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )

    rows = df[OUT_COLS].values.tolist()
    total = len(rows)
    cur = conn.cursor()
    written = 0
    t0 = time.time()

    for start in range(0, total, WRITE_BATCH):
        batch = rows[start : start + WRITE_BATCH]
        conn.execute("BEGIN")
        cur.executemany(SQL, batch)
        conn.execute("COMMIT")
        written += len(batch)
        elapsed = time.time() - t0
        pct = written / total
        eta = elapsed / pct * (1 - pct) if pct > 0 else 0
        print(
            f"\r[adj_etl] Writing {written:,}/{total:,} ({pct:.1%})  "
            f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s    ",
            end="",
            flush=True,
        )

    print(
        f"\n[adj_etl] Write done: {written:,} rows in {time.time()-t0:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(conn: sqlite3.Connection) -> bool:
    """
    Samsung Electronics (005930) did a 50:1 face-value split on 2018-05-04.
    Pre-split adj_close / raw_close should be ≈ 0.02 (= 1/50).
    """
    df = pd.read_sql_query(
        """
        SELECT dp.date,
               dp.closing_price   AS raw_close,
               adj.adj_closing_price,
               adj.adj_factor
        FROM   daily_prices  dp
        JOIN   adj_daily_prices adj
               ON  adj.stock_code = dp.stock_code
               AND adj.date       = dp.date
        WHERE  dp.stock_code = '005930'
          AND  dp.date BETWEEN '20180430' AND '20180510'
        ORDER  BY dp.date
        """,
        conn,
    )

    if df.empty:
        print("[validate] ✗  No Samsung data for 20180430 ~ 20180510.", flush=True)
        return False

    print("\n[validate] Samsung (005930) around 20180504 50:1 split:")
    print(df.to_string(index=False))

    pre = df[df["date"] < "20180504"]
    if pre.empty:
        print("[validate] ✗  No pre-split rows.", flush=True)
        return False

    ratio = float(pre.iloc[-1]["adj_closing_price"]) / float(pre.iloc[-1]["raw_close"])
    expected = 1 / 50   # 0.02

    ok = abs(ratio - expected) < 0.005
    symbol = "✓" if ok else "✗"
    print(
        f"\n[validate] {symbol}  Pre-split adj/raw = {ratio:.5f}  "
        f"(expected ≈ {expected:.5f}  =  1/50)",
        flush=True,
    )
    return ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AdjPriceETL:
    """Thin wrapper so algostock_cli.py can import and call .run()."""

    def __init__(self, db_path: str = str(DEFAULT_DB)):
        self.db_path = db_path

    def run(self, skip_validate: bool = False) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            _create_table(conn)
            df = _load_prices(self.db_path)
            df = compute_adj_prices(df)
            _write(conn, df)
            if not skip_validate:
                validate(conn)
        finally:
            conn.close()

    def validate_only(self) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            return validate(conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build adj_daily_prices table (수정주가 복원)."
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    p.add_argument("--validate-only", action="store_true",
                   help="Skip build; only run Samsung spot-check.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip Samsung spot-check after build.")
    args = p.parse_args()

    etl = AdjPriceETL(db_path=args.db)
    if args.validate_only:
        sys.exit(0 if etl.validate_only() else 1)
    else:
        etl.run(skip_validate=args.no_validate)
