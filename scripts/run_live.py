#!/usr/bin/env python3
"""
Live trading runner — checks rebalancing schedule and submits orders via Kiwoom REST API.

Usage
-----
  # 오늘 리밸런싱 필요한지 확인만
  python3 scripts/run_live.py --run myrun

  # 확인 + 주문까지 (모의투자)
  python3 scripts/run_live.py --run myrun --execute

  # 여러 run 중 선택
  python3 scripts/run_live.py

Setup
-----
  환경변수로 API 키 설정 (또는 .env 파일):
    KIWOOM_APP_KEY=...
    KIWOOM_APP_SECRET=...
    KIWOOM_ACCOUNT=12345678-01   # 계좌번호
    KIWOOM_MOCK=true             # 모의투자 여부
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

RUNS_DIR  = Path("runs")
LIVE_DIR  = Path("live")
STATE_FILE = LIVE_DIR / "state.json"

# ---------------------------------------------------------------------------
# Trading calendar (Korean market)
# ---------------------------------------------------------------------------

def _get_krx_calendar():
    try:
        import exchange_calendars as ec
        return ec.get_calendar("XKRX")
    except Exception:
        return None


def _trading_days_between(start: str, end: str) -> list[str]:
    """Return list of KRX trading days (YYYYMMDD) between start and end inclusive."""
    cal = _get_krx_calendar()
    if cal is None:
        # Fallback: weekdays only
        dates = pd.bdate_range(start, end)
        return [d.strftime("%Y%m%d") for d in dates]
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    sessions = cal.sessions_in_range(s, e)
    return [d.strftime("%Y%m%d") for d in sessions]


def _add_trading_days(date_str: str, n: int) -> str:
    """Return the date that is n KRX trading days after date_str."""
    cal = _get_krx_calendar()
    ts = pd.Timestamp(date_str)
    if cal is None:
        result = ts + pd.offsets.BDay(n)
        return result.strftime("%Y%m%d")
    result = cal.session_offset(ts, n)
    return result.strftime("%Y%m%d")


def _tomorrow_str() -> str:
    return (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")


def _today_str() -> str:
    return datetime.today().strftime("%Y%m%d")


def _next_trading_day(date_str: str) -> str:
    return _add_trading_days(date_str, 1)

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted([
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "results.csv").exists() and (d / "picks.csv").exists()
    ])


def pick_run(run_name: str) -> str:
    """Interactively pick a run if not specified."""
    runs = list_runs()
    if not runs:
        print("ERROR: No completed runs found in runs/")
        sys.exit(1)
    if run_name and run_name in runs:
        return run_name
    if run_name and run_name not in runs:
        print(f"WARNING: run '{run_name}' not found.")

    print("\nAvailable runs:")
    for i, r in enumerate(runs, 1):
        results_csv = RUNS_DIR / r / "results.csv"
        picks_csv   = RUNS_DIR / r / "picks.csv"
        try:
            res = pd.read_csv(results_csv)
            last_date = res["date"].max()
            n_rebals  = len(res)
            horizon   = _extract_horizon(picks_csv)
            print(f"  [{i}] {r:<20}  last_rebal={last_date}  rebals={n_rebals}  horizon={horizon}d")
        except Exception:
            print(f"  [{i}] {r}")

    while True:
        choice = input("\nSelect run number (or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(runs):
            return runs[int(choice) - 1]
        if choice in runs:
            return choice
        print("Invalid choice, try again.")


def _extract_horizon(picks_csv: Path) -> int:
    """Extract horizon from forward_return column name, e.g. forward_return_42d → 42."""
    try:
        cols = pd.read_csv(picks_csv, nrows=0).columns.tolist()
        for c in cols:
            m = re.search(r"forward_return_(\d+)d", c)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 21  # fallback

# ---------------------------------------------------------------------------
# Schedule logic
# ---------------------------------------------------------------------------

def compute_next_rebal(run_dir: Path) -> dict:
    """
    Given a run directory, compute the next rebalancing signal date.

    Returns dict with:
        last_rebal      : last rebalancing signal date (YYYYMMDD)
        horizon         : horizon in trading days
        next_rebal      : next rebalancing signal date (YYYYMMDD)
        next_exec       : execution date = next_rebal + 1 trading day (YYYYMMDD)
        trading_days_left : trading days from today until next_exec
        status          : 'today' | 'tomorrow' | 'future' | 'overdue'
    """
    picks_csv   = run_dir / "picks.csv"
    results_csv = run_dir / "results.csv"

    results = pd.read_csv(results_csv, dtype={"date": str})
    last_rebal = str(results["date"].max()).replace("-", "")

    horizon = _extract_horizon(picks_csv)
    today   = _today_str()

    # Advance rebal signal date until next_exec is today or in the future.
    # Track skipped rebalancings so we can warn the user.
    candidate      = last_rebal
    skipped_rebals = []
    while True:
        next_rebal = _add_trading_days(candidate, horizon)
        next_exec  = _next_trading_day(next_rebal)
        if next_exec >= today:
            break
        skipped_rebals.append({"signal": next_rebal, "exec": next_exec})
        candidate = next_rebal

    tomorrow  = _tomorrow_str()
    days_left = len(_trading_days_between(today, next_exec)) - 1

    if next_exec == today:
        status = "today"
    elif next_exec == tomorrow:
        status = "tomorrow"
    else:
        status = "future"

    return {
        "last_rebal":         last_rebal,
        "horizon":            horizon,
        "next_rebal":         next_rebal,
        "next_exec":          next_exec,
        "trading_days_left":  days_left,
        "status":             status,
        "skipped_rebals":     skipped_rebals,
    }


def get_current_holdings(run_dir: Path) -> set[str]:
    """Return the set of stock codes held from the last rebalancing."""
    picks_csv = run_dir / "picks.csv"
    picks = pd.read_csv(picks_csv, dtype={"stock_code": str, "date": str})
    last_date = picks["date"].max()
    return set(picks[picks["date"] == last_date]["stock_code"].tolist())


# ---------------------------------------------------------------------------
# Live state management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load live trading state from live/state.json."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "last_executed_rebal": None,   # 마지막으로 실행한 리밸런싱 신호일
            "current_holdings": [],        # 현재 실제 보유 종목 코드 목록
            "run_name": None,
        }
    import json
    return json.loads(STATE_FILE.read_text())


def save_state(state: dict) -> None:
    import json
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def save_order_log(exec_date: str, sell_orders: list, buy_orders: list, new_holdings: list) -> None:
    import json
    log = {
        "exec_date":    exec_date,
        "sell_orders":  sell_orders,
        "buy_orders":   buy_orders,
        "new_holdings": new_holdings,
        "logged_at":    datetime.now().isoformat(),
    }
    order_path = LIVE_DIR / "orders" / f"{exec_date}.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"  주문 내역 저장 → {order_path}")

# ---------------------------------------------------------------------------
# Kiwoom REST API client
# ---------------------------------------------------------------------------

class KiwoomClient:
    """
    Thin wrapper around Kiwoom REST API (모의투자 / 실거래).

    공식 문서: https://apiportal.kiwoom.com
    환경변수:
        KIWOOM_APP_KEY     : 앱 키
        KIWOOM_APP_SECRET  : 앱 시크릿
        KIWOOM_ACCOUNT     : 계좌번호 (e.g. "12345678-01")
        KIWOOM_MOCK        : "true" → 모의투자, "false" → 실거래 (기본: true)
    """

    MOCK_BASE = "https://mockapi.kiwoom.com"   # 모의투자 엔드포인트
    REAL_BASE = "https://openapi.kiwoom.com"   # 실거래 엔드포인트

    def __init__(self):
        self.app_key    = os.environ.get("KIWOOM_APP_KEY", "")
        self.app_secret = os.environ.get("KIWOOM_APP_SECRET", "")
        self.account    = os.environ.get("KIWOOM_ACCOUNT", "")
        self.mock       = os.environ.get("KIWOOM_MOCK", "true").lower() != "false"
        self.base_url   = self.MOCK_BASE if self.mock else self.REAL_BASE
        self._token: str = ""

    def _check_credentials(self) -> bool:
        if not self.app_key or not self.app_secret or not self.account:
            print(
                "\n[Kiwoom] 환경변수가 설정되지 않았습니다.\n"
                "  export KIWOOM_APP_KEY=...\n"
                "  export KIWOOM_APP_SECRET=...\n"
                "  export KIWOOM_ACCOUNT=12345678-01\n"
                "  export KIWOOM_MOCK=true\n"
            )
            return False
        return True

    def authenticate(self) -> bool:
        """OAuth2 token 발급."""
        if not self._check_credentials():
            return False
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/oauth2/token",
                headers={"Content-Type": "application/json"},
                json={
                    "grant_type":    "client_credentials",
                    "appkey":        self.app_key,
                    "appsecretkey":  self.app_secret,
                },
                timeout=10,
            )
            resp.raise_for_status()
            self._token = resp.json().get("access_token", "")
            print(f"[Kiwoom] 인증 성공 ({'모의투자' if self.mock else '실거래'})")
            return True
        except Exception as e:
            print(f"[Kiwoom] 인증 실패: {e}")
            return False

    def _headers(self) -> dict:
        return {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self._token}",
            "appkey":        self.app_key,
            "appsecretkey":  self.app_secret,
        }

    def get_holdings(self) -> pd.DataFrame:
        """현재 보유 종목 조회."""
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/v1/account/balance",
                headers=self._headers(),
                params={"account": self.account},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            # 응답 구조는 Kiwoom 문서 기준으로 파싱 — 필요시 조정
            holdings = data.get("output", [])
            return pd.DataFrame(holdings)
        except Exception as e:
            print(f"[Kiwoom] 보유종목 조회 실패: {e}")
            return pd.DataFrame()

    def order_sell(self, stock_code: str, quantity: int, price: int = 0) -> bool:
        """매도 주문 (price=0 → 시장가)."""
        return self._order(stock_code, quantity, price, side="sell")

    def order_buy(self, stock_code: str, quantity: int, price: int = 0) -> bool:
        """매수 주문 (price=0 → 시장가)."""
        return self._order(stock_code, quantity, price, side="buy")

    def _order(self, stock_code: str, quantity: int, price: int, side: str) -> bool:
        try:
            import requests
            payload = {
                "account":    self.account,
                "stock_code": stock_code,
                "order_type": "01" if side == "buy" else "02",   # 01=매수, 02=매도
                "quantity":   quantity,
                "price":      price,        # 0=시장가
                "price_type": "01" if price == 0 else "00",      # 00=지정가, 01=시장가
            }
            resp = requests.post(
                f"{self.base_url}/v1/order/stock",
                headers=self._headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            order_no = result.get("order_no", "?")
            print(f"  [주문완료] {side.upper()} {stock_code} x{quantity}  주문번호={order_no}")
            return True
        except Exception as e:
            print(f"  [주문실패] {side.upper()} {stock_code}: {e}")
            return False

# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

def build_orders(
    new_picks: pd.DataFrame,
    current_holdings: set[str],
    top_n: int,
    portfolio_krw: int,
) -> tuple[list[dict], list[dict]]:
    """
    Compare current holdings vs new picks to generate buy/sell lists.

    Returns (sell_orders, buy_orders).
    Each order: {stock_code, name, quantity, price}
    """
    new_codes  = set(new_picks.head(top_n)["stock_code"].tolist())
    sell_codes = current_holdings - new_codes
    buy_codes  = new_codes - current_holdings

    per_stock_krw = portfolio_krw // max(len(new_codes), 1)

    sell_orders = []
    for code in sell_codes:
        sell_orders.append({
            "stock_code": code,
            "name":       new_picks[new_picks["stock_code"] == code]["name"].values[0]
                          if code in new_picks["stock_code"].values else code,
            "quantity":   0,    # 전량매도: 실제 보유수량은 API 잔고 조회로 채워야 함
            "price":      0,    # 시장가
        })

    buy_orders = []
    for _, row in new_picks[new_picks["stock_code"].isin(buy_codes)].iterrows():
        price = int(row.get("closing_price", row.get("매수가", 0)))
        qty   = per_stock_krw // max(price, 1) if price > 0 else 0
        buy_orders.append({
            "stock_code": str(row["stock_code"]),
            "name":       str(row.get("name", "")),
            "quantity":   qty,
            "price":      0,    # 시장가
        })

    return sell_orders, buy_orders


def print_order_summary(
    schedule: dict,
    current_holdings: set[str],
    new_picks: pd.DataFrame,
    sell_orders: list[dict],
    buy_orders:  list[dict],
    top_n: int,
) -> None:
    new_codes  = set(new_picks.head(top_n)["stock_code"].tolist())
    hold_codes = current_holdings & new_codes

    print("\n" + "=" * 60)
    print("  REBALANCING PLAN")
    print("=" * 60)
    print(f"  다음 리밸런싱 신호일: {schedule['next_rebal']}")
    print(f"  실행일 (T+1):        {schedule['next_exec']}")
    print(f"  현재 보유 종목:      {len(current_holdings)}개")
    print(f"  신규 포트폴리오:     {top_n}개")

    if hold_codes:
        print(f"\n  📌 유지 종목 ({len(hold_codes)}개): {', '.join(sorted(hold_codes))}")

    if sell_orders:
        print(f"\n  🔴 매도 ({len(sell_orders)}개):")
        for o in sell_orders:
            print(f"     {o['stock_code']}  {o['name']}")

    if buy_orders:
        print(f"\n  🟢 매수 ({len(buy_orders)}개):")
        for o in buy_orders:
            print(f"     {o['stock_code']}  {o['name']}  수량={o['quantity']}주  시장가")

    print("=" * 60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live rebalancing runner with Kiwoom REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python3 scripts/run_live.py                        # run 선택 후 스케줄 확인
  python3 scripts/run_live.py --run myrun            # 특정 run 스케줄 확인
  python3 scripts/run_live.py --run myrun --execute  # 주문 실행까지
        """,
    )
    parser.add_argument("--run",       type=str, default="",
                        help="Run name (runs/ 하위 폴더명). 없으면 목록에서 선택")
    parser.add_argument("--execute",   action="store_true",
                        help="실제로 주문 제출 (기본: dry-run만)")
    parser.add_argument("--top",       type=int, default=0,
                        help="포트폴리오 종목 수 (0=model.pkl에서 자동)")
    parser.add_argument("--portfolio", type=int, default=100_000_000,
                        help="총 투자금액 KRW (기본: 1억)")
    parser.add_argument("--no-update", action="store_true",
                        help="update-all 건너뜀")
    parser.add_argument("--force", action="store_true",
                        help="이미 실행된 리밸런싱도 강제 재실행")
    args = parser.parse_args()

    # ── 1. Run 선택 ──────────────────────────────────────────────────────
    run_name = pick_run(args.run)
    run_dir  = RUNS_DIR / run_name
    print(f"\n[Run] {run_name}")

    # ── 2. DB 최신화 ─────────────────────────────────────────────────────
    if not args.no_update:
        print("\n[1/4] DB 업데이트 (update-all)...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/algostock_cli.py", "update-all"],
            check=False,
        )
        if result.returncode != 0:
            print("  WARNING: update-all 일부 실패. 계속 진행합니다.")
    else:
        print("\n[1/4] DB 업데이트 건너뜀 (--no-update)")

    # ── 3. 리밸런싱 스케줄 확인 ─────────────────────────────────────────
    print("\n[2/4] 리밸런싱 스케줄 확인...")
    schedule = compute_next_rebal(run_dir)
    horizon  = schedule["horizon"]

    # 이미 실행된 리밸런싱인지 확인
    state = load_state()
    already_done = (state.get("last_executed_rebal") == schedule["next_rebal"])

    print(f"  마지막 백테스트 리밸런싱: {schedule['last_rebal']}")
    print(f"  Horizon:                 {horizon} 거래일")

    # 놓친 리밸런싱 경고
    skipped = schedule.get("skipped_rebals", [])
    if skipped:
        print(f"\n  ⚠️  놓친 리밸런싱 {len(skipped)}회 (이미 지남):")
        for s in skipped:
            print(f"     신호일 {s['signal']} → 실행일 {s['exec']}  ← 지나갔음 (미실행)")
        print(f"  → 처음 시작이라면 정상입니다. 다음 리밸런싱부터 실행하세요.")

    print(f"\n  다음 리밸런싱 신호일: {schedule['next_rebal']}")
    print(f"  ⏰ 주문 실행일 (시초가): {schedule['next_exec']}  09:00 KST 전 실행 권장")

    if state.get("current_holdings"):
        print(f"  현재 보유 종목: {len(state['current_holdings'])}개  {state['current_holdings']}")

    if already_done:
        print(f"\n  ✅ 이미 실행 완료된 리밸런싱입니다 (신호일: {schedule['next_rebal']})")
        print(f"     강제 재실행: --force 플래그 추가")
        if not getattr(args, "force", False):
            return

    status = schedule["status"]
    if status == "future":
        days = schedule["trading_days_left"]
        print(f"\n  ⏳ {days} 거래일 후 실행일입니다. 아직 주문 불필요.")
        print(f"     실행일 전날 저녁 or 당일 09:00 전에 --execute로 실행하세요.")
        return
    elif status == "tomorrow":
        print(f"\n  📅 내일({schedule['next_exec']})이 실행일입니다.")
        print(f"     오늘 픽을 미리 계산하고, 내일 09:00 전에 --execute로 주문하세요.")
        # 픽 계산은 계속 진행 (주문은 안 함)
    elif status == "today":
        print(f"\n  ✅ 오늘({schedule['next_exec']})이 실행일입니다! 시초가 주문을 진행합니다.")

    # ── 4. 신규 픽 계산 ──────────────────────────────────────────────────
    print("\n[3/4] 신규 포트폴리오 계산...")

    from ml.models.base import BaseRanker
    from ml.features import FeatureEngineer

    model_path = run_dir / "model.pkl"
    if not model_path.exists():
        print(f"ERROR: model.pkl not found in {run_dir}")
        sys.exit(1)

    model = BaseRanker.load(str(model_path))
    meta  = model.metadata or {}

    # top_n
    top_n = args.top or meta.get("top_n", 10)

    # market cap
    min_cap = meta.get("min_market_cap", 500_000_000_000)
    max_cap = meta.get("max_market_cap", None)

    # sector neutral
    sector_neutral = meta.get("sector_neutral_score", True)

    print(f"  top_n={top_n}  min_cap={min_cap:,}  max_cap={max_cap}  sector_neutral={sector_neutral}")

    fe = FeatureEngineer()
    today = _today_str()

    pred_df = fe.prepare_prediction_data(
        end_date=today,
        target_horizon=horizon,
        min_market_cap=min_cap,
        max_market_cap=max_cap,
    )
    if pred_df.empty:
        print("ERROR: 예측 데이터 없음. DB 업데이트 확인 필요.")
        sys.exit(1)

    # Filter suspended stocks
    if "value" in pred_df.columns:
        pred_df = pred_df[pred_df["value"] > 0].copy()

    # Score
    pred_df["score"] = model.predict(pred_df)
    if sector_neutral and "sector" in pred_df.columns:
        sec_mean = pred_df.groupby("sector")["score"].transform("mean")
        sec_std  = pred_df.groupby("sector")["score"].transform("std").replace(0, np.nan)
        pred_df["score_rank"] = (
            (pred_df["score"] - sec_mean) / sec_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        pred_df["score_rank"] = pred_df["score"]

    pred_df["rank"] = pred_df["score_rank"].rank(ascending=False, method="first").astype(int)
    new_picks = pred_df.sort_values("rank")

    # 현재 보유 종목 — live/state.json 우선, 없으면 picks.csv 기반
    if state.get("current_holdings") and state.get("run_name") == run_name:
        current_holdings = set(state["current_holdings"])
        print(f"  [State] 보유 종목 {len(current_holdings)}개 (live/state.json 기준)")
    else:
        current_holdings = get_current_holdings(run_dir)
        print(f"  [State] 보유 종목 {len(current_holdings)}개 (picks.csv 기준 — 첫 실행)")

    # 주문 생성
    sell_orders, buy_orders = build_orders(
        new_picks=new_picks,
        current_holdings=current_holdings,
        top_n=top_n,
        portfolio_krw=args.portfolio,
    )

    print_order_summary(schedule, current_holdings, new_picks, sell_orders, buy_orders, top_n)

    # ── 5. 주문 실행 ──────────────────────────────────────────────────────
    if status == "tomorrow" and not args.execute:
        print("\n  [Tomorrow] 픽 계산 완료. 내일 아침 09:00 전에 아래 명령어로 주문하세요:")
        print(f"  python3 scripts/run_live.py --run {run_name} --execute")
        return

    if not args.execute:
        print("\n  [Dry-run] --execute 플래그 없음. 주문 제출 안 함.")
        print(f"  실제 주문: python3 scripts/run_live.py --run {run_name} --execute")
        return

    print("\n[4/4] 주문 제출...")

    # 최종 확인
    total = len(sell_orders) + len(buy_orders)
    confirm = input(f"\n  매도 {len(sell_orders)}건 + 매수 {len(buy_orders)}건 ({total}건) 주문하겠습니까? (y/n): ").strip().lower()
    if confirm != "y":
        print("  취소했습니다.")
        return

    client = KiwoomClient()
    if not client.authenticate():
        print("  인증 실패. 주문 중단.")
        return

    # 매도 먼저
    print(f"\n  매도 주문 ({len(sell_orders)}건):")
    for o in sell_orders:
        client.order_sell(o["stock_code"], o["quantity"], price=0)

    # 매수
    print(f"\n  매수 주문 ({len(buy_orders)}건):")
    for o in buy_orders:
        client.order_buy(o["stock_code"], o["quantity"], price=0)

    # state 저장
    new_holdings = list(set(new_picks.head(top_n)["stock_code"].tolist()))
    save_order_log(schedule["next_exec"], sell_orders, buy_orders, new_holdings)
    save_state({
        "last_executed_rebal": schedule["next_rebal"],
        "current_holdings":    new_holdings,
        "run_name":            run_name,
        "last_updated":        datetime.now().isoformat(),
    })

    print("\n  ✅ 주문 완료.")


if __name__ == "__main__":
    main()
