#!/usr/bin/env python3
"""Interactive quant dashboard — self-contained HTML (Plotly JS embedded, no CDN needed).

Usage:
  python3 scripts/dashboard.py --results results_twap5.csv --db krx_stock_data.db
  python3 scripts/dashboard.py --results results_execlag1.csv
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pof

DARK   = "plotly_dark"
C_PORT  = "#00b4d8"
C_BENCH = "#6c757d"
C_GREEN = "#2dc653"
C_RED   = "#e63946"
C_ACC   = "#f72585"
C_YEL   = "#ffd60a"


# ──────────────────────────────────────────────────────────── loaders ──────

def resolve_run_dir(run_arg: str) -> Path:
    """Accept either a run name ('twap5') or full path ('runs/twap5')."""
    p = Path(run_arg)
    if p.is_dir():
        return p
    candidate = Path("runs") / run_arg
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Run folder not found: tried '{p}' and '{candidate}'")


def load_results(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results.csv"
    df   = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_sidecar(run_dir: Path, filename: str) -> pd.DataFrame:
    p = run_dir / filename
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def parse_top_picks(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in results.iterrows():
        raw = r.get("top_picks", "")
        if pd.isna(raw):
            continue
        for rank, token in enumerate(str(raw).split(" | "), 1):
            try:
                left, ret_s = token.rsplit(":", 1)
                code = left.split("(")[0].strip()
                name = left.split("(")[1].rstrip(")").strip() if "(" in left else ""
                ret  = float(ret_s.replace("%", "").replace("+", "").strip()) / 100
                rows.append({
                    "date":     r["date"],
                    "date_str": str(r["date"])[:10],
                    "year":     int(str(r["date"])[:4]),
                    "rank":     rank,
                    "code":     code,
                    "name":     name,
                    "return":   ret,
                })
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def query_universe(db_path: str, date_yyyymmdd: list[str],
                   min_mc: int = 100_000_000_000) -> pd.DataFrame:
    if not db_path or not Path(db_path).exists():
        print("[Dashboard] DB not found — skipping market-cap/volume charts")
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        ph   = ", ".join(f"'{d}'" for d in date_yyyymmdd)
        df   = pd.read_sql_query(
            f"SELECT date, stock_code, market_cap, value "
            f"FROM daily_prices "
            f"WHERE date IN ({ph}) AND market_cap >= {min_mc} AND value > 0",
            conn,
        )
        conn.close()
        print(f"[Dashboard] DB: {len(df):,} rows across {df['date'].nunique()} dates")
        return df
    except Exception as e:
        print(f"[Dashboard] DB query failed: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────── figures ──────

def fig_cumret(results: pd.DataFrame) -> go.Figure:
    cum_p = (1 + results["portfolio_return"]).cumprod()
    cum_b = (1 + results["benchmark_return"]).cumprod()
    xs    = results["date"].dt.strftime("%Y-%m-%d").tolist()
    alpha = (cum_p - cum_b) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=((cum_p - 1) * 100).tolist() + ((cum_b - 1) * 100).tolist()[::-1],
        fill="toself", fillcolor="rgba(0,180,216,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=(cum_b - 1) * 100, name="Benchmark (KRX)",
        line=dict(color=C_BENCH, width=1.5, dash="dot"),
        hovertemplate="%{x}<br>Benchmark: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=(cum_p - 1) * 100, name="Portfolio",
        mode="lines+markers",
        line=dict(color=C_PORT, width=2.5),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="%{x}<br>Portfolio: %{y:.1f}%<br>Alpha: %{customdata:+.1f}%<extra></extra>",
        customdata=alpha,
    ))
    fig.update_layout(
        title="📈 누적 수익률 — Portfolio vs Benchmark",
        xaxis=dict(
            title="리밸런싱 날짜",
            rangeslider=dict(visible=True, thickness=0.07),
            rangeselector=dict(bgcolor="#21262d", buttons=[
                dict(count=1, label="2023", step="year", stepmode="todate"),
                dict(count=2, label="~2024", step="year", stepmode="todate"),
                dict(step="all", label="전체"),
            ]),
        ),
        yaxis_title="누적 수익률 (%)",
        hovermode="x unified",
        template=DARK, height=420,
        legend=dict(x=0.01, y=0.97, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=50, b=10),
    )
    return fig


def fig_3d_picks(picks: pd.DataFrame) -> go.Figure:
    """3D scatter: Rebalance Date × Portfolio Rank × Realized Return."""
    if picks.empty:
        fig = go.Figure()
        fig.add_annotation(text="top_picks 데이터 없음<br>results CSV의 top_picks 컬럼에서 자동 파싱됩니다",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color="#8b949e"))
        fig.update_layout(title="🎯 3D Picks Scatter", template=DARK, height=560)
        return fig

    date_labels = sorted(picks["date_str"].unique())
    date_map    = {d: i for i, d in enumerate(date_labels)}
    xi = picks["date_str"].map(date_map).values
    yi = picks["rank"].values
    zi = (picks["return"] * 100).values

    texts = picks.apply(
        lambda r: f"<b>{r['code']}</b> ({r['name']})<br>Rank #{int(r['rank'])}<br>Return: {r['return']:+.1%}",
        axis=1,
    ).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xi, y=yi, z=zi,
        mode="markers",
        marker=dict(
            size=8,
            color=zi,
            colorscale=[
                [0.0, "#e63946"], [0.35, "#ff6b6b"],
                [0.5, "#ffd60a"], [0.65, "#4cc9f0"],
                [1.0, "#2dc653"],
            ],
            cmin=-25, cmid=0, cmax=25,
            colorbar=dict(title="수익률 %", x=1.02, len=0.8),
            opacity=0.90,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Add zero-plane (Z=0) as a light grid
    date_idx  = list(range(len(date_labels)))
    rank_idx  = list(range(1, picks["rank"].max() + 2))
    z_zero    = [[0] * len(date_idx) for _ in rank_idx]
    fig.add_trace(go.Surface(
        x=date_idx, y=rank_idx, z=z_zero,
        colorscale=[[0, "rgba(255,255,255,0.05)"], [1, "rgba(255,255,255,0.05)"]],
        showscale=False, opacity=0.15, hoverinfo="skip",
    ))

    fig.update_layout(
        title="🎯 3D Picks — 날짜 × 순위 × 수익률  (드래그: 회전 | 스크롤: 줌)",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(
                title="리밸런싱 날짜",
                tickvals=list(range(len(date_labels))),
                ticktext=[d[5:] for d in date_labels],   # MM-DD
                gridcolor="#30363d", zerolinecolor="#30363d",
            ),
            yaxis=dict(
                title="포트폴리오 순위",
                gridcolor="#30363d", zerolinecolor="#30363d",
            ),
            zaxis=dict(
                title="수익률 (%)",
                gridcolor="#30363d", zerolinecolor="#30363d",
            ),
            camera=dict(eye=dict(x=2.0, y=-1.6, z=0.8)),
        ),
        template=DARK, height=580,
        margin=dict(t=50, b=10),
    )
    return fig


def fig_3d_quintile(results: pd.DataFrame) -> go.Figure:
    """3D surface: Date × Quintile × Return — shows signal monotonicity over time."""
    q_cols = [c for c in ["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"] if c in results.columns]
    if len(q_cols) < 3:
        fig = go.Figure()
        fig.add_annotation(text="q1_ret ~ q5_ret 컬럼이 없습니다",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color="#8b949e"))
        fig.update_layout(title="📊 3D Quintile Surface", template=DARK, height=560)
        return fig

    dates = results["date"].dt.strftime("%Y-%m-%d").tolist()
    z     = results[q_cols].values * 100              # (n_dates, n_q)
    xi    = list(range(len(dates)))
    yi    = list(range(1, len(q_cols) + 1))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z.T,
        x=xi, y=yi,
        colorscale=[
            [0.0,  "#e63946"], [0.3,  "#ff6b6b"],
            [0.45, "#2d2d2d"], [0.5,  "#ffd60a"],
            [0.55, "#4cc9f0"], [0.7,  "#00b4d8"],
            [1.0,  "#2dc653"],
        ],
        cmin=-20, cmid=0, cmax=20,
        colorbar=dict(title="수익률 %", x=1.02, len=0.8),
        opacity=0.92,
        hovertemplate="날짜: %{x}<br>Q%{y}<br>수익률: %{z:.1f}%<extra></extra>",
    ))
    fig.update_traces(
        contours_z=dict(show=True, usecolormap=True, highlightcolor="rgba(255,255,255,0.4)",
                        project_z=True, width=2),
        contours_x=dict(show=True, color="rgba(255,255,255,0.08)", width=1),
        contours_y=dict(show=True, color="rgba(255,255,255,0.08)", width=1),
    )

    tick_step = max(1, len(dates) // 6)
    fig.update_layout(
        title="📊 3D Quintile Surface — 날짜별 신호 단조성  (드래그: 회전 | 스크롤: 줌)",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(
                title="리밸런싱",
                tickvals=list(range(0, len(dates), tick_step)),
                ticktext=[dates[i][2:] for i in range(0, len(dates), tick_step)],
                gridcolor="#30363d",
            ),
            yaxis=dict(
                title="Quintile",
                tickvals=yi,
                ticktext=[f"Q{i}" for i in yi],
                gridcolor="#30363d",
            ),
            zaxis=dict(title="수익률 (%)", gridcolor="#30363d"),
            camera=dict(eye=dict(x=1.6, y=-2.2, z=1.1)),
        ),
        template=DARK, height=580,
        margin=dict(t=50, b=10),
    )
    return fig


def fig_3d_risk_return(results: pd.DataFrame, universe_df: pd.DataFrame) -> go.Figure:
    """3D scatter: Market Return × Portfolio Return × IC — quant alpha decomposition."""
    has_ic = "ic_spearman" in results.columns
    x = results["benchmark_return"] * 100
    y = results["portfolio_return"] * 100
    z = results["ic_spearman"] if has_ic else pd.Series(np.zeros(len(results)))
    years   = results["year"].tolist() if "year" in results.columns else [""] * len(results)
    dates   = results["date"].dt.strftime("%Y-%m-%d").tolist()
    palette = {2023: "#00b4d8", 2024: "#f72585", 2025: "#2dc653"}

    fig = go.Figure()
    for yr, grp in results.groupby("year") if "year" in results.columns else [(None, results)]:
        mask  = results["year"] == yr if "year" in results.columns else pd.Series([True] * len(results))
        color = palette.get(yr, C_PORT)
        xi    = (results.loc[mask, "benchmark_return"] * 100).values
        yi    = (results.loc[mask, "portfolio_return"] * 100).values
        zi    = results.loc[mask, "ic_spearman"].values if has_ic else np.zeros(mask.sum())
        ds    = results.loc[mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        alpha_vals = yi - xi
        txt = [f"<b>{d}</b><br>벤치마크: {bx:+.1f}%<br>포트폴리오: {by:+.1f}%<br>알파: {a:+.1f}%<br>IC: {ic:.3f}"
               for d, bx, by, a, ic in zip(ds, xi, yi, alpha_vals, zi)]
        fig.add_trace(go.Scatter3d(
            x=xi, y=yi, z=zi,
            mode="markers",
            name=str(yr),
            marker=dict(
                size=10,
                color=alpha_vals,
                colorscale="RdYlGn", cmin=-15, cmid=0, cmax=15,
                opacity=0.9,
                line=dict(width=1, color=color),
                symbol="circle",
            ),
            text=txt,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Diagonal: where portfolio = benchmark (zero alpha line)
    lim = max(abs(x.max()), abs(x.min()), 20)
    z_ic_mean = float(z.mean()) if has_ic else 0
    fig.add_trace(go.Scatter3d(
        x=[-lim, lim], y=[-lim, lim], z=[z_ic_mean, z_ic_mean],
        mode="lines", name="Zero Alpha",
        line=dict(color="rgba(255,255,255,0.3)", width=2, dash="dash"),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title="🔬 3D Alpha Decomposition — 시장수익률 × 포트수익률 × IC  (드래그: 회전)",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(title="시장 수익률 (%)", gridcolor="#30363d"),
            yaxis=dict(title="포트폴리오 수익률 (%)", gridcolor="#30363d"),
            zaxis=dict(title="IC (Spearman)", gridcolor="#30363d"),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.0)),
        ),
        template=DARK, height=560,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=50, b=10),
    )
    return fig


def fig_return_dist(results: pd.DataFrame) -> go.Figure:
    palette = {2023: C_PORT, 2024: C_ACC, 2025: C_GREEN}
    years   = sorted(results["year"].unique()) if "year" in results.columns else [None]
    fig     = go.Figure()
    for yr in years:
        sub = results[results["year"] == yr] if yr is not None else results
        ret = sub["portfolio_return"] * 100
        fig.add_trace(go.Histogram(
            x=ret, name=str(yr) if yr else "전체",
            nbinsx=12, opacity=0.75,
            marker_color=palette.get(yr, C_PORT),
            histnorm="probability",
            hovertemplate=f"{yr}<br>%{{x:.1f}}%<br>비율: %{{y:.2f}}<extra></extra>",
        ))
        # Mean line
        fig.add_vline(
            x=float(ret.mean()), line_dash="dot",
            line_color=palette.get(yr, C_PORT), line_width=1.5,
            annotation_text=f"{yr} avg {ret.mean():+.1f}%",
            annotation_font_color=palette.get(yr, C_PORT),
            annotation_font_size=10,
        )
    fig.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
    fig.update_layout(
        title="📉 수익률 분포 — Return Distribution by Year",
        xaxis_title="리밸런싱 수익률 (%)", yaxis_title="비율",
        barmode="overlay", template=DARK, height=360,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    return fig


def fig_ic_bar(results: pd.DataFrame) -> go.Figure:
    if "ic_spearman" not in results.columns:
        fig = go.Figure()
        fig.update_layout(title="IC — 없음", template=DARK, height=320)
        return fig
    xs     = results["date"].dt.strftime("%Y-%m-%d").tolist()
    ic     = results["ic_spearman"]
    mean   = float(ic.mean())
    colors = [C_GREEN if v >= 0 else C_RED for v in ic]
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=ic, marker_color=colors, name="IC",
        hovertemplate="%{x}<br>IC: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=mean, line_color=C_YEL, line_dash="dash",
                  annotation_text=f"Mean IC = {mean:.4f}",
                  annotation_font_color=C_YEL,
                  annotation_position="top right")
    fig.add_hline(y=0, line_color=C_BENCH, line_width=1)
    fig.update_layout(
        title="🎯 IC (Spearman) — 리밸런싱별",
        xaxis_title="날짜", yaxis_title="IC",
        template=DARK, height=320,
    )
    return fig


def fig_annual_sharpe(results: pd.DataFrame) -> go.Figure:
    """Per-year and overall Sharpe bars (more meaningful than rolling-12 with only 18 rows)."""
    rows = []
    yearly = results.groupby("year") if "year" in results.columns else [(None, results)]
    rebals_per_year = max(len(results) / results["year"].nunique(), 1) if "year" in results.columns else len(results)
    for yr, grp in yearly:
        ret  = grp["portfolio_return"]
        tr   = (1 + ret).prod() - 1
        vol  = ret.std() * np.sqrt(len(grp))       # annualized with actual rebal count
        sh   = tr / vol if vol > 0 else 0
        rows.append({"Year": str(yr), "Sharpe": sh, "Return": tr * 100})
    # Overall
    ret  = results["portfolio_return"]
    n_yr = results["year"].nunique() if "year" in results.columns else 1
    vol  = ret.std() * np.sqrt(len(ret) / n_yr)
    tr   = (1 + ret).prod() - 1
    ann  = (1 + tr) ** (1 / n_yr) - 1
    sh   = ann / vol if vol > 0 else 0
    rows.append({"Year": "전체", "Sharpe": sh, "Return": tr * 100})

    df     = pd.DataFrame(rows)
    colors = [C_GREEN if v >= 1 else C_YEL if v >= 0 else C_RED for v in df["Sharpe"]]
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Year"], y=df["Sharpe"], marker_color=colors, name="Sharpe",
        text=[f"{v:.2f}" for v in df["Sharpe"]], textposition="outside",
        hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1, line_color=C_GREEN, line_dash="dot",
                  annotation_text="Sharpe=1 (목표)", annotation_position="top right",
                  annotation_font_color=C_GREEN)
    fig.update_layout(
        title="⚡ 연도별 Sharpe Ratio",
        xaxis_title="연도", yaxis_title="Sharpe",
        template=DARK, height=320,
        yaxis=dict(range=[min(0, df["Sharpe"].min() - 0.3), df["Sharpe"].max() + 0.5]),
    )
    return fig


def fig_drawdown(results: pd.DataFrame) -> go.Figure:
    cum      = (1 + results["portfolio_return"]).cumprod()
    drawdown = (cum / cum.cummax() - 1) * 100
    xs       = results["date"].dt.strftime("%Y-%m-%d").tolist()
    fig      = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=drawdown, fill="tozeroy",
        fillcolor="rgba(230,57,70,0.25)",
        line=dict(color=C_RED, width=1.5),
        name="Drawdown",
        hovertemplate="%{x}<br>DD: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=float(drawdown.min()), line_color=C_RED, line_dash="dot",
                  annotation_text=f"Max DD = {drawdown.min():.1f}%",
                  annotation_font_color=C_RED, annotation_position="bottom right")
    fig.update_layout(
        title="📉 Drawdown",
        xaxis_title="날짜", yaxis_title="Drawdown (%)",
        template=DARK, height=280,
    )
    return fig


def fig_turnover(results: pd.DataFrame) -> go.Figure:
    xs  = results["date"].dt.strftime("%Y-%m-%d").tolist()
    fig = go.Figure()
    if "turnover" in results.columns:
        fig.add_trace(go.Bar(
            x=xs, y=results["turnover"] * 100,
            marker_color=C_ACC, name="Turnover (%)",
            hovertemplate="%{x}<br>Turnover: %{y:.0f}%<extra></extra>",
        ))
    if "transaction_cost" in results.columns:
        fig.add_trace(go.Scatter(
            x=xs, y=results["transaction_cost"] * 100,
            mode="lines+markers",
            line=dict(color=C_YEL, width=2),
            marker=dict(size=6),
            name="Tx Cost (%)",
            yaxis="y2",
            hovertemplate="%{x}<br>Tx Cost: %{y:.3f}%<extra></extra>",
        ))
    fig.update_layout(
        title="🔄 Turnover & Transaction Cost",
        xaxis_title="날짜",
        yaxis=dict(title="Turnover (%)"),
        yaxis2=dict(title="Tx Cost (%)", overlaying="y", side="right", showgrid=False),
        barmode="group", template=DARK, height=300,
    )
    return fig


def fig_sector_bar(sector_df: pd.DataFrame) -> go.Figure:
    if sector_df.empty or "sector" not in sector_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="sector_attribution CSV 없음",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#8b949e"))
        fig.update_layout(title="🏭 섹터 귀인", template=DARK, height=380)
        return fig
    agg    = sector_df.groupby("sector")["contribution"].sum().sort_values()
    colors = [C_RED if v < 0 else C_GREEN for v in agg.values]
    fig    = go.Figure(go.Bar(
        x=agg.values, y=agg.index,
        orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in agg.values], textposition="outside",
        hovertemplate="%{y}<br>기여도: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="🏭 섹터 귀인 — Cumulative Sector Attribution",
        xaxis_title="누적 기여도",
        template=DARK, height=420,
        margin=dict(l=160),
    )
    return fig


def fig_marketcap_3d(universe_df: pd.DataFrame) -> go.Figure:
    """3D surface: Date × MarketCap tier × Stock count — shows universe composition over time."""
    if universe_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="--db 경로 지정 필요 (DB 쿼리 결과 없음)",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#8b949e"))
        fig.update_layout(title="🏢 시가총액 3D 분포", template=DARK, height=500)
        return fig

    dates  = sorted(universe_df["date"].unique())
    # Log-spaced market cap bins
    edges   = np.logspace(np.log10(1e10), np.log10(5e13), 13)   # 100억 ~ 50조, 12 bins
    labels  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1e12:
            labels.append(f"{lo/1e8:.0f}억~{hi/1e8:.0f}억")
        else:
            labels.append(f"{lo/1e12:.1f}조~{hi/1e12:.1f}조")

    z_count = []   # (n_dates, n_bins)
    for d in dates:
        day = universe_df[universe_df["date"] == d]["market_cap"].values
        counts, _ = np.histogram(day, bins=edges)
        z_count.append(counts.tolist())

    z_arr = np.array(z_count, dtype=float)   # (n_dates, n_bins)

    fig = go.Figure(go.Surface(
        z=z_arr.T,   # (n_bins, n_dates) — bins on Y, dates on X
        x=list(range(len(dates))),
        y=list(range(len(labels))),
        colorscale="Viridis",
        colorbar=dict(title="종목 수", x=1.02),
        opacity=0.92,
        hovertemplate="날짜 #%{x}<br>시총 구간 #%{y}<br>종목 수: %{z}<extra></extra>",
    ))
    fig.update_traces(
        contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
    )
    tick_step = max(1, len(dates) // 6)
    fig.update_layout(
        title="🏢 3D 시가총액 분포 — 날짜 × 시총 구간 × 종목 수  (드래그: 회전)",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(
                title="리밸런싱 날짜",
                tickvals=list(range(0, len(dates), tick_step)),
                ticktext=[str(dates[i])[:8] for i in range(0, len(dates), tick_step)],
                gridcolor="#30363d",
            ),
            yaxis=dict(
                title="시가총액 구간",
                tickvals=list(range(0, len(labels), 2)),
                ticktext=[labels[i] for i in range(0, len(labels), 2)],
                gridcolor="#30363d",
            ),
            zaxis=dict(title="종목 수", gridcolor="#30363d"),
            camera=dict(eye=dict(x=1.8, y=-2.0, z=1.2)),
        ),
        template=DARK, height=560,
        margin=dict(t=50, b=10),
    )
    return fig


def fig_volume_box(universe_df: pd.DataFrame) -> go.Figure:
    """Box plot of daily trading value (거래대금) per rebalance date."""
    if universe_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="--db 경로 지정 필요",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#8b949e"))
        fig.update_layout(title="💹 거래대금 분포", template=DARK, height=380)
        return fig

    dates = sorted(universe_df["date"].unique())
    n     = max(len(dates), 1)
    fig   = go.Figure()
    for i, d in enumerate(dates):
        day     = universe_df[universe_df["date"] == d]
        val_log = np.log10(day["value"].clip(lower=1))
        color   = f"hsl({int(270 + i * 90 / n)},70%,55%)"
        fig.add_trace(go.Box(
            y=val_log, name=str(d)[:8],
            boxpoints="outliers",
            marker=dict(color=color, size=3, opacity=0.5),
            line_color=color, fillcolor=color.replace("55%", "25%"),
            hovertemplate=f"<b>{d}</b><br>log₁₀(거래대금): %{{y:.2f}}<extra></extra>",
        ))
    tickvals = [np.log10(v) for v in [1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 1e11]]
    ticktext = ["1천만", "5천만", "1억", "5억", "10억", "50억", "100억", "1천억"]
    fig.update_layout(
        title="💹 거래대금 분포 — Daily Trading Value by Rebalance Date",
        yaxis=dict(title="거래대금 (로그 KRW)", tickvals=tickvals, ticktext=ticktext),
        xaxis_title="리밸런싱 날짜",
        template=DARK, height=420,
    )
    return fig


def fig_marketcap_box(universe_df: pd.DataFrame) -> go.Figure:
    """Box plot of market cap per rebalance date."""
    if universe_df.empty:
        fig = go.Figure()
        fig.update_layout(title="🏢 시가총액 분포 (Box) — DB 없음", template=DARK, height=380)
        return fig
    dates = sorted(universe_df["date"].unique())
    n     = max(len(dates), 1)
    fig   = go.Figure()
    for i, d in enumerate(dates):
        day    = universe_df[universe_df["date"] == d]
        mc_log = np.log10(day["market_cap"].clip(lower=1))
        color  = f"hsl({int(200 + i * 80 / n)},70%,55%)"
        fig.add_trace(go.Box(
            y=mc_log, name=str(d)[:8],
            boxpoints="outliers",
            marker=dict(color=color, size=3, opacity=0.5),
            line_color=color, fillcolor=color.replace("55%", "25%"),
            hovertemplate=f"<b>{d}</b><br>log₁₀(시가총액): %{{y:.2f}}<extra></extra>",
        ))
    tickvals = [np.log10(v) for v in [1e10, 5e10, 1e11, 5e11, 1e12, 5e12, 1e13]]
    ticktext = ["100억", "500억", "1천억", "5천억", "1조", "5조", "10조"]
    fig.update_layout(
        title="🏢 시가총액 분포 (Box) — Market Cap by Rebalance Date",
        yaxis=dict(title="시가총액 (로그 KRW)", tickvals=tickvals, ticktext=ticktext),
        xaxis_title="리밸런싱 날짜",
        template=DARK, height=420,
    )
    return fig


# ──────────────────────────────────────────────────────── HTML builder ──────

_CSS = """
* { box-sizing: border-box; }
body { background: #0d1117; color: #e6edf3;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 0; }
h1   { text-align: center; padding: 28px 0 4px; color: #58a6ff;
       font-size: 1.8em; letter-spacing: .05em; margin: 0; }
p.sub { text-align: center; color: #8b949e; margin: 0 0 16px; font-size: .88em; }
.grid { display: grid; grid-template-columns: 1fr 1fr;
        gap: 14px; padding: 14px; max-width: 1900px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 8px; overflow: hidden; }
.full { grid-column: 1 / -1; }
@media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
"""

# Layout: (key, is_full_width)
LAYOUT = [
    ("cumret",        True),
    ("3d_picks",      True),
    ("3d_quintile",   True),
    ("3d_alpha",      True),
    ("3d_mc",         True),
    ("return_dist",   False),
    ("ic",            False),
    ("annual_sharpe", False),
    ("drawdown",      False),
    ("turnover",      False),
    ("sector",        False),
    ("mc_box",        False),
    ("vol_box",       False),
]


def build_html(figs: dict[str, go.Figure], title: str) -> str:
    # Embed Plotly JS inline (works without internet / CDN)
    plotly_js = pof.get_plotlyjs()

    cards = []
    for key, full in LAYOUT:
        if key not in figs:
            continue
        fig   = figs[key]
        div   = fig.to_html(full_html=False, include_plotlyjs=False)
        cls   = "card full" if full else "card"
        cards.append(f'<div class="{cls}">{div}</div>')

    return (
        "<!DOCTYPE html><html><head>"
        f'<meta charset="utf-8"><title>{title}</title>'
        f"<script>{plotly_js}</script>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        f'<h1>📊 Quant Dashboard</h1>'
        f'<p class="sub">AlgoStock &nbsp;·&nbsp; {title} &nbsp;·&nbsp; Dark Mode</p>'
        f'<div class="grid">{"".join(cards)}</div>'
        "</body></html>"
    )


# ────────────────────────────────────────────────────────────── main ──────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive quant dashboard → self-contained HTML",
        epilog="Example:  python3 scripts/dashboard.py twap5",
    )
    ap.add_argument("run", nargs="?", default="",
                    help="Run name (e.g. 'twap5') — loads from runs/twap5/")
    ap.add_argument("--db",             default="krx_stock_data.db",
                    help="SQLite DB for market-cap/volume (optional)")
    ap.add_argument("--output",         default="",
                    help="Output HTML path (default: runs/<name>/dashboard.html)")
    ap.add_argument("--min-market-cap", type=int, default=100_000_000_000,
                    help="Universe floor for DB query (default 1000억)")
    args = ap.parse_args()

    if not args.run:
        ap.error("Specify a run name, e.g.:  python3 scripts/dashboard.py twap5")

    run_dir  = resolve_run_dir(args.run)
    out_path = args.output or str(run_dir / "dashboard.html")

    print(f"[Dashboard] Run folder: {run_dir}")
    results   = load_results(run_dir)
    sector_df = load_sidecar(run_dir, "sector_attribution.csv")
    picks_df  = parse_top_picks(results)
    print(f"[Dashboard] Picks parsed: {len(picks_df)} stock-rebalance rows")

    rebal_dates = results["date"].dt.strftime("%Y%m%d").tolist()
    print(f"[Dashboard] Querying DB for {len(rebal_dates)} rebalance dates ...")
    universe_df = query_universe(args.db, rebal_dates, args.min_market_cap)
    if universe_df.empty:
        print("[Dashboard] DB unavailable — market cap & volume charts will show placeholder.")

    print("[Dashboard] Building charts ...")
    figs: dict[str, go.Figure] = {
        "cumret":        fig_cumret(results),
        "3d_picks":      fig_3d_picks(picks_df),
        "3d_quintile":   fig_3d_quintile(results),
        "3d_alpha":      fig_3d_risk_return(results, universe_df),
        "3d_mc":         fig_marketcap_3d(universe_df),
        "return_dist":   fig_return_dist(results),
        "ic":            fig_ic_bar(results),
        "annual_sharpe": fig_annual_sharpe(results),
        "drawdown":      fig_drawdown(results),
        "turnover":      fig_turnover(results),
        "sector":        fig_sector_bar(sector_df),
        "mc_box":        fig_marketcap_box(universe_df),
        "vol_box":       fig_volume_box(universe_df),
    }

    html = build_html(figs, title=run_dir.name)
    Path(out_path).write_text(html, encoding="utf-8")
    size_mb = Path(out_path).stat().st_size / 1_048_576
    print(f"[Dashboard] ✅ Saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  macOS:   open {out_path}")
    print(f"  Windows: start {out_path}")


if __name__ == "__main__":
    main()
