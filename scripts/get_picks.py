#!/usr/bin/env python3
"""Generate latest long/avoid picks with the unified model.

Fixes vs previous version:
  1. Prediction uses today's data (not ~21 days stale)
  2. Uses same target as backtest (risk-adjusted rank)
  3. Loads backtest model by default; retraining uses proper val split + early stopping
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _truncate_text(value: object, max_len: int) -> str:
    text = "" if pd.isna(value) else str(value)
    if max_len <= 0 or len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    return text[: max_len - 1] + "…"


def _format_market_cap_short(value: object) -> str:
    if pd.isna(value):
        return ""
    v = float(value)
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for base, suffix in units:
        if abs(v) >= base:
            return f"{v / base:.2f}{suffix}"
    return f"{v:.0f}"


def _format_display_df(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Format numeric values for terminal display."""
    out = df.copy()
    if "sector" in out.columns:
        out["sector"] = out["sector"].map(
            lambda v: "" if pd.isna(v) else str(v).split("_")[-1]
        )
    if "name" in out.columns:
        out["name"] = out["name"].map(lambda v: _truncate_text(v, 14))
    if "sector" in out.columns:
        out["sector"] = out["sector"].map(lambda v: _truncate_text(v, 16))
    if "market_cap" in out.columns:
        out["market_cap"] = out["market_cap"].map(_format_market_cap_short)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{decimals}f}")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    return out


def _print_df_vertical(title: str, df: pd.DataFrame, decimals: int) -> None:
    """Print one pick per block for narrow terminals."""
    display_df = _format_display_df(df, decimals=decimals)
    print(f"\n=== {title} ===")
    if display_df.empty:
        print("(no rows)")
        return
    for idx, row in enumerate(display_df.to_dict("records"), start=1):
        print(f"\n[{title} #{idx}]")
        for k, v in row.items():
            print(f"- {k}: {v}")


def _print_df_table(
    title: str,
    df: pd.DataFrame,
    decimals: int = 2,
    max_columns: int = 6,
    id_columns: tuple[str, ...] = ("rank",),
    first_page_context_columns: tuple[str, ...] = ("stock_code", "name", "sector"),
) -> None:
    """Print DataFrame as one or more readable terminal tables."""
    if df.empty:
        print(f"\n=== {title} ===")
        print("(no rows)")
        return

    display_df = _format_display_df(df, decimals=decimals)
    cols = [str(c) for c in display_df.columns]
    max_columns = max(1, int(max_columns))

    id_cols = [c for c in id_columns if c in cols]
    context_cols = [c for c in first_page_context_columns if c in cols and c not in id_cols]

    if len(cols) <= max_columns:
        parts = [cols]
    elif not id_cols:
        parts = [cols[i:i + max_columns] for i in range(0, len(cols), max_columns)]
    else:
        # First page: include context columns once.
        first_base = id_cols + context_cols
        if len(first_base) >= max_columns:
            first_part = first_base[:max_columns]
            rest_cols = [c for c in cols if c not in first_part]
        else:
            remaining_cols = [c for c in cols if c not in first_base]
            first_room = max(0, max_columns - len(first_base))
            first_part = first_base + remaining_cols[:first_room]
            rest_cols = remaining_cols[first_room:]

        parts = [first_part] if first_part else [id_cols]

        # Later pages: repeat only identifiers.
        follow_room = max(1, max_columns - len(id_cols))
        for i in range(0, len(rest_cols), follow_room):
            parts.append(id_cols + rest_cols[i:i + follow_room])

    try:
        from prettytable import PrettyTable
    except ImportError:
        for idx, part_cols in enumerate(parts, start=1):
            part_title = title if len(parts) == 1 else f"{title} ({idx}/{len(parts)})"
            print(f"\n=== {part_title} ===")
            print(display_df[part_cols].to_string(index=False))
        return

    for idx, part_cols in enumerate(parts, start=1):
        part_title = title if len(parts) == 1 else f"{title} ({idx}/{len(parts)})"
        print(f"\n=== {part_title} ===")
        table = PrettyTable()
        table.field_names = part_cols
        table.align = "l"
        table.max_width = 18
        for row in display_df[part_cols].itertuples(index=False, name=None):
            table.add_row([v if pd.notna(v) else "" for v in row])
        print(table)


def _select_target_col(df: pd.DataFrame, horizon: int) -> str:
    """Match the backtest's target: residual rank → 5 integer levels (0-4) for LambdaRank."""
    residual_rank_col = f"target_residual_rank_{horizon}d"
    rank_label_col = f"target_rank_label_{horizon}d"
    base_col = residual_rank_col if residual_rank_col in df.columns else f"target_rank_{horizon}d"
    if base_col in df.columns and rank_label_col not in df.columns:
        df[rank_label_col] = np.clip((df[base_col] * 5).astype(int), 0, 4)
    return rank_label_col


def _retrain_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    model_path: str,
    time_decay: float,
    patience: int,
    learning_rate: float,
    n_estimators: int,
    model_name: str = "lgbm",
) -> MLRanker:
    """Retrain with proper val split + early stopping (matches backtest logic)."""
    from ml.models import get_model_class

    train_years = sorted(df["date"].str[:4].unique())
    val_year = train_years[-1]
    sub_train = df[df["date"].str[:4] != val_year].copy()
    val_df = df[df["date"].str[:4] == val_year].copy()
    if sub_train.empty:
        sub_train, val_df = df.copy(), None

    print(f"[Retrain] train years={train_years[:-1]}, val year={val_year}")
    print(f"[Retrain] train rows={len(sub_train):,}, val rows={len(val_df) if val_df is not None else 0:,}")

    ModelClass = get_model_class(model_name)
    model = ModelClass(
        feature_cols=feature_cols,
        target_col=target_col,
        time_decay=time_decay,
        patience=patience,
    )
    params = model.BEST_PARAMS.copy()
    params["learning_rate"] = learning_rate
    params["n_estimators"] = n_estimators
    params["n_jobs"] = max(1, (os.cpu_count() or 4) // 2)

    model.train(sub_train, val_df, params=params)
    model.save(model_path)
    print(f"[Retrain] saved model to {model_path}")
    return model


def _print_feature_group_importance(model) -> None:
    """Print LightGBM gain-based importance aggregated by feature group."""
    try:
        imp_df = model.feature_importance()
    except Exception as e:
        print(f"[Feature Importance] Could not compute: {e}")
        return

    from ml.features.registry import get_feature_group_map
    group_map = get_feature_group_map()

    imp_df["group"] = imp_df["feature"].map(group_map).fillna("other")
    group_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)
    total = group_imp.sum()

    # Also collect top-3 features per group
    top_features = (
        imp_df.sort_values("importance", ascending=False)
        .groupby("group")
        .head(3)
        .groupby("group")["feature"]
        .apply(list)
    )

    print("\n" + "=" * 60)
    print("  Feature Group Importance (Gain-based)")
    print("=" * 60)
    bar_scale = 40
    for grp, imp in group_imp.items():
        pct = imp / total * 100 if total > 0 else 0.0
        bar = "█" * int(pct / 100 * bar_scale)
        feats = ", ".join(top_features.get(grp, [])[:3])
        print(f"  {grp:<22} {pct:>5.1f}%  {bar}")
        if feats:
            print(f"  {'':22}        top: {feats}")
    print("=" * 60)
    print(f"  Total features: {len(imp_df)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Get latest picks from unified model")
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "xgboost", "catboost"],
                        help="Model type to use (default: lgbm)")
    parser.add_argument("--db", default="krx_stock_data.db")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD (default: today)")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--bottom", type=int, default=10)
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000)
    parser.add_argument("--max-market-cap", type=int, default=None, help="Maximum market cap (e.g. 5000000000000 for SMID-cap)")
    parser.add_argument("--model-path", default="models/lgbm_unified.pkl",
                        help="Path to pre-trained model from backtest")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain instead of loading saved model")
    parser.add_argument("--train-start", default="20120101",
                        help="Training data start date when --retrain is used")
    parser.add_argument("--time-decay", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--display-decimals", type=int, default=2,
                        help="Decimal places for numeric values in terminal tables")
    parser.add_argument("--max-columns", type=int, default=6,
                        help="Maximum columns per printed table (wide tables are split)")
    parser.add_argument("--display-style", choices=["table", "vertical"], default="table",
                        help="How to render picks in terminal")
    parser.add_argument("--view", choices=["compact", "full"], default="full",
                        help="Column set for terminal display")
    parser.add_argument("--target-return", type=float, default=0.05,
                        help="매도가 = 매수가 × (1 + target_return). Default: 0.05 (5%%)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--feature-importance", action="store_true",
                        help="Print feature group importance (gain) after loading model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    from ml.features import FeatureEngineer

    from datetime import datetime
    end_date = args.end or datetime.now().strftime("%Y%m%d")

    fe = FeatureEngineer(args.db)

    # --- Load or retrain model ---
    model_file = Path(args.model_path)
    if not args.retrain and model_file.exists():
        from ml.models.base import BaseRanker
        print(f"Loading pre-trained model from {model_file}")
        model = BaseRanker.load(str(model_file))
        feature_cols = model.feature_cols
        print(f"Model target: {model.target_col}, features: {len(feature_cols)}")

        # Sync all backtest config from model metadata
        meta = model.metadata
        if meta:
            # market cap — only override if user left the default sentinel
            if "min_market_cap" in meta and args.min_market_cap == 500_000_000_000:
                args.min_market_cap = meta["min_market_cap"]
            if "max_market_cap" in meta and getattr(args, "max_market_cap", None) is None:
                args.max_market_cap = meta["max_market_cap"]
            # top_n — only override if user left the default (20)
            if "top_n" in meta and args.top == 20:
                args.top = meta["top_n"]
            # sector neutral score
            if "sector_neutral_score" in meta:
                args._sector_neutral_score = meta["sector_neutral_score"]
            else:
                args._sector_neutral_score = True  # backtest default
            # liquidity
            if "min_daily_value" in meta and not hasattr(args, "_min_daily_value"):
                args._min_daily_value = meta["min_daily_value"]
            else:
                args._min_daily_value = getattr(args, "_min_daily_value", 0)
            print(
                f"[Model metadata] "
                f"min_market_cap={args.min_market_cap:,}  "
                f"max_market_cap={args.max_market_cap}  "
                f"top_n={args.top}  "
                f"sector_neutral={args._sector_neutral_score}  "
                f"min_daily_value={args._min_daily_value}  "
                f"horizon={meta.get('horizon', '?')}  "
                f"backtest_end={meta.get('backtest_end', '?')}"
            )
        else:
            # No metadata (old model) — use safe defaults matching backtest
            args._sector_neutral_score = True
            args._min_daily_value = 0
    elif args.retrain:
        args._sector_neutral_score = True   # backtest default
        args._min_daily_value = 0
        print("Retraining model...")
        train_df = fe.prepare_ml_data(
            start_date=args.train_start,
            end_date=end_date,
            target_horizon=args.horizon,
            min_market_cap=args.min_market_cap,
            use_cache=not args.no_cache,
            n_workers=args.workers,
        )
        if train_df.empty:
            print("No training data available.")
            return
        feature_cols = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in train_df.columns]
        target_col = _select_target_col(train_df, args.horizon)
        print(f"Target: {target_col}")
        model = _retrain_model(
            df=train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=args.horizon,
            model_path=str(model_file),
            time_decay=args.time_decay,
            patience=args.patience,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            model_name=args.model,
        )
    else:
        print(f"No model found at {model_file}. Run backtest first or use --retrain.")
        return

    # --- Feature group importance ---
    if args.feature_importance:
        _print_feature_group_importance(model)

    # --- Build prediction features for the latest date (no forward return needed) ---
    pred_df = fe.prepare_prediction_data(
        end_date=end_date,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
        max_market_cap=getattr(args, "max_market_cap", None),
    )
    if pred_df.empty:
        print("No prediction data available.")
        return

    # Check feature alignment
    missing_features = [c for c in feature_cols if c not in pred_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} model features missing from prediction data: {missing_features[:5]}")
        feature_cols = [c for c in feature_cols if c in pred_df.columns]

    # ── Filters matching backtest universe construction ───────────────────
    # 1. Exclude suspended stocks (거래정지: value == 0)
    if "value" in pred_df.columns:
        suspended_before = len(pred_df)
        pred_df = pred_df[pred_df["value"] > 0].copy()
        suspended = suspended_before - len(pred_df)
        if suspended:
            print(f"[Filter] Removed {suspended} suspended stocks (value=0)")

    # 2. Minimum daily trading value (liquidity filter)
    _min_dv = getattr(args, "_min_daily_value", 0)
    if _min_dv > 0 and "value" in pred_df.columns:
        before = len(pred_df)
        pred_df = pred_df[pred_df["value"] >= _min_dv].copy()
        print(f"[Filter] min_daily_value={_min_dv:,} removed {before - len(pred_df)} stocks")

    # ── Scoring ──────────────────────────────────────────────────────────
    pred_df["score"] = model.predict(pred_df)

    # Sector-neutral score (replicates backtest --sector-neutral-score)
    _sn = getattr(args, "_sector_neutral_score", True)
    if _sn and "sector" in pred_df.columns:
        sec_mean = pred_df.groupby("sector")["score"].transform("mean")
        sec_std  = pred_df.groupby("sector")["score"].transform("std").replace(0, np.nan)
        pred_df["score_rank"] = (
            (pred_df["score"] - sec_mean) / sec_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        pred_df["score_rank"] = pred_df["score"]

    pred_df["rank"] = pred_df["score_rank"].rank(ascending=False, method="first").astype(int)

    # 매수가 / 매도가
    pred_df["매수가"] = pred_df["closing_price"].round(0).astype(int)
    pred_df["매도가"] = (pred_df["closing_price"] * (1 + args.target_return)).round(0).astype(int)

    latest_date = pred_df["date"].max()

    compact_cols = [
        "rank",
        "stock_code",
        "name",
        "sector",
        "매수가",
        "매도가",
        "closing_price",
        "market_cap",
        "score",
        "score_rank",
        "roe",
        "mom_21d",
    ]
    full_cols = [
        "rank",
        "stock_code",
        "name",
        "sector",
        "매수가",
        "매도가",
        "closing_price",
        "market_cap",
        "score",
        "score_rank",
        "roe",
        "gpa",
        "mom_21d",
        "sector_relative_momentum_21d",
        "sector_relative_momentum_20d",
        "sector_breadth_21d",
        "liquidity_decay_score",
        "low_price_trap",
        "sector_dispersion_21d",
    ]
    display_cols = compact_cols if args.view == "compact" else full_cols
    display_cols = [c for c in display_cols if c in pred_df.columns]
    export_cols = [c for c in full_cols if c in pred_df.columns]

    longs = pred_df.nsmallest(args.top, "rank")[display_cols]
    avoids = pred_df.nlargest(args.bottom, "rank")[display_cols]

    print(f"\nBase date: {latest_date}")
    print(f"Universe: {len(pred_df)}")

    if args.display_style == "vertical":
        _print_df_vertical("Top Picks", longs, decimals=args.display_decimals)
        _print_df_vertical("Avoid Picks", avoids, decimals=args.display_decimals)
    else:
        _print_df_table("Top Picks", longs, decimals=args.display_decimals, max_columns=args.max_columns)
        _print_df_table("Avoid Picks", avoids, decimals=args.display_decimals, max_columns=args.max_columns)

    out_file = Path(f"picks_unified_{latest_date}.csv")
    pred_df[export_cols].sort_values("rank").to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved ranking to {out_file}")


if __name__ == "__main__":
    main()
