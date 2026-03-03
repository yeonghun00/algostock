# AlgoStock Overview

A quantitative stock-picking system for Korean equities (KOSPI/KOSDAQ). Pulls market data from KRX, engineers 87 features across 10 groups, trains a LightGBM ranking model via walk-forward validation, and outputs ranked stock picks for live rebalancing.

## Architecture

```
KRX APIs / Raw Financial ZIPs
        |
   ETL Pipelines  ──►  krx_stock_data.db (SQLite)
        |
  ml/features/_pipeline.py   (data loading + merging)
        |
  ml/features/registry.py    (10 feature groups, @register pattern)
        |
  ml/models/lgbm.py          (LightGBM Huber ranker, default)
  ml/models/xgboost.py       (XGBoost alternative)
  ml/models/catboost.py      (CatBoost alternative)
        |
   ┌────┴──────────┐
Backtest           Live
scripts/           scripts/
run_backtest.py    run_live.py
   |                   |
runs/<name>/       Kiwoom REST API
  results.csv       (orders)
  picks.csv
  model.pkl
```

## Directory Structure

```
algostock/
├── etl/                          # Data ingestion
│   ├── krx_api.py                # KRX API client
│   ├── clean_etl.py              # Prices + stock master
│   ├── index_constituents_etl.py # Index membership snapshots
│   ├── delisted_stocks_etl.py    # Delisted stock list
│   ├── adj_price_etl.py          # Adjusted price chain
│   └── financial_etl.py          # IFRS financial statements
├── ml/
│   ├── features/                 # 10 feature groups (registry pattern)
│   │   ├── registry.py           # FeatureGroup base + @register
│   │   ├── _pipeline.py          # DB loading, merging, orchestration
│   │   ├── momentum.py
│   │   ├── momentum_academic.py
│   │   ├── volume.py
│   │   ├── volatility.py
│   │   ├── fundamental.py
│   │   ├── market.py
│   │   ├── sector.py
│   │   ├── sector_neutral.py
│   │   ├── sector_rotation.py
│   │   ├── distress.py
│   │   └── macro_interaction.py
│   └── models/
│       ├── base.py               # BaseRanker (save/load/predict)
│       ├── lgbm.py               # LGBMRanker (default)
│       ├── xgboost.py
│       └── catboost.py
├── scripts/
│   ├── run_backtest.py           # Walk-forward backtest + model save
│   ├── get_picks.py              # Today's picks from saved model
│   ├── run_live.py               # Rebalance schedule + Kiwoom orders
│   ├── run_etl.py                # Unified ETL runner
│   ├── algostock_cli.py          # CLI interface
│   ├── dashboard.py              # HTML dashboard
│   ├── auto_live.sh              # Daily cron/launchd wrapper
│   └── setup_scheduler.sh        # Scheduler install/remove
├── verification/
│   ├── verify_backtest.py        # Independent result cross-check
│   └── README.md
├── runs/                         # One folder per backtest run
│   └── <run_name>/
│       ├── results.csv
│       ├── picks.csv
│       ├── model.pkl
│       ├── report.png
│       ├── rolling_sharpe.csv
│       ├── quintiles.csv
│       ├── sector_attribution.csv
│       └── stat_significance.csv
├── live/
│   ├── state.json                # Current holdings + last rebal
│   ├── logs/                     # Daily execution logs
│   └── orders/                   # Per-date order JSON logs
└── krx_stock_data.db
```

## End-to-End Workflow

### Step 1: ETL — Refresh Data

```bash
python3 scripts/run_etl.py update --markets kospi,kosdaq --workers 4
```

Or via CLI:

```bash
python3 scripts/algostock_cli.py update-all
```

For full historical backfill:

```bash
python3 scripts/run_etl.py backfill --start-date 20100101 --end-date 20251231
```

See [ETL.md](ETL.md) for pipeline details, DB schema, and individual pipeline commands.

### Step 2: Train + Backtest

```bash
python3 scripts/run_backtest.py \
  --start 20100101 --end 20260101 \
  --horizon 42 --top-n 20 \
  --train-years 3 \
  --min-market-cap 100000000000 --max-market-cap 1000000000000 \
  --buy-rank 10 --hold-rank 120 \
  --buy-fee 0.05 --sell-fee 0.25 \
  --patience 100 --no-cache \
  --output myrun --save-picks
```

Builds features, runs walk-forward folds, simulates portfolio with transaction costs, saves model to `runs/myrun/model.pkl`.

See [MODEL.md](MODEL.md) for all CLI flags, feature group reference, and model params.

### Step 3: Get Picks (manual)

```bash
python3 scripts/get_picks.py --model-path runs/myrun/model.pkl --top 20
```

Scores today's universe with the saved model, prints ranked picks and avoid list, saves `picks_unified_YYYYMMDD.csv`.

### Step 4: Live Rebalancing

```bash
# Check schedule (dry-run)
python3 scripts/run_live.py --run myrun

# Execute orders on rebalance day
python3 scripts/run_live.py --run myrun --execute
```

Logic:
- Runs ETL update first
- Reads last rebalance date from `runs/myrun/results.csv`
- Computes next execution date = last_rebal + horizon + 1 trading day
- If today: places sell → buy orders via Kiwoom REST API
- Saves state to `live/state.json` and order log to `live/orders/YYYYMMDD.json`

### Step 5: Automate

```bash
# Install daily launchd scheduler (07:30 local time by default)
./scripts/setup_scheduler.sh start --run myrun --hour 7 --min 30

# Also wake Mac from sleep 5 min before
sudo pmset repeat wakeorpoweron MTWRF 07:25:00

# Check scheduler status
./scripts/setup_scheduler.sh status

# Stop
./scripts/setup_scheduler.sh stop
sudo pmset repeat cancel
```

## Key Design Principles

**Point-in-time (PIT) safety** — Financial data only used after its `available_date` (45/90-day rule). No future information leaks into training or evaluation. See [BIAS.md](BIAS.md).

**Walk-forward validation** — Model never tested on training data. Rolling N-year training window, tested on the next calendar year. 43-day embargo between train and test.

**Transaction-cost-aware** — Buy/sell fees on every rebalance. Hysteresis (buy-rank / hold-rank) reduces unnecessary turnover.

**Sector-aware** — Sector z-scores, sector relative momentum, breadth, rotation signals. Scoring sector-neutralized by default.

**Survivorship-bias-free** — Delisted stocks included in universe up to their delisting date.

## Output Files

| File | Description |
|------|-------------|
| `runs/<name>/results.csv` | Per-rebalance returns, alpha, IC, benchmark |
| `runs/<name>/picks.csv` | All stock picks with scores and forward returns |
| `runs/<name>/model.pkl` | Trained model artifact (LightGBM) |
| `runs/<name>/report.png` | Visual backtest report |
| `runs/<name>/rolling_sharpe.csv` | Rolling Sharpe over time |
| `runs/<name>/quintiles.csv` | Q1–Q5 average returns |
| `runs/<name>/sector_attribution.csv` | Sector contribution breakdown |
| `runs/<name>/stat_significance.csv` | t-stats, bootstrap CI, IC t-stat |
| `live/state.json` | Current holdings + last executed rebalance date |
| `live/logs/YYYYMMDD.log` | Full daily execution log |

## Key Metrics to Watch

- **Mean IC**: Rank correlation between model scores and forward returns. 0.12+ is strong.
- **IC IR**: IC / std(IC). > 1.5 is very good.
- **Quintile monotonicity**: Q5 return > Q4 > Q3 > Q2 > Q1. If not, model ranking is broken.
- **Down capture**: < 0.7 is good defense. 0.07 means near-independent from market drops.
- **Beta**: < 0.5 means returns are largely alpha-driven, not market-beta.
- **Sharpe t-stat**: > 2.0 (5% significance). Prefer Newey-West HAC version.
