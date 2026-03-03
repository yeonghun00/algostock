# AlgoStock

Korean equity quantitative trading system — KRX data pipeline, LightGBM ranking model, walk-forward backtest, and automated live rebalancing via Kiwoom REST API.

**Backtest results (2018–2025, out-of-sample)**
- Total return: 195.9% vs KOSPI 200 benchmark 153.2% (+42.7% alpha)
- Sharpe: 0.96 | Calmar: 0.82 | Max drawdown: -17.8% | Ann. return: 14.5%
- IC: 0.1268 | IC IR: 1.76 | Down capture: 0.07 | Beta: 0.38
- Statistical significance: 4/4 t-tests pass at 1% (Newey-West, Bootstrap CI [0.36, 1.83])

---

## How to Use

### 1. Update data

```bash
python3 scripts/run_etl.py update --markets kospi,kosdaq --workers 4
```

Or via the unified CLI:

```bash
python3 scripts/algostock_cli.py update-all
```

### 2. Run a backtest

```bash
python3 scripts/run_backtest.py \
  --start 20100101 --end 20260101 \
  --horizon 42 --top-n 20 \
  --train-years 3 --min-market-cap 100000000000 --max-market-cap 1000000000000 \
  --buy-rank 10 --hold-rank 120 \
  --buy-fee 0.05 --sell-fee 0.25 \
  --patience 100 --no-cache \
  --output myrun --save-picks
```

Results saved to `runs/myrun/`. Includes `results.csv`, `picks.csv`, `model.pkl`, `report.png`.

### 3. Get today's picks (manual)

```bash
python3 scripts/get_picks.py --model-path runs/myrun/model.pkl --top 20
```

### 4. Live rebalancing — check schedule

```bash
# Dry-run: update DB, check if today is rebalance day, show plan
python3 scripts/run_live.py --run myrun
```

Shows one of:
- `⏳ N 거래일 후 실행일` — not yet, nothing to do
- `📅 내일이 실행일` — tomorrow, previews picks
- `✅ 오늘이 실행일` — today, shows buy/sell orders

### 5. Live rebalancing — execute orders

```bash
python3 scripts/run_live.py --run myrun --execute
```

Requires Kiwoom API credentials in `.env` (see below).

---

## Automated Scheduling

### Start (runs daily at 07:30 local time, auto-selects latest run)

```bash
./scripts/setup_scheduler.sh start
```

### Start with specific run and time

```bash
./scripts/setup_scheduler.sh start --run myrun --hour 7 --min 30
```

### Stop everything

```bash
./scripts/setup_scheduler.sh stop
sudo pmset repeat cancel    # if you set the wake schedule
```

### Check status

```bash
./scripts/setup_scheduler.sh status
```

### Wake Mac from sleep before scheduled time (optional)

```bash
# Wake 5 min before — e.g. for 07:30 schedule:
sudo pmset repeat wakeorpoweron MTWRF 07:25:00
```

> **Timezone note (HKT = UTC+8):** Korean market opens 9:00 AM KST = 8:00 AM HKT.
> Run before 8:00 AM HKT to place opening orders. Running after 8:00 AM HKT means market is closed and Kiwoom will reject orders.

---

## Kiwoom API Setup

Create `.env` in the project root (already in `.gitignore`):

```
KIWOOM_APP_KEY=your_app_key
KIWOOM_APP_SECRET=your_app_secret
KIWOOM_ACCOUNT=12345678-01
KIWOOM_MOCK=true       # true = paper trading, false = real money
```

---

## DB Validation

```bash
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM daily_prices;"
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM index_constituents;"
sqlite3 krx_stock_data.db "SELECT COUNT(*) FROM financial_periods;"
```

---

## Verify Backtest Results Independently

```bash
python3 verification/verify_backtest.py --run myrun --tolerance 0.05
```

Cross-checks picks against Naver Finance adjusted prices. See `verification/README.md`.

---

## Project Structure

```
algostock/
├── etl/                          # Data ingestion pipelines
│   ├── krx_api.py                # KRX API client (rate-limited, parallel)
│   ├── clean_etl.py              # Prices + stock master
│   ├── index_constituents_etl.py # Index membership snapshots
│   ├── delisted_stocks_etl.py    # Delisted stock list
│   ├── adj_price_etl.py          # Adjusted price chain builder
│   └── financial_etl.py          # Financial statements (BS/PL/CF)
├── ml/
│   ├── features/                 # Feature engineering (registry pattern)
│   │   ├── registry.py           # @register decorator + topological sort
│   │   ├── _pipeline.py          # Data loading, merging, orchestration
│   │   ├── momentum.py           # Price momentum features
│   │   ├── momentum_academic.py  # 52w proximity, MA ratios
│   │   ├── volume.py             # Amihud illiquidity, turnover
│   │   ├── volatility.py         # Rolling vol, beta
│   │   ├── fundamental.py        # ROE, GPA (PIT-safe)
│   │   ├── market.py             # Market regime, index count
│   │   ├── sector.py             # Sector momentum, breadth
│   │   ├── sector_neutral.py     # Sector z-scores
│   │   ├── sector_rotation.py    # Dispersion, rotation signal
│   │   ├── distress.py           # Liquidity decay, low-price trap
│   │   └── macro_interaction.py  # Value/momentum regime interactions
│   └── models/                   # Model backends
│       ├── base.py               # BaseRanker (save/load/predict)
│       ├── lgbm.py               # LightGBM (default)
│       ├── xgboost.py            # XGBoost
│       └── catboost.py           # CatBoost
├── scripts/                      # Entry points
│   ├── run_backtest.py           # Walk-forward backtest + model training
│   ├── get_picks.py              # Generate picks from trained model
│   ├── run_live.py               # Live rebalancing + Kiwoom orders
│   ├── run_etl.py                # Unified ETL runner
│   ├── algostock_cli.py          # CLI (etl status/update/backfill)
│   ├── dashboard.py              # HTML dashboard generator
│   ├── auto_live.sh              # Daily automation wrapper
│   └── setup_scheduler.sh        # Install/remove launchd scheduler
├── verification/
│   ├── verify_backtest.py        # Independent result verification
│   └── README.md
├── runs/                         # Backtest outputs (per run)
│   └── <run_name>/
│       ├── results.csv           # Per-rebalance returns + alpha
│       ├── picks.csv             # All stock picks with scores
│       ├── model.pkl             # Trained model artifact
│       ├── report.png            # Visual report
│       └── ...
├── live/                         # Live trading state
│   ├── state.json                # Current holdings + last rebal date
│   └── logs/                     # Daily run logs (YYYYMMDD.log)
├── docs/                         # Reference documentation
│   ├── OVERVIEW.md               # Architecture + end-to-end workflow
│   ├── ETL.md                    # ETL pipeline details + DB schema
│   ├── MODEL.md                  # Features, model params, CLI reference
│   └── BIAS.md                   # Bias reduction mechanisms
└── krx_stock_data.db             # SQLite database (all market data)
```

---

## Docs

| Doc | Contents |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Architecture, end-to-end workflow |
| [docs/ETL.md](docs/ETL.md) | ETL pipelines, DB schema, backfill commands |
| [docs/MODEL.md](docs/MODEL.md) | Features (87 total), model params, CLI flags |
| [docs/BIAS.md](docs/BIAS.md) | Look-ahead bias, PIT safety, survivorship controls |
| [verification/README.md](verification/README.md) | Independent backtest verification tool |

---

## Disclaimer

For educational and research purposes only. Past performance does not guarantee future results. Not financial advice.
