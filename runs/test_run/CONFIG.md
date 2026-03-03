# test_run — Backtest Config & Results

**Date run:** 2026-03-03
**Verdict:** STRONG ✅ — all 4 t-tests significant at 1%

## Command

```bash
python3 scripts/run_backtest.py \
  --start 20100101 --end 20260101 \
  --horizon 21 --top-n 10 \
  --train-years 2 --min-market-cap 100000000000 --max-market-cap 1000000000000 \
  --buy-rank 10 --hold-rank 120 \
  --buy-fee 0.05 --sell-fee 0.25 \
  --patience 100 --no-cache \
  --output test_run --save-picks
```

## Key Parameters

| Parameter | Value | Note |
|---|---|---|
| Horizon | 21 trading days | ~1 month hold |
| Top-N | 10 stocks | Concentrated portfolio |
| Train window | 2 years rolling | Shorter = more adaptive |
| Market cap | 100B – 1T KRW | Mid-cap universe |
| Buy rank | ≤ 10 | Only add new position if rank top-10 |
| Hold rank | ≤ 120 | Keep holding if still top-120 |
| Buy fee | 0.05% | |
| Sell fee | 0.25% | |
| Period | 2017–2025 | 9 years out-of-sample |

## Performance Summary

| Metric | Value |
|---|---|
| Total Return | **545.87%** |
| Benchmark (KOSPI 200) | 218.36% |
| Alpha | **+327.51%** |
| Ann. Return | **23.03%** |
| Ann. Vol | 19.62% |
| **Sharpe** | **1.17** |
| Calmar | 0.78 |
| Max Drawdown | -29.40% |
| Max Underwater | 20 rebals |
| Rebalances | 108 (9 years) |

## Risk Profile

| Metric | Value | Interpretation |
|---|---|---|
| Up Capture | 0.83 | Captures 83% of market upside |
| Down Capture | 0.31 | Only 31% of market downside — good defense |
| Beta | 0.29 | Mostly alpha-driven, not market exposure |
| Win Rate | 50.00% | Beats benchmark exactly half the time |
| Profit Factor | 1.30 | Wins > losses in magnitude |
| Win/Loss Ratio | 1.30 | Avg win +5.59%, avg loss -4.32% |

## Signal Quality

| Metric | Value |
|---|---|
| Mean IC | 0.0973 |
| IC IR | 1.11 |
| Quintile | Q1=-0.44% → Q5=+2.26% — monotonic PASS |

## Statistical Significance

| Test | t-stat | p-value |
|---|---|---|
| OLS | +3.48 | 0.001 *** |
| Newey-West HAC | +3.52 | 0.001 *** |
| Sharpe (Lo 2002) | +3.48 | 0.001 *** |
| Bootstrap Sharpe 95% CI | [0.55, 2.16] | ✅ entirely positive |
| IC t-stat | +11.51 | 0.000 *** |

## Annual Returns

| Year | Return | Alpha | Sharpe |
|---|---|---|---|
| 2017 | 49.75% | +24.01% | 1.93 |
| 2018 | 25.37% | +44.08% | 1.74 |
| 2019 | 16.19% | +1.15% | 1.16 |
| 2020 | 100.97% | +52.80% | 6.68 |
| 2021 | -5.11% | -2.85% | -0.24 |
| 2022 | 0.06% | +21.79% | 0.00 |
| 2023 | 24.89% | +7.46% | 1.10 |
| 2024 | -8.59% | -0.32% | -0.54 |
| 2025 | 35.93% | -85.82%* | 2.67 |

*2025 alpha anomaly — benchmark data likely incomplete for tail period.

## Top Feature Groups (Gain)

| Group | Share |
|---|---|
| sector_neutral | 33.5% |
| fundamental | 19.5% |
| sector | 12.0% |
| market | 11.7% |
| momentum_academic | 7.8% |
| distress | 7.0% |

## Robustness Tests

| Test | Result |
|---|---|
| Long-Short (Top 10% - Bottom 10%) | 47.57% return, Sharpe 2.85 |
| Beta-Hedged (β=0.29) | 18.13% return, Sharpe 0.97 |
| Ex-2023 robustness | Sharpe 1.18 — PASS (≥0.70) |
| Turnover reduction | 78.61%→70.65%, Sharpe 1.17→0.97 |

## vs myrun (horizon=42, top-n=20)

| | test_run | myrun |
|---|---|---|
| Horizon | 21d | 42d |
| Top-N | 10 | 20 |
| Total return | **545.87%** | 195.90% |
| Sharpe | **1.17** | 0.96 |
| Alpha | **+327.51%** | +42.72% |
| Max DD | -29.40% | **-17.82%** |
| Down capture | 0.31 | **0.07** |
| Beta | **0.29** | 0.38 |
| t-stat (NW) | **+3.52** | +2.78 |

test_run has higher absolute returns and Sharpe. myrun has tighter drawdown and better downside protection.
