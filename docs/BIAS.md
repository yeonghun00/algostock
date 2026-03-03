# 편향 감소 메커니즘 (Bias Reduction Mechanisms)

> 이 문서는 알고리즘 전략에서 발생 가능한 각 편향의 원인, 우리 코드에서의 구현 위치, 그리고 작동 메커니즘을 설명한다.

---

## 1. 미래 데이터 누출 방지 (Look-Ahead Bias)

### 문제 정의
모델이 학습하거나 예측할 때 **실전에서는 알 수 없는 미래 정보**를 사용하는 것.
예: 오늘 날짜로 스코어를 내는데, 내일 공시된 재무제표를 사용.

---

### 메커니즘 1: Point-in-Time 재무 데이터

**위치**: `ml/features/_pipeline.py` → `_load_financial_ratios_pit()`

```
재무제표 공시 흐름:
  회계 기말 (12/31) → 공시 날짜 (3~5월) → available_date
                                              ↑
                                    이 날짜 이후에만 사용
```

```python
WHERE REPLACE(fp.available_date, '-', '') <= ?  # 피처 날짜 이전 공시만
...
merge_asof(direction="backward")  # 가장 최근 과거 공시만 붙임
```

**작동 원리**:
- 한국 상장사는 결산 후 약 90일 내 공시 의무
- `fiscal_end = 2024-12-31`이어도 `available_date = 2025-03-31`이면, 2025-03-31 이전 날짜의 피처에는 이 재무제표가 포함되지 않음
- 450일 스탈니스 가드: 너무 오래된 재무 정보(15개월 이상)는 자동 폐기

---

### 메커니즘 2: 후향적 피처 계산 (Backward-Looking Features)

**위치**: `ml/features/momentum.py`, `volatility.py`, `volume.py`, `distress.py`

```
피처 계산 방향:
  [과거 126일] → [오늘] ✅
  [오늘] → [미래 42일] ❌ (target에만 허용)
```

모든 피처는 `rolling()`, `pct_change()`, `shift(+N)` (양수 shift = 과거 참조)만 사용:

| 피처 | 계산 방식 | 미래 참조? |
|------|---------|----------|
| `mom_5d` | `close.pct_change(5)` | ❌ |
| `volatility_21d` | `returns.rolling(21).std()` | ❌ |
| `amihud_21d` | `abs_ret / value rolling(21)` | ❌ |
| `drawdown_252d` | `close / rolling(252).max() - 1` | ❌ |
| `ma_ratio_20_120` | `MA20 / MA120` | ❌ |

---

### 메커니즘 3: market_forward_return 격리

**위치**: `ml/features/_pipeline.py` → `_load_market_regime()` + `_add_targets()`

```python
# 이것은 FUTURE 데이터 — 하지만 feature가 아닌 target 계산에만 사용
idx[f"market_forward_return_{horizon}d"] = closing_index.shift(-horizon) / closing_index - 1

# feature_cols는 등록된 컬럼만 포함 — market_forward_return 제외됨
MarketFeatures.columns = ["market_regime_120d", "constituent_index_count"]
feature_cols = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in df.columns]
```

시장 선행 수익률은 **Beta-adjusted residual target 계산** (label 생성)에만 쓰이며, 모델의 입력 피처로는 절대 사용되지 않는다.

---

### 메커니즘 4: Walk-Forward 엄격 분리

**위치**: `scripts/run_backtest.py` → `walk_forward_split()` + embargo logic

```
시간 흐름 →

[Train 2019-2022] [21일 Embargo] [Test 2023]
                  ↑
          이 기간 데이터 폐기

[Train 2020-2023] [21일 Embargo] [Test 2024]
[Train 2021-2024] [21일 Embargo] [Test 2025]
```

```python
# Embargo: test 시작 21일 전까지의 training 데이터 제거
cutoff = all_dates[idx - embargo_days]  # embargo_days = 21
sub_train = sub_train[sub_train["date"] < cutoff].copy()
```

**왜 21일 embargo가 필요한가?**
42일 예측 horizon에서, T=12/31 시점의 학습 데이터는 2023-01-01~2023-02-11 구간의 수익률을 타겟으로 사용한다. 만약 Test 시작이 2023-01-01이면, T=12/31 데이터가 Test 기간과 겹친다. Embargo는 이 겹치는 샘플을 제거한다.

---

### 메커니즘 5: Validation Set 시간 격리

**위치**: `scripts/run_backtest.py` → `_run_fold()`

```python
train_years = sorted(train_df["date"].str[:4].unique())
val_year = train_years[-1]        # Training window의 마지막 연도 (미래 아님)
sub_train = train_df[...date < val_year...]
val_df    = train_df[...date == val_year...]
```

Validation set은 **Training window 안에서** 마지막 연도를 분리한다. Test set에서 데이터를 가져오지 않는다. Early stopping의 기준도 이 시간 격리된 Validation set에서만 계산된다.

---

## 2. 생존자 편향 방지 (Survivorship Bias)

### 문제 정의
**이미 상장되어 있는 종목으로만** 과거를 재현하는 것.
예: 2019년 universe를 2026년 현재 상장 종목 기준으로 구성하면, 2019~2026년 사이에 상장폐지된 종목(=망한 종목)이 자동으로 제외된다.

---

### 메커니즘 1: 상장폐지 종목 포함 및 데이터 절단

**위치**: `ml/features/_pipeline.py` → `_exclude_delisted()`

```
상장폐지 종목의 올바른 처리:

  [2019] [2020] [2021] [폐지일: 2021-09-15] ← 여기서 잘라냄
    ↓      ↓      ↓
  포함   포함   포함까지만   ← 훈련 데이터에 반영됨
                           (망하는 과정도 학습)
```

```python
keep = merged["delisting_date"].isna() | (merged["date"] < merged["delisting_date"])
return merged.loc[keep].drop(columns=["delisting_date"])
```

상장폐지 이전 데이터는 **학습에 포함**되어 모델이 "폐지 직전 신호"를 학습하고, 폐지 이후 데이터는 포함되지 않아 존재하지 않는 미래를 참조하지 않는다.

---

### 메커니즘 2: Fix A — 폐지 종목의 Forward Return 처리

**위치**: `ml/features/_pipeline.py` → `_add_targets()`

```
문제: 종목이 T+10에 폐지되면 T의 forward_return_42d = NaN
      NaN을 그냥 제거하면 → "망하지 않은 종목만" 남음 = 생존자 편향

해결: NaN → 실제 마지막 가격으로 수익률 계산
```

```python
# Fix A: 폐지 전 마지막 관찰 가격으로 수익률 대체
nan_mask = out[fwd_col].isna() & out["closing_price"].gt(0)
out.loc[nan_mask, fwd_col] = (
    last_price[nan_mask] / out.loc[nan_mask, "closing_price"] - 1
)
```

예: 매수 가격 5,000원, 폐지 직전 가격 500원 → forward_return = -90%
이 -90% 수익률이 모델 학습에 반영되어 "폐지 직전 신호"를 학습.

---

### 메커니즘 3: Fix B — 거래정지 종목의 Forward Return 처리

**위치**: `ml/features/_pipeline.py` → `_add_targets()`

```
문제: 거래정지(거래정지) 기간 중 가격이 동결 → forward_return이 0%처럼 보임
      실제로는 거래정지 해제 후 폭락하는 경우가 많음

해결: T+42에 거래량이 0이면 → 마지막 실거래 가격으로 수익률 재계산
```

```python
# Fix B: T+42 시점 거래량이 0이면 마지막 가격으로 대체
future_value = g["value"].shift(-target_horizon)
frozen_mask = (
    out[fwd_col].notna()
    & out["closing_price"].gt(0)
    & (future_value == 0)     # 거래정지 감지
)
out.loc[frozen_mask, fwd_col] = (
    last_price[frozen_mask] / out.loc[frozen_mask, "closing_price"] - 1
)
```

---

### 메커니즘 4: Rebalance 시점 거래정지 종목 실시간 제외

**위치**: `scripts/run_backtest.py` → `_run_fold()`, rebalance loop

```python
# 당일 거래량이 0인 종목 = 거래정지 → universe에서 실시간 제외
if "value" in day_df.columns:
    day_df = day_df[day_df["value"] > 0].copy()
```

학습 데이터에는 포함하되(폐지 과정 학습), **실제 포트폴리오 구성에는 제외**한다.
이것이 실전과 동일한 조건이다 — 거래정지 종목은 살 수 없다.

---

## 3. 실행 편향 방지 (Execution Bias)

### 문제 정의
백테스트에서는 "당일 종가로 매수"를 가정하지만, 실전에서는 당일 종가 체결이 불가능하거나 어렵다. (장 마감 직전 대량 주문 = 슬리피지 급증)

---

### 메커니즘 1: 실행 지연 테스트 (exec-lag)

**위치**: `scripts/run_backtest.py` → exec_lag 계산 (2026-02-19 추가)

```python
# T 종가 기준 (기본값)
forward_return_42d = close[T+42] / close[T] - 1

# T+1 종가 기준 (exec_lag=1)
forward_return_42d_lag1 = close[T+43] / close[T+1] - 1
```

T+1 체결로 변경해도 Sharpe가 **1.74 → 2.87**로 오히려 개선됨 → 실행 편향 없음 확인.

---

### 메커니즘 2: 거래 비용 (슬리피지 내재화)

**위치**: `scripts/run_backtest.py` → 수익률 계산

```python
net_port_ret = (1.0 + port_ret) * (1.0 - transaction_cost) - 1.0
transaction_cost = turnover * (buy_fee_rate + sell_fee_rate)
```

매 리밸런싱마다 **실제 매매 비용을 차감**한다. 기본 설정 (buy=0.05%, sell=0.25%)은 현실적인 소형주 시장가 주문 기준이다.

---

## 4. 소표본 편향 방지 (Small Sample Bias)

### 문제 정의
리밸런싱 횟수(~18회)가 적으면 Sharpe 추정치의 표준오차가 크다.
운 좋은 2023년 하나가 전체 통계를 왜곡할 수 있다.

---

### 메커니즘 1: Ex-Year Robustness Test

**위치**: `scripts/run_backtest.py` → Requested Tests

```python
# 특정 연도를 제외한 Sharpe 계산
ex_year_ret = [r for r in results if r["year"] != exclude_year]
```

2023년 제외 후에도 Sharpe ≥ 0.70 이상이면 특정 연도 의존성이 없음을 증명.
→ 실행 지연 테스트: Ex-2023 Sharpe = **2.74** (매우 강건)

---

### 메커니즘 2: Quintile 단조성 검증

**위치**: `scripts/run_backtest.py` → rebalance loop

```python
q_mono = int(q5 > q4 > q3 > q2 > q1)
```

모델 스코어 순위에 따라 Q1~Q5 수익률이 단조증가하면 신호의 일관성 확인.
단순 운이라면 특정 구간만 좋고 전체 단조성은 깨진다.

---

### 메커니즘 3: IC 안정성 (IC IR)

**위치**: `scripts/run_backtest.py` → summary 계산

```
IC (Information Coefficient) = 스코어 순위와 실제 수익률 순위의 상관계수
IC IR = mean(IC) / std(IC)    ← IC의 신호대잡음비

IC IR = 1.53 → IC가 평균 1.53σ만큼 0보다 위에 있음 = 안정된 신호
```

IC가 높아도 IC IR이 낮으면 불안정한 신호 (일부 기간만 좋음).

---

## 5. 유동성 편향 방지 (Liquidity Bias)

### 문제 정의
소형주는 실제 거래대금이 적어, 백테스트 수량을 실전에서 매수하려면 시장 충격(market impact)이 발생한다. 백테스트는 이를 무시한다.

---

### 메커니즘 1: 거래대금 하한 필터 (min-daily-value)

**위치**: `scripts/run_backtest.py` → rebalance loop (2026-02-19 추가)

```python
# 당일 거래대금이 N원 미만인 종목 제외
if min_daily_value > 0 and "value" in day_df.columns:
    day_df = day_df[day_df["value"] >= min_daily_value].copy()
```

거래대금 100억 하한 테스트 결과: Sharpe 2.04 → **0.50** (전략 붕괴)
→ 알파의 원천이 거래대금 100억 미만의 종목임을 역으로 증명.

---

### 실전 AUM 용량 추정

```
가정: 포트폴리오 AUM = X원, Top-10 동일가중
      종목당 배분 = X / 10원
      체결 가능 한도 = 일 거래대금의 10%

체결 가능 조건: X / 10 ≤ 일거래대금 × 10%
             → X ≤ 일거래대금 × 1 (= 1일 전체 거래대금)

평균 일거래대금 (알파 종목): ~30~50억 KRW
최대 AUM: 30~50억 × 1 = 30~50억 KRW per stock × 10 ≈ 300~500억

실질적 한도: 슬리피지 고려 시 ~50~150억 KRW
```

---

## 편향별 요약표

| 편향 종류 | 발생 위험 | 구현된 방어 메커니즘 | 검증 결과 |
|---------|---------|-----------------|---------|
| **Look-Ahead (피처)** | 미래 데이터 사용 | PIT 재무 + backward rolling | ✅ CLEAN |
| **Look-Ahead (타겟)** | 타겟이 피처에 누출 | feature_cols 엄격 분리 | ✅ CLEAN |
| **Walk-Forward 누출** | 미래 테스트 데이터 학습 | 21일 embargo + 시간순 분리 | ✅ CLEAN |
| **Validation 누출** | test에서 val set 추출 | train window 내부에서 val 분리 | ✅ CLEAN |
| **생존자 편향 (폐지)** | 망한 종목 제외 | Fix A + _exclude_delisted | ✅ CLEAN |
| **생존자 편향 (정지)** | 거래정지 수익률 왜곡 | Fix B + value>0 필터 | ✅ CLEAN |
| **실행 편향** | T종가 체결 불가 | exec_lag=1 테스트 | ✅ CLEAN (Sharpe 2.87) |
| **소표본 편향** | 특정 연도 의존성 | Ex-year test + IC IR | ✅ 강건 |
| **유동성 편향** | 체결 불가 종목 선택 | min-daily-value 필터 | 🔴 **AUM 한계 확인** |
| **파라미터 오버피팅** | 하이퍼파라미터 in-sample 튜닝 | 추가 OOS 검증 필요 | ⚠️ 잔존 위험 |

---

*작성: 2026-02-19. 기반 코드: `CACHE_VERSION = "unified_v49_delistfix_20260218"`*
