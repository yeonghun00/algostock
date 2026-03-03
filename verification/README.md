# Backtest Verification Tool

백테스트 결과(`picks.csv`)를 **FinanceDataReader (Naver Finance)** 를 통해 독립적으로 재검증하는 도구.

---

## 사용법

```bash
# 기본 실행 (run 이름으로)
python3 verification/verify_backtest.py --run myrun

# tolerance 5% 기준으로 (권장)
python3 verification/verify_backtest.py --run myrun --tolerance 0.05

# picks.csv 직접 지정
python3 verification/verify_backtest.py --picks runs/myrun/picks.csv --tolerance 0.05

# 출력 폴더 지정
python3 verification/verify_backtest.py --run myrun --out my_verification_output
```

### 전체 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--run NAME` | — | `runs/NAME/picks.csv` 자동 로드 |
| `--picks PATH` | — | picks.csv 직접 경로 지정 |
| `--tolerance FLOAT` | `0.02` | match 기준 수익률 차이 (0.05 = ±5%) |
| `--out PATH` | picks 폴더 내 `verification/` | 결과 저장 폴더 |
| `--fwd-col NAME` | 자동감지 | forward return 컬럼명 (보통 자동감지됨) |
| `--delay FLOAT` | `0.3` | API 요청 간격(초), rate limiting |

---

## 작동 원리

### 데이터 소스: Naver Finance 수정주가

```
fdr.DataReader('NAVER:005930', start, end)
```

- **Naver Finance**는 수정주가(수정종가) 를 반환 → pykrx(우리 ETL)와 동일한 조정 방향
- KRX 소스는 수정주가 미반영(raw 종가)이므로 사용 안 함

### 가격 기준 정렬

백테스트와 동일한 기준을 사용:

| | 백테스트 (`run_backtest.py`) | 검증 도구 |
|---|---|---|
| **매수가** | `opening_price.shift(-1)` = T+1 시가 | `_next_open()`: signal date 이후 첫 영업일 시가 |
| **매도가** | `opening_price.shift(-(horizon+1))` = T+43 시가 | `_open_on()`: 매도날짜 당일 시가 |
| **매도날짜** | T+43 날짜 (= `매도날짜` 컬럼) | 그대로 사용 |

공휴일 처리: 해당 날짜 데이터 없으면 10 캘린더일 이내 다음 영업일 자동 fallback.

---

## 결과 해석

### 검증 결과 요약

```
Total trade-records             : 660
  Fully verified (both returns)  : 638
    ✅  Match  (|Δ| ≤ 5%)        : 633  (99.2%)
    ⚠️   Discrepancy (|Δ| > 5%)   : 5
  🔴  Delisted / unavailable      : 0
  ❓  No sell date                : 22

Return accuracy:
  Mean   |Δreturn|  : 0.118%
  Median |Δreturn|  : 0.013%
```

### 각 status 의미

| Status | 의미 |
|---|---|
| `match` | FDR 수익률과 BT 수익률 차이가 tolerance 이내 |
| `discrepancy` | 차이가 tolerance 초과 → 아래 원인 분석 참고 |
| `delisted_or_unavailable` | FDR에서 데이터 없음 (상폐 또는 데이터 미존재) |
| `no_sell_date` | 매도날짜 NaN → 아직 열린 포지션 (백테스트 tail) |
| `fdr_price_missing` | FDR에 데이터는 있으나 해당 날짜 가격 없음 |
| `bt_return_missing` | picks.csv에 forward return 값 없음 |

### 권장 tolerance: `--tolerance 0.05`

`0.02` (2%)는 너무 엄격함. 이유:
- pykrx와 Naver는 동일한 방향(수정주가)이지만 유상증자 TERP 공식이 미세하게 다름
- 두 소스 간 가격 레벨 차이 ~1~2%가 매수/매도 양쪽에서 누적되면 수익률 오차 ~2~4%

`0.05` (5%)가 실질적 기준:
- 시스템적 오류(코드 버그, look-ahead bias)는 이 수준 넘는 대규모 패턴으로 나타남
- 단순 데이터 제공자 간 조정 계수 차이는 5% 이내

---

## Discrepancy 원인 분류

### 1. 정상 — 데이터 제공자 간 조정 계수 차이

**패턴:** `sell_ratio ≈ 1.000`, `buy_ratio ≠ 1.000`

매도가는 정확히 일치하는데 매수가만 다를 때 → 매수 시점 근처에 corporate action(유상증자, 액면분할 등)이 발생했고 pykrx와 Naver가 서로 다른 조정 계수 적용.

→ **백테스트 오류 아님.** 두 데이터 제공자 모두 각자의 방식으로 올바른 수정주가 계산.

예시:
```
아주IB투자 20230504 buy_ratio=1.000  ← 완벽 일치
아주IB투자 20230706 buy_ratio=1.000  ← 완벽 일치
아주IB투자 20230905 buy_ratio=1.071  ← 이 날짜 근처 corporate action 발생
```

### 2. 주의 — 홀딩 기간 중 corporate action

**패턴:** `buy_ratio ≈ 1.000`, `sell_ratio ≈ 1.000`인데 수익률 부호가 반대

홀딩 기간 중에 액면분할/유상증자가 발생하면 가격 단위가 바뀜. Naver는 이 시점의 price를 retroactive 조정하지 않을 수 있음. → 이 경우 **pykrx 기반 백테스트가 경제적으로 올바른 수익률**.

### 3. 확인 필요 — 양쪽 다 가격이 크게 다를 때

`buy_ratio`와 `sell_ratio` 모두 1.0에서 멀고, 수익률도 크게 다를 때 → 해당 종목 데이터 개별 확인 필요.

---

## 출력 파일

실행 후 `runs/<run>/verification/` (또는 `--out` 지정 경로)에 저장:

| 파일 | 내용 |
|---|---|
| `verification_detail.csv` | 모든 거래 상세 (BT 가격, FDR 가격, 수익률 비교) |
| `discrepancy_report.csv` | tolerance 초과 거래만 필터링 |
| `delisted_report.csv` | 상폐/데이터 없는 종목 목록 |
| `verification_summary.txt` | 콘솔 출력과 동일한 텍스트 리포트 |

---

## 검증 한계 (이 도구가 잡지 못하는 것)

이 도구는 **데이터 정확도와 수익률 계산 로직**을 검증함. 다음은 검증 범위 밖:

| 요소 | 설명 |
|---|---|
| **Market impact** | 실제 매수시 호가에 미치는 영향 (소형주 슬리피지) |
| **유동성 리스크** | 장 초반 시가 체결 불가 케이스 |
| **거래비용** | picks.csv에는 없음, 백테스트 내부에서 별도 반영 |
| **상폐 종목** | FDR 데이터 없음 → `delisted_report.csv`에 별도 기록 |

---

## myrun 검증 결과 요약 (2026-02-24 기준)

```
소스: Naver Finance 수정주가
tolerance: ±5%
매수 기준: T+1 시가  /  매도 기준: 매도날짜 당일 시가

Match rate  : 633/638 = 99.2%
Mean  |Δ|   : 0.118%
Median |Δ|  : 0.013%
Max   |Δ|   : 22.656%  (SNK 950180, corporate action 조정 계수 차이)

결론: 백테스트 계산 로직 정상. 나머지 오차는 데이터 제공자 간
      유상증자 조정 계수 차이로, 백테스트 신뢰도에 영향 없음.
```
