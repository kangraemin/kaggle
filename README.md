# Kaggle 실험 레포

대회별 trial/submission 관리. 모든 대회는 `TRIAL_GUIDE.md` 규칙을 따른다.

## 대회 목록

| 폴더 | 대회 | 상태 | Best Public |
|------|------|------|-------------|
| `ts-forecasting/` | Hedge fund - Time series forecasting | 진행중 | 0.1499 (sub_01) |
| `churn/` | Playground S6E3 - Binary Classification with Bank Churn | 진행중 | 0.91404 (sub_03, 04) |

---

## ts-forecasting (Hedge fund - Time series forecasting)

### 대회 개요
- **데이터**: 36,923개 금융 시계열, 각 144 step 관측. KEY = (code, sub_code, sub_category, horizon)
- **타겟**: weighted_rmse_score (max 1.0)
- **핵심 난관**: test 시리즈 89%가 신규(train에 없는 조합). lag feature 쓰면 test에서 NaN.

### Submissions

| sub | Best Trial | Val | Public | 왜 이걸 골랐나 | 결과 |
|-----|-----------|-----|--------|----------------|------|
| 01 | trial_001 | 0.1163 | **0.1499** | 아무것도 모를 때 raw feature만으로 시작. baseline | public > val. raw feature가 일반화 잘됨 |
| 02 | trial_010 | 0.8923 | **0.0000** | val score가 제일 높아서 선택했는데… | lag feature가 test에서 NaN → 모든 예측 0 → 0점 |
| 03 | trial_058 | 0.5912 | **TBD** | weight^0.05로 학습 균형 맞추고 danger 자동 패치 | pending |

### 실험 흐름

#### sub_01 — "일단 뭐가 있는지 보자"
raw feature + LightGBM으로 시작. val 0.1163, public 0.1499. 생각보다 잘 됨.

#### sub_02 — "AR(1)=0.86이면 lag가 핵심 아닌가?" → 완전 실패
lag_1 correlation 0.86을 발견하고 흥분해서 lag feature를 마구 추가했다. val score가 0.89까지 올라갔다.
그런데 val은 train 내부 split이라 y_target이 있어서 lag가 동작했던 것. test에서는 y_target 없음 → lag=NaN → 모델이 0만 예측 → 0점.

> **교훈**: val score가 높다고 제출하면 안 된다. val이 test 상황을 재현하는지 확인 필수.

#### sub_02 이후 — "그럼 어떻게 하지?"

**1단계: lag 없이 raw feature만 (trial_012)**
- lag 빼고 target encoding만 → test에서 89% 신규 시리즈라 series_mean도 NaN. val 0.30.

**2단계: cold-start val 설계 (trial_015~021)**
- ts_index split은 거짓말. 시리즈 10% holdout으로 "처음 보는 시리즈" 상황 재현.
- cold-start val 0.64 달성. 그런데 제출하니 또 0점. 83EG83KQ(weight=13조)에 6.37 예측 → 분자 폭발.

**3단계: 고가중치 시리즈 문제 발견 (danger_ratio)**
- 83EG83KQ: y_target≈0, weight=13조. 조금이라도 크게 예측하면 ratio가 백만 배 → 0점.
- save_submission()에 자동 danger 패치 추가. 이제 어떤 trial도 0점 안 나옴.

**4단계: known series에 last_y feature (trial_033)**
- test에서 10.9%는 known series (training에 있는 KEY). 그 시리즈의 마지막 y_target 추가.
- val score 0.1145 → 0.2054 (+0.085 점프). known series에서 last_y가 압도적 1위 feature.

**5단계: 더 많은 시계열 features + mixed val (trial_034~042)**
- last_y_1/2/3 + rolling stats + series_mean 추가 → val 0.36~0.49
- val이 100% known series라 unknown 상황 반영 안 됨 → cold-start 15% 섞기 → mixed val
- 수렴: trial_042 val 0.4873

**6단계: weight 변환 발견 (trial_054~058)** ← 진짜 breakthrough
- 훈련 weight가 13조까지 있어서 모델이 극소수 시리즈에 과적합됨
- sqrt → 0.5227, ^0.25 → 0.5536, ^0.1 → 0.5849, ^0.05 → **0.5912** (sweet spot)
- weight^0.05로 압축하면 모든 시리즈를 비슷하게 중요하게 학습 → 일반화 폭발

### 핵심 인사이트

**효과 있었던 것:**
- `last_y` (마지막 training y값) → known series 예측의 핵심
- mixed val (시간분할 + cold-start 15%) → unknown series 반영
- **`weight^0.05` 변환** → 극단 weight 압축, 균형있는 학습
- 자동 danger 패치 (`save_submission` 내부)

**효과 없었던 것:**
- lag/rolling/ewm → test에서 NaN
- series_mean 단독 → 89% NaN
- target normalization, pseudo-labeling, clustering → 모두 0점 또는 하락
- AR 예측값 feature → last_y와 중복 정보

---

## churn (Playground S6E3 - Bank Customer Churn)

### 대회 개요
- **데이터**: 은행 고객 이탈 예측. tenure, MonthlyCharges, TotalCharges 등 금융 피처.
- **타겟**: AUC-ROC (max 1.0)
- **특징**: 피처 설계 싸움. public 0.914 근방에서 치열.

### Submissions

| sub | Best Trial | Val | Public | 왜 이걸 골랐나 | 결과 |
|-----|-----------|-----|--------|----------------|------|
| 01 | trial_001 | 0.91613 | **0.91377** | baseline — LightGBM raw feature | val-public gap 안정적 (0.0024) |
| 02 | trial_004 | 0.91663 | **0.91393** | Optuna 하이퍼파라미터 + target encoding | +0.002 개선 |
| 03 | trial_014 | 0.91677 | **0.91404** | 5모델 OOF blend (grid search로 최적 비율) | 앙상블 효과 |
| 04 | trial_017 | 0.91682 | **0.91404** | 5모델 blend (xgb_opt 주도) | 동일 public |
| 05 | trial_024 | 0.91690 | **0.91395** | XGB 7seeds × 5fold = 35 models | val 최고였는데 public 오히려 하락 |
| 06 | trial_028 | 0.91683 | **0.91400** | - | - |

### 실험 흐름

#### sub_01 — baseline
LightGBM + LabelEncoding + 5-Fold. val 0.916, public 0.914. 시작은 좋음.
sub_01 reflection에서 발견: LabelEncoding이 범주형 관계를 못 반영. → target encoding으로 교체.

#### sub_02 — 피처 설계 집중
- `AvgMonthlyCharge = TotalCharges / (tenure+1)` — tenure 대비 납부액 비율
- `ChargeGap = MonthlyCharges - AvgMonthlyCharge` — 최근 요금이 평균보다 높으면 churn 가능성↑
- Optuna로 하이퍼파라미터 탐색 → trial_004가 최고 val → 제출

#### sub_03 — 앙상블 탐색
단일 모델의 한계를 느끼고 여러 모델 조합:
- LightGBM, XGBoost, CatBoost, 각각 다른 피처 조합
- OOF 예측으로 grid search해서 최적 blend 비율 찾기 (trial_014)
- public 0.91404 달성 → 이후 trial들이 val은 높아도 public은 여기서 안 벗어남

#### sub_04~06 — 더 이상 안 올라감
multi-seed 앙상블, meta-stacking, 추가 피처들 다 해봤지만 public 0.914 벽 못 넘음.
val은 0.91690까지 올라갔는데 public은 오히려 0.91395로 하락 → 과적합.

### 핵심 인사이트

**효과 있었던 것:**
- `ChargeGap`, `AvgMonthlyCharge` 피처 설계
- Optuna 하이퍼파라미터 튜닝
- OOF 기반 모델 앙상블

**효과 없었던 것:**
- groupby 집계 83피처 → 노이즈
- 원본 Telco 데이터 추가 → 분포 차이로 오히려 하락
- multi-seed 35 models → val↑ but public↓ (과적합)
- meta-stacking → 개선 없음

---

## 제출 전 필수 체크

```python
# ts-forecasting: utils.py의 save_submission이 자동 실행
# danger_ratio > 0.1 코드는 train mean으로 자동 패치
```

자세한 규칙: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)
