# Kaggle 실험 레포

| 폴더 | 대회 | Best Public |
|------|------|-------------|
| `ts-forecasting/` | Hedge fund - Time series forecasting | 0.1499 |
| `churn/` | Playground S6E3 - Churn Classification | 0.91404 |

---

## ts-forecasting

**대회**: 금융 시계열 예측. KEY=(code,sub_code,sub_category,horizon). test 89%가 신규 시리즈.

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 lgbm_baseline | 아무것도 모르니 raw feature + LightGBM부터 | public 0.1499. 생각보다 잘 됨 | lag feature 추가해보자 (AR(1) 발견) |
| 002~010 lag/rolling/cross | AR(1)=0.86 발견 → lag가 핵심일 것 | val 0.89까지 올랐지만 **public 0.0000** | test에서 lag=NaN임을 몰랐다. val이 test 상황 반영 못 함 |
| 012~014 no-lag | lag 빼고 raw+target encoding만 | val 0.30. test 89% 신규라 series_mean도 NaN | val split 자체를 test처럼 바꿔야 함 |
| 015~021 cold-start val | 시리즈 holdout으로 "처음 보는 시리즈" 상황 재현 | cold-start val 0.64. 제출 → **또 0점** | 83EG83KQ(weight=13조, y≈0)에 6.37 예측 → 분자 폭발 |
| 031~032 danger 패치 | 고가중치 시리즈 예측 폭발이 0점 원인 → 자동 패치 추가 | save_submission에 danger_ratio 자동 검사 | known series에 last_y feature 넣어보자 |
| 033 last_y | test known series(10.9%)에 마지막 training y값 추가 | val 0.1145 → **0.2054** (+0.085 점프) | lag 2,3 + rolling stats도 추가 |
| 034~042 rolling+mixed val | lag 1~20, rolling stats + val에 cold-start 15% 혼합 | val 0.36 → 0.49. cold-start 15%가 최적 | weight가 너무 극단적이어서 학습이 편중될 것 같음 |
| 054~058 weight 변환 | weight 최대 13조 → 모델이 극소수 시리즈에 과적합 → sqrt/^0.25/^0.1/^0.05 시도 | **weight^0.05 = val 0.5912** (sweet spot) | cold-start 비율 재조정, 앙상블 |

**제출 현황**

| sub | trial | public | 교훈 |
|-----|-------|--------|------|
| 01 | 001 | 0.1499 | raw feature도 괜찮음 |
| 02 | 010 | 0.0000 | val이 높아도 test 상황 재현 안 하면 의미 없음 |
| 03 | 058 | TBD | weight^0.05 + danger 자동 패치 |

---

## churn (Playground S6E3)

**대회**: 은행 고객 이탈 예측. AUC-ROC. 0.914 근방에서 치열.

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 lgbm_baseline | LightGBM + LabelEncoding + 5-Fold baseline | val 0.91613, public 0.91377 | LabelEncoding이 범주형 관계 못 반영 → target encoding으로 교체 |
| 002 feature_eng | tenure 대비 실제 납부액(AvgMonthlyCharge), 요금 이탈(ChargeGap) 신호 확인 | val 0.91621 | Optuna로 하이퍼파라미터 탐색 |
| 004 lgbm_tuned | Optuna 50 trials + target encoding 조합 | val 0.91663 → **제출 → public 0.91393** | 단일 모델 한계. 앙상블 시도 |
| 005~008 앙상블 | LGBM + XGBoost + CatBoost OOF blend | val 0.91668~0.91677 | OOF grid search로 최적 비율 찾기 |
| 014 mega_blend | 5모델 OOF grid search blend | val 0.91677 → **public 0.91404** | 이게 ceiling인지 확인 |
| 017 final_blend | XGBoost 비중 높인 5모델 blend | val 0.91682 → public 0.91404 | 동일. 피처 더 필요 |
| 018~021 추가 피처들 | groupby 집계, 원본 데이터 추가, 상대적 위치 피처 | 전부 하락 | 노이즈가 많음. multi-seed 앙상블 시도 |
| 024 xgb_multiseed | XGB 7seeds×5fold = 35 models → variance 감소 | val 0.91690 (최고) → **public 0.91395 (하락)** | val 최고 ≠ public 최고. 0.914 벽이 있음 |

**제출 현황**

| sub | trial | public | 교훈 |
|-----|-------|--------|------|
| 01 | 001 | 0.91377 | baseline도 나쁘지 않음 |
| 02 | 004 | 0.91393 | 피처+튜닝으로 +0.002 |
| 03 | 014 | 0.91404 | 앙상블이 효과 있음 |
| 04 | 017 | 0.91404 | 동일. 이미 ceiling |
| 05 | 024 | 0.91395 | val 높다고 public 높은 거 아님 |

---

자세한 규칙: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)
