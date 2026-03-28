## Submission 03 Reflection

### 결과
- Val score: 0.91677 (trial_014)
- Public score: 0.91404
- Val-Public gap: -0.00273

### Gap 원인 분석
- gap이 sub_02(-0.0027)와 거의 동일 → 앙상블 조합을 바꿔도 gap 구조는 변하지 않음
- val 개선(+0.00019)이 public 개선(+0.00011)으로 약하게 연결됨
- 앙상블 한계 도달: trial_014→017 val은 올랐으나 public은 동일(0.91404)

### 효과 있었던 것
- XGBoost Optuna 튜닝 (trial_015): +0.0001 개선
- 5모델 weighted blend (trial_014/017): 단일 모델 대비 소폭 개선
- OOF grid search weight 탐색: 단순 평균보다 나음

### 효과 없었던 것
- 학습률 낮추기 (0.05→0.01): 하락
- External data (원본 Telco 7천행): 하락 — 59만 행 대비 너무 작음
- "No internet service"→"No" 통합: 하락
- 10-Fold: 5-Fold보다 낮음
- groupby 집계 83 피처: 하락 — 노이즈
- Meta-learner stacking: weighted blend보다 낮음

### 버려야 할 것
- 단순 피처 추가 접근 — 이 데이터에서 피처 엔지니어링 효과 미미
- groupby 집계 — synthetic 데이터라 실제 집계 패턴이 의미 없음

### 유지해야 할 것
- XGB + LGBM + CatBoost 앙상블 구조
- OOF grid search weight 탐색
- target encoding (CV-safe)

### 다음 가설
- Neural Network (MLP/TabNet) — GBDT와 완전히 다른 inductive bias로 앙상블 다양성 확보
- 피처보다 모델 다양성이 핵심인 것 같음
