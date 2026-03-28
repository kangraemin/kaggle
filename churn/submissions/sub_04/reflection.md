## Submission 04 Reflection

### 결과
- Val score: 0.91682 (trial_017)
- Public score: 0.91404
- Val-Public gap: -0.00278

### Gap 원인 분석
- sub_03(0.91404)와 동일한 public score → val 개선(+0.00005)이 public으로 전혀 연결 안 됨
- 앙상블 조합 미세 조정은 더 이상 효과 없음 — 근본적인 모델 다양성이 필요한 시점

### 효과 있었던 것
- XGBoost Optuna 30 trials (trial_015): val +0.0001
- 5모델 final blend (trial_017): val 소폭 개선

### 효과 없었던 것
- meta-learner stacking (LR): weighted blend보다 낮음
- groupby 집계 피처 83개: 하락 — synthetic 데이터라 집계 패턴 의미 없음

### 버려야 할 것
- 앙상블 조합 미세 조정 — public 개선 없음
- 피처 엔지니어링 (groupby, 도메인 피처) — 이 데이터에서 일관되게 효과 없음

### 유지해야 할 것
- LGBM + XGB + CatBoost 3모델 구조
- OOF weighted blend

### 다음 가설
- Neural Network (MLP/TabNet) 추가 — GBDT와 다른 inductive bias로 앙상블 다양성 확보
- 모델 다양성이 핵심, 피처/파라미터 튜닝은 한계 도달
