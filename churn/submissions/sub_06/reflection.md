## Submission 06 Reflection

### 결과
- Val score: 0.91683 (trial_028 lgbm_reg_multiseed)
- Public score: 0.91400
- Val-Public gap: -0.00283

### Gap 원인 분석
- sub_05(-0.00295)보다 gap 축소 → 정규화 강화가 효과 있었음
- 하지만 all-time public best(0.91404, sub_03/04)에는 여전히 못 미침
- val이 높다고 public이 높다는 보장 없음 — 024는 val best였지만 public은 하락
- 정규화 강화(lambda=2.0)가 gap을 줄이는 올바른 방향임을 확인

### 효과 있었던 것
- **강한 정규화** (lambda=2.0, num_leaves=31→31, min_child_samples 20→50): gap -0.00295 → -0.00283으로 축소
- **EDA**: 피처 독립성 확인, 고위험 조합 피처 발굴 (SeniorCitizen×Electronic check 등)
- **Adversarial validation**: train/test 분포 차이 없음 확인 → gap 원인이 분포 차이 아님 확인

### 효과 없었던 것
- Pseudo-labeling: 하락
- ExtraTrees: 큰 하락 (0.91258)
- Smoothed target encoding: val 하락 (효과 미미)
- CatBoost Optuna: 기본 대비 미미한 개선
- 원본 데이터 컬럼 merge (Chris Deotte 방식): 하락

### 버려야 할 것
- val 기준으로만 best 판단 — public과 방향 다를 수 있음
- 새 모델/피처 계속 추가하는 방향 — 이미 한계 도달
- ExtraTrees, pseudo-labeling

### 유지해야 할 것
- 강한 정규화 기조 (lambda=2.0)
- multi-seed 앙상블
- LGBM + XGB + CatBoost 3모델 구조

### 다음 가설
- 정규화 더 강화 (lambda=5.0 등) 해볼 것
- 036 ultimate blend (9모델 scipy 최적화) 결과 sub_07로 제출
- val이 아닌 public gap을 줄이는 방향으로 집중
