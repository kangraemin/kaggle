## Submission 05 Reflection

### 결과
- Val score: 0.91690 (trial_024 XGB multi-seed)
- Public score: 0.91395
- Val-Public gap: -0.00295

### Gap 원인 분석
- Val은 역대 best였으나 public은 오히려 하락 (-0.00009 vs sub_04)
- Val-public gap이 -0.0027에서 -0.00295로 확대 → val overfitting 심화
- XGB multi-seed가 train OOF에서 분산을 줄였지만 test 일반화는 개선 안 됨
- val과 public 방향이 달라지기 시작 → val 지표 신뢰도 낮아짐

### 버려야 할 것
- **val 기준으로만 best 판단** — public과 방향이 달라질 수 있음
- 피처 추가 시도 (019~021 전부 하락)
- Logistic Regression — 이 데이터에선 효과 없음 (0.91030)

### 유지해야 할 것
- LGBM + XGB + CatBoost 3모델 구조
- OOF weighted blend 방식

### 다음 가설
- CatBoost multi-seed (025) sub_06에 편입
- val보다 public gap을 줄이는 방향 — 정규화 강화 또는 simpler 모델
- 제출 횟수 아껴서 확실한 것만 제출
