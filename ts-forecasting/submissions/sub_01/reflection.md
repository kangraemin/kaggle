# Submission 01 Reflection

## 결과
- Val score: 0.1163
- Public score: 0.1499
- Val-Public gap: +0.0336 (public이 val보다 높음)

## Gap 원인 분석

**Val < Public (gap +0.034)**
- Val split이 학습 분포와 비슷하지 않을 가능성. ts_index > 2880 기준으로 나눴는데, public test set의 시계열 특성이 val보다 오히려 단순할 수 있음.
- 어차피 0.11~0.15 구간은 거의 랜덤 수준 — gap 분석이 무의미할 정도로 점수 자체가 낮음.

**왜 이렇게 낮은가**
- raw features(horizon, sub_code, feature_a 등)만으로는 시계열의 자기상관 구조를 전혀 못 잡음.
- weighted_rmse_score는 고가중 시리즈(83EG83KQ 등 y_target ≈ 0)에 민감한데, lag 없이는 이 시리즈 예측이 불가능.
- 사실상 시계열 예측 문제에서 lag feature 없는 모델은 baseline조차 아님.

## 버려야 할 것
- lag 없는 모델은 어떤 조합으로도 경쟁력 없음. 다시 시도할 이유 없음.

## 유지해야 할 것
- LightGBM 자체 구조는 유효. feature만 문제.
- val split 방식(ts_index 기준)은 일단 유지. 단, val-public gap 추이 계속 모니터링 필요.

## 다음 가설
- sub_02에서 trial_007(val 0.891, cross_horizon lag features)을 제출.
- cross_horizon feature가 top importance를 독점했으므로 public에서도 유효한지 검증.
- val-public gap이 이번처럼 양수면 val split이 너무 쉬운 것일 수 있음 → 필요시 val 전략 재검토.
