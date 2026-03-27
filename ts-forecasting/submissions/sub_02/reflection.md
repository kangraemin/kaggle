# Submission 02 Reflection

## 결과
- Best trial: trial_010 (target encoding)
- Val score: 0.8923
- Public score: 0.0000
- Val-Public gap: 완전 붕괴

## 원인 분석

**왜 0.0000이 나왔나**
- lag_1 = shift(1) of y_target → test rows의 99.9%에서 NaN
- train: ts_index 1~3601 / test: ts_index 3602~4376 (시리즈당 평균 131개 rows)
- test 첫 번째 row만 lag_1 유효 (train 마지막값), 나머지 774개는 NaN
- LightGBM이 NaN branch로 예측 → 거의 0 수렴 → weighted_rmse_score = 0

**왜 val score는 0.89나 나왔나**
- val = train 내부 split (ts_index > 2880) → y_target 있음 → lag 정상 동작
- val이 test 상황을 전혀 반영 못 함 → 완전히 뻥튀기된 score

**sub_02 trial 전체가 쓰레기인 이유**
- trial_002~010 전부 lag/rolling/cross_horizon 기반 → 동일한 문제
- val score 0.84~0.89 전부 신뢰 불가

## 버려야 할 것
- lag/rolling/ewm/cross_horizon feature를 그냥 쓰는 모델
- 현재 val split 방식 (test 상황을 재현 못 함)

## 유지해야 할 것
- LightGBM 구조 자체는 유효
- target encoding (series_mean/std/median) — train 통계 기반이라 test에서도 유효
- raw feature_* — 항상 유효

## 다음 가설 (sub_03)
- trial_012: raw feature + target encoding만 → test에서 확실히 동작, baseline 넘는지 확인
- trial_013: recursive prediction → lag feature 유지하되 step-by-step 예측으로 NaN 해결
- val도 recursive 방식으로 검증해야 진짜 score 확인 가능

## 핵심 교훈
val split을 test처럼 구성해야 함. val rows y_target을 NaN으로 만든 뒤 feature engineering → test 상황 재현.
