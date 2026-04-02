# Sub 02 Reflection — trial_002_fe_catboost

## 결과
- Val: 0.9853 / Public: 0.9609
- Gap: -0.0244 (sub_01 대비 gap 소폭 개선)
- Val +0.0009, Public +0.0020 향상

## 잘된 점
- domain FE(ET_proxy, water_balance, interactions)가 public에서 더 큰 폭으로 개선 → 일반화에 기여
- XGB가 개별 모델 중 최고 (0.9851) — engineered feature 활용도가 높음
- 앙상블 weight grid search로 최적 비율(4:5:1) 도출

## 문제점
- CatBoost가 0.9824로 가장 낮음 — cat_features 활용이 오히려 방해?
- 여전히 gap -0.024 존재
- FE 11개 추가했지만 val 개선 0.09%p로 미미

## 배운 점
- 이 데이터셋은 baseline이 이미 높아서 FE 단독으로는 큰 개선 어려움
- public 개선폭(+0.002)이 val(+0.0009)보다 큼 → FE가 일반화에 더 효과적
- CatBoost의 auto_class_weights=Balanced가 이 데이터에는 안 맞을 수 있음

## 다음 방향
- target encoding / frequency encoding 시도
- CatBoost 튜닝 또는 제외하고 LGBM+XGB 2-model 앙상블
- original data(있으면) blending
- pseudo labeling (confidence 높은 test 샘플 활용)
