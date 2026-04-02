# Sub 01 Reflection — trial_001_lgbm_baseline

## 결과
- Val: 0.9844 / Public: 0.9589
- Gap: -0.0255 (val 대비 public 낮음)

## 잘된 점
- Fold std 0.0002로 매우 안정적인 CV
- 단순 label encoding + LightGBM으로 98.4% 달성 — 데이터 자체가 깨끗함

## 문제점
- Val-Public gap이 -0.025로 꽤 큼
- feature engineering 없이 raw feature만 사용

## 배운 점
- 3-class 분류에서 baseline이 이미 높아서 개선 여지가 작음
- gap이 크다는 건 train/test 분포 차이가 있을 수 있음 — FE로 일반화 필요

## 다음 방향
- domain FE 추가 (ET_proxy, water_balance 등)
- multi-model ensemble로 diversity 확보
