## Submission 02 Reflection

### 결과
- Val score: 0.91663
- Public score: 0.91393
- Val-Public gap: -0.00270

### Gap 원인 분석
- sub_01(-0.00236)보다 gap 소폭 확대 → Optuna 튜닝이 val에 살짝 더 과적합됐을 가능성
- target encoding이 CV 내부에서만 fit되지만, 파라미터 탐색 자체가 val 기준으로 최적화되어 test 일반화가 덜 됨
- charge interaction feature(ChargeGap 등)는 val에서 효과적이었으나 test 분포 차이 가능성

### 버려야 할 것
- Optuna를 val AUC 기준으로만 튜닝하는 방식 → public gap 고려한 정규화 강화 필요

### 유지해야 할 것
- charge interaction features (ChargeGap, AvgMonthlyCharge, ChargeRatio) — top feature로 활약
- target encoding (CV-safe)
- Stratified 5-Fold

### 다음 가설
- 서비스 개수(num_services), tenure 구간, fiber+monthly 조합 등 도메인 피처 추가
- XGBoost, CatBoost 다양한 모델 앙상블로 분산 감소
- OOF 기반 weighted blend로 최적 앙상블 가중치 탐색
