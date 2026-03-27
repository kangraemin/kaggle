## Submission 01 Reflection

### 결과
- Val score: 0.91613
- Public score: 0.91377
- Val-Public gap: -0.00236

### Gap 원인 분석
- Val이 public보다 0.0024 높음 → 약한 overfitting 또는 val/test 분포 미세 차이
- 5-Fold Stratified CV 자체는 잘 설계됐으나, public test set이 train과 분포가 완전히 동일하지 않을 가능성
- LabelEncoding은 범주형의 실제 관계를 반영 못함 → 모델이 val에서 암기한 패턴이 test에서 일반화 안 됐을 수 있음

### 버려야 할 것
- LabelEncoding (순서 없는 범주형에 부적절, target encoding으로 교체)

### 유지해야 할 것
- Stratified 5-Fold CV (val-public gap이 0.0024로 안정적)
- LightGBM (기본 세팅으로도 0.916, 충분히 강한 베이스)
- TotalCharges / MonthlyCharges / tenure 중심 접근

### 다음 가설
- `AvgMonthlyCharge = TotalCharges / (tenure + 1)` — tenure 대비 실제 납부액 비율이 churn 신호일 것
- `ChargeGap = MonthlyCharges - AvgMonthlyCharge` — 최근 요금이 평균보다 높으면 churn 가능성↑
- 범주형 target encoding — LabelEncoding 대비 churn rate 직접 반영으로 AUC 개선 기대
