# Sub 05 Reflection

## Submission 09 — Voting Ensemble
- **Public**: 0.9712 (vote3), 0.9712 (nina)
- **결과**: 008b(0.9721)보다 하락

### 교훈
- submission 간 agreement가 99.4~99.6%라 voting 효과 거의 없음
- 바뀐 0.4~0.6%가 오히려 맞던 걸 틀리게 만듦
- **버려야 할 것**: 유사한 모델끼리의 voting — 다양성 부족하면 의미 없음
- **유지해야 할 것**: 없음

## Submission 10 — Multi-seed XGB (다른 세션)
- **Public**: 0.9720
- **결과**: 008b(0.9721)와 거의 동일 (-0.0001)

### 교훈
- 5-seed 평균이 안정성은 올리지만 public 개선은 미미
- threshold(High×4.6)이 008b(High×3.7)보다 더 공격적인데도 public 미갱신
- val best(0.9741)가 public best는 아님 → threshold 과적합 가능성

## Submission 10b — Multi-seed CAT (이 세션)
- **Public**: 0.9687
- **결과**: 008b 대비 -0.0034 하락

### 교훈
- CatBoost 3-seed avg 0.9702 > XGB 0.9689 (val 기준)
- 하지만 threshold(High×2.2)가 약해서 public에서 효과 부족
- 008b의 High×3.7이 최적에 가까움 — threshold는 3.5~4.0이 sweet spot

### 다음 가설
- CatBoost proba에 008b 수준 threshold(High×3.7) 적용하면 개선될 수 있음
- XGB+CAT proba-level 앙상블 (label voting 아닌 확률 평균)
- Binary threshold features를 pairwise에 포함 (C(23,2)=253 pairs)
- LR stacking meta-learner
