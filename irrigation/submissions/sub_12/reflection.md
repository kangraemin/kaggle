# Sub 12 Reflection — trial_012_extend_rounds

## 결과
- Val: 0.9793 (bal_acc) / Public: 미제출
- trial_011(0.9794) 대비 -0.0001 하락

## 핵심 변경점 (trial_011 대비)
- n_estimators: 4000 → 15000
- early_stopping_rounds: 200 추가
- 실제 avg rounds: 6406 (early stop에 의해)

## 교훈

### mlogloss early stop ≠ balanced_accuracy optimal
- mlogloss는 6406 round에서 최적이지만, bal_acc는 그보다 일찍 최적
- 4000 hard cap(trial_011)이 6406 early stop(trial_012)보다 val이 높음
- **proxy metric(mlogloss)으로 early stopping하면 target metric(bal_acc)과 어긋남**

### 더 많은 round가 항상 좋은 건 아님
- 4000 → 6406으로 60% 더 학습했지만 오히려 -0.0001
- overfitting이 아니라 mlogloss 최적화가 bal_acc 최적화와 방향이 다른 것

## 버려야 할 것
- mlogloss early stopping — bal_acc custom eval이 필요하거나 hard cap이 나음

## 유지해야 할 것
- trial_011의 4000 rounds hard cap이 현재 최적
