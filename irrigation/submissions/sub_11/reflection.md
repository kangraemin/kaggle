# Sub 11 Reflection — trial_011_slow_xgb_deeper_trees

## 결과
- Val: **0.9794** (bal_acc) / Public: **0.97799**
- 이전 best (008b): Public 0.9721 → **+0.0059 개선**
- Gap: -0.0014 (전체 trial 중 가장 안정적)
- 1등(0.9803)과 gap: 0.0023

## 핵심 변경점 (008b 대비)
1. **sklearn TargetEncoder(multiclass) on 171 pairwise** → 513 TE cols 추가 (총 750 features)
2. **lr=0.01, 4000 rounds hard cap** (early stopping 없음)
3. **threshold: Low×0.8, Med×0.7, High×4.6** (더 공격적)

## 교훈

### sklearn TE가 manual TE보다 훨씬 나음
- 이전 세션에서 manual multiclass TE로 537 cols 넣었을 때 mlogloss 0.82로 완전 실패
- sklearn TargetEncoder는 내부 CV + smoothing이 제대로 작동 → 같은 513 cols인데 성공
- **manual TE의 버그가 문제였지, 접근 자체가 틀린 게 아니었음**

### hard cap이 early stopping보다 나음
- trial_012에서 early_stopping=200 넣었더니 avg 6406 rounds에서 멈춤
- 하지만 val 0.9793 < 0.9794 (trial_011) — mlogloss 기준 early stop이 bal_acc 최적이 아님
- **bal_acc 최적 round는 mlogloss 최적보다 더 일찍** → hard cap이 차라리 나음

### threshold sweet spot
- High×4.6은 trial_010(High×4.6)에서도 썼지만 그때는 base 모델이 약해서 효과 없었음
- base 모델이 0.976(acc)으로 충분히 강해야 threshold가 먹힘

## 버려야 할 것
- manual multiclass TE — sklearn TargetEncoder로 대체
- mlogloss early stopping — bal_acc와 불일치

## 유지해야 할 것
- sklearn TE(multiclass) on 171 pairwise
- lr=0.01 slow learning + hard cap rounds
- threshold Low×0.8, Med×0.7, High×4.6

## 다음 가설
- multi-seed (3~5 seeds) + trial_011 아키텍처
- LGBM 추가 앙상블 (proba-level)
- original data append + log-odds bias tuning
