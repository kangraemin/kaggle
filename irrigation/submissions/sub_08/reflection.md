# Sub 08 Reflection — trial_008_sklearn_multiclass_te

## 결과
- Val: **0.9712** (bal_acc) / Public: **0.9692**
- best(sub_04, 0.9721) 대비 -0.0029 낮음
- Gap: -0.0020 (sub_04 -0.0017과 유사)

## 핵심 변경
- sklearn TargetEncoder(multiclass, cv=5) 적용 — 265개 source cols
- full original merge (append 대신 TE source로만 사용)
- non-linear FE + binary threshold를 pairwise에 포함 (253 pairs)
- XGB single model (5k rounds) + bias tuning
- bias: Low -1.23, Medium -0.78, High 0.0

## 평가
- trial_008b(0.9738)와 동일 방향이지만 pairwise에 TE를 과하게 적용
- 265 TE cols가 noise를 추가 — 008b는 cat_cols 8개에만 TE를 한정해서 더 좋았음
- XGB raw bal_acc 0.969 → bias tuning으로 +0.002 끌어올림
- sub_04(008b)의 "pairwise는 factorize, TE는 원본 cat에만" 교훈 재확인
