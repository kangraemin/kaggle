# Sub 07 Reflection — trial_007_bias_tuned_stacking

## 결과
- Val: **0.9707** (bal_acc) / Public: 미제출
- trial_006(0.9699) 대비 +0.0008

## 핵심 변경
- Original data append (weight=0.35) — trial_006은 TE only
- Ridge meta-learner on 9 OOF prob cols (LGBM+XGB+CAT × 3 class)
- Bias tuning: argmax(log(p) + bias) coordinate descent
- CatBoost 우세 (ensemble weights 0:2:7)

## 배운 것
- Bias tuning이 +0.002 정도 효과 있음 — log-prob 공간에서 class boundary 조정
- Ridge meta > simple weight grid search (0.9700 vs 0.9675)
- Original data append가 TE only보다 유효 — 실제 데이터 분포 주입 효과
- CatBoost가 이 세팅에서 XGB를 압도 (depth=8, balanced weights)
