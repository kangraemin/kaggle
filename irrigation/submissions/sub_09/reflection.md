# Sub 09 Reflection — trial_009_stat_group_features

## 결과
- Val: **0.9710** (bal_acc) / Public: 미제출
- trial_008(0.9712) 대비 -0.0002

## 핵심 변경
- Statistical group features: 88 pairs × 4 stats (mean/std/min/max) = 352 cols
- Original TE on CATS + NUMS (57 cols, smoothing alpha=10)
- XGB 15k iterations / early stopping 500
- Bias tuning

## 배운 것
- Statistical group features가 이 데이터셋에서는 효과 없음
- 352개 통계 feature 추가가 오히려 noise — feature 수 증가 ≠ 성능 향상
- trial_008b의 "pairwise factorize + cat TE만" 접근이 여전히 최적
- 과도한 feature engineering은 tabular 데이터에서 역효과
