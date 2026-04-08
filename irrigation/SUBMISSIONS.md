# Submissions — playground-series-s6e4

| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-04-02 | trial_001_lgbm_baseline | 0.9844 (acc) | 0.9589 | - | -0.0255 | baseline |
| 02 | 2026-04-02 | trial_002_fe_catboost | 0.9853 (acc) | 0.9609 | - | -0.0244 | - |
| 03 | 2026-04-04 | trial_003_balanced_blend | 0.9711 (bal_acc) | 0.9691 | - | -0.0020 | - |
| 03-nothresh | 2026-04-04 | trial_003 (no threshold) | 0.9678 (bal_acc) | 0.9652 | - | -0.0026 | threshold 비교용 |
| 06 | 2026-04-04 | trial_006_full_pairwise_ensemble | 0.9699 (bal_acc) | 0.9668 | - | -0.0031 | - |
| 08 | 2026-04-04 | trial_008_sklearn_multiclass_te | 0.9712 (bal_acc) | 0.9692 | - | -0.0020 | - |
| 08b | 2026-04-05 | trial_008b_multiclass_te_fullpair | 0.9738 (bal_acc) | 0.9721 | - | -0.0017 | - |
| 09-vote | 2026-04-04 | vote3 (008b+008+003 majority) | - | 0.9712 | - | - | voting 실험 |
| 09-nina | 2026-04-04 | nina blend | - | 0.9712 | - | - | blending 실험 |
| 10 | 2026-04-05 | trial_010_multiseed_xgb | 0.9741 (bal_acc) | 0.9720 | - | -0.0021 | - |
| 10b | 2026-04-06 | trial_010_multiseed_cat | 0.9713 (bal_acc) | 0.9687 | - | -0.0026 | - |
| 11 | 2026-04-06 | trial_011_slow_xgb_deeper_trees | 0.9794 (bal_acc) | 0.97799 | - | -0.0014 | - |
| 13 | 2026-04-09 | trial_013_multiseed_lgbm_orig_append | 0.9796 (bal_acc) | 0.97833 | - | -0.0013 | ✅ best public |

**Gap** = Public - Val. 양수면 val이 보수적, 음수면 overfitting 의심.

## 분석
- trial_013: gap -0.0013으로 역대 가장 안정적
- multi-seed averaging이 val(+0.0002)보다 public(+0.00034)에서 더 효과적 — 분산 감소가 unseen data에서 발현
- top LB 0.9808과 gap: 0.0025
- 최종 제출 후보: trial_013 (val+public best)
