# Submissions — playground-series-s6e4

| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-04-02 | trial_001_lgbm_baseline | 0.9844 (acc) | 0.9589 | - | -0.0255 | baseline |
| 02 | 2026-04-02 | trial_002_fe_catboost | 0.9853 (acc) | 0.9609 | - | -0.0244 | - |
| 03 | 2026-04-04 | trial_003_balanced_blend | 0.9711 (bal_acc) | 0.9691 | - | -0.0020 | - |
| 03-nothresh | 2026-04-04 | trial_003 (no threshold) | 0.9678 (bal_acc) | 0.9652 | - | -0.0026 | threshold 비교용 |
| 06 | 2026-04-04 | trial_006_full_pairwise_ensemble | 0.9699 (bal_acc) | 0.9668 | - | -0.0031 | - |
| 04 | 2026-04-05 | trial_008b_multiclass_te_fullpair | 0.9738 (bal_acc) | 0.9721 | - | -0.0017 | ✅ best public |

**Gap** = Public - Val. 양수면 val이 보수적, 음수면 overfitting 의심.
