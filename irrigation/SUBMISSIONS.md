# Submissions — playground-series-s6e4

| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-04-02 | trial_001_lgbm_baseline | 0.9844 (acc) | 0.9589 | - | -0.0255 | baseline |
| 02 | 2026-04-02 | trial_002_fe_catboost | 0.9853 (acc) | 0.9609 | - | -0.0244 | - |
| 03 | 2026-04-04 | trial_003_balanced_blend | 0.9711 (bal_acc) | 0.9691 | - | -0.0020 | - |
| 03-nothresh | 2026-04-04 | trial_003 (no threshold) | 0.9678 (bal_acc) | 0.9652 | - | -0.0026 | threshold 비교용 |
| 06 | 2026-04-04 | trial_006_full_pairwise_ensemble | 0.9699 (bal_acc) | 0.9668 | - | -0.0031 | - |
| 08 | 2026-04-04 | trial_008_sklearn_multiclass_te | 0.9712 (bal_acc) | 0.9692 | - | -0.0020 | - |
| 08b | 2026-04-05 | trial_008b_multiclass_te_fullpair | 0.9738 (bal_acc) | 0.9721 | - | -0.0017 | ✅ best public |
| 09-vote | 2026-04-04 | vote3 (008b+008+003 majority) | - | 0.9712 | - | - | voting 실험 |
| 09-nina | 2026-04-04 | nina blend | - | 0.9712 | - | - | blending 실험 |
| 10 | 2026-04-05 | trial_010_multiseed_xgb | 0.9741 (bal_acc) | 0.9720 | - | -0.0021 | val best, public 미갱신 |
| 10b | 2026-04-06 | trial_010_multiseed_cat | 0.9713 (bal_acc) | 0.9687 | - | -0.0026 | CAT only, threshold 약해서 하락 |
| 11 | 2026-04-06 | trial_011_slow_xgb_deeper_trees | 0.9794 (bal_acc) | 0.97799 | - | -0.0014 | ✅ best public — slow XGB lr=0.01 + sklearn 171 pairwise TE |

**Gap** = Public - Val. 양수면 val이 보수적, 음수면 overfitting 의심.

## 분석
- trial_008b가 val-public gap -0.0017로 가장 안정적
- trial_010은 val 최고(0.9741)지만 public 0.9720으로 008b(0.9721)보다 미세 하락
- threshold 과적합 가능성: 010 High×4.6 vs 008b High×3.7
- **trial_011: val 0.9794, public 0.97799 — gap -0.0014로 가장 안정적, 새 best**
- 최종 제출 후보: trial_011 (val+public best)
