# Trials Log — ts-forecasting

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.1163 | 0.1499 | LightGBM, raw features only | ✅ submitted |
| 002 | lgbm_lags | 0.8432 | - | + lag 1~20, rolling mean/std | ✅ done |
| 003 | lgbm_more_lags | 0.8422 | - | + lag 50, 100, ewm | ✅ done |
| 004 | lgbm_cross_horizon | 0.8913 | - | + other horizons' y_target at same ts_index | ✅ done |
| 005 | lgbm_ewm | 0.8433 | - | + exponential weighted mean (span 5,20,50) | ✅ done |
| 006 | lgbm_per_horizon | 0.8441 | - | separate model per horizon | ✅ done |
| 007 | lgbm_all_features | 0.8912 | - | all of above combined | ✅ done |
| 008 | cross_h_lags | - | - | cross_horizon lag1/2/3 extended | ⏭ skip |
| 009 | hparam_tuning | 0.8913 | - | num_leaves=127, lr=0.03, regularization | ✅ done |
| 010 | target_enc | 0.8923 | - | + series mean/std/median (train only) | ✅ done |
| 011 | clean_cross_h | - | - | cross_horizon + 간소화 features | ⏭ skip |

## Notes
- Val split: ts_index <= 2880 (train), > 2880 (val)
- Metric: weighted_rmse_score (higher = better, max 1.0)
- lag_1 feature importance >> all others → series is AR(1)-like
- trial_007: cross_horizon lag features가 top importance 독점 → 핵심 신호
- cross_horizon val score inflated 주의: val rows는 y_target 있어서 작동하지만 test rows는 NaN → 실제 public score는 ~0.84 예상
- trial_010: series_mean/std (target encoding)은 test에서도 유효 → 추가 signal
- High-weight series (83EG83KQ) have y_target ≈ 0 → small errors matter
