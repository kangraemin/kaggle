# Trials Log — ts-forecasting

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.1163 | 0.1499 | LightGBM, raw features only | ✅ submitted |
| 002 | lgbm_lags | 0.8432 | - | + lag 1~20, rolling mean/std | ✅ done |
| 003 | lgbm_more_lags | 0.8422 | - | + lag 50, 100, ewm | ✅ done |
| 004 | lgbm_cross_horizon | - | - | + other horizons' y_target at same ts_index | ⏳ pending |
| 005 | lgbm_ewm | 0.8433 | - | + exponential weighted mean (span 5,20,50) | ✅ done |
| 006 | lgbm_per_horizon | - | - | separate model per horizon | ⏳ pending |
| 007 | lgbm_all_features | 0.8912 | - | all of above combined | ✅ done |

## Notes
- Val split: ts_index <= 2880 (train), > 2880 (val)
- Metric: weighted_rmse_score (higher = better, max 1.0)
- lag_1 feature importance >> all others → series is AR(1)-like
- trial_007: cross_horizon lag features가 top importance 독점 → 핵심 신호
- High-weight series (83EG83KQ) have y_target ≈ 0 → small errors matter
