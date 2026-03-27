# Trials — playground-series-s6e3

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.91613 | 0.91377 | raw features + LabelEncoding + 5-Fold | ✅ submitted |
| 002 | feature_eng | 0.91621 | - | target encoding + ChargeGap/AvgMonthlyCharge/ChargeRatio | ✅ done |
| 003 | catboost | 0.91624 | - | CatBoost native categorical + charge features | ✅ done |
| 004 | lgbm_tuned | 0.91663 | - | Optuna 50 trials + target encoding + charge features | ✅ done |
| 005 | ensemble | ~0.91623 | - | trial_002 + trial_003 단순 평균 | ✅ done |
| 006 | advanced_features | 0.91636 | - | num_services/tenure_bin/fiber_monthly + target encoding | ✅ done |
| 007 | xgboost | 0.91659 | - | XGBoost + advanced features + target encoding | ✅ done |
| 008 | stacking | 0.91668 | - | 006×0.35 + 007×0.65 weighted blend | ✅ done |

## 메트릭
- Task: classification
- Metric: AUC-ROC
- Direction: higher is better
