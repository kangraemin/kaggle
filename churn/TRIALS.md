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
| 009 | low_lr | 0.91647 | - | lr=0.01, 5000 iterations | ❌ worse |
| 010 | external_data | 0.91628 | - | 원본 Telco 7천행 추가 | ❌ worse |
| 011 | no_internet_fix | 0.91626 | - | No internet service→No 통합 + Contract_ord | ❌ worse |
| 012 | catboost_oof | 0.91628 | - | CatBoost + No internet→No + OOF 저장 | ❌ worse |
| 013 | 10fold | 0.91636 | - | 10-Fold + No internet→No + Contract_ord | ❌ worse |
| 014 | mega_blend | 0.91677 | 0.91404 | 5모델 OOF grid search blend (XGB×0.4) | ✅ submitted |
| 015 | xgb_tuned | 0.91669 | - | XGBoost Optuna 30 trials | ✅ done |
| 016 | meta_stacking | 0.91674 | - | LR meta-learner on 4모델 OOF | ✅ done |
| 017 | final_blend | 0.91682 | 0.91404 | 5모델 blend (xgb_opt×0.4 주도) | ✅ submitted |
| 018 | groupby_features | 0.91621 | - | groupby 집계 83 피처 + tenure flags | ❌ worse |
| 019 | orig_as_cols | 0.91633 | - | 원본 데이터 groupby mean → 컬럼 merge (Chris Deotte 방식) | ❌ worse |
| 020 | logistic_regression | 0.91030 | - | Logistic Regression + target encoding | ❌ worse |
| 021 | relative_features | 0.91608 | - | 그룹 내 상대적 위치 피처 + service_combo | ❌ worse |
| 022 | multi_seed_lgbm | 0.91670 | - | LGBM 7seeds × 5fold = 35 models | ✅ done |
| 023 | final_blend | 0.91686 | - | XGB_opt + XGB + CB + LGBM_multi blend | ✅ done |
| 024 | xgb_multiseed | 0.91690 | 0.91395 | XGB 7seeds × 5fold = 35 models | ✅ submitted |

## 메트릭
- Task: classification
- Metric: AUC-ROC
- Direction: higher is better
