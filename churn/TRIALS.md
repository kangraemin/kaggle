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
| 025 | cb_multiseed | 0.91642 | - | CatBoost 7seeds × 5fold | ✅ done |
| 026 | high_risk_features | 0.91654 | - | EDA 기반 고위험 조합 + 정규화 강화 | ✅ done |
| 028 | lgbm_reg_multiseed | 0.91683 | 0.91400 | lambda=2.0 + 7seeds | ✅ submitted |
| 029 | smoothed_te | 0.91630 | - | Bayesian smoothed TE | ❌ worse |
| 031 | xgb_smooth_multi | 0.91690 | - | XGB + smoothed TE + high risk + 7seeds | ✅ done |
| 032 | lgbm_reg_smooth | 0.91683 | - | LGBM reg + smoothed TE + 7seeds | ✅ done |
| 033 | catboost_optuna | 0.91632 | - | CatBoost Optuna 20 trials | ✅ done |
| 034 | pseudo_label | 0.91634 | - | Pseudo-labeling (48% test samples) | ❌ worse |
| 035 | extratrees | 0.91258 | - | ExtraTrees | ❌ worse |
| 036 | ultimate_blend | 0.91695 | - | 9모델 grid search blend | ✅ done |
| 037 | lgbm_reg5 | 0.91681 | - | lambda=5.0 + 7seeds | ✅ done |
| 038 | xgb_reg5 | 0.91673 | - | XGB reg=5.0 + 7seeds | ✅ done |
| 039 | cb_reg | 0.91609 | - | CatBoost l2=10 + 7seeds | ❌ worse |
| 040 | lgbm_reg10 | 0.91681 | - | lambda=10.0 + 7seeds | ✅ done |
| 041 | ridge | 0.90341 | - | RidgeClassifier | ❌ worse |
| 042 | mlp | 0.91327 | - | PyTorch MLP 3-layer MPS | ❌ worse |
| 044 | histgbm | 0.91606 | - | HistGradientBoosting + 7seeds | ❌ worse |
| 045 | feat_select | 0.91515 | - | top 15 features only | ❌ worse |
| 047 | realmlp | 0.91247 | - | RealMLP (PyTabKit) | ❌ worse |
| 048 | dart | 0.91482 | - | LGBM DART + 7seeds | ❌ worse |
| 049 | lr_onehot | 0.91078 | - | LR + one-hot encoding | ❌ worse |
| 051 | xgb_wide_optuna | 0.91693 | - | XGB Optuna 100 trials 넓은 범위 | ✅ done |
| 052 | all_rank_blend | 0.91694 | - | 9모델 rank averaging blend | ✅ done |
| 055 | woe | 0.91685 | - | WoE encoding + LGBM multi-seed | ✅ done |
| 056 | repeated_kfold | 0.91682 | - | RepeatedStratifiedKFold 3×5=15 folds | ✅ done |
| 057 | calibrated | 0.91693 | - | Platt scaling on best blend | ✅ done |
| 058 | xgb_depth1 | 0.91375 | - | XGB max_depth=1 + 7seeds | ❌ worse |
| 059 | lgbm_depth1 | 0.91373 | - | LGBM num_leaves=2 + 7seeds | ❌ worse |
| 060 | realmlp_kaggle | 0.91938 | 0.91683 | RealMLP 20-fold (Kaggle CPU fork) | ✅ submitted |
| 061 | realmlp_xgb_blend | 0.91945 | 0.91686 | RealMLP×0.85 + XGB×0.15 | ✅ submitted |

## 메트릭
- Task: classification
- Metric: AUC-ROC
- Direction: higher is better
