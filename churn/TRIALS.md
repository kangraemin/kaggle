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
| 027 | blend | 0.91692 | - | 4모델 blend (XGB_multi×0.7 주도) | ✅ done |
| 028 | lgbm_reg_multiseed | 0.91683 | 0.91400 | lambda=2.0 + 7seeds | ✅ submitted |
| 029 | smoothed_te | 0.91630 | - | Bayesian smoothed TE | ❌ worse |
| 030 | mega_blend | 0.91694 | - | 6모델 blend (XGB_multi×0.6 + LGBM_reg×0.3) | ✅ done |
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
| 043 | mlp_blend | 0.91695 | - | 8모델 scipy optimize blend (MLP 포함, 기여도 미미) | ✅ done |
| 044 | histgbm | 0.91606 | - | HistGradientBoosting + 7seeds | ❌ worse |
| 045 | feat_select | 0.91515 | - | top 15 features only | ❌ worse |
| 046 | final_blend | 0.91694 | - | 5모델 scipy optimize blend (XGB×0.62 + LGBM5×0.35) | ✅ done |
| 047 | realmlp | 0.91247 | - | RealMLP (PyTabKit) | ❌ worse |
| 048 | dart | 0.91482 | - | LGBM DART + 7seeds | ❌ worse |
| 049 | lr_onehot | 0.91078 | - | LR + one-hot encoding | ❌ worse |
| 050 | rank_blend | 0.91694 | - | 5모델 rank averaging blend (XGB×0.6 + LGBM5×0.4) | ✅ done |
| 051 | xgb_wide_optuna | 0.91693 | - | XGB Optuna 100 trials 넓은 범위 | ✅ done |
| 052 | all_rank_blend | 0.91694 | - | 9모델 rank averaging blend | ✅ done |
| 053 | geo_blend | 0.91694 | - | 4모델 geometric mean blend (XGB×0.6 + LGBM5×0.4) | ✅ done |
| 054 | oof_stacking | 0.91188 | - | LGBM stacking: original features + OOF predictions | ❌ worse |
| 055 | woe | 0.91685 | - | WoE encoding + LGBM multi-seed | ✅ done |
| 056 | repeated_kfold | 0.91682 | - | RepeatedStratifiedKFold 3×5=15 folds | ✅ done |
| 057 | calibrated | 0.91693 | - | Platt scaling on best blend | ✅ done |
| 058 | xgb_depth1 | 0.91375 | - | XGB max_depth=1 + 7seeds | ❌ worse |
| 059 | lgbm_depth1 | 0.91373 | - | LGBM num_leaves=2 + 7seeds | ❌ worse |
| 060 | realmlp_kaggle | 0.91938 | 0.91683 | RealMLP 20-fold (Kaggle CPU fork) | ✅ submitted |
| 061 | realmlp_xgb_blend | 0.91945 | 0.91686 | RealMLP×0.85 + XGB×0.15 | ✅ submitted |
| 062 | interaction_te | 0.91692 | - | XGB + 12 interaction pair TE + 7seeds | ❌ worse |
| 063 | freq_encoding | 0.91688 | - | XGB + frequency encoding + TE + 7seeds | ❌ worse |
| 064 | adversarial_val | 0.91594 | - | Adversarial validation으로 top 4 피처 제거 + XGB 7seeds | ❌ worse |
| 065 | ridge_xgb_ngram | 0.91658 | - | Ridge→XGB two-stage + N-gram TE + 7seeds | ❌ worse |
| 066 | all_combos_pseudo | 0.91710 | - | All 2-way cat combos (120) + pseudo labels (0.999) + orig stats + freq enc + 7seeds XGB | ✅ done |
| 067 | distribution_digit | 0.91850 | 0.91571 | Distribution features (pctrank/zscore vs churner) + quantile distance + digit features + ORIG stats + XGB 7seeds | ⚠️ overfit |
| 068 | dist_combos_20fold | 0.91846 | - | Distribution + All 2-way combos (120) + pseudo labels + digit + ORIG stats + freq enc + 20-fold CV + XGB 7seeds | ❌ worse |
| 069 | knn_stacking | 0.91840 | - | KNN prob (K=20,50,100,200 OOF) + distribution + digit + ORIG stats + XGB 7seeds | ❌ worse |
| 070 | ngram_direct | 0.91843 | - | Bi-gram(15) + Tri-gram(20) TE + distribution + digit + ORIG stats + XGB 7seeds | ❌ worse |
| 071 | optuna_deep | 0.91855 | - | Optuna 200 trials XGB + distribution/digit/ORIG + 7seeds | ⚠️ distribution 포함 → overfit 위험 |
| 073 | optuna_clean | 0.91844 | - | Optuna 30 trials XGB clean features (no distribution) + 7seeds | ✅ done |
| 074 | te_std_enriched | 0.91762 | - | All 2-way combos (120) + TE mean/std + digit + ORIG stats + freq enc + 7seeds XGB (no dist) | ✅ done |
| 075 | hill_climbing | 0.91964 | - | 53모델 OOF Hill Climbing rank blend → RealMLP×0.67 + KNN×0.33 | ⚠️ KNN(069)이 distribution feature 사용 → overfit 위험, 미제출 |
| 078 | realmlp_te_std_blend | 0.91950 | 0.91690 | RealMLP(060)×0.80 + XGB_TE_std(074)×0.20 rank blend | ✅ submitted |
| 079 | hill_climbing_v2 | 0.91958 | 0.91693 | 52모델 Hill Climbing (dist 제외) → RealMLP_blend×0.4 + RealMLP×0.4 + Optuna_clean×0.2 | ✅ submitted |
| 080 | ridge_ensemble_v2 | 0.91961 | 0.91702 | 51 OOFs Ridge (alpha=100, dist 제외, +trial_074) | ✅ submitted |
| 081 | te_std_optuna_params | 0.91771 | - | trial_074 features + Optuna params (lr=0.012, colsample=0.38, gamma=0.61) + 10k rounds + 7seeds | ✅ done |
| 082 | realmlp_xgb081_blend | 0.91951 | 0.91689 | RealMLP(060)×0.79 + XGB081×0.21 rank blend | ✅ submitted |
| 083 | ridge_ensemble_v3 | 0.91962 | 0.91701 | 53 OOFs Ridge (alpha=100, dist 제외, +trial_074/081) | ✅ submitted |
| 084 | ridge_ensemble_v4 | 0.91964 | 0.91704 | 56 OOFs Ridge (alpha=50, ALL models incl dist, +trial_074/081) | ✅ submitted |

## 메트릭
- Task: classification
- Metric: AUC-ROC
- Direction: higher is better
