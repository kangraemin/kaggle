# Trials — playground-series-s6e4

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.9844 (acc) | 0.9589 | LightGBM default, label encoding, 5-fold CV | done |
| 002 | fe_catboost | 0.9853 (acc) | 0.9609 | FE(ET_proxy,water_balance,interactions) + LGBM+XGB+CAT ensemble(4:5:1) | done |
| 003 | balanced_blend | **0.9711 (bal_acc)** | **0.9691** | 메트릭 수정(bal_acc) + class_weight balanced + orig 10K blend + pairwise interaction TE + threshold opt(High×2.6) | ✅ best |
| 004 | target_enc_catpairs | 0.9852 (acc) | - | 28 pairwise cat combos + target encoding(72 TE cols) + freq encoding + LGBM+XGB+CAT(5:5:1). 잘못된 메트릭 | done |
| 005 | ext_data_balanced_acc | 0.9692 (bal_acc) | - | XGB + original data TE + append(w=0.1) + Deotte formula features + balanced weights + pairwise TE | done |
| 006 | full_pairwise_ensemble | 0.9699 (bal_acc) | 0.9668 | Full pairwise(171 NUMS_bin+CATS) + multiclass TE + XGB/LGBM/CAT ensemble(XGB only best) + orig TE + freq enc + Ektarr FE | done |
| 007 | bias_tuned_stacking | 0.9707 (bal_acc) | - | Orig append(w=0.35) + Ridge meta-learner + bias tuning(log-prob) + XGB+CAT(0:2:7) + depth=8 | done |
| 008 | sklearn_multiclass_te | 0.9712 (bal_acc) | 0.9692 | sklearn TE(multiclass,cv=5) on 265 cols + full orig merge + non-linear FE + binary in pairwise + XGB single(5k) + bias tuning | done |
| 008b | multiclass_te_fullpair | **0.9738 (bal_acc)** | **0.9721** | 171 pairwise factorize + cat_cols multiclass TE(24) + Deotte binary + orig TE only(no append) + XGB(1:0) + threshold(High×3.7) | ✅ best public |
| 009 | stat_group_features | 0.9710 (bal_acc) | - | Stat group feat(352 cols, 88 pairs x 4 stats, alpha=10) + orig TE CATS+NUMS(57 cols) + XGB 15k/500 + bias tuning | done |
| 010 | multiseed_xgb | **0.9741 (bal_acc)** | 0.9720 | Multi-seed XGB (5 seeds x 5 folds = 25 models) on trial_008b arch + threshold(High×4.6) | best val |
| 010b | multiseed_cat | 0.9713 (bal_acc) | 0.9687 | 3-seed XGB+CAT, CAT only(0.9702) best, threshold(High×2.2) 약함 | done |
| 011 | slow_xgb_deeper_trees | **0.9794 (bal_acc)** | - | lr=0.01, 4000 rounds (hard cap) + sklearn TE(multiclass) on 171 pairwise(513) + manual cat TE(24) = 750 features, threshold(Low×0.8,Med×0.7,High×4.6) | ✅ best val |

## 메트릭
- Task: classification (3-class: Low / Medium / High)
- Metric: balanced_accuracy_score (대회 공식 메트릭)
- Direction: higher_is_better
