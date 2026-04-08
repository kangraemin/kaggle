<div align="center">

# Kaggle Experiments

**Tracking every trial, every failure, every lesson across Kaggle competitions.**

[한국어](README.ko.md)

</div>

---

| Folder | Competition | Best Public | Status |
|--------|-------------|-------------|--------|
| `churn/` | Playground S6E3 — Customer Churn | 0.91707 (private 0.91815) | 84+ trials, 15 subs, ended |
| `irrigation/` | Playground S6E4 — Irrigation Need | 0.97833 | 14 trials, 13 subs, in progress |
| `birdclef/` | BirdCLEF+ 2026 — Bird Species | 0.929 | 23 trials, 12 subs, in progress |
| `ts-forecasting/` | Hedge Fund — Time Series | 0.1499 | 4 subs, 3 scored zero |
| `march-mania/` | March Mania 2026 — NCAA Basketball | not submitted | missed deadline |

---

## churn (Playground S6E3)

**TL;DR**: Predict telecom customer churn. AUC-ROC metric. 4,142 teams.
**Key challenge**: Top scores packed in 0.914–0.917. Broke through the GBDT ceiling by forking a RealMLP notebook on Kaggle.
**Final**: Best public **0.91707**, private **0.91815**. 15 submissions, 84+ trials.

### Experiment flow

| Trial | Why | Result | Next |
|-------|-----|--------|------|
| 001 LightGBM | Baseline | val 0.9161, **public 0.9138** | Target encoding + tuning |
| 002–004 FE+tuning | ChargeGap, Optuna 50 rounds | val 0.9166 → **0.9139** | Ensemble |
| 005–008 ensemble | LGBM + XGB + CatBoost blends | val 0.9167 → **0.9140** | Weight optimization |
| 009–013 failures | External data, 10-fold, feature merging | **All degraded** | Drop external data |
| 014–017 optimal blend | 5-model OOF grid search | val 0.9168 → **0.91404** — GBDT ceiling | Multi-seed, NN |
| 018–024 all-in | 83 groupby vars, multi-seed (7×5=35 models) | val 0.9169 → **stuck at 0.9140** | Need NN |
| 025–059 big search | MLP, RealMLP, Ridge, DART, pseudo-labeling... 35 trials | **All failed to break through** locally | Fork Kaggle notebook |
| 060 RealMLP fork | RealMLP 20-fold on Kaggle notebook | val 0.9194 → **public 0.91683** — **+0.003 breakthrough** | Blend with XGB |
| 061 RealMLP+XGB | RealMLP×0.85 + XGB×0.15 | **public 0.91686** | Ridge ensemble |
| 074 TE std enriched | TE mean+std + 120 combos + digit features | val **0.91762** — local GBDT best | Ridge input |
| 075–084 Ridge ensemble | 55 OOFs into Ridge(alpha=100). No filtering | val 0.9196 → **public 0.91707** → **private 0.91815** | Competition ended |

### Submissions

| sub | trial | public | what happened |
|-----|-------|--------|---------------|
| 01 | 001 baseline | 0.91377 | Raw features only |
| 03 | 014 ensemble | **0.91404** | 5-model blend. GBDT ceiling |
| 07 | 060 RealMLP | **0.91683** | Kaggle notebook fork. NN broke GBDT wall |
| 10 | Ridge 55 OOFs | **0.91707** | 55 OOF Ridge ensemble. **best public** |
| 11–15 | 078–084 | 0.91689–0.91704 | Ridge variants. **private 0.91815** |

---

## irrigation (Playground S6E4)

**TL;DR**: Classify irrigation need (Low/Medium/High) from soil/weather/crop data. Balanced accuracy metric.
**Key challenge**: Metric was balanced_accuracy, not accuracy. High class is only 3.3%. Slow learning rate (lr=0.01) + sklearn pairwise TE on 171 combinations were the key breakthroughs.

### Experiment flow

| Trial | Why | Result | Next |
|-------|-----|--------|------|
| 001 LightGBM baseline | Baseline | val 0.9844(acc), **public 0.9589** | Fix metric to bal_acc |
| 002 FE + ensemble | Domain features + 3-model blend | val 0.9853(acc), **public 0.9609** | bal_acc re-evaluation |
| 003 balanced blend | Fixed metric + class_weight + threshold(High×2.6) | val **0.9711**(bal_acc), **public 0.9691** | Expand pairwise TE |
| 004–006 TE exploration | Various target encoding approaches (28–171 pairs) | val 0.9692–0.9699 | Factorize pairs, TE on cats only |
| 007 stacking | Ridge meta-learner + bias tuning | val 0.9707 | Switch to sklearn TE |
| 008b fullpair | 171 pairwise factorize + cat TE(24) + threshold(High×3.7) | val **0.9738**, **public 0.9721** | Slow LR + full pairwise TE |
| 011 slow XGB | lr=0.01, 4000 rounds hard cap + sklearn TE on 171 pairwise (750 features) | val **0.9794**, **public 0.97799** | Multi-seed |
| 013 multiseed | 3-seed XGB + orig append + coord descent bias tuning | val **0.9796**, **public 0.97833** | Pseudolabeling, 5-seed |

### Submissions

| sub | trial | public | what happened |
|-----|-------|--------|---------------|
| 01 | 001 baseline | 0.9589 | Raw features + LightGBM |
| 03 | 003 balanced | 0.9691 | Metric fix + threshold. **+0.008 jump** |
| 08b | 008b fullpair | 0.9721 | 171 pairwise + multiclass TE + threshold |
| 11 | 011 slow XGB | 0.97799 | lr=0.01 + sklearn TE 750 features. **+0.006 jump** |
| 13 | 013 multiseed | **0.97833** | 3-seed + bias tuning. **best** |

---

## birdclef (BirdCLEF+ 2026)

**TL;DR**: Classify 234 bird/frog/insect species from 60s field recordings (5s segments). Macro-averaged ROC-AUC.
**Key challenge**: Code Competition — submissions only via Kaggle notebooks. CPU 90min limit.

> Competition in progress — approaches hidden until competition ends.

### Submissions

| sub | public | status |
|-----|--------|--------|
| 01 | 0.912 | first valid |
| 02 | 0.910 | - |
| 03 | 0.904 | - |
| 04 | **0.928** | - |
| 05–08 | failed | notebook errors |
| 09 | 0.928 | - |
| 10 | 0.925 | - |
| 11 | 0.928 | - |
| 12 | **0.929** | **best** |

---

## ts-forecasting (Hedge Fund)

**TL;DR**: Predict 36,923 financial time series. 89% of test series are unseen. Weighted RMSE.
**Key challenge**: One series (weight 13 trillion) could zero out the entire score if predicted wrong.

### Three zeros

1. **Val didn't reflect test** (sub_02) — Test had no lag features, val did. val 0.89 → public 0.0000
2. **High-weight series explosion** (sub_03) — Predicted 6.37 for a series worth 0.000009 → error 5.6×10¹⁴
3. **Group mean is poison** (sub_04) — Training mean (-0.67) on new series (true ≈ 0) is worse than predicting zero

**Finding**: Competition host said "public 0.5+ scores likely use future data (cheating)". Honest ceiling is 0.3–0.5.

---

## march-mania (March Machine Learning Mania 2026)

**TL;DR**: Predict NCAA tournament win probabilities. Brier Score.
**Key challenge**: Missed submission deadline. "9 days to go" meant days until tournament results, not submission cutoff.

Best local score: Men 0.161, Women 0.132 — would have been competitive.

---

## Lessons learned

1. **Don't trust val blindly** — Verify val reflects the actual test scenario (ts-forecasting: 3 zeros)
2. **Check prediction distribution before submitting** — Score of 0 means prediction explosion/bias
3. **Fork Kaggle notebooks when hitting local limits** — Broke GBDT ceiling with RealMLP fork (+0.003), BirdCLEF also solved by forking
4. **Don't filter weak models** — Ridge ensemble with all 55 OOFs beat cherry-picked subsets
5. **Code Competition = half the work is environment** — Don't build from scratch, fork and extend
6. **Check the actual deadline first** — "X days to go" might not mean submission deadline
7. **Never submit without OOF validation** — Even post-processing needs local val first
8. **Improving val is the right direction** — In churn, val-public gap was negative but private beat public. Val is the more accurate indicator
9. **Slow learning rate + hard cap > early stopping** — lr=0.01 with fixed 4000 rounds beat mlogloss-based early stopping for balanced_accuracy (irrigation trial_011 vs 012)
10. **Multi-seed averaging helps public more than val** — Variance reduction shows on unseen data (irrigation trial_013: val +0.0002, public +0.0003)

---

Details: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)
