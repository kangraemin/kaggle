# Strategy: trial_013_multiseed_lgbm_orig_append

## Approach Name
Multi-seed XGB+LGBM Ensemble + Original Data Append (w=0.35) + Log-odds Bias Tuning

## Motivation
trial_011 (XGB single seed, lr=0.01, 4000 rounds) -> OOF 0.979367.
Top LB achieves 0.9808+ using:
1. Multi-model diversity (XGB + LGBM)
2. Multi-seed averaging (reduces variance)
3. Original data APPEND with w=0.35 (we only use orig for TE, never append the rows)
4. Coordinate descent bias tuning in log-odds space (current threshold sweep is less principled)

## Key Changes vs trial_011

### 1. Multi-seed XGB (3 seeds x 5 folds = 15 XGB models)
- Seeds: [42, 123, 456]
- XGB params: same as trial_011 (lr=0.01, 4000 rounds hard cap, colsample_bytree=0.4)
- Average OOF/test across seeds before blending

### 2. LightGBM (3 seeds x 5 folds = 15 LGBM models)
- num_leaves=127, lr=0.03, n_estimators=3000, early_stopping=100
- min_child_samples=50, colsample_bytree=0.6, subsample=0.8
- reg_alpha=0.1, reg_lambda=1.0
- Same sample_weight=balanced
- Same 750 feature set as trial_011

### 3. Original Data APPEND (sample_weight=0.35)
- Append the 10K original rows to each fold's training set
- Assign sample_weight=0.35 to original rows, balanced weights to synthetic rows
- Val does NOT include orig rows (OOF measured on synthetic val only)
- Note: also keep existing orig TE features (global, leakage-free) as before

### 4. Coordinate Descent Bias Tuning (log-odds space, UtaAzu method)
Replace current threshold multiplication with:

    def tune_bias(proba, y_true):
        best_bias = np.zeros(3)
        best_score = balanced_accuracy_score(y_true, proba.argmax(1))
        for step in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
            improved = True
            while improved:
                improved = False
                for ci in range(3):
                    for d in (-1.0, 1.0):
                        c = best_bias.copy()
                        c[ci] += d * step
                        preds = np.argmax(np.log(proba + 1e-15) + c, axis=1)
                        s = balanced_accuracy_score(y_true, preds)
                        if s > best_score + 1e-9:
                            best_bias, best_score, improved = c, s, True
        return best_bias, best_score

### 5. Hill Climbing Ensemble Weights
After XGB and LGBM OOF are ready:
- Search alpha in [0.0, 1.0] at 0.05 steps: blend = alpha * xgb_oof + (1-alpha) * lgbm_oof
- Apply coordinate descent bias tuning on the blended OOF

## Feature Set (unchanged from trial_011)
Same 750 features as trial_011:
- 213 base (num_cols + le_cats + binary_threshold + orig_TE on 19 cols)
- 24 manual multiclass TE on cat_cols (inside fold)
- 513 sklearn TargetEncoder(multiclass) on ALL 171 pairwise (inside fold)

## Architecture Summary
30 total models: 3 seeds x 5 folds x 2 model types (XGB + LGBM)
OOF: seed-averaged per model type -> hill climb blend (alpha)
Post: coordinate descent log-odds bias tuning on blended OOF

## Expected Impact
- Multi-seed variance reduction: +0.0005~0.001
- LGBM diversity: +0.001~0.002 OOF lift
- Orig append w=0.35: +0.001 (more signal for High class)
- Combined expected OOF: 0.980~0.981

## What NOT to Change
- 750 feature set (proved in trial_011)
- XGB lr=0.01, 4000 rounds hard cap (trial_012 confirmed mlogloss early stop hurts)
- 5-fold StratifiedKFold
- target_map = {Low:0, Medium:1, High:2}

## Implementation Notes
- orig rows get sample_weight=0.35 (not balanced-adjusted)
- sklearn TE is fit on training fold (incl. orig rows) only; transform val/test
- LGBM early_stopping on val logloss, cap at 3000 rounds
- Total training time estimate: ~4-6 hours (use caffeinate -s)

## File Location
submissions/sub_13/trial_013_multiseed_lgbm_orig_append/trial_013_multiseed_lgbm_orig_append.py
