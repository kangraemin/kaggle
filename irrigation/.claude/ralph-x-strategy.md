# Strategy: trial_010_multiseed_xgb

## Approach
Multi-seed XGB averaging on trial_008b architecture — 5 seeds, averaged OOF/test predictions, then threshold optimization.

## Research Summary

### Why Multi-Seed?
- trial_008b best: 0.9738 OOF, 0.9721 public. Single seed=42, XGB only.
- Top LB (Mahog 0.9793, W-Bruno 0.9804) all use multi-seed ensembles.
- Raw XGB OOF = 0.9689; threshold adds +0.005 (partially overfit on OOF).
- Multi-seed averaging reduces variance → more stable threshold → true gain on LB.

### Base: trial_008b (no changes to features)
- 171 pairwise factorize (all CATS+NUMS combos)
- Multiclass TE on cat_cols only (8 cols x 3 classes = 24 TE per fold)
- Deotte binary threshold features (soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10)
- Original data TE on all_raw_cols (19 cols)
- XGB only (LGBM dropped, weight was 0 in trial_008b)
- sample_weight="balanced", 5-fold StratifiedKFold

### Key Change: 5 Seeds
Seeds: [42, 123, 456, 789, 2024]
- Each seed: re-split folds + re-initialize XGB(random_state=seed)
- 25 total models (5 seeds x 5 folds)
- OOF avg: mean of 5 per-seed OOF arrays
- Test avg: mean of 5 per-seed test arrays
- Threshold optimization applied on averaged OOF

## Expected Impact
- Variance reduction: +0.002-0.004 raw bal_acc
- Expected OOF: 0.974-0.977 (from 0.9738 with threshold)
- LB estimate: 0.974+

## Sub Folder
submissions/sub_10/trial_010_multiseed_xgb/
