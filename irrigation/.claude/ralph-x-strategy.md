# Strategy: trial_011_slow_xgb_deeper_trees

## Approach Name
Slow XGBoost Training (lr=0.01, 15k rounds) + Sklearn Multiclass TE on ALL 171 Pairwise

## Hypothesis
trial_010의 XGB 파라미터(lr=0.1, n_estimators=2000, early_stopping=50)는 실제로 약 400개 트리에서 조기종료 → underfit.
Chris Deotte(0.9808), Mahog(0.9793) 모두 lr=0.01 + 수만 라운드 사용. lr을 10배 낮추고 더 많은 트리를 학습시키는 것이 gap의 주요 원인.

## Key Changes from trial_010

### 1. XGBoost 파라미터 (핵심)
- lr: 0.1 → 0.01 (10배 낮춤)
- n_estimators: 2000 → 15000
- early_stopping_rounds: 50 → 200
- colsample_bytree: 0.6 → 0.4 (더 많은 feature 대응)
- 나머지 동일 (max_depth=6, subsample=0.8)

### 2. sklearn Multiclass TE on ALL 171 Pairwise (추가 신호)
- trial_010: 8 cat_cols × 3 classes = 24 TE features
- trial_011: 171 pairwise × 3 classes = 513 TE features (sklearn TargetEncoder within-fold)
- TargetEncoder(target_type="multiclass", smooth="auto") from sklearn
- 기존 8 cat manual TE(24) 유지 + 추가 513 sklearn pairwise TE

### 3. Single seed (시간 균형)
- trial_010: 5 seeds × 5 folds = 25 models (lr=0.1 fast)
- trial_011: 1 seed × 5 folds = 5 models (lr=0.01 slow ~2-3시간)
- 결과 좋으면 multi-seed 확장

## Architecture
- 171 pairwise factorize (CATS+NUMS 전체)
- 8 cat_cols manual multiclass TE (24 per fold) — 유지
- 171 pairwise sklearn TE (513 per fold) — 신규
- Deotte 4 binary features
- Original data TE (19 cols)
- Label encoded cat_cols
- Total: ~213 base + 24 manual TE + 513 sklearn TE ≈ 750 features

## Expected Impact
- Slow XGB (lr=0.01 vs 0.1): +0.003~0.005 raw bal_acc
- sklearn pairwise TE: +0.002~0.004
- Combined expected OOF raw: 0.972~0.976
- After threshold: 0.975~0.979

## Implementation Notes
- sklearn TargetEncoder: fold 내 train으로만 fit, val/test에 transform
- pairwise factorized int codes → TargetEncoder 입력으로 사용
- training time: ~2-3시간 (caffeinate -s 필수)

## File Location
submissions/sub_11/trial_011_slow_xgb_deeper_trees/trial_011_slow_xgb_deeper_trees.py
