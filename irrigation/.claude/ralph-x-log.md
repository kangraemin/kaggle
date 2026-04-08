# Ralph-X Work Log
Task: irrigation 대회 trial & 개선
Current best val: 0.9699 bal_acc (trial_006)
Current best public: 0.9609 (trial_002)

## [2026-04-03 21:00] STRATEGY: trial_004_target_enc_catpairs

### 리서치 결과
- Kaggle discussion 페이지는 JS-rendered라 직접 크롤링 불가
- NVIDIA Kaggle Grandmasters 블로그에서 7가지 핵심 기법 확인:
  1. Pairwise categorical combinations (8개 col -> 28개 interaction)
  2. Target encoding (within CV fold)
  3. Multi-level stacking
  4. Pseudo-labeling
  5. Hill climbing ensemble
  6. Multi-seed averaging
  7. Full-data retraining
- 대회 진행 중이라 top solution writeup 아직 없음

### 데이터 분석
- 630K train / 270K test, 3-class (Low 59%, Medium 38%, High 3.3%)
- 8 categorical (low cardinality: 2~6 unique values)
- 12 numeric features
- 현재 best: trial_002 (OOF 0.9853, Public 0.9609)

### 기존 시도 정리
- trial_001: LGBM baseline (0.9844 / 0.9589)
- trial_002: domain FE + 3-model ensemble (0.9853 / 0.9609) ← current best
- trial_003: multiseed stacking / balanced blend → 코드 작성만 완료, 미실행

### 전략 결정: Target Encoding + Pairwise Categorical Combinations
- trial_002 대비 핵심 차별점: categorical interaction을 명시적으로 feature화
- 8개 categorical에서 28개 pairwise combination 생성
- 각 categorical (원본+pairwise)에 target encoding (within CV fold, smoothing)
- Frequency encoding 추가
- 기존 domain FE 유지, 모델 구조 동일 (LGBM+XGB+CAT)
- 전략 파일: .claude/ralph-x-strategy.md


## [2026-04-04 00:41] RESULT: trial_004_target_enc_catpairs

### 결과
- OOF accuracy: **0.9852** (ensemble, weights 5:5:1)
- Individual: LGBM 0.9847, XGB 0.9850, CAT 0.9825
- Features: 168 (base + 72 TE + 28 pair + freq)

### EVALUATION
- **NO IMPROVEMENT**: 0.9852 < 0.9853 (trial_002)
- 168개 feature로 늘렸으나 오히려 미세 하락. TE + pairwise가 이 데이터셋에서 accuracy 기준으로는 효과 없음.


## [2026-04-04 00:55] STRATEGY: trial_005_ext_data_balanced_acc

### 리서치 결과
- Kaggle leaderboard 확인: Top LB 0.97996 (W-Bruno), Chris Deotte 0.97975, 우리 0.96085
- Chris Deotte 'Original Data Exact Formula' 노트북: 원본 데이터는 6개 feature로 100% 설명
  - Binary thresholds: Soil_Moisture<25, Temperature_C>30, Rainfall_mm<300, Wind_Speed_kmh>10
  - Categoricals: Crop_Growth_Stage, Mulching_Used
- Mahog 'XGB CV 0.97723 LB 0.97644': original data + TE + pairwise + balanced weights
- UtaAzu 'External Data LB 0.970': adversarial validation + weighted original data append
- Ali Afzal 'Stacked LGB+XGB+CAT LB 0.970': stacking approach

### 핵심 발견
1. **대회 메트릭이 balanced_accuracy_score** (우리는 regular accuracy 사용 중이었음!)
2. Original/external data 활용이 필수 (모든 top solution이 사용)
3. Deotte의 binary threshold features가 원본 데이터의 핵심
4. Balanced class weights가 High class recall에 필수

### 전략 결정: External Data + Formula Features + Balanced Accuracy
- 메트릭 수정: balanced_accuracy_score
- Original data TE + append (weight=0.1)
- Binary threshold features 추가
- Balanced class weights
- Pairwise combinations + sklearn TargetEncoder
- XGBoost single model (Mahog approach)
- 전략 파일: .claude/ralph-x-strategy.md


## [2026-04-04 04:33] STRATEGY: trial_006_full_pairwise_ensemble

### 리서치 결과 (Top Notebook 코드 분석)
- Mahog (LB 0.97644): ALL pairwise (NUMS+CATS ~190 combos), sklearn multiclass TE, bal_acc early stopping
- Ektarr (LB 0.97444): orig data append, extensive domain FE, pairwise on NUMS+CATS+binary, freq encoding
- Kospintr: statistical features (mean/std/skew per binned feature), 10-fold, HGBC/XGB/LGBM/CatB/RealMLP
- Leaderboard top: W-Bruno 0.97996, Chris Deotte 0.97975, Georgios 0.97910

### 기존 시도 vs Top 솔루션 Gap
1. Pairwise scope: 우리 CATS only (28) vs top NUMS+CATS (~190)
2. TargetEncoder: 수동 단일값 TE vs sklearn multiclass TE
3. Early stopping: mlogloss vs balanced_accuracy custom metric
4. Model: trial_005 single XGB vs multi-model ensemble + bal_acc optimization
5. Frequency encoding 미사용

### 전략 결정: Full Pairwise + Multiclass TE + Multi-Model Ensemble
- NUMS+CATS 전체 pairwise combinations (cardinality filter)
- sklearn TargetEncoder(multiclass, cv=5) within fold
- balanced_accuracy custom eval metric for early stopping
- LGBM + XGB + CatBoost ensemble, bal_acc weight search
- Richer domain FE (evaporation, water_deficit, heat_stress, drying_force)
- Original data TE only (Mahog approach, append 제거)
- Frequency encoding on CATS
- 전략 파일: .claude/ralph-x-strategy.md


## [2026-04-04 08:23] RESULT: trial_006_full_pairwise_ensemble

### 결과
- OOF balanced_accuracy: **0.9699** (ensemble best = XGB only)
- OOF accuracy: 0.9845
- Individual bal_acc: LGBM 0.9628, XGB 0.9699, CAT 0.9686
- Ensemble weights: (0, 1, 0)
- Features: 243 base + 24 fold TE = 267 per fold
- Pairwise: 171 cols (NUMS_binned + CATS, card limit 500)
- High class recall: 94%

### 분석
- trial_005 대비 미세 개선 (0.9692 -> 0.9699)
- XGB 단독 우세

### EVALUATION
- **IMPROVED**: bal_acc 0.9699 > 0.9692 (trial_005), accuracy 0.98454 > 0.9853 (trial_002)
- 메트릭 전환: regular accuracy → balanced_accuracy (대회 공식 메트릭)
- best-score.txt를 balanced_accuracy 기준으로 갱신: 0.9699
- **현재 best: trial_006_full_pairwise_ensemble** (bal_acc 0.9699)


## [2026-04-04 13:00] STRATEGY: trial_007_bias_tuned_stacking

### 리서치 결과
- Kaggle API로 top 20 LB 확인: W-Bruno 0.98032, Chris Deotte 0.98022, Giba 0.97931
- Ali Afzal notebook (LB 0.97779) 코드 분석: sklearn TE on all pairwise + LR stacking
- AidenSong notebook (LB 0.97796) 코드 분석: orig append(w=0.35) + Ridge/LGB meta stacking + bias tuning
- Nina notebook (LB 0.97783): majority voting ensemble of top submissions
- Kospintr notebook: statistical features (mean/std/skew per binned feature), 10-fold, 5 models

### 핵심 발견 (trial_006 대비 gap)
1. **Bias tuning on log-probabilities** — 완전히 새로운 기법. argmax(log(p)+bias) coordinate descent
2. **Original data APPEND with weight=0.35** — trial_006은 TE only. Top은 실제 데이터 append
3. **sklearn TargetEncoder on ALL pairwise** — trial_006은 manual TE on CATS only (8 cols)
4. **Stacking meta-learner** — Ridge/LGB on 9 OOF prob cols instead of simple weight search
5. **Model params**: CatBoost depth 6->8, LGB leaves 63->127

### 전략 결정: Bias-Tuned Stacking + Orig Append + sklearn TE
- 4가지 핵심 개선 동시 적용
- Expected: bal_acc 0.975+, LB 0.975+
- 전략 파일: .claude/ralph-x-strategy.md



## [2026-04-04 15:40] RESULT: trial_007_bias_tuned_stacking

### 결과
- OOF balanced_accuracy: **0.9707** (ridge_meta + bias tuning)
- OOF accuracy: 0.9803
- Individual bal_acc: LGBM 0.9623, XGB 0.9674, CAT 0.9672
- Grid blend weights: (0, 2, 7), bal_acc=0.9675
- Ridge meta: 0.9700, LGB meta: 0.9644
- Bias tuning: [-0.5, -0.8, 0.0] -> 0.9707
- High class recall: 96% (up from 94%)

### EVALUATION
- **IMPROVED**: bal_acc 0.9707 > 0.9699 (trial_006)
- best: trial_007_bias_tuned_stacking (bal_acc 0.9707)


## [2026-04-04 20:30] STRATEGY: trial_008_sklearn_multiclass_te

### 리서치 결과 (Kaggle 노트북 코드 분석)
- Kaggle API로 top 20 LB 확인: Chris Deotte 0.98082, W-Bruno 0.98042, Mahog 0.97929
- Ali Afzal notebook (LB 0.97779) 코드 분석: sklearn TargetEncoder(multiclass, cv=5) on ALL pairwise
- Ektarr notebook (LB 0.97444) 코드 분석: full orig merge + non-linear FE + XGB bal_acc eval callback
- Kospintr notebook: statistical features (mean/std/skew per binned feature), RealMLP, 10-fold

### 핵심 발견 (trial_007 대비 gap)
1. **sklearn multiclass TE on ALL pairwise** - 우리는 8 CATS만 manual TE. Top은 171+ pairwise에 multiclass TE (3 values per col)
2. **XGB bal_acc eval metric for early stopping** - 우리는 mlogloss. Top은 bal_acc callback으로 직접 최적화
3. **Full original data merge** - trial_003 (best public 0.9691)은 full merge. trial_007은 0.35 weight로 downweight
4. **Non-linear features** - squared, log1p, rank features. 한 번도 안 해봄
5. **Binary features in pairwise pool** - Ektarr 방식
6. **XGB 15k iterations** (vs 우리 5k)

### 전략 결정: sklearn Multiclass TE + bal_acc Eval + Full Orig + Non-Linear FE
- 6가지 핵심 개선 동시 적용. Single XGB + bias tuning
- Expected: bal_acc 0.975+, LB 0.975+
- 전략 파일: .claude/ralph-x-strategy.md


## [2026-04-04 22:14] RESULT: trial_008_sklearn_multiclass_te

### 결과
- OOF balanced_accuracy: **0.9712** (XGB + bias tuning)
- OOF accuracy: 0.9805
- XGB raw bal_acc: 0.9690
- Bias: [-1.23, -0.78, 0.0]
- Per-fold bal_acc: [0.9682, 0.9692, 0.9702, 0.9689, 0.9687]
- Features: 334 base + 795 fold TE = 1129 per fold
- Pairwise: 253 cols (CATS+NUMS_bin+binary)
- sklearn TE on 265 cols (multiclass, cv=5)
- High class recall: 96%

### EVALUATION
- **IMPROVED**: bal_acc 0.9712 > 0.9707 (trial_007)
- best: trial_008_sklearn_multiclass_te (bal_acc 0.9712)


## [2026-04-05] EVALUATION: trial_008_sklearn_multiclass_te

- Latest trial: trial_008_sklearn_multiclass_te
- New score: 0.971155 (oof_balanced_accuracy)
- Previous best: 0.9699 (trial_006_full_pairwise_ensemble, from best-score.txt)
- **IMPROVED**: 0.971155 > 0.9699
- best-score.txt updated to 0.971155
- **Current best: trial_008_sklearn_multiclass_te** (bal_acc 0.971155)


## [2026-04-05] STRATEGY: trial_009_stat_group_features

### 리서치 결과 (gap analysis after trial_008)
- Current best: trial_008 OOF 0.9712
- Leaderboard: Chris Deotte 0.9808, W-Bruno 0.9804, Mahog 0.9793, Ali Afzal 0.9778
- Gap: ~0.008 to Mahog (closest single-model approach)

### Gap 분석
1. **Statistical group features** - Kospintr 방식: (binned_numeric x cat) 그룹별 label 분포 통계
   - mean/std/freq per (numeric_bin x cat) group
   - 우리 trial에서 한 번도 시도 안 함
   - 완전히 새로운 feature signal
2. **Orig TE on numerics** - Ali Afzal: CATS+NUMS 19개 전체에 원본 TE 적용 (우리: CATS 8개만)
   - 57 prior features vs 24: 11 numeric 피처에 원본 분포 정보 추가
3. **XGB iterations 5k → 15k** - 상위권 모두 15k-50k 사용. 우리는 5k 상한

### 전략 결정: Statistical Group Features + Orig TE on Numerics + Extended XGB
- 세 가지 개선 동시 적용
- Statistical group features: 88 pairs x 4 stats = ~352 cols
- Orig TE: 19 x 3 = 57 cols (기존 24)
- XGB: 15k rounds, early stopping 500
- 나머지는 trial_008 그대로
- Expected OOF: 0.974-0.977


## [2026-04-05] RESULT: trial_009_stat_group_features

### 결과
- OOF balanced_accuracy: **0.9710** (XGB + bias tuning)
- OOF accuracy: 0.9799
- XGB raw bal_acc: 0.9689
- Bias: [-1.382, -0.80, 0.002]
- Per-fold bal_acc: [0.9674, 0.9689, 0.9705, 0.9683, 0.9694] avg 0.9689
- Features: 367 base + 352 stat_group + 795 fold TE = 1514 per fold
- Stat group features: 352 cols (88 pairs x 4 stats)
- Orig TE extended: 57 cols (8 CATS x 3 + 11 NUMS x 3)

### 분석
- stat group features 352개 추가 -> per-fold raw 소폭 향상
- OOF raw 0.9689 vs trial_008 0.9690 -- 거의 동일
- Bias 후: 0.9710 vs 0.9712 -- 미세 하락

### EVALUATION
- **NO IMPROVEMENT**: 0.9710 < 0.9712 (trial_008)
- best: trial_008_sklearn_multiclass_te (bal_acc 0.9712) 유지


## [2026-04-05] STRATEGY: trial_010_multiseed_xgb

### 전략
- trial_008b 아키텍처 기반 (171 pairwise factorize, 24 multiclass TE on cats, Deotte binary, orig TE)
- 5 seeds x 5 folds = 25 total models (seeds: 42, 123, 456, 789, 2024)
- OOF/Test: per-seed 평균 -> averaged OOF에 threshold 최적화
- XGB: lr=0.1, early_stopping=50 (속도 최적화)


## [2026-04-05] RESULT: trial_010_multiseed_xgb

### 결과
- OOF bal_acc (raw, multi-seed avg): **0.9682**
- OOF bal_acc (after threshold): **0.9741**
- Threshold weights: [0.7, 0.5, 4.6]
- Per-seed OOF: 42=0.9681, 123=0.9683, 456=0.9681, 789=0.9681, 2024=0.9681
- Total models: 25 (5 seeds x 5 folds)
- Features: 213 base + 24 TE = 237

### 분석
- Multi-seed raw OOF 0.9682 vs trial_008b raw 0.9689 (lr=0.1 조기종료로 약간 낮음)
- Threshold 후: 0.9741 vs trial_008b 0.9738 — 미세 개선 (+0.0003)
- Multi-seed variance reduction 효과: 각 seed간 raw OOF 편차 매우 작음 (0.00022 std)
- 기대 대비 낮음: 예상 0.974-0.977 vs 실제 0.9741

### EVALUATION
- **MARGINAL IMPROVEMENT**: 0.9741 > 0.9738 (trial_008b)
- best: trial_010_multiseed_xgb (bal_acc 0.9741)
- best-score.txt 갱신: 0.9741


## [2026-04-05] STRATEGY: trial_011_slow_xgb_deeper_trees

### 리서치 결과 (Discussion 크롤링)
- Kaggle discussion 페이지 직접 크롤링 성공
- Chris Deotte (LB 0.9808): lr=0.01, n_estimators=50000, early_stopping=500
- Mahog (LB 0.9793): ALL pairwise TE (171 × sklearn multiclass) + orig 0.35 weight + slow XGB
- Ali Afzal (LB 0.9778): sklearn TargetEncoder(multiclass, cv=5) on ALL 171 pairwise
- AidenSong (LB 0.9780): bias tuning via coordinate descent in log-odds space
- 핵심 공식: True signal = 4 binary thresholds + 2 categorical (6 features total)

### Gap 분석 (trial_010 OOF 0.9741 vs LB top 0.9808)
1. **XGB learning rate/iterations**: trial_010 lr=0.1 early_stop=50 → ~400 trees (underfit)
   Top: lr=0.01, 50k rounds, early_stop=500 → 5000~20000 trees
2. **Pairwise TE scope**: trial_010 = 8 cats × 3 classes = 24 TE
   Top: ALL 171 pairwise × 3 classes = 513 TE (sklearn multiclass)
3. **Bias tuning**: coordinate descent in log-odds space (step sizes: 1.0→0.002 decreasing)

### 전략 결정: Slow XGB + sklearn ALL Pairwise TE
- XGB: lr=0.01, n_estimators=15000, early_stopping=200
- sklearn TargetEncoder(multiclass) on ALL 171 pairwise within-fold → 513 new features
- 기존 24 cat TE 유지, single seed (시간 대비 결과 확인 목적)
- Expected OOF raw: 0.972~0.976, after threshold: 0.975~0.979


## [2026-04-07] RESULT: trial_011_slow_xgb_deeper_trees

### 결과
- OOF balanced_accuracy (after threshold): **0.979367**
- OOF accuracy: 0.976026
- Threshold weights: [0.8, 0.7, 4.6]
- XGB: lr=0.01, n_estimators=4000 (hard cap, no early stop)
- Features: 213 base + 24 manual TE + 513 sklearn pairwise TE = 750

### EVALUATION
- **IMPROVED**: 0.979367 > 0.9741 (trial_010)
- **Current best: trial_011_slow_xgb_deeper_trees** (bal_acc 0.979367)
- best-score.txt 갱신: 0.9741 → 0.979367


## [2026-04-07] RESULT: trial_012_extend_rounds

### 결과
- OOF balanced_accuracy (after threshold): **0.979274**
- OOF accuracy: 0.97546
- Threshold weights: [0.7, 0.5, 4.8]
- XGB: lr=0.01, n_estimators_max=15000, early_stopping=200 (mlogloss)
- Best rounds per fold: avg 6406.2
- Features: 213 base + 24 manual TE + 513 sklearn pairwise TE = 750

### 분석
- trial_011 (4000 hard cap, 0.979367) > trial_012 (15k + mlogloss early_stop, 0.979274)
- mlogloss 기반 early stop이 balanced_accuracy 최적화에 역효과 확인
- mlogloss 기준 최적 round != bal_acc 기준 최적 round

### EVALUATION
- **NO IMPROVEMENT vs trial_011**: 0.979274 < 0.979367
- **IMPROVED vs previous best (trial_010)**: 0.979274 > 0.9741
- Current best: **trial_011_slow_xgb_deeper_trees** (bal_acc 0.979367)
- best-score.txt updated: 0.9741 → 0.979367 (trial_011 기준)


## [2026-04-07] STRATEGY: trial_013_multiseed_lgbm_orig_append

### 리서치 결과 (Discussion 크롤링)
- Chris Deotte (LB 0.98082): magic formula 6개 feature로 100% 정확도 역공학, logit anchor features
- wguesdon (LB 0.9788): 14-model GBDT ensemble (5 XGB + 3 LGB + 6 CAT), hill climbing weights, DE threshold 최적화
- BlamerX (CV 0.979): LGBM + orig append(w=0.35) + pseudolabeling (max_prob>=0.92)
- UtaAzu (0.977): coordinate descent bias tuning in log-odds space, step=1.0->0.002
- Mahog (LB 0.9793): ALL pairwise TE (171 x sklearn multiclass) + orig 0.35 weight + slow XGB
- 핵심: Top 솔루션은 단일 모델이 아닌 멀티 모델 앙상블 + 멀티 시드 사용

### Gap 분석 (trial_011 OOF 0.979367 vs LB top 0.9808)
1. **Single seed**: trial_011은 seed=42만 사용. Top은 3-5 seeds 평균 (분산 감소)
2. **Single model**: trial_011은 XGB만. Top은 XGB + LGBM + CAT 앙상블
3. **Original data**: trial_011은 orig에서 TE만. Top은 orig rows APPEND (weight=0.35)
4. **Bias tuning**: trial_011 threshold sweep vs top coordinate descent log-odds (더 정밀)
5. **Pseudolabeling**: 한 번도 시도 안 함 (BlamerX 방식)

### 전략 결정: Multi-seed XGB+LGBM + Original Append + Log-odds Bias Tuning
- 3 seeds x 5 folds x 2 models (XGB + LGBM) = 30 total models
- XGB: trial_011 동일 (lr=0.01, 4000 rounds hard cap)
- LGBM: num_leaves=127, lr=0.03, n_estimators=3000, early_stopping=100
- Original rows APPEND (w=0.35) to each training fold
- Hill climbing blend alpha on XGB/LGBM OOF
- Coordinate descent log-odds bias tuning (UtaAzu method)
- Expected OOF: 0.980~0.981


## [2026-04-09] RESULT: trial_013_multiseed_lgbm_orig_append

### 결과
- OOF balanced_accuracy (after bias tuning): **0.979573**
- Seed-averaged OOF: XGB=0.976280, LGBM=0.974683
- Hill climbing: alpha=1.00 (XGB only, LGBM didn't contribute)
- Bias: [-0.939, -0.98, 1.0] -> 0.979573 (was 0.976280)
- Per-seed XGB OOF: seed42=0.975982, seed123=0.975999, seed456=0.976077
- Per-seed LGBM OOF: seed42=0.974679, seed123=0.974501, seed456=0.974303
- Total models: 30 (3 seeds x 5 folds x 2 types)
- Features: 750 (213 base + 24 manual TE + 513 sklearn pairwise TE)

### 분석
- Multi-seed XGB 평균 (0.97628) vs trial_011 단일 시드 (raw ~0.976) — 미세 개선
- LGBM (num_leaves=127, lr=0.03, early_stop=100) < XGB 성능 → alpha=1.0 (XGB only)
- Orig append (w=0.35) 효과: XGB raw 0.97628 vs trial_011 ~0.976 — 수치상 비슷
- Bias tuning (coord descent): 0.97628 -> 0.979573 (+0.0033 lift)
- trial_011 threshold: 0.979367, trial_013 bias tuning: 0.979573 — +0.000206 개선

### EVALUATION
- **IMPROVED**: 0.979573 > 0.979367 (trial_011) — 미세 개선 (+0.000206)
- **Current best: trial_013_multiseed_lgbm_orig_append** (bal_acc 0.979573)
- 핵심: LGBM이 XGB를 이기지 못함 (num_leaves=127 + early_stop=100이 너무 약함)
- 다음 방향: LGBM 강화 (num_leaves=255, lr=0.01, 더 많은 rounds) 또는 다른 접근
