# Sub 13 Reflection — trial_013_multiseed_lgbm_orig_append + trial_014_ridge_stacking

## 결과
- trial_013: Val **0.9796** / Public **0.97833** (new best!)
- trial_014: Val 0.9785 / Public 미제출
- 이전 best (011): Public 0.97799 → **+0.00034 개선**
- Gap: -0.0013 (역대 가장 안정적)

## trial_013 핵심 변경점 (trial_011 대비)
1. **3-seed averaging** (42, 123, 456) × 5-fold × 2 models = 30 models
2. **LGBM 추가** — num_leaves=127, lr=0.03, 3000 rounds, early_stopping=100
3. **Original data append** (weight=0.35)
4. **Coordinate descent bias tuning** (threshold sweep 대체)

## trial_014 (ridge stacking)
- 9개 trial OOF를 LR meta-learner로 stacking → 0.9785
- 단일 모델(trial_013 XGB 0.9763)보다 하락
- stacking이 이 데이터셋에서 효과 없음 확인

## 교훈

### multi-seed는 public에서 더 효과적
- val: +0.0002 (미미), public: +0.00034 (의미 있음)
- seed 분산 감소가 unseen data에서 발현 — val에서는 과소평가됨

### LGBM은 XGB보다 약함
- LGBM avg bal_acc 0.9747 vs XGB 0.9763
- hill climbing이 XGB alpha=1.0 선택 → LGBM blend 기여 0
- 이 데이터셋에서 XGB single model이 최적

### coord descent bias > threshold sweep
- bias [-0.939, -0.98, 1.0] — log-odds 공간에서 더 정밀한 조정
- gap -0.0013으로 가장 안정적

### stacking은 역효과
- 다양한 trial OOF를 합쳐도 단일 강한 모델을 이기지 못함
- 모든 trial이 비슷한 feature set 기반이라 diversity 부족

## 버려야 할 것
- LGBM 앙상블 — XGB 단독이 나음
- OOF stacking — diversity 없이는 역효과
- 30개 모델 학습 (며칠 소요) — 효과 대비 비용 과다

## 유지해야 할 것
- 3-seed XGB averaging (val 미미하지만 public에서 효과)
- coord descent bias tuning
- orig append(w=0.35)
- sklearn TE(multiclass) on 171 pairwise + lr=0.01 hard cap

## 다음 가설
- pseudolabeling (high confidence test samples → train에 추가)
- Chris Deotte magic formula features (logit anchor)
- 5-seed로 확장 (3→5 seed)
