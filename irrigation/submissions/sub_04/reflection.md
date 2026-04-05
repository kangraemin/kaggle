# Sub 04 Reflection

## 결과
- trial_008b_multiclass_te_fullpair: Val **0.9738** (bal_acc) / Public **0.9721**
- 이전 best (sub_03): Public 0.9691 → **+0.0030 개선**
- Gap: -0.0017 (sub_03 -0.0020에서 소폭 개선)
- 1등(0.9803)과 gap: 0.0082

## 핵심 발견

### 1. 171 pairwise factorize가 핵심
- C(19,2)=171 pairwise를 factorize만 하고 TE는 cat_cols(8개)에만 적용
- 이전 시도(v1)에서 171 pairwise 전부에 multiclass TE를 먹였더니 537개 TE가 noise가 됨 (mlogloss 0.82, bal_acc 0.42)
- **pairwise는 factorize, TE는 원본 cat에만** — 이게 정답

### 2. Deotte binary threshold features 효과
- soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10 (4개)
- 이 4개가 원본 데이터의 생성 공식 핵심 feature
- base features에 추가하니 XGB가 이걸 잘 활용

### 3. Original data append 제거 효과
- trial_003: original 10K를 train에 concat → 640K행
- trial_008: original은 TE source로만 사용 → 630K행 (순수 train만)
- append 안 하는 게 더 나음 — synthetic data에 original을 섞으면 distribution drift

### 4. XGB 단독이 앙상블보다 나음
- 앙상블 weight search 결과: XGB:LGBM = 1:0 (XGB only)
- XGB 0.9689 > LGBM 0.9668
- 171 pairwise feature를 XGB가 LGBM보다 잘 활용

### 5. threshold 공격적일수록 효과적
- trial_003: High ×2.6, Medium ×0.75 → +0.0033
- trial_008: High ×3.7, Medium ×0.55 → +0.0049
- minority class에 더 강한 가중치가 public에서도 먹힘

## 삽질 기록
- v1: 171 pairwise × 3 class = 537 TE → mlogloss 0.82로 완전 실패 (11시간 낭비)
- v2: feature_cols에 string cat_cols 포함 → XGB가 못 읽어서 bal_acc 0.42
- v3: pairwise factorize only + TE on cats only → 정상 작동

## 다음에 해볼 것
- CatBoost 추가 (trial_003에서 CAT가 개별 1등이었음)
- Multi-seed 앙상블 (3~5 seeds)
- Binary threshold를 pairwise에 포함 (C(23,2)=253 pairs)
- Submission voting ensemble (003 + 008b + 다른 세션 결과)
- LR stacking meta-learner
