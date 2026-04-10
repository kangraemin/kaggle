# Sub 13 Reflection — trial_024_effnet5fold_lse

## 결과
- trial_024: Public **0.922** (best 0.929 대비 **-0.007 하락**)

## 변경사항 (sub_12 대비)
1. EffNet 1-fold → 5-fold 앙상블
2. LSE inference pooling 추가 (forward_features → freq pool → head → LSE over time)
3. BLEND_EFFNET 0.08 → 0.10

## 교훈

### LSE는 학습-추론 불일치로 역효과
- head가 global pool feature(1280-dim 벡터)로 학습됨
- 추론 시 forward_features → freq mean → temporal features(8×1280)로 head에 전달
- head weights가 전혀 다른 feature 분포를 받게 되어 출력 품질 급락
- 이론상 "같은 dim이라 괜찮다"지만 실제로는 global pool과 temporal feature의 분포가 다름
- **LSE 효과를 보려면 LSE 구조로 처음부터 학습해야 함**

### 5-fold 단독 효과는 불명확
- LSE와 묶여 있어 5-fold 자체 효과를 분리하지 못함
- 이전 `effnet-5fold-blend` v1 (LSE 없이 5-fold, 0.08 blend) = **0.929** → 5-fold 자체는 효과 없거나 미미

### blend weight 0.10은 중립
- 5-fold 품질 자체가 1-fold와 비슷하다면 0.10이나 0.08이나 큰 차이 없음

## 버려야 할 것
- **LSE inference-only 적용** — 학습과 추론의 feature 분포 불일치로 역효과
- `forward_features` 기반 추론 (학습은 global pool인 경우)

## 유지해야 할 것
- Perch 90-92% + EffNet 8-10% 블렌딩 구조
- 5-fold EffNet (단, global pool로 추론)
- best: 0.929 (sub_12, trial_023)

## 다음 가설
- **trial_025**: 5-fold global pool 추론 + blend 0.08 (LSE 제거, sub_12 구조로 복구) — 5-fold 단독 효과 확인
- **trial_026**: LSE를 처음부터 학습한 모델 (train_effnet_lse_5fold.py, Kaggle T4 GPU로) — MPS 문제 없음
- **trial_027**: 현재 학습 중인 증강 강화 모델(SpecAugment) 완료 후 블렌딩
