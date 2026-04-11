# Sub 14 Reflection — trial_025_effnet5fold_global

## 결과
- trial_025: Public **0.929** (best 동일, 변화 없음)

## 변경사항 (sub_13 대비)
1. LSE inference 완전 제거 (`_BirdEffNetLSE` → `_BirdEffNet` global pool)
2. BLEND_EFFNET 0.10 → 0.08 복구
3. EffNet 5-fold 유지 (fold 0-4)

## 교훈

### 5-fold는 단독으로 효과 없음
- 1-fold (sub_12, 0.929) vs 5-fold (sub_14, 0.929) — 동일
- 분산 감소 효과가 있더라도 public score 차이로는 드러나지 않음
- 단, private score에서는 안정성 이점이 있을 수 있음

### LSE 제거로 sub_12 수준 복구
- sub_13 (LSE, 0.922) → sub_14 (global pool, 0.929) +0.007 회복
- LSE inference-only가 -0.007을 만들었던 것 재확인

## 버려야 할 것
- 5-fold 단독 블렌딩 튜닝 (효과 없음)

## 유지해야 할 것
- Perch + EffNet 블렌딩 구조 (92:8)
- Global pool 추론
- best: 0.929 (sub_12=sub_14)

## 다음 가설
- **trial_026**: SpecAugment 5-fold (effnet_5fold_aug, fold2-4 학습 완료 후) — augmentation이 CNN 품질 향상 → 블렌딩 효과 증가
- **trial_027**: 블렌딩 가중치 탐색 (0.08→0.12) — 5-fold라 품질 올라가면 비중 높여도 OK
- **trial_028**: LSE 처음부터 학습 (effnet_lse_5fold) — 추론 시에만 아닌 학습부터 LSE 구조
