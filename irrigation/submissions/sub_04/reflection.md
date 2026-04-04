# Sub 04 Reflection — trial_004_target_enc_catpairs

## 결과
- Val: 0.9852 (accuracy) / Public: 미제출
- 168 features (28 pairwise + 72 TE + 기존)

## 평가
- accuracy 메트릭으로 최적화 — 대회 메트릭(balanced_accuracy)과 불일치
- trial_002 대비 val 변화 거의 없음 (0.9853 → 0.9852)
- pairwise + TE feature가 168개로 폭증했지만 accuracy 개선 없음
- **교훈**: 메트릭부터 확인해야 함. 잘못된 메트릭으로 튜닝하면 시간 낭비
