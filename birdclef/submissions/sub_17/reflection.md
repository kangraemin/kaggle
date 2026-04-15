# Sub 17 Reflection — trial_030_blend_sweep

## 결과
- trial_030: Public **0.930** (best 동일, 변화 없음)

## 변경사항 (sub_16 대비)
1. BLEND_EFFNET 0.10 → 0.15

## 교훈

### BLEND 0.08 ~ 0.15 범위에서 점수 차이 없음
- 0.08, 0.10, 0.15 모두 0.930 — blend 비중 조정은 이 distill 모델에서 무의미
- Perch가 워낙 강해서 EffNet 비중이 크게 달라져도 final score에 영향 미미
- blend 탐색보다 CNN 품질 자체를 올리는 방향이 더 유효

## 버려야 할 것
- BLEND 비중 fine-tuning (0.08~0.15 범위 내 조정)

## 유지해야 할 것
- BLEND_EFFNET ~0.08 (기본값으로 복귀)
- distill 5-fold 구조

## 다음 가설
- **trial_031**: SpecAugment + distillation 재학습 — CNN 품질 자체 개선
- **trial_032**: 더 많은 distill 에폭 / 더 강한 supervision
- **trial_033**: 다른 블렌딩 방식 — sigmoid 후 blend vs logit blend 비교
