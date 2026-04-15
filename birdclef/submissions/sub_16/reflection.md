# Sub 16 Reflection — trial_029_blend_sweep

## 결과
- trial_029: Public **0.930** (best 동일, 변화 없음)

## 변경사항 (sub_15 대비)
1. BLEND_EFFNET 0.08 → 0.10

## 교훈

### BLEND 0.10은 0.08 대비 개선 없음
- 0.08과 0.10이 동일한 0.930 — 이 범위에서 blend 비중은 무의미한 차이
- 더 큰 폭(0.15, 0.20)을 시도하거나 반대로 낮춰봐야 최적값 파악 가능

### 로컬 노트북에 경로 fallback 반영 필수
- v8에서 /tmp/에만 수정했던 BASE/ONNX fallback이 로컬 노트북에 없어서 v9 ERROR 발생
- 앞으로 수정은 항상 로컬 notebooks/에 먼저 반영 후 /tmp/로 복사

## 버려야 할 것
- 미세한 blend 조정 (±0.02 수준) — 차이 없음

## 유지해야 할 것
- Perch + EffNet distill 구조 (BLEND_EFFNET ~0.08~0.10)
- Global pool 5-fold, dataset_sources 방식

## 다음 가설
- **trial_030**: BLEND_EFFNET 0.15 — 더 큰 폭으로 탐색
- **trial_031**: BLEND_EFFNET 0.05 — 반대 방향 탐색 (Perch 비중 더 높이기)
- **trial_032**: SpecAugment + distillation 재학습 — CNN 품질 자체 개선
