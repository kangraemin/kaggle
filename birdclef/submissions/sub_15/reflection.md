# Sub 15 Reflection — trial_028_distill_5fold

## 결과
- trial_028: Public **0.930** (new best, +0.001 vs 이전 best 0.929)

## 변경사항 (sub_14 대비)
1. EffNet 모델을 knowledge distillation으로 재학습 — Perch 임베딩 L2-MSE 매칭 (proj head 학습)
2. 커널 dataset_sources 방식으로 전환 (stale embedded datasetId 문제 해결)
3. ONNX_PATH fallback 추가 (`/kaggle/input/datasets/` vs `/kaggle/input/` 양쪽 탐색)
4. competition BASE 경로 fallback 추가
5. BLEND_EFFNET = 0.08, 5-fold global pool 유지

## 교훈

### Distillation이 CNN 품질을 실질적으로 높임
- 동일한 블렌딩 비율(0.08)에서 0.929 → 0.930으로 향상
- Perch 임베딩을 target으로 한 L2-MSE distillation이 EffNet 표현력을 개선
- 5-fold는 단독으로 효과 없지만, distillation과 결합 시 +0.001

### Kaggle kernel-metadata.json dataset_sources 마운트 경로
- `dataset_sources: ["owner/slug"]` → `/kaggle/input/{slug}/` 에 마운트
- NOT `/kaggle/input/datasets/{owner}/{slug}/` (이 경로는 embedded datasetId 방식)
- embedded datasetId는 dataset 삭제/재생성 시 stale되어 마운트 실패

### BirdCLEF 2026은 Notebook-only 제출
- `kaggle competitions submit` CLI로 제출 불가 — "This competition only accepts Submissions from Notebooks"
- Kaggle 웹에서 커널 Submit 버튼으로만 제출 가능

## 버려야 할 것
- CLI `kaggle competitions submit` 시도 (notebook-only 대회에서 무의미)
- embedded datasetId 방식 (stale 위험)

## 유지해야 할 것
- Perch + EffNet distillation 구조
- BLEND_EFFNET = 0.08 (Perch 92% + EffNet 8%)
- Global pool 추론
- dataset_sources 방식 (kernel-metadata.json)
- ONNX_PATH 및 BASE 경로 fallback

## 다음 가설
- **trial_029**: BLEND_EFFNET 탐색 (0.08→0.12~0.15) — distillation으로 CNN 품질 올랐으니 비중 높여도 OK
- **trial_030**: Distillation 추가 에폭 학습 / 더 많은 fold — distillation 품질 자체를 더 높임
- **trial_031**: SpecAugment + distillation 조합 — augmentation으로 일반화 추가 향상
