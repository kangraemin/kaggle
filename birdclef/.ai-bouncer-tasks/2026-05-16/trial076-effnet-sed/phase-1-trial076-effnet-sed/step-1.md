## 구현 목표
- 변경 대상: `notebooks/birdclef2026-effnet-5fold-blend.ipynb`, `submissions/sub_69/trial_076_effnet_sed/meta.json`
- 핵심 변경:
  - cells[62]: `BLEND_EFFNET = 0.0` → `BLEND_EFFNET = 0.15  # trial_076: EffNet5fold 복원 + SED 유지`; `effnet_logits = np.zeros(...)` 라인 제거
  - cells[63] 신규 삽입: `_EffSpec`, `_BirdEffNet` 클래스 + epoch50 5-fold 로드 + `effnet_logits` 추론 루프
  - cells[68] (old cells[67]): `# === blend: Perch 63% + EffNet5fold 15%... (trial_076 ...)` 로 헤더 갱신
  - meta.json 신규: trial_076, sub_69, kernel v67, BLEND_EFFNET=0.15
- 참고: plan.md ## Phase 1 / Step 1

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1  | happy | cells[62]에 BLEND_EFFNET = 0.15 존재 | assert pass | ✅ |
| TC-2  | happy | cells[62]에 effnet_logits = np.zeros 없음 (placeholder 제거) | assert pass | ✅ |
| TC-3  | regression | cells[62] BLEND_SED = 0.15 유지 | assert pass | ✅ |
| TC-4  | regression | cells[62] distill_logits placeholder 유지 | assert pass | ✅ |
| TC-5  | regression | cells[62] _perch_scores_raw 유지 | assert pass | ✅ |
| TC-6  | happy | cells[63] _BirdEffNet 클래스 정의 포함 | assert pass | ✅ |
| TC-7  | happy | cells[63] effnet_logits 변수 할당 포함 | assert pass | ✅ |
| TC-8  | happy | cells[63] birdclef2026-effnet-5fold-epoch50 경로 포함 | assert pass | ✅ |
| TC-9  | happy | cells[68] trial_076 레이블 포함 | assert pass | ✅ |
| TC-10 | happy | cells[68] EffNet5fold 15% 포함 | assert pass | ✅ |

## 실행 출력

```
Total cells: 73
PASS
```
