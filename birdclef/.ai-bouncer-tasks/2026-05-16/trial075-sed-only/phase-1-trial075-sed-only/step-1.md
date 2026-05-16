## 구현 목표
- 변경 대상: `notebooks/birdclef2026-effnet-5fold-blend.ipynb`
- 핵심 변경:
  - cells[62]: `BLEND_PROTO = 0.65  # trial_074: ProtoSSM up 50→65%...` → `BLEND_PROTO = 0.0  # trial_075: ProtoSSM 제거. SED 15% 단독 격리. Perch 65%→78%.`
  - cells[67]: header `# === blend: Perch 13% + ProtoSSM 65%... (trial_074 ...)` → `# === blend: Perch 78% + ProtoSSM 0%... (trial_075 ...)`
- 참고: plan.md ## Phase 1 / Step 1

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1  | 변경 | cells[62] BLEND_PROTO=0.0 | assert pass | ✅ |
| TC-2  | 회귀 | cells[62] BLEND_SED=0.15 유지 | assert pass | ✅ |
| TC-3  | 회귀 | cells[62] BLEND_EFFNET=0.0 유지 | assert pass | ✅ |
| TC-4  | 회귀 | cells[62] _perch_scores_raw 유지 | assert pass | ✅ |
| TC-5  | 회귀 | cells[62] distill_logits 유지 | assert pass | ✅ |
| TC-6  | 변경 | cells[67] trial_075 포함 | assert pass | ✅ |
| TC-7  | 변경 | cells[67] ProtoSSM 0% 포함 | assert pass | ✅ |
