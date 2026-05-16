# Round 2: 회귀 검증

## 확인 사항

cells[62] 전체 내용 검토:
- BLEND_PROTO = 0.0 ✅ (0.65에서 변경됨)
- BLEND_PSEUDOMIX = 0.05 ✅ (유지)
- BLEND_EFFNET = 0.0 ✅ (유지)
- BLEND_MWF0 = 0.02 ✅ (유지)
- BLEND_SED = 0.15 ✅ (유지, override 버그 없음)
- effnet_logits placeholder ✅
- distill_logits placeholder ✅
- _perch_scores_raw ✅

cells[67] 헤더:
- `# === blend: Perch 78% + ProtoSSM 0% + mwf0 2% + pmix 5% + SED 15% (trial_075 — ProtoSSM 제거, SED 15% 격리 테스트) ===` ✅

## 결론
통과
