# E2E Verification Result

## TC 실행 결과

10-TC python assert 스크립트 출력: `PASS`

| TC-ID | 결과 |
|-------|------|
| TC-1: BLEND_EFFNET = 0.15 | ✅ |
| TC-2: effnet_logits placeholder 제거 | ✅ |
| TC-3: BLEND_SED = 0.15 유지 | ✅ |
| TC-4: distill_logits placeholder 유지 | ✅ |
| TC-5: _perch_scores_raw 유지 | ✅ |
| TC-6: cells[63] _BirdEffNet 클래스 포함 | ✅ |
| TC-7: cells[63] effnet_logits 변수 포함 | ✅ |
| TC-8: cells[63] birdclef2026-effnet-5fold-epoch50 경로 포함 | ✅ |
| TC-9: cells[68] trial_076 레이블 포함 | ✅ |
| TC-10: cells[68] EffNet5fold 15% 포함 | ✅ |

## 결론
통과
