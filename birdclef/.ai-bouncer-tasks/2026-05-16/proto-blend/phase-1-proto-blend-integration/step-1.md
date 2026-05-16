## 구현 목표
- 변경 대상: `notebooks/birdclef2026-effnet-5fold-blend.ipynb` (cells[60], cells[62], cells[67])
- 핵심 변경:
  - cells[60] 끝: `LOGS["test_inference"] = test_logs` 다음에 `proto_logits = final_test_scores.copy()` 추가
  - cells[62]: BLEND_PSEUDOMIX 줄 다음에 `BLEND_PROTO = 0.50` 추가
  - cells[67]: blend formula에 `- BLEND_PROTO` 추가 + `+ BLEND_PROTO * proto_logits` 라인 추가, print 메시지 trial_073으로 업데이트
- 참고: plan.md ## Phase 1 / Step 1

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1  | happy | cells[60] source에 `proto_logits = final_test_scores.copy()` 포함 | assert 통과 | ✅ |
| TC-2  | happy | cells[62] source에 `BLEND_PROTO = 0.50` 포함 | assert 통과 | ✅ |
| TC-3  | happy | cells[67] source에 `BLEND_PROTO * proto_logits` 포함 | assert 통과 | ✅ |
| TC-4  | happy | cells[67] source에 `BLEND_SED - BLEND_PROTO` 포함 (Perch 계산 반영) | assert 통과 | ✅ |
| TC-5  | regression | cells[60]의 기존 `LOGS["test_inference"] = test_logs` 라인 유지 | assert 통과 | ✅ |

## 실행출력

```
PASS
```

검증 명령어: `python3 -c "import json; nb=json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb')); s60=''.join(nb['cells'][60]['source']); s62=''.join(nb['cells'][62]['source']); s67=''.join(nb['cells'][67]['source']); assert 'proto_logits = final_test_scores.copy()' in s60; assert 'BLEND_PROTO = 0.50' in s62; assert 'BLEND_PROTO * proto_logits' in s67; assert 'BLEND_SED - BLEND_PROTO' in s67; assert 'LOGS[\"test_inference\"] = test_logs' in s60; print('PASS')"`
