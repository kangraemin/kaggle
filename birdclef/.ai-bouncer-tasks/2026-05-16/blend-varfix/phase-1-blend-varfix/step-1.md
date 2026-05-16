## 구현 목표
- 변경 대상: `notebooks/birdclef2026-effnet-5fold-blend.ipynb` (cells[62], cells[67])
- 핵심 변경:
  - cells[62] 끝: `BLEND_EFFNET = 0.0`, `BLEND_MWF0 = 0.02`, `effnet_logits = np.zeros(...)` 추가
  - cells[67] 첫 줄 주석: "Perch 13% + EffNet 5fold 15%" → "Perch 28%" 수정
- 참고: plan.md ## Phase 1 / Step 1

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1  | happy | cells[62]에 `BLEND_EFFNET = 0.0` 포함 | assert 통과 | ✅ |
| TC-2  | happy | cells[62]에 `BLEND_MWF0 = 0.02` 포함 | assert 통과 | ✅ |
| TC-3  | happy | cells[62]에 `effnet_logits = np.zeros` 포함 | assert 통과 | ✅ |
| TC-4  | regression | cells[67]의 `BLEND_PROTO * proto_logits` 유지 | assert 통과 | ✅ |

검증 명령어: `python3 -c "import json; nb=json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb')); s62=''.join(nb['cells'][62]['source']); s67=''.join(nb['cells'][67]['source']); assert 'BLEND_EFFNET = 0.0' in s62; assert 'BLEND_MWF0 = 0.02' in s62; assert 'effnet_logits = np.zeros' in s62; assert 'BLEND_PROTO * proto_logits' in s67; print('PASS')"`

## 실행출력

```
PASS
```
