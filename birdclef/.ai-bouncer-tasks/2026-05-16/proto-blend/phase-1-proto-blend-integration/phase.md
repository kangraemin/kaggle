## 목표
ProtoSSM v4 출력(`final_test_scores` from cell 61)을 cell 68 blend formula에 반영. 3개 cell 수정.

## 기술 접근
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cell 61 끝: `final_test_scores` → `proto_logits`로 저장
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cell 63: `BLEND_PROTO=0.50` 상수 추가
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cell 68: blend formula에 `+ BLEND_PROTO * proto_logits` 추가, Perch 63%→13%

## Steps
- Step 1: 3개 cell 수정 후 검증 명령 통과 확인
