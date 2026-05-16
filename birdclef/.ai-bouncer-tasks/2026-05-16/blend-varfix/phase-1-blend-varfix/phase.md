## 목표
cell 63에 누락된 `BLEND_EFFNET`, `BLEND_MWF0`, `effnet_logits` 변수를 추가하여 kernel v61 NameError crash 방지.

## 기술 접근
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cell 63 (cells[62]): BLEND_PROTO 뒤에 BLEND_EFFNET=0.0, BLEND_MWF0=0.02, effnet_logits placeholder 추가
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cell 68 (cells[67]): 주석 "Perch 13%"→"Perch 28%" 수정

## Steps
- Step 1: cell 63 변수 추가 + cell 68 주석 수정 — 검증 명령 PASS 확인
