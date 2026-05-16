## 목표
cells[62] BLEND_PROTO 0.65→0.0, cells[67] 헤더 코멘트 trial_074→trial_075 갱신

## 기술 접근
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[62]: `BLEND_PROTO = 0.65` → `BLEND_PROTO = 0.0` (ProtoSSM 완전 제거, Perch 13%→78%)
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[67]: 헤더 코멘트 trial_074 → trial_075, Perch 78%, ProtoSSM 0%

## Steps
- Step 1: cells[62]/cells[67] 수정 + 검증 PASS — python 7-TC assert 스크립트 PASS
