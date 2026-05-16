## 목표
cells[62] BLEND_EFFNET 0→0.15 + effnet_logits placeholder 제거, cells[63]에 EffNet5fold epoch50 추론 셀 삽입(cells[63-67]을 [64-68]로 shift), cells[68] blend 헤더 trial_075→trial_076, meta.json 신규 생성

## 기술 접근
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[62]: `BLEND_EFFNET = 0.0` → `BLEND_EFFNET = 0.15`, `effnet_logits = np.zeros(...)` 라인 제거
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[63] 신규 삽입: _EffSpec + _BirdEffNet 클래스 + epoch50 5-fold 모델 로드 + effnet_logits 추론
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[68] (old cells[67]): 헤더 코멘트 Perch 78%→63%, EffNet5fold 15%, trial_075→trial_076
- `submissions/sub_69/trial_076_effnet_sed/meta.json`: 신규 생성

## Steps
- Step 1: 노트북 수정 (3개 변경 + cell 삽입) + meta.json 생성 + 10-TC 검증 PASS
