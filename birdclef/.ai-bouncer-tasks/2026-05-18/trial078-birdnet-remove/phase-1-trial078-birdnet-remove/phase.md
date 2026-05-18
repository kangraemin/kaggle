## 목표
jarturo/birdnet 제거로 trial_077 silent reject 수정. BLEND_BIRDNET=0, Perch 23%. blend 공식 수정 효과 첫 번째 유효 격리 검증.

## 기술 접근
- `notebooks/kernel-metadata.json`: dataset_sources에서 "jarturo/birdnet" 항목 제거
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[62]: BLEND_BIRDNET 0.10→0.0, 주석 trial_077→trial_078
- `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[69]: 헤더 주석 Perch 13%→23%, trial_077→trial_078
- `submissions/sub_71/trial_078_birdnet_remove/meta.json`: 신규 생성

## Steps
- Step 1: 3개 파일 수정 + meta.json 생성 + 6-TC PASS — python3 verify 스크립트 PASS (6 TCs)
