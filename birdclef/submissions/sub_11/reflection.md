## Submission 11 Reflection

**Base**: trial_020 (ONNX Perch 0.928) + V18 파라미터 + audio features
**Trial**: trial_022 (full_upgrade)

### 결과
- Public: **0.928** (best와 동일, 변화 없음)

### 변경사항 (sub_09 대비)
- V18 CFG: d_model 256→320, n_ssm_layers 3→4, n_epochs 60→80
- MLP (128)→(256,128), PCA 64→128
- Audio feature engineering: energy, silence ratio, ZCR, spectral centroid
- Energy-weighted scoring

### 교훈
- **V18 파라미터 변경은 dry-run 환경에서 효과 없음** — dry-run 20파일로는 모델이 제대로 학습 안 됨. 실제 test에서 효과가 상쇄됐을 가능성
- **audio features (energy weighting)도 효과 없음** — Perch 임베딩이 이미 audio 특성을 충분히 캡처하고 있어서 redundant
- 파라미터 튜닝만으로는 0.928 벽을 못 넘음

### 버려야 할 것
- 단순 파라미터 튜닝으로 점수 올리려는 시도
- audio energy weighting (Perch와 redundant)

### 유지해야 할 것
- trial_020 ONNX 파이프라인 (0.928 안정)

### 다음 가설
- 완전히 다른 모델(SED CNN) 블렌딩이 유일한 돌파구
- 또는 Discussion의 "multi-context head" 접근 (Tom 8위, hengck23)
