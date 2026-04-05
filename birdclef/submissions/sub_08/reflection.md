## Submission 08 Reflection

**Base**: yukiZ 0.926 fork — clean seed1891 (API push v5) + Version 1 재제출
**Trial**: trial_018 (clean) + Version 1 재검증

### 결과
- seed1891 clean (v5): **Notebook Timeout**
- Version 1 재제출: **Notebook Timeout** — 이전에 0.928 나왔던 바로 그 버전

### 근본 원인: hidden test 데이터 증가
- test set ~600개 soundscape 파일 (Discussion 확인)
- Perch v2 CPU 추론: 파일당 ~10초 → 600파일 = ~100분
- 120분 제한에 train/model 학습 시간 포함하면 초과
- 34% public + 66% private 전체에서 120분 (Tom Denton 확인)
- **Kaggle hidden test는 대회 중 점진적으로 커짐** — 4/4에 성공한 Version 1이 4/5에 타임아웃

### 교훈
- 이전에 성공한 제출이 나중에 타임아웃 될 수 있음
- dry-run(20파일) 성공 ≠ 채점(600+파일) 성공 — 30배 차이
- API push 시 kernel_sources 노트북 output 마운트 안 됨 → dataset으로 변환 필요
- BirdCLEF 2023에서도 같은 문제 → ONNX/TFLite 양자화로 해결

### 버려야 할 것
- 현재 Perch CPU 파이프라인 그대로 제출 (타임아웃 확정)

### 유지해야 할 것
- yukiZ 0.926 fork 모델 아키텍처 (정확도 0.928 검증됨)
- Perch 임베딩 품질
- train 데이터 사전 캐시 구조

### 다음 가설
- Perch TFLite 변환 — CPU 추론 2~3배 가속
- ONNX Runtime 양자화 — TFLite 대안
- 타임 버짓 fallback — 시간 부족 시 prior로 예측
