## Submission 09 Reflection

**Base**: yukiZ 0.926 fork + Perch ONNX Runtime 변환
**Trial**: trial_020 (onnx_perch)

### 결과
- Public: **0.928** (best 유지, sub_04와 동일)

### 변경사항 (sub_08 대비)
- Perch SavedModel → ONNX Runtime 변환 (HuggingFace justinchuby/Perch-onnx)
- TF 2.20 설치 불필요 → onnxruntime wheel만 dataset에 포함
- infer_fn → ort.InferenceSession으로 교체
- TFLite INT8 시도 → Kaggle OOM → ONNX로 전환

### 교훈
- **TFLite + SELECT_TF_OPS는 Kaggle에서 OOM** — TF 전체 런타임을 메모리에 올려서 16GB 초과
- **ONNX Runtime이 정답** — CPU 추론 2배 빠르고, 메모리 적고, TF 의존성 없음
- **HuggingFace에 이미 변환된 ONNX 모델이 있었음** — 직접 변환할 필요 없었음
- **API push에서 dataset_sources 경로는 /kaggle/input/datasets/{owner}/{slug}/** — 디버그 셀로 확인
- **onnxruntime은 Kaggle에 기본 설치 안 됨** — wheel을 dataset에 포함해야 함
- **로컬 벤치마크**: SavedModel 0.64s/window → ONNX 0.31s/window (2x)

### 버려야 할 것
- TFLite 변환 방향 (Kaggle 환경에서 OOM)
- TF 2.20 의존성 (ONNX로 불필요해짐)

### 유지해야 할 것
- **ONNX Perch 파이프라인** — 타임아웃 해결 + 0.928 유지 확인
- yukiZ 0.926 fork 모델 아키텍처 (ProtoSSM + 벡터화 MLP + 배치 TTA)
- Kaggle dataset: ramkang/perch-v2-onnx (ONNX 모델 + onnxruntime wheel)

### 다음 가설
- seed 변경 실험 (이제 타임아웃 없으니 가능)
- ProtoSSM 하이퍼파라미터 튜닝 (epoch, lr, d_model)
- 배치 사이즈 32로 증가 (ONNX에서 추가 가속 가능)
- CNN 블렌딩 (EfficientNetV2 학습 완료, 25MB)
