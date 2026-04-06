## Submission 10 Reflection

**Base**: pantanal-distill-improvement-a4dc68 (0.930) + ONNX 변환
**Trial**: trial_021 (093_onnx_fork)

### 결과
- Public: **0.925** (best 0.928 대비 -0.003)

### 변경사항 (sub_09 대비)
- 0.93 공개노트북(a4dc68) 전체 fork
- 동일하게 ONNX Runtime 변환 적용
- V18 CFG 동일 + hardcoded per-class thresholds 포함
- ProtoSSM 아키텍처가 우리 노트북과 약간 다름 (meta_dim=8, d_input+d_scores concat)

### 교훈
- **ONNX 변환이 Perch 출력에 미세 차이를 만듦** — 같은 모델이어도 TF SavedModel vs ONNX Runtime 결과가 동일하지 않음
- 0.93 노트북이 0.930 나온 건 TF SavedModel 기반. ONNX로 바꾸면 -0.003~0.005 손실 가능
- per-class thresholds�� 원본 모델 출력에 최적화돼있어서, ONNX 출력에는 안 맞을 수 있음

### 버려야 할 것
- 다른 노트북의 hardcoded thresholds를 그대로 복붙하는 것 (모델 출력이 다르면 무의미)

### 유지해야 할 것
- trial_020 (우리 ONNX 파이프라인, 0.928) — 현재 best
- ONNX Runtime 기반 추론

### 다음 가설
- 우리 노트북(trial_020) 기반으로 V18 파라미터 적용 (0.93 노트북 fork 대신)
- ONNX 출력 기반으로 per-class threshold 재최적화
- CNN 블렌딩 추가
