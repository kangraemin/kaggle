## Submission 12 Reflection

**Base**: trial_020 (ONNX Perch 0.928) + EfficientNetV2-B0 블렌딩
**Trial**: trial_023 (effnet_blend)

### 결과
- Public: **0.929** (new best! +0.001)

### 변경사항 (sub_09 대비)
- EfficientNetV2-B0 1-fold (25MB, 로컬 학습) 추론 추가
- mel-spectrogram (torchaudio) → timm tf_efficientnetv2_b0 → logits
- Perch 92% + EffNet 8% weighted average (mlnjsh v75 참고)
- 블렌딩은 final_test_scores 이후, submission 직전에 적용

### 교훈
- **CNN 블렌딩이 +0.001 효과** — 다른 feature space(mel-spectrogram vs Perch embedding)라 앙상블 다양성 확보
- **1-fold만으로도 효과 있음** — mlnjsh는 5-fold로 더 큰 효과, 우리는 1-fold이지만 여전히 개선
- **블렌딩 가중치 8%가 적절** — 1-fold이라 품질이 낮으므로 너무 높이면 역효과
- **timm vs torchvision 아키텍처 차이 주의** — state_dict 키 구조가 완전히 다름
- **spec 키 필터링 필요** — 학습 시 Spectrogram이 모델에 포함돼있으면 추론 시 제거해야

### 버려야 할 것
- 없음

### 유지해야 할 것
- **Perch + EffNet 블렌딩 구조** — 첫 best 갱신
- ONNX Perch 파이프라인
- BLEND_EFFNET = 0.08

### 다음 가설
- EffNet 5-fold 학습 → 더 강한 CNN → 블렌딩 효과 증가
- 블렌딩 가중치 탐색 (0.05~0.15)
- multi-context head (hengck23, Tom 8위) 적용
- EffNet 학습 데이터 augmentation 강화
