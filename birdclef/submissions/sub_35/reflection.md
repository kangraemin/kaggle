# Sub 35 Reflection — trial_042 gaussian_smoothing (v31)

**Base**: Perch 65% + EffNet 15% + prior mask blend (trial_038 base, v31)
**Trial**: trial_042 v31

## 결과
- Public: **PENDING** (채점 중, 2026-05-04 제출)
- ref: 52301956

## 변경사항 (sub_25/sub_34 대비)
- sub_25 (prior mask, 0.930) 대비: Gaussian temporal smoothing 추가
- sub_34 (ConvNeXt timeguard, silent) 대비: ConvNeXt 제거 → 2-way Perch+EffNet blend 복귀 + smoothing 추가
- prior mask 이후, submission pd.DataFrame 생성 전에 `gaussian_filter1d(sigma=1.0, axis=0)` per soundscape 적용
- scipy.ndimage 사용, 별도 학습 없음

## 교훈
- (결과 확인 후 작성)

## 버려야 할 것
- ConvNeXt 3-way blend — 타임아웃 위험 너무 높음, 효과도 검증 안 됨

## 유지해야 할 것
- Perch + EffNet 2-way blend 기반 (안정 0.930)
- prior mask 후처리
- per-soundscape 처리 패턴

## 다음 가설
- Gaussian smoothing 효과 확인 후:
  - 효과 있으면: sigma 튜닝 (0.5, 1.5, 2.0) 또는 다른 temporal smoothing 방법 탐색
  - 효과 없으면: threshold per-class 최적화 또는 pseudolabel 재시도
