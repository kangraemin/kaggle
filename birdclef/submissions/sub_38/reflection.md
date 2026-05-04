# Sub 38 Reflection — trial_045 effnet_mw_retry

**Base**: EffNet multiwindow standalone (fold0, ep16, Val AUC 0.9794)
**Trial**: trial_045 kernel v9

## 결과
- Public: **0.836** ❌ (best 0.930 대비 -0.094)
- ref: 52313149, 2026-05-04 제출

## 변경사항 (sub_33 대비)
- enable_gpu: true → false 수정 (BirdCLEF 2026 GPU max=0 — 이전 v8에서 이 버그가 있었음)
- IS_DRY_RUN v8 로직 유지 (test_soundscapes 없으면 zero-pred)
- push 전용 디렉토리 신규 생성

## 교훈
- Standalone EffNet (Val AUC 0.9794)이 LB에서 0.836 — Perch 없이는 크게 하락
- Perch가 제공하는 사전학습 임베딩이 BirdCLEF에서 핵심적 역할을 함
- EffNet은 Perch 블렌드의 보조 역할로만 유효 (trial_023부터 확인된 사실 재확인)

## 버려야 할 것
- Standalone EffNet 단독 제출 — Perch 없이 경쟁력 없음 (0.836)

## 유지해야 할 것
- IS_DRY_RUN 자동 감지 패턴 — 코드 컴피티션 노트북 표준
- enable_gpu 사전 확인 — BirdCLEF 2026 CPU 전용 대회

## 다음 가설
- 점수 확인 후: standalone EffNet 성능이 0.920+ 이면 blend에 통합 고려
