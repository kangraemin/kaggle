# Sub 38 Reflection — trial_045 effnet_mw_retry

**Base**: EffNet multiwindow standalone (fold0, ep16, Val AUC 0.9794)
**Trial**: trial_045 kernel v9

## 결과
- Public: **PENDING** (채점 중)
- ref: 52313149, 2026-05-04 제출

## 변경사항 (sub_33 대비)
- enable_gpu: true → false 수정 (BirdCLEF 2026 GPU max=0 — 이전 v8에서 이 버그가 있었음)
- IS_DRY_RUN v8 로직 유지 (test_soundscapes 없으면 zero-pred)
- push 전용 디렉토리 신규 생성

## 교훈
- (결과 확인 후 작성)

## 버려야 할 것
- (결과 확인 후 작성)

## 유지해야 할 것
- IS_DRY_RUN 자동 감지 패턴 — 코드 컴피티션 노트북 표준
- enable_gpu 사전 확인 — BirdCLEF 2026 CPU 전용 대회

## 다음 가설
- 점수 확인 후: standalone EffNet 성능이 0.920+ 이면 blend에 통합 고려
