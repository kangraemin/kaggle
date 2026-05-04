# Sub 35 Reflection — trial_042 gaussian_smoothing (v31)

**Base**: Perch 65% + EffNet 15% + prior mask blend (trial_038 base, v31)
**Trial**: trial_042 v31

## 결과
- Public: **silent reject** (COMPLETE, publicScore 없음)
- ref: 52301956, 2026-05-04 제출 → 약 100분 후 COMPLETE

## 변경사항 (sub_25/sub_34 대비)
- sub_25 (prior mask, 0.930) 대비: Gaussian temporal smoothing 추가
- sub_34 (ConvNeXt timeguard, silent) 대비: ConvNeXt 제거 → 2-way Perch+EffNet blend 복귀 + smoothing 추가
- prior mask 이후, submission pd.DataFrame 생성 전에 `gaussian_filter1d(sigma=1.0, axis=0)` per soundscape 적용
- scipy.ndimage 사용, 별도 학습 없음

## 교훈
- Gaussian smoothing 코드 자체는 정상 삽입됐으나 플랫폼 채점 이상으로 결과 미확인
- sub_26(2026-04-30)부터 sub_35까지 10연속 silent reject — 채점 시스템이 우리 노트북을 정상 채점하지 못하는 구조적 문제
- sub_32 blend v28 diagnostic(안정적 0.930 기록)도 점수 없음 → 코드 문제 아닌 플랫폼 이상

## 버려야 할 것
- ConvNeXt 3-way blend — 타임아웃 위험 너무 높음, 효과도 검증 안 됨

## 유지해야 할 것
- Perch + EffNet 2-way blend 기반 (안정 0.930)
- prior mask 후처리
- per-soundscape 처리 패턴

## 다음 가설
- **플랫폼 이상 원인 파악이 선결 과제** — 채점이 안 되는 이상 어떤 시도도 검증 불가
- 포럼에서 다른 참가자들도 같은 현상인지 확인 필요
- Gaussian smoothing 효과는 플랫폼 정상화 후 재검증 예정
