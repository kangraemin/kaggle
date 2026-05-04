# Sub 34 Reflection — trial_041 convnext_timeguard (v30)

**Base**: ConvNeXt-Base XCL 3-way blend (Perch 65%+EffNet 15%+ConvNeXt 20%) v28
**Trial**: trial_041 v30

## 결과
- Public: **silent reject** (COMPLETE, publicScore 없음)
- 2026-05-03 14:19 제출

## 변경사항 (sub_33 대비)
- 기존 notebook → blend notebook으로 베이스 전환
- ConvNeXt deadline: `8 * 3600` → `_WALL_START + 114 * 60` (114분 hard deadline)
- v30 push

## 교훈
- 114분 timeguard로도 silent reject — 플랫폼 채점 시스템이 여전히 이상하거나, 경쟁 eval 환경에서 114분도 초과됨
- kernel log 기준 95분이지만 경쟁 eval이 더 느려서 114분도 부족할 수 있음
- ConvNeXt 3-way blend는 타임아웃 위험이 있어 실질적으로 운영 불가 판단

## 버려야 할 것
- ConvNeXt 3-way blend 시도 — 타임아웃 위험이 너무 높음
- deadline 튜닝 반복 — 근본 문제(ConvNeXt 추론 속도)를 해결하지 않는 한 의미 없음

## 유지해야 할 것
- Perch + EffNet 2-way blend (안정적 0.930)
- timeguard 패턴 자체는 유용 — 다른 모델에도 적용 가능

## 다음 가설
- 2-way blend에 post-processing 개선 (Gaussian smoothing, threshold tuning)
- ConvNeXt는 경량화(pruning, INT8) 없이는 포기
