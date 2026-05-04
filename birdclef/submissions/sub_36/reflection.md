# Sub 36 Reflection — trial_043 convnext_remove (v32)

**Base**: Perch+EffNet 2-way blend + prior mask + Gaussian smoothing
**Trial**: trial_043 v32

## 결과
- Public: **❌ silent** (30분+ PENDING → 플랫폼 채점 이상 지속)
- ref: 52304806, 2026-05-04 제출
- ConvNeXt 제거 효과 미확인 (채점 불가)

## 변경사항 (sub_35 대비)
- ConvNeXt 완전 제거 (Cell 63: 244줄 → 3줄)
- 3-way blend → 2-way blend: Perch 85% + EffNet 15%
- Gaussian smoothing(σ=1.0) 유지

## 교훈
- 플랫폼 채점 이상이 4/30~5/4 총 10회 이상 연속으로 지속됨 (sub_26~36)
- ConvNeXt 제거 자체는 올바른 방향이나 플랫폼 이상으로 효과 검증 불가
- 채점 시스템 정상화 전까지 sub당 1회 정도만 테스트하고 결과 기다리는 것이 최선

## 버려야 할 것
- ConvNeXt — 경쟁 eval에서 89분 이상 소요, 2시간 제한 초과 불가피
- 3-way blend 구조 (ConvNeXt 포함) — timeout 위험 내재

## 유지해야 할 것
- Perch + EffNet 2-way blend (sub_25에서 0.930 달성)
- prior mask 후처리
- Gaussian smoothing(σ=1.0) — 효과 검증 목적으로 유지

## 다음 가설
- 채점 정상화 확인 후: sigma 튜닝 (0.5, 1.5, 2.0) 또는 per-class threshold 최적화
- BLEND_EFFNET 조정 (0.10, 0.20) 시도
