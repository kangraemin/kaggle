# Sub 36 Reflection — trial_043 convnext_remove (v32)

**Base**: Perch+EffNet 2-way blend + prior mask + Gaussian smoothing
**Trial**: trial_043 v32

## 결과
- Public: **0.928** (best 0.930 대비 -0.002)
- ref: 52304806, 2026-05-04 제출
- 채점 정상화 확인 (30분 PENDING 후 COMPLETE)

## 변경사항 (sub_35 대비)
- ConvNeXt 완전 제거 (Cell 63: 244줄 → 3줄)
- 3-way blend → 2-way blend: Perch 85% + EffNet 15%
- Gaussian smoothing(σ=1.0) 유지

## 교훈
- 채점 정상화됨 — ConvNeXt 제거(2-way blend 복귀)로 Notebook Timeout 해소 확인
- Gaussian smoothing(σ=1.0)은 효과 없음 혹은 역효과 (0.930→0.928, -0.002)
- sub_25(0.930) 기준선과 동일 구조인데 점수가 낮음 → Gaussian smoothing이 원인일 가능성 높음

## 버려야 할 것
- ConvNeXt — 경쟁 eval에서 89분 이상 소요, 2시간 제한 초과 불가피 ✅ 제거 완료
- Gaussian smoothing(σ=1.0) — 오히려 -0.002 하락. 제거 필요

## 유지해야 할 것
- Perch + EffNet 2-way blend (sub_25에서 0.930 달성)
- prior mask 후처리
- Gaussian smoothing(σ=1.0) — 효과 검증 목적으로 유지

## 다음 가설
- **trial_044**: Gaussian smoothing 제거 → sub_25(0.930) 구조로 완전 복귀 확인
- sub_25 기준선 재현 후: BLEND_EFFNET 조정 (0.10, 0.20) 또는 EffNet 추가 fold 시도
