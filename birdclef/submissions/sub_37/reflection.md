# Sub 37 Reflection — trial_044 gaussian_remove (v33)

**Base**: Perch+EffNet 2-way blend + prior_mask (Gaussian smoothing 제거)
**Trial**: trial_044 v33

## 결과
- Public: **0.930** ✅ (best 동률 복귀)
- ref: 52306531, 2026-05-04 제출

## 변경사항 (sub_36 대비)
- Gaussian smoothing(σ=1.0) 블록 14줄 완전 제거
- prior_mask, BLEND_EFFNET(0.15), temperature scaling 유지

## 교훈
- Gaussian smoothing은 soundscape 시계열에 적용 시 오히려 역효과 (-0.002)
- sub_36(0.928) vs sub_37(0.930) 대조로 인과 확인 완료
- 간단한 후처리 추가 전 A/B 실험이 중요

## 버려야 할 것
- Gaussian smoothing (σ=1.0, per soundscape) — 확인된 -0.002 역효과

## 유지해야 할 것
- Perch+EffNet 2-way blend (sub_25 기준)
- prior_mask 후처리

## 다음 가설
- 0.930 복귀 확인 후: BLEND_EFFNET 조정 (0.10, 0.20) 또는 EffNet fold 추가 시도
