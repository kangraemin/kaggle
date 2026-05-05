# Sub 40 Reflection — trial_047 effnet_mw_blend

**Base**: Perch 77% + EffNet distill 5fold 15% + EffNet fold0 8% (sub_39 변형)
**Trial**: trial_047 kernel v35

## 결과
- Public: **0.932** ➖ best 동률 (trial_046과 동점)

## 변경사항 (sub_39 대비)
- BLEND_MWF0 0.05 → 0.08 (fold0 비중 상향)
- Perch 비중 0.80 → 0.77

## 교훈
- fold0 비중 5% → 8%로 올려도 LB 점수 변화 없음 (0.932 = 0.932)
- 최적 fold0 비중은 이미 5%에서 포화된 듯 — 더 올려도 추가 diversity 기여 없음
- Perch 80% → 77%로 줄인 것도 neutral (손해도 이득도 없음)

## 버려야 할 것
- BLEND_MWF0 비중 단순 상향: 효과 없음 확인

## 유지해야 할 것
- 3-way blend 구조 자체는 유효 (0.932 유지)
- fold0 5% 비중이 최적인 것으로 보임

## 다음 가설
- fold1~4 standalone 추가 (5-way blend): 더 많은 diversity source 추가
- distill 5fold epochs 증가 (30→50): OOF 개선 시도
- 새로운 아키텍처 모델 추가 (EfficientNetV2-S, B1 등)
