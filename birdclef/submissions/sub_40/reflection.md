# Sub 40 Reflection — trial_047 effnet_mw_blend

**Base**: Perch 77% + EffNet distill 5fold 15% + EffNet fold0 8% (sub_39, 0.932)
**Trial**: trial_047 kernel v35

## 결과
- Public: **TBD** (PENDING)

## 변경사항 (sub_39 대비)
- BLEND_MWF0 0.05 → 0.08 (fold0 비중 상향)
- Perch 비중 0.80 → 0.77

## 가설
fold0 5%에서 +0.002 효과 확인. 8%로 올리면 추가 개선 가능성 있음.

## 교훈
- (점수 확인 후 업데이트)

## 버려야 할 것
- (점수 확인 후 업데이트)

## 유지해야 할 것
- (점수 확인 후 업데이트)

## 다음 가설
- 0.08이 0.932 이상이면: 0.10도 시도
- 0.08이 0.932 미만이면: 최적 비중은 0.05, 다른 방향 탐색
