## Submission 78 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_085 effnet_readd

### 결과
- Public: **0.938** (best 동률)
- 제출: 2026-05-31 17:21 → COMPLETE

### 변경사항 (이전 sub 대비)
- BLEND_EFFNET 0→0.05 (EffNet5fold 재도입), BLEND_PROTO 0.50→0.45 (-5pp)
- SED 0.40/Perch 0.10 유지, EffNet 진단 print 추가
- kernel v76→v77

### 진단 결과 (4개 컴포넌트 logit 분포 완성)
| 컴포넌트 | mean | std | range |
|---|---|---|---|
| proto | -0.93 | 2.42 | [-9.9, 10.2] |
| sed | -7.26 | 1.82 | [-13.1, 3.6] |
| perch | -1.15 | 4.13 | [-13.6, 12.8] |
| effnet | **-9.04** | 3.66 | [-19.6, 2.2] |

### 교훈
- EffNet5fold 5% 재도입도 **무변화**. 다른 모델 가족(EfficientNetV2-B0)이지만 5% 비중으로는 row-rank를 충분히 못 바꿈.
- 0.938 천장이 매우 견고: trial_082(SED↑)·083(Perch↑)·084(정규화)·085(EffNet 5%) 전부 동률/하락.
- 가능성: ① 5% 비중이 너무 작아 효과 미미, ② Proto 50%가 이미 강해 EffNet 신호가 묻힘, ③ EffNet 자체가 Proto/SED 대비 약해 기여 못함.

### 버려야 할 것
- 소량(5%) 컴포넌트 추가로 천장 돌파 기대 (trial_059 distill 3%, trial_071 SED 5% 무효과와 동일 패턴 반복).

### 유지해야 할 것
- trial_080 best 0.938 (SED40/Proto50/Perch10).
- 4개 컴포넌트 logit 진단값 (분포 파악 완료).

### 다음 가설
1. **trial_086 EffNet 비중 10%로 push** — 5% 무효과가 "비중 부족" 때문인지 검증. BLEND_EFFNET 0.05→0.10, Proto 0.45→0.40. 효과 나오면 EffNet 신호 유효, 동률이면 EffNet 기여 한계 확정. **다음 실행**.
2. **mwf0+pmix+EffNet 동시 재도입** — 단일 소량이 아니라 검증된 보조 3종을 합쳐 15%로 다양성 묶음 투입.
3. **Proto TTA shift 확대** — 최대 비중 Proto 자체 품질 향상 (모델 추론 변경).
