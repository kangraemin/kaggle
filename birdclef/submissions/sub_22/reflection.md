# Sub 22 Reflection — trial_035_softauc

## 결과
- Public: **0.930** (best 동률, trial_028 이후 4번째 0.930)
- OOF (5-fold mean): **0.9815** (vs trial_028 baseline 0.9792, +0.0023)

## 변경사항 (sub_21 대비)
- SoftAUC hybrid loss 도입: `loss_cls = 0.5 * BCE + 0.5 * SoftAUC`
- SoftAUCLoss: pairwise ranking loss, gamma=1.0, Mixup soft target 지원
- 스펙트로그램 pre-compute: 전체 35549개 → float16 npy 캐시 (4.7GB), epoch당 ~1095s → ~170s (6.5x 속도 향상)
- EffNet distill 5fold weights 교체: distill → softauc

## 무슨 일이 있었나
- SoftAUC는 BirdCLEF 2025 1위가 사용한 기법. AUC 메트릭과 학습 목표를 직접 정렬
- OOF는 +0.0023 개선됐지만 LB는 0.930 동률 — EffNet 15% blend 안에서의 개선이 전체 점수에 미치는 영향 한계
- BLEND_EFFNET=0.15 → EffNet이 기여하는 비중이 낮아서 EffNet 자체 성능 개선이 LB에 잘 반영 안 됨
- sub_15(0.930) 이후 sub_16, 17, 22가 모두 동률 — 0.930 벽은 Perch+ProtoSSM 파이프라인 자체의 ceiling일 가능성

## 교훈
- **EffNet 학습 개선만으로는 0.930 돌파 어렵다** — 15% blend 한계. EffNet 기여 폭이 너무 작음
- spectrogram pre-compute는 학습 속도 6.5x 향상 → 앞으로 모든 EffNet 실험에 재사용 가능
- OOF 개선 ≠ LB 개선. 특히 blend가 작을수록 더 그럼

## 버려야 할 것
- EffNet 학습 개선(loss/regularization)만 반복하는 전략 — BLEND 15%에서 효과 한계 확인
- OOF 0.001~0.002 수준의 미세 개선을 기대하고 제출 낭비하는 패턴

## 유지해야 할 것
- SoftAUC hybrid loss — OOF 개선 효과 있음, BLEND를 더 올릴 경우 LB 반영 가능성 있음
- 스펙트로그램 pre-compute 패턴 (6.5x 속도향상, 모든 이후 실험에 재사용)
- BLEND_EFFNET=0.15 기준선 유지 (더 강한 모델로 올릴 수도 있음)

## 다음 가설
1. **BLEND 비율 올리기** (0.15 → 0.25~0.30) — SoftAUC 모델이 distill보다 강하니 기여 비중 올리면 효과 가능성. 단 Perch 비중 낮아지는 리스크
2. **Perch 파이프라인 직접 개선** — ProtoSSM을 올해 데이터로 직접 학습 (작년 weight 쓰면 일반화 실패 확인됨, sub_19)
3. **더 많은 epoch + 다양한 augmentation** — spec pre-compute로 epoch 30→50 가능. CutMix/SpecAugment
4. **EffNet BLEND 없이 독립 제출** — EffNet 단독 성능 파악용 (진단 목적)
