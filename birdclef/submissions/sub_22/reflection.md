# Sub 22 Reflection — trial_035_softauc

## 결과
- Public: **0.930** (best 동률)
- OOF (5-fold mean): **0.9815** (vs trial_028 baseline 0.9792, +0.0023)

## 변경사항 (sub_21 대비)
- SoftAUC hybrid loss 도입: `loss_cls = 0.5 * BCE + 0.5 * SoftAUC`
- SoftAUCLoss: pairwise ranking loss, gamma=1.0, Mixup soft target 지원
- 스펙트로그램 pre-compute: 전체 35549개 → float16 npy 캐시 (4.7GB), epoch당 ~1095s → ~170s (6.5x 속도 향상)
- EffNet distill 5fold weights 교체: `birdclef2026-effnet-5fold-distill` → `birdclef2026-effnet-5fold-softauc`

## 무슨 일이 있었나
- SoftAUC는 BirdCLEF 2025 1위가 사용한 기법. AUC 메트릭과 학습 목표를 직접 정렬
- BCE는 calibration에 최적화, SoftAUC는 ranking에 직접 최적화 → hybrid로 안정성 + 성능 확보
- Fold별 OOF: 0.9846 / 0.9799 / 0.9827 / 0.9778 / 0.9822 (baseline 0.9792 대비 +0.0023)
- pre-compute 시 예상 크기(357MB) vs 실제(4.7GB) — 2715 → 35549 샘플 추정 오류

## 교훈
- SoftAUC hybrid loss는 OOF에서 유의미한 개선 (+0.0023). LB 반영 여부는 이번 제출로 확인
- Spectrogram pre-compute으로 epoch 속도 6.5x 향상 → 더 많은 epoch/실험 가능해짐
- 대회 메트릭(AUC)에 직접 정렬된 loss는 OOF 개선에 실질적으로 기여

## 버려야 할 것
- SoftAUC만으로 LB 돌파는 어려움 — OOF 개선이 LB에 직결되지 않음 확인

## 유지해야 할 것
- SoftAUC hybrid loss (OOF +0.0023, LB 반영 여부 확인 필요)
- 스펙트로그램 pre-compute 패턴 (이후 모든 EffNet 실험에 재사용)
- BLEND_EFFNET=0.15 (sub_16~17에서 검증된 값)

## 다음 가설
1. **BLEND weight 조정** — SoftAUC 모델이 더 강하면 0.15 → 0.20~0.25 올려볼 수 있음
2. **더 많은 epoch** — 속도 향상으로 30→50 epoch 가능해짐
3. **label smoothing + SoftAUC** — over-confident 방지
