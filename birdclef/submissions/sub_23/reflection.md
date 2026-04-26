# Sub 23 Reflection — trial_036_epoch50_blend25

## 결과
- Public: **0.929** (best 0.930 대비 -0.001)
- OOF (5-fold mean): **0.9823** (vs trial_035 0.9815, +0.0008)

## 변경사항 (sub_22 대비)
- N_EPOCHS 30→50 (spec precompute 덕분에 epoch당 ~170s 유지, 총 ~2.4h 추가 학습)
- BLEND_EFFNET 0.15→0.25 (EffNet 기여 비중 +10%)
- spec_indices mmap 패턴: 전체 mmap + 정수 인덱스 배열로 fancy indexing copy 방지

## 무슨 일이 있었나
- BLEND를 올리면 SoftAUC 개선이 LB에 반영될 것이라 기대했지만 오히려 -0.001 하락
- OOF는 0.9823으로 trial_035(0.9815) 대비 소폭 상승 — epoch 50의 효과는 있었음
- 그러나 BLEND=0.25는 Perch+ProtoSSM 비중(0.75)을 낮춰 전체 점수 하락을 유발한 것으로 보임
- 최적 BLEND는 여전히 0.08~0.15 범위로 추정 — EffNet이 Perch보다 약한 상황에서 비중 올리면 역효과

## 학습 과정 삽질
- num_workers=4 시도 → macOS spawn 방식으로 4.7GB 배열 pickle × 4 = 288s/epoch (역효과)
- mmap_mode='r' 단독 → fancy indexing `all_specs[train_idx]`가 3.5GB copy 생성 → 9GB swap → 505s/epoch
- spec_indices 패턴으로 해결: 단일 정수 인덱싱은 view (copy 없음) → ~170s/epoch 유지
- num_workers=0이 macOS 환경에서 최적 (단순 numpy slice에서는 worker overhead가 이득보다 큼)

## 교훈
- **BLEND 올리는 것 = EffNet 강화가 아니라 Perch 약화** — EffNet이 Perch보다 약한 한 BLEND 올리면 손해
- 두 가지 변수(epoch+1, BLEND+1)를 동시에 변경해서 어느 쪽이 원인인지 분리 불가
- 0.930 벽은 EffNet 개선이나 BLEND 조정으로는 돌파 어려움 — 파이프라인 근본적 변경 필요

## 버려야 할 것
- BLEND 0.25 이상 — Perch 비중 감소로 역효과 확인
- EffNet epoch 늘리기 + BLEND 올리기 동시 시도 — 원인 분리 안 됨

## 유지해야 할 것
- SoftAUC hybrid loss (OOF 개선 효과 있음)
- spec_indices mmap 패턴 (메모리 효율 + 속도 유지)
- num_workers=0 on macOS (spawn 방식 환경에서 최적)
- BLEND_EFFNET=0.10~0.15 범위 (최적 BLEND 범위)

## 다음 가설
1. **Perch 파이프라인 직접 개선** — ProtoSSM을 올해 데이터로 직접 학습. 현재 파이프라인 ceiling 돌파의 근본 해법
2. **EffNet 단독 제출 진단** — EffNet 자체 LB 성능 확인 (BLEND 없이). EffNet이 실제로 얼마나 강한지 파악
3. **더 강한 augmentation** — CutMix/SpecAugment 추가. OOF 개선이 BLEND 0.15 내에서 LB 반영되는지 확인
4. **BLEND 최적값 재탐색** — 0.08 vs 0.10 vs 0.15 단일 변수로 A/B 비교
