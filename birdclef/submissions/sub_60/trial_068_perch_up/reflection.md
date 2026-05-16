# trial_068 reflection — perch_up

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-16 ~05:50 UTC

## 결과 분석

mwf0 0.04→0.02 원복 + pmix 0.09→0.07 (-2pp) → Perch 0.72→0.76 (+4pp) → **0.934 동률**.

**Perch 비중 방향도 완전 포화:**

| Perch weight | EffNet | mwf0 | pmix | 결과 |
|---|---|---|---|---|
| 0.72 | 0.15 | 0.02 | 0.11 | 0.934 (trial_062 base) |
| 0.72 | 0.15 | 0.03 | 0.10 | 0.934 (trial_066) |
| 0.72 | 0.15 | 0.04 | 0.09 | 0.934 (trial_067) |
| 0.76 | 0.15 | 0.02 | 0.07 | 0.934 (trial_068) |

Perch 4pp 증량도 무효과.

## blend weight space 완전 소진 총정리

| 방향 | 시도 | 결과 |
|---|---|---|
| pmix 0.08~0.11 | 전 범위 | 0.934 동률 |
| pmix 0.13 | 1회 | 0.933 ❌ |
| mwf0 제거 | 1회 | 0.933 ❌ |
| mwf0 0.02~0.04 | 전 범위 | 0.934 동률 |
| distill 0.03 | 1회 | 0.934 동률 |
| EffNet 0.15~0.17 | 2단계 | 0.934 동률 |
| EffNet 0.19 | 1회 | 0.933 ❌ |
| ConvNeXt 0.03 | 2회 | TIMEOUT |
| Perch 0.72~0.76 | 여러 조합 | 전부 0.934 동률 |

**결론**: 6-component blend의 weight space에서 **0.934 천장 완전 확정**. 추가 weight 조정으로 돌파 불가.

## 시사점

weight space 탐색 종료. 새로운 돌파구 필요:

1. **Post-processing**: 현재 `CLASS_PRIOR_MASK` (비사이트 종 0.3× 억제) 및 temperature scaling (T_AVES=1.10) 적용 중 → 이 파라미터 조정이 다음 탐색 방향
2. **새 모델 컴포넌트**: mwf0 fold1~4 추가 학습, 더 큰 backbone
3. **학습 전략 개선**: augmentation, 더 많은 epochs

## 다음 trial_069

**Prior mask 완화** (0.3→0.5): 비사이트 종 억제를 덜 강하게
- cell 17의 `mask[i] = 0.3` → `mask[i] = 0.5`
- 가설: 0.3이 너무 강해서 실제로 존재하는 비사이트 종의 신호를 억누르고 있을 가능성
