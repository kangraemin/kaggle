# trial_064 reflection — effnet_weight_up

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-15 13:44 UTC (67 min PENDING)

## 결과 분석

EffNet 15%→17% (+2pp), Perch 72%→70% (-2pp) → 0.934 동률.

**EffNet weight 방향 LB 무감각 확인:**
- trial_064: EffNet 15%→17% → 0.934 동률 (개선 없음)
- 방향 자체가 neutral. Perch ↔ EffNet 교환이 LB에 영향 없음.

**현재까지 탐색 정리:**
| 방향 | 시도 | 결과 |
|---|---|---|
| pmix weight up | 0.08→0.11→0.13 | 모두 0.934 이하 |
| mwf0 제거 | 0.02→0.00 | -0.001 (trial_063) |
| distill 추가 | 0.03 | 동률 (trial_059) |
| EffNet weight up | 0.15→0.17 | 동률 (trial_064) |
| ConvNeXt 추가 | 0.03 | TIMEOUT (인프라 문제) |

## 시사점

현재 블렌드가 다양한 방향에서 0.934에 수렴 → **blend weight 탐색의 포화** 가능성.
breakthrough를 위해 모델 자체(새 모델, 새 학습 방식) 개선이 필요할 수 있음.
단기적으로는 EffNet 0.19 push로 이 방향의 포화 여부 최종 확인.

## 다음 trial_065

BLEND_EFFNET 0.17→0.19 (+2pp), Perch 0.70→0.68 (-2pp).
mwf0 0.02, pmix 0.11 고정.
목적: EffNet weight 방향의 포화 지점 확인.
