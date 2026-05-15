# trial_065 reflection — effnet_further_up

- score: **0.933** ❌ (-0.001 vs best 0.934)
- scored_at: 2026-05-15 14:58 UTC (67 min PENDING)

## 결과 분석

EffNet 17%→19% (+2pp), Perch 70%→68% (-2pp) → 0.933 하락.

**EffNet weight 방향 확정적으로 역효과:**

| trial | EffNet | Perch | score | result |
|---|---|---|---|---|
| 064 | 17% | 70% | 0.934 | ➖ 동률 |
| 065 | 19% | 68% | 0.933 | ❌ -0.001 |

결론: **EffNet weight ↑ → Perch weight ↓ 방향은 15%에서 포화.** 17%는 동률(noise), 19%는 명백히 역효과.

**현재까지 blend weight 탐색 종합:**
- pmix: 0.08→0.11→0.13 = 모두 0.934 이하 (포화)
- mwf0: 제거 시 -0.001 (필요한 컴포넌트)
- distill: 동률
- EffNet: 0.15→0.17→0.19 = 동률→동률→하락 (15% optimal)
- ConvNeXt: TIMEOUT (인프라 문제)

## 시사점

blend weight 공간에서 거의 모든 방향이 포화됨.
아직 미시험 방향: **mwf0 증량** (0.02→0.03). mwf0 제거(trial_063)가 -0.001이었으니 반대 방향 가능성.

## 다음 trial_066

원복: EffNet 0.15, Perch 0.72.
신규: mwf0 0.02→0.03 (+1pp), pmix 0.11→0.10 (-1pp, 합 유지).
