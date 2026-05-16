# trial_069 reflection — prior_mask_relax

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-16 ~08:10 UTC

## 결과 분석

CLASS_PRIOR_MASK 비사이트 종 억제 0.3→0.5 완화 → **0.934 동률**.

prior mask 완화도 LB 무반응. 0.3 suppression이 적절했거나, 이 방향 자체가 무의미한 것으로 확인.

## 이번 세션 탐색 요약 (trial_062~069)

| trial | 변경 | 결과 |
|---|---|---|
| 062 | pmix 원복 베이스라인 | 0.934 ➖ |
| 063 | mwf0 제거 | 0.933 ❌ |
| 064 | EffNet 0.15→0.17 | 0.934 ➖ |
| 065 | EffNet 0.17→0.19 | 0.933 ❌ |
| 066 | mwf0 0.02→0.03 | 0.934 ➖ |
| 067 | mwf0 0.03→0.04 | 0.934 ➖ |
| 068 | Perch 0.72→0.76 | 0.934 ➖ |
| 069 | prior mask 0.3→0.5 | 0.934 ➖ |

8연속 0.934 이하. 모든 blend weight 방향 + prior mask 방향 포화.

## 다음 방향 (다음 세션)

- temperature scaling (T_AVES 1.10→1.0)
- prior mask 완전 비활성화 (1.0)
- 새 모델 학습 (mwf0 fold1~4, EffNet-L, 더 많은 epochs)
