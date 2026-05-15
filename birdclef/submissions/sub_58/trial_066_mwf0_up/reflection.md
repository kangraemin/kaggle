# trial_066 reflection — mwf0_up

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-15 16:26 UTC (80 min PENDING)

## 결과 분석

mwf0 0.02→0.03 (+1pp), pmix 0.11→0.10, EffNet 0.15, Perch 0.72 → 0.934 동률.

**blend weight 전 방향 포화 확정:**

| 방향 | 시도 결과 |
|---|---|
| pmix up | 0.08/0.11/0.10 = 0.934, 0.13 = 0.933 |
| mwf0 제거 | -0.001 (trial_063) |
| mwf0 up | 0.03 = 동률 (trial_066) |
| distill | 0.03 = 동률 (trial_059) |
| EffNet up | 0.17 = 동률, 0.19 = -0.001 |
| ConvNeXt | TIMEOUT |

모든 방향에서 0.934 ceiling 돌파 불가.

## 시사점

현재 blend 구성(Perch 72% + EffNet5fold 15% + mwf0 ~2-3% + pmix ~10-11%)이 weight space에서 optimal에 도달. 추가 개선을 위해서는:
1. 새로운 모델 컴포넌트 (더 강력한 백본)
2. 학습 전략 개선 (더 많은 데이터, 더 나은 augmentation)
3. 앙상블 방식 변경 (logit 합 외의 다른 fusion)

## 다음 trial_067

mwf0 0.04까지 push — 포화 경계 확정 후 전략 재검토.
BLEND_MWF0 0.03→0.04, pmix 0.10→0.09, Perch/EffNet 고정.
