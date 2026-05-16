# trial_067 reflection — mwf0_further_up

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-16 ~01:05 UTC

## 결과 분석

mwf0 0.03→0.04 (+1pp), pmix 0.10→0.09 (-1pp), EffNet/Perch 고정(0.15/0.72) → **0.934 동률**.

**mwf0 증량 방향 전체 포화 확정:**

| mwf0 weight | pmix | 결과 |
|---|---|---|
| 0.02 | 0.11 | 0.934 (trial_058 base) |
| 0.02 | 0.11 | 0.934 (trial_062) |
| 0.00 | 0.13 | 0.933 ❌ (trial_063, 제거 역효과) |
| 0.03 | 0.10 | 0.934 ➖ (trial_066) |
| 0.04 | 0.09 | 0.934 ➖ (trial_067) |

mwf0 0.00~0.04 전 범위 탐색 완료. 0.00 제거만 하락, 0.02~0.04는 전부 동률.

## blend weight space 포화 총정리

| 방향 | 시도 결과 |
|---|---|
| pmix up | 0.08/0.10/0.11 = 0.934, 0.13 = 0.933 |
| mwf0 제거 | -0.001 (trial_063) |
| mwf0 0.02→0.04 | 전부 동률 |
| distill 추가 | 동률 (trial_059) |
| EffNet 15%→17% | 동률 |
| EffNet 15%→19% | -0.001 (trial_065) |
| ConvNeXt | TIMEOUT (2회) |

**결론**: 현재 6-component blend의 weight space에서 0.934 ceiling 완전 확정. 추가 blend weight 조정으로 돌파 불가.

## 시사점

weight space 탐색이 종료됐다. 다음 단계는 두 가지 방향:

1. **Perch 비중 증가 테스트**: mwf0 0.04→0.02 원복 + pmix 0.09→0.07 감소 → Perch 0.72→0.76으로 증량 (주요 컴포넌트 방향 재탐색)
2. **새 모델 컴포넌트**: 더 강력한 backbone (EfficientNet-L 등), 더 많은 folds, 다른 사전학습 모델

## 다음 trial_068

mwf0 0.04→0.02 (원복) + pmix 0.09→0.07 (-2pp) → Perch 0.72→0.76 (+4pp).
BLEND_MWF0=0.02, BLEND_PSEUDOMIX=0.07, Perch implicit=0.76, EffNet=0.15.
