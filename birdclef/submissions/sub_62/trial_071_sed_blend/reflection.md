# trial_071 reflection — sed_blend

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-16 ~10:10 UTC (~92분 소요)

## 결과 분석

Tucker Distilled-SED 5-fold ONNX BLEND_SED=0.05 추가 → **0.934 동률**.

SED 모델 (EfficientNetB0 + Perch 지식증류)이 5% 비중에서 LB 무반응.

## 가능한 해석

1. **SED 5%가 너무 작음**: 단 5%는 Perch 72%에 희석되어 신호 미미
2. **SED ≈ Perch의 distillation**: Tucker SED가 Perch로 지식증류됨 → 예측이 Perch와 유사 → diversity 미확보
3. **SED preprocessing mismatch**: librosa mel 파라미터가 test 환경에서 미세하게 다를 수 있음

## 세션 루프 결론 (trial_062~071)

| 범주 | 시도 | 결과 |
|---|---|---|
| blend weight 조정 | 8회 (trial_062~069) | 전부 0.934 |
| post-processing | 1회 (trial_069 prior_mask) | 0.934 |
| temperature scaling | 1회 (trial_070 T_AVES) | 0.934 |
| SED diversity | 1회 (trial_071 SED 5%) | 0.934 |

**10연속 0.934 천장 확정**. Top LB 0.961 vs 현재 0.934 = 0.027 gap.

## trial_072 방향

**가설 A**: SED 비중 증량 (0.05→0.15) — SED 신호가 있는지 확인
**가설 B**: SED 제거 후 완전히 다른 방향 — 로컬 학습 (mwf0 fold1~4)
**가설 C**: 다른 public SED/모델 탐색

일일 한도 5/5 소진. UTC midnight (KST 09:00) 리셋 후 제출 가능.
