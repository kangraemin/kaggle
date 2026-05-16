# trial_070 reflection — temp_scale_off

- score: **0.934** ➖ (best 동률)
- scored_at: 2026-05-16 ~09:46 UTC

## 결과 분석

T_AVES 1.10→1.0 제거 → **0.934 동률**.

ROC-AUC는 rank-based metric이라 temperature scaling이 이론적으로 불필요하다는 가설 → 실험으로 확인.
T_AVES 1.10이든 1.0이든 LB 점수 동일 = temperature scaling이 rank에 아무 영향 없음 (예상대로).

## 의미

- T_AVES는 완전히 무의미한 파라미터였음 (rank 보존)
- 1.0으로 고정 유지 (1.10도 동일하지만 1.0이 더 자연스러움)
- 이 방향도 포화

## 다음

trial_071 (SED 5-fold ONNX) 채점 대기 중.
SED diversity가 실제로 도움이 되는지 확인 후 trial_072 방향 결정.
