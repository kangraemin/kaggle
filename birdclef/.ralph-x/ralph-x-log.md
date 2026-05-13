# Ralph-X Work Log — BirdCLEF 2026
Current best public LB: 0.933 (trial_050: iter1에서 달성)
Kernel: ramkang/birdclef2026-effnet-5-fold-pseudo-blend

## 이전 결과
- iter 1: trial_050 → 0.933 NEW BEST
- iter 2: trial_051 → 0.933 동률 (fold 추가 무효과)
- iter 3: rate limit으로 중단 → 재개

## iter 9 (재개): trial_059 distill_add
- 시작: 2026-05-13 16:46 KST
- 전략: distill_5fold(KD L2-MSE) 별도 5번째 컴포넌트 추가, BLEND_DISTILL=0.03, mwf0 0.05→0.02
- 근거: sub_48 reflection "EffNet 28% 예산 내부 재분배 데드 엔드, 새 모델/diversity 축 필요". trial_055(같은 슬롯 평균 -0.001)와 다른 실험.
- kernel v46 COMPLETE (wall 259.7s, +124s vs trial_058 136s)
- log highlights: "Loaded 5 distill folds (KD L2-MSE)", "BLEND_DISTILL=0.03", "blend (...): Perch 72% + EffNet5fold 15% + fold0-B0 2% + fold0-S 0% + pmix 8% + distill 3%", submission mean 0.0452
- 제출: 2026-05-13 17:02 KST, PENDING
- 발견: trial_058 = public 0.934 확정 (best 동률, logit-blend 복원 성공)
- 채점 완료: 2026-05-13 18:08 KST (제출 후 ~66분), **public 0.934** ➖ best 동률 (변화 없음)
- 결론: distill_5fold 별도 weight 추가도 LB 무효과. trial_055(같은 슬롯 평균 -0.001)와 다른 경로지만 결과 같음. EffNet 28% 예산 내 모든 재분배·KD 손실 diversity 시도가 0.934 천장 못 깸 (trial_054/055/056/058/059 6연속 동률 또는 회귀).
- 다음(it.10) 방향: EffNet 풀 안에서 흔들기 중단, 새 1차 모델 컴포넌트(AudioMAE/BEATs probe 또는 EffNetV2-S Xeno-pretrain 또는 SED attention head) 또는 Perch 멀티윈도우 추론으로 새 diversity 축 필요. 자세한 가설은 sub_51 reflection.md.
