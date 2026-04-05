# Sub 10 Reflection — trial_010_multiseed_xgb

## 결과
- Val: **0.9741** (bal_acc) / Public: **0.9720**
- trial_008b(val 0.9738, pub 0.9721) 대비: val +0.0003, pub -0.0001

## 핵심 변경
- Multi-seed XGB: 5 seeds × 5 folds = 25 models averaging
- trial_008b 아키텍처 기반 (pairwise factorize + cat TE + Deotte binary)
- Threshold 최적화: High × 4.6 (008b는 3.7)

## 배운 것
- Multi-seed averaging이 val에서 +0.0003 효과 — 미미함
- Seed 간 분산이 매우 낮아 추가 seed 효과 제한적
- **Threshold 과적합 의심**: val 올랐지만 public은 오히려 0.0001 하락
  - High×4.6이 val fold 분포에 과적합, test에서는 High×3.7이 더 안정
- 최종 제출은 trial_008b가 더 안전한 선택
- Gap 비교: 008b -0.0017 vs 010 -0.0021 → 008b가 일관적
