# Sub 06 Reflection — trial_006_full_pairwise_ensemble

## 결과
- Val: 0.9699 (balanced_accuracy) / Public: 0.9668
- 243 features (171 pairwise + 24 TE + 기존)

## 평가
- 앙상블 가중치 (0:1:0) — XGB 단독이 best. LGBM/CAT는 앙상블에 기여 못함
- XGB 단독 bal_acc 0.9699 > LGBM 0.9627, CAT 0.9686
- 171개 pairwise feature가 오히려 LGBM을 약화시킴 (과도한 feature)
- trial_003 대비 val은 비슷(0.9699 vs 0.9678)이지만 threshold opt 없어서 public은 0.9668 < 0.9691
- **교훈**: feature 수를 늘리는 것보다 threshold optimization이 더 효과적
