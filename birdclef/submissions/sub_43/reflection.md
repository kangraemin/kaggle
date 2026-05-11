# Sub 43 Reflection — trial_051 pmix_2fold_blend

**Base**: trial_050 best 0.933 (Perch 77% + epoch50-5fold 15% + mwf0-fold0 5% + pmix-fold0 3%)
**Trial**: trial_051 kernel v38 (ralph-x iteration 2/10)

## 변경사항 (trial_050 대비)
- **pseudo-mix 컴포넌트를 2-fold 앙상블로 확장**: pmix fold0 단일 → fold0+fold1 평균
  - fold1은 로컬 MPS 학습 완료 (5/12, 30 epochs, val AUC ~0.981 — fold0과 동급)
  - 동일 아키텍처(EffNetF0: stem_conv 1→3 + tf_efficientnetv2_b0 + Linear(1280,234)), 동일 spec(`_mwf0_spec`)
- **BLEND_PSEUDOMIX = 0.03 고정** — fold1 추가 효과만 격리하기 위해 weight는 trial_050과 동일하게 둠
- Kaggle dataset `ramkang/birdclef2026-effnet-pseudo-mix` 새 버전 (best_fold0.pth + best_fold1.pth)
- 노트북: Cell63에서 `_pmix_model` 단일 → `_pmix_models` 리스트 (glob `best_fold*.pth`), Cell64에서 pmix 추론을 fold 평균으로
- 추론 비용: pmix forward가 1→2회로 증가하지만 같은 spec 재사용, 병목은 Perch ONNX·EffNet 5-fold이므로 미미
- 4-way blend 공식 불변: **Perch 77% + epoch50-5fold 15% + mwf0-fold0 5% + pmix(fold0+fold1) 3%**

## 가설
단일 fold0보다 fold0+fold1 평균이 분산이 작고 일반화가 좋아 동일 3% weight에서 약간의 추가 이득(+0.000~0.001) 기대. trial_047(mwf0 weight 0.05→0.08 무효과)에서 보듯 weight sweep 단독은 효과가 거의 없으므로, 컴포넌트 품질 자체를 올리는 게 정석.

## 결과
- Public: **0.933** ➖ best 동률 (trial_050과 동일, 변화 없음)
- kernel v38 run 검증 통과: pmix 2 folds 로드 확인, blend 비율 확인, submission.csv 240 rows mean 0.0579 (trial_050 0.0577과 거의 동일 — 이미 이 시점에 변화가 미세함이 보였음)

## 가설 검증 (반증됨)
"2-fold 앙상블이 단일 fold0보다 분산이 작아 +0.000~0.001" → **무효과**. 이유 후보:
1. **3% weight가 너무 작음** — pmix가 전체의 3%만 차지하므로 그 안에서 fold0→fold0+fold1로 노이즈가 줄어도 최종 logit에 미치는 영향이 측정 한계 이하
2. **fold0이 이미 충분히 안정적** — 30 epoch 학습한 EffNet은 단일 fold도 OOF AUC ~0.98로 안정적이라 추가 fold의 한계 이득이 작음
3. **public LB 240 rows의 해상도 한계** — macro-AUC, 종마다 양성 sample 적어 0.001 미만 차이는 노이즈에 묻힘
- 회귀(regression)는 아님 — diversity 컴포넌트로서 가치는 유지, 다만 "fold 더 쌓기"만으로는 LB가 안 움직임 ⇒ **다음엔 weight를 올리거나(0.03→0.05+) 새로운 종류의 컴포넌트를 추가하는 게 정석**

## 교훈
- **같은 종류 컴포넌트의 fold 추가는 diminishing returns** — mwf0 fold0(+0.002) → pmix fold0(+0.001) → pmix fold1(+0.000). 새 정보가 적으면 LB가 안 움직임
- 0.0577→0.0579 같은 submission mean의 미세 변화가 사전 신호 — kernel 로그의 final mean을 비교하면 LB 결과를 어느 정도 예측 가능
- pmix 5-fold 완성을 기다리기보다, **다른 축의 diversity**(EffNetV2-S, 다른 spec, 다른 loss)나 **weight 재배분**(Perch 비중을 더 줄이고 EffNet 계열을 합쳐 25~30%) 쪽이 더 유망

## 다음 가설 (ralph it.3+) — 우선순위 재조정
1. **EffNet 계열 weight 재배분** (빠름, 데이터 변경 0): 현재 Perch 77% + (EffNet 5fold 15% + mwf0 5% + pmix 3% = 23%). Perch 비중을 70~73%로 더 줄이고 EffNet 합산을 27~30%로 — trial_036(BLEND 0.25)은 5fold 한 컴포넌트만 키워서 실패했지만, 지금은 4종 EffNet 컴포넌트가 있어 다를 수 있음. 또는 pmix만 0.03→0.05/0.07
2. **EffNetV2-S + Xeno pretrain** (trial_049, kernel v12 학습 중) 완성 시 BLEND_EFFNET_S 활성화 — 아키텍처가 다른(S vs B0) 진짜 새 diversity
3. pmix fold2~4 추가 → 5-fold (로컬 fold2 학습 중) — 단, it.2에서 2-fold가 무효과였으니 우선순위 낮춤. 5-fold면 다를 수도 있으나 기대 낮음
4. prior_mask 외 도메인 후처리 / soundscape-level aggregation 변경

## 유지해야 할 것
- Perch ONNX (backbone): 압도적 기여
- epoch50-5fold 15%: 안정적 (distill30보다 우수, trial_048에서 확인)
- mwf0-fold0 5% + pmix 3%: 누적 diversity 전략 (0.930 → 0.932 → 0.933)
- prior_mask (Tier C ×0.3)
