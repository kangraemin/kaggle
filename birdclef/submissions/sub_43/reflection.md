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
- Public: **PENDING** (kernel v38 제출, 채점 대기 중)
- kernel v38 dry-run/run 검증 통과: pmix 2 folds 로드 확인, blend 비율 확인, submission.csv 240 rows mean 0.0579 (trial_050 0.0577과 거의 동일)

## 다음 가설 (ralph it.3+)
- pmix fold2~4 추가 → pmix 5-fold 완전 앙상블 (로컬에서 fold2 학습 중, 5/12 기준 epoch 11/30)
- pmix 2-fold(또는 5-fold)가 안정적이면 BLEND_PSEUDOMIX sweep (0.03 → 0.05) 또는 mwf0와 합쳐 0.08~0.10
- EffNetV2-S + Xeno pretrain (trial_049, kernel v12 학습 중) 완성 시 BLEND_EFFNET_S 활성화 (5번째 컴포넌트)
- prior_mask 외 다른 도메인 후처리 (현재 Tier C ×0.3은 neutral~약간 도움)

## 유지해야 할 것
- Perch ONNX (backbone): 압도적 기여
- epoch50-5fold 15%: 안정적 (distill30보다 우수, trial_048에서 확인)
- mwf0-fold0 5% + pmix 3%: 누적 diversity 전략 (0.930 → 0.932 → 0.933)
- prior_mask (Tier C ×0.3)
