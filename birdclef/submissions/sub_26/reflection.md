# Sub 26 Reflection — trial_039_convnext_5fold

**Base**: sub_24 trial_037 ConvNeXt 3-way blend (0.930) — fold1 single
**Trial**: trial_039
**Hypothesis**: ConvNeXt fold1 single → fold0~4 전체 5-fold mean으로 교체. 모델 분산 감소 → 0.930 돌파 기대.

## 결과
- Public: **PENDING** (2026-04-30 제출, 노트북 v23 실행 중)
  - 채점 완료 시 업데이트 예정

## 변경사항 (sub_25 대비)
- ConvNeXt 모델: fold2 single → fold0~4 전체 5-fold (glob 자동 로드)
- Kaggle 데이터셋 `ramkang/birdclef2026-convnext-5fold` 업데이트 (fold0,1,2,3,4 전부 추가)
- 노트북 v23 push (코드 변경 없음, glob 패턴이 자동 처리)
- 블렌드 비율 유지: Perch 65% + EffNet 15% + ConvNeXt 20%
- prior mask 코드 그대로 포함 (효과 미미하지만 제거 안 함)

## 학습 결과 (convnext_5fold)
- Fold 1: 0.9895, Fold 2: 0.9891, Fold 3: 0.9920, Fold 4: 0.9908, Fold 5: 0.9913
- Mean AUC: **0.9905**
- 총 소요: 4460.7분 (~74시간)

## 교훈
- (채점 후 작성 예정)

## 버려야 할 것
- (채점 후 작성 예정)

## 유지해야 할 것
- 3-way blend 골격 (Perch + EffNet distill + ConvNeXt XCL)
- ConvNeXt XCL backbone (다른 아키텍처 + 조류 특화 pretrain)

## 다음 가설
1. **trial_040 v2_train (HIGH)**: multi-window cache + labels_v2 (secondary 0.3 + soundscape 1478) 통합 학습. labeled soundscape 도메인 갭 직격.
2. **trial_041 effnet_pseudo_blend (LOW)**: effnet_5fold_pseudo (CV 0.9792) 교체 시도. distill 0.930 LB 기준으로 낮을 가능성 높아 후순위.
