# Sub 52 Reflection — trial_060 convnext_add

**Base**: trial_058/059 (Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 8% + distill 3%) — 0.934 best 동률
**Trial**: trial_060 kernel v47 (ralph-x iteration 10)

## 결과
- Public: **미확정 (TIMEOUT_90MIN)** — kernel v47 정상 완료했으나 Kaggle 신규 submission이 90분 폴링 중 등장하지 않음. score 미수령.
- Private: N/A
- kernel v47 COMPLETE (wall 445s = 7.4min, +185s vs trial_059 259.7s — ConvNeXt-Base fold0 추론 1.8min 추가, 예상 +10~20min 추정의 하단). submission.csv 240×235, no NaN, range 0.000~0.997, Final mean 0.0432 (trial_059 0.0452 -0.0020 — convnext logit이 약간 작은 평균 logit을 만들어 sigmoid 후 살짝 낮은 mean).
- kernel 로그 핵심: `Loaded 5 EffNet folds (epoch50 SoftAUC)`, `Loaded 5 distill folds (KD L2-MSE)`, `EffNetF0 fold0 loaded`, `EffNet Pseudo-Mix loaded 5 folds`, **`ConvNeXt fold0 loaded (best_fold0.pth)`**, **`ConvNeXt fold0 inference done: 1.8min`**, blend 라인 `blend (logit-space weighted sum, trial_060): Perch 72% + EffNet5fold 15% + fold0-B0 2% + fold0-S 0% + pmix 8% + distill 0% + convnext 3%`, final_test_scores logit range [-11.998, 7.079] mean -2.977 (trial_058 [-11.924, 6.958] 대비 살짝 넓은 분포 → convnext logit이 일부 클래스에서 더 강한 신호).

## 변경사항
- **ConvNeXt-Base XCL fold0를 6번째 별도 컴포넌트로 추가** (trial_059의 zero-effect distill 3% 슬롯을 1:1 치환):
  - Cell65 신규: `transformers.ConvNextForImageClassification.from_pretrained(local_files_only=True)` → backbone+커스텀 head(1024→234) 추출. `/kaggle/input/datasets/denden12/birdset-convnext-base-xcl` (또는 fallback `/kaggle/input/birdset-convnext-base-xcl`) + `/kaggle/input/datasets/ramkang/birdclef2026-convnext-5fold/best_fold0.pth`(strict=True). 입력 `_mwf0_spec(B,1,128,T)` — XCL training과 동일 mel(n_fft=1024, n_mels=128, top_db=80) + XCL norm. inference loop는 mwf0/pmix 패턴(per-file N_WINDOWS chunks, batch forward).
  - Cell66 fusion: `BLEND_DISTILL=0.0` (drop, was 0 효과 in trial_059), `BLEND_CONVNEXT=0.03` 신규. fusion 식 `+ BLEND_CONVNEXT * convnext_logits` 추가, Perch 계수 `(1 - 0.15 - 0.02 - 0.08 - 0.03) = 0.72` 유지.
  - Cell62: distill loader/inference 유지 (BLEND_DISTILL=0이지만 logit은 계속 계산 — 향후 재활성화 옵션 보전, +~1min wall 지불).
- 모델·다른 컴포넌트·prior_mask·downstream 후처리(온도/file-level/rank-aware/delta-shift) 전부 trial_058/059 그대로 → 백본 가족 다양화 효과만 격리

## 검증
- ConvNeXt fold0 정상 로드+추론 1.8min, transformers Kaggle base image 기본 포함 확인
- blend print 라인이 모든 컴포넌트 weight를 정확히 출력, fusion 분기 정상 동작
- submission.csv 240×235(헤더 포함 235종 = 234 + row_id) 정상 생성, NaN 없음

## TIMEOUT 진단 (점수 미수령 원인)
- kaggle competitions submissions CSV에 trial_060 행 미등장 (90분 폴링, 5분 간격 18회 모두 trial_059가 latest)
- 시도: `kaggle competitions submit -k ramkang/birdclef2026-effnet-5fold-blend -v 47 -m "..."` → **400 Bad Request** (CreateCodeSubmission API 거부)
- 추정 원인: Kaggle 코드 컴페티션 submission은 kernel push만으론 트리거되지 않고, kernel 페이지의 "Submit to Competition" 액션이 hidden test 마운트 후 재실행을 만들어야 채점 생성됨. 이 단계가 자동화되지 않은 듯. trial_058/059는 동일 워크플로우에서 ~5분 내 submission 등장했었는데, 이번 kernel v47만 미반영. ralph-x 자동화의 submit 트리거 단계 누락 또는 Kaggle API 일시 거부 가능성.
- kernel 자체는 정상 (출력 다운로드 가능, submission.csv 형식 OK)

## 가설 (미검증 — 점수 없음)
"ConvNeXt-Base XCL fold0 3% 별도 weight"가 trial_058/059의 0.934 천장을 깨는가? — 검증 보류.

근거(예상 시나리오):
- **긍정**: trial_037에서 같은 ConvNeXt가 20% blend로 0.930(당시 baseline 동률)였음 → 5개 EffNet 변종 ensemble에 추가하면 ortho 신호가 더 명확. transformer-style 백본 + 9736-species XCL pretrain은 EffNet(BirdEffNet, n_mels=256)/Perch(audio embedding)와 입력 패딩·feature space가 모두 달라 logit-space에서 독립 정보 가능.
- **부정**: distill 5fold(같은 EffNet 백본, 다른 손실)도 별도 weight 3%로 0 효과. blend weight 3%는 LB granularity(~0.001) 안에서 신호 미반영일 수도. ConvNeXt single-fold(fold0)는 5fold 앙상블 대비 noise 큼 — sub_44/45 reflection의 "EffNet 단일 fold는 LB 무영향"과 같은 패턴이면 0.

## 교훈
- **자동 제출 트리거가 노이즈 변수**: ralph-x iter 10에서 처음으로 "kernel 정상이지만 Kaggle submission 미생성" 발생. iter 1~9는 모두 kernel push → 수분 내 submission 등장이었으나, 이번엔 90분 후에도 미등장. 워크플로우의 submit step이 어떤 메커니즘으로 작동하는지(자동 vs UI 클릭 vs CLI) 확인 필요. (4월 30일~5월 1일의 silent reject 시기와는 다름 — 그때는 submission은 생성되었으나 publicScore가 빈칸이었음. 이번엔 submission 자체가 생성 안 됨.)
- **kaggle CLI submit -k는 코드 컴페티션에서 400** — 일반적인 file upload용 submit 명령으론 코드 컴페티션 submission 생성 안 됨. UI 또는 다른 API 필요.
- **ConvNeXt-Base CPU 추론 wall 1.8min**: 예상 +10~20min의 하단. CPU에서 ConvNeXt-Base 234클래스 inference가 transformer-style이라도 합리적인 시간 — 향후 fold 추가(2~5fold)도 wall budget(9h) 내 여유 충분.

## 다음 가설 (ralph it.11+, 우선순위)
1. **[필수] submit 트리거 수동 처리** — kaggle.com에서 kernel v47 페이지의 "Submit to Competition" 액션 수동 실행. 자동화 워크플로우의 submit step 점검. 점수 회수 시 0.934±0이면 백본 가족 다양화도 ineffective로 결론 → 다음 axis 탐색. +0.001 이상이면 ConvNeXt fold0..4 확장 또는 weight 0.03→0.05~0.10 ramp.
2. **ConvNeXt 5fold 앙상블 확장** (점수 +라면): trial_039 시기에 fold0..4 학습 완료된 체크포인트(`birdclef2026-convnext-5fold/best_fold[0..4].pth`) 모두 활용. fold0 단일 3% → 5fold 평균 5~10% blend로 noise 감소.
3. **새 1차 모델 다른 축** (점수 0이라면):
   - **AudioMAE 또는 BEATs probe** (sub_51 reflection rec #1): Perch 외 다른 사전학습 오디오 임베딩 frozen probe 학습.
   - **EfficientNetV2-S Xeno-pretrain** (sub_51 reflection rec #2 idle): trial_049에서 보류, 학습 비용 필요.
   - **Perch 멀티윈도우 TTA** (sub_51 reflection rec #3): Perch 자체 정확도 강화 — 5초 ±shift 2~3회 평균. wall budget 위험 점검.
4. **submit automation 견고화**: ralph-x 워크플로우에 submission 등장 확인 + 누락시 알람·재시도 단계 추가. iter 10처럼 점수 못 받고 90분 낭비하는 케이스 방지.
