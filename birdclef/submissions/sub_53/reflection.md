# Sub 53 Reflection — trial_061 convnext_5fold

**Base**: trial_060 (Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 8% + distill 0% + ConvNeXt fold0 3%) — TIMEOUT, 점수 미확정
**Trial**: trial_061 kernel v48 (ralph-x iteration 11)

## 결과
- Public: **미확정 (TIMEOUT_90MIN)** — kernel v48 정상 COMPLETE했으나 Kaggle 신규 submission이 20:29~21:55 KST 90분 폴링(5분 간격 20회) 중 등장하지 않음. trial_060에 이어 ConvNeXt-axis 2연속 TIMEOUT.
- Private: N/A
- kernel v48 KernelWorkerStatus.COMPLETE 확인. submission CSV의 latest 행은 여전히 trial_059(2026-05-13 08:02). trial_061 행 자체가 생성되지 않음.

## 변경사항
- **ConvNeXt 컴포넌트 fold0 single → fold0..4 5-fold 평균으로 확장** (1:1 weight 유지):
  - Cell 65: `_ck_path = .../best_fold0.pth` → `_convnext_ck_paths = sorted(glob(.../best_fold*.pth))` (5개 매칭).
  - 단일 `_convnext_model` → `_convnext_models[]` 리스트, 5개 fold 각각 `ConvNextForImageClassification.from_pretrained(local_files_only=True) → _BirdConvNeXt(_clf) → load_state_dict(_sd, strict=True)` (per-fold from_pretrained으로 backbone 참조 공유 방지).
  - Inference loop: `convnext_logits[...] = _convnext_model(_spec).numpy()` → `_cnx_preds = np.stack([_m(_spec).numpy() for _m in _convnext_models]); convnext_logits[...] = _cnx_preds.mean(axis=0)` (pmix 패턴 그대로 차용).
  - 로깅: per-file ConvNeXt 5-fold avg progress wall 출력.
- Cell 66 fusion: trial-id label만 trial_060→trial_061. blend 식·weight·prior_mask 전부 동일 (Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 8% + distill 0% + ConvNeXt 3%).
- kernel-metadata.json: 변경 없음 (`ramkang/birdclef2026-convnext-5fold` + `denden12/birdset-convnext-base-xcl` 이미 dataset_sources).
- 모델·weight·downstream 후처리(온도/file-level/rank-aware/delta-shift) 전부 trial_060 그대로 → ConvNeXt fold-차원 신호만 격리

## 검증
- kernel v48 push 성공, KernelWorkerStatus.COMPLETE 확인
- ConvNeXt 5-fold ckpts 5개 (`ramkang/birdclef2026-convnext-5fold/best_fold[0..4].pth`) Kaggle dataset에 사전 존재 확인
- **Kaggle CSV 채점 행 미등장**: 5분 간격 20회 폴링(20:29, 20:34, ..., 21:55 KST) 동안 latest 행은 모두 trial_059(2026-05-13 08:02). trial_061 행 자체가 생성되지 않음.

## TIMEOUT 진단 (점수 미수령 원인, trial_060과 동일 증상)
- **kernel은 정상 COMPLETE** (KernelWorkerStatus.COMPLETE) but **Kaggle submission 행 미생성**.
- trial_060 reflection의 진단(="kaggle competitions submit -k -v 47 → 400 Bad Request, 코드 컴페티션 submission은 kernel push만으론 트리거되지 않음, kernel 페이지의 'Submit to Competition' 액션이 hidden test 마운트 후 재실행을 만들어야 채점 생성됨")이 그대로 재현. 워크플로우의 자동 submit 트리거가 ConvNeXt 컴포넌트가 들어간 kernel에서만 누락되는 패턴 — 동일 버전 카운터 v47/v48에서 연속 발생.
- 대조: trial_058 v45, trial_059 v46는 정상적으로 submission 행 생성·채점됨 (수분 내). ConvNeXt 추가 이후 두 번 연속 누락 → 데이터셋(`birdset-convnext-base-xcl` 또는 `birdclef2026-convnext-5fold`) 마운트가 코드 컴페티션 hidden re-run 단계에서 거부될 가능성? 또는 v47/v48 모두 ConvNeXt-Base CPU 추론으로 wall 늘어 hidden test inference 시 timeout 처리되어 submission 생성 단계까지 못 가는 가능성.
- ralph it.10/it.11 두 번 다 90분 폴링 만료. 점수 회수는 수동 UI 트리거 필요.

## 가설 (미검증 — 점수 없음)
"ConvNeXt-Base XCL fold0..4 5-fold 평균 3% 별도 weight"가 trial_058/059의 0.934 천장을 깨는가? — 검증 보류.

근거(예상 시나리오):
- **긍정**: pmix 컴포넌트의 fold 확장 패턴(trial_051 2-fold neutral → trial_053 4-fold +0.001)이 ConvNeXt에도 적용된다면, fold0 single noise 너머 ConvNeXt-XCL 9736-species pretrain의 ortho 신호가 LB에 register 가능. trial_037에서 같은 ConvNeXt가 20% blend로 0.930 동률 → 5개 EffNet 변종 ensemble에 3% 보조로 추가하면 distill(같은 EffNet 백본)보다 강한 가족 diversity.
- **부정**: trial_060 fold0 single이 TIMEOUT이라 5-fold 확장의 한계 효용을 비교할 baseline 자체가 없음. distill 5fold도 0.03 weight에서 0 효과 — 3pp 슬롯이 LB 0.001 granularity에서 의미 없는 가능성. ConvNeXt 입력 mel 스케일(n_mels=128 XCL pretrain)이 EffNet(n_mels=256)/Perch(audio embedding) 대비 다른 도메인 → ensemble 시 logit magnitude calibration 불일치로 신호 묽어짐 가능.

## 교훈
- **2연속 TIMEOUT으로 ConvNeXt-axis 운영 불가 확정**: ralph it.10/it.11 모두 동일 증상 (kernel COMPLETE + Kaggle submission 행 미생성). 워크플로우의 ConvNeXt 컴포넌트 추가 후 자동 submit 트리거가 작동하지 않음. trial_060 reflection의 "[필수] submit 트리거 수동 처리" 권장이 it.11에서도 자동화 없이 재현 — automation 정책 결정 없이는 ConvNeXt 신호 검증 자체가 막힘.
- **fold 확장 자체는 코드 검증됨**: pmix 패턴 그대로 차용(per-fold from_pretrained, np.stack mean), 메모리·wall 모두 안전 범위. kernel 정상 COMPLETE → 코드 결함은 아님. 채점 단계의 인프라 이슈로 확정.
- **2연속 ConvNeXt-axis 시행으로 ralph 비용 30분+** (kernel 두 번, 폴링 두 번 90분씩) — 채점 미발생을 일찍 감지하는 단계(예: kernel COMPLETE 후 10분 내 submission CSV에 행이 생기지 않으면 즉시 fallback)가 필요. 현 워크플로우는 90분 만료까지 기다림.
- **submission 채점 행 vs kernel worker COMPLETE 분리 모니터링 필요**: 워커가 정상 종료해도 코드 컴페티션의 hidden test re-run으로 별도 submission 행이 만들어진다는 점이 trial_060/061에서 명확. ralph-x가 kernel COMPLETE만 보고 "성공"으로 판단하면 안 됨.

## 다음 가설 (ralph it.12+, 우선순위)
1. **[필수] ConvNeXt-axis 잠정 동결 + trial_058(0.934) 베이스 복귀** — ConvNeXt 컴포넌트 cell 65/66 둘 다 제거, fusion 식을 trial_058 byte-identical로 복원(Perch 72% + EffNet5fold 15% + mwf0 5% + pmix 8%). 자동 submit 인프라가 ConvNeXt 데이터셋·wall과 호환되는 형태로 정리될 때까지 다른 축에 자원 집중.
2. **EffNet 백본 내부 다양성 강화 (1차 후보, ralph 비용 0)**:
   - sub_51 reflection rec #3 재시도 — **Perch 멀티윈도우 TTA** (5초 ±shift 2~3회 평균, Perch logit 자체 정확도 강화). wall budget 점검: 현재 kernel v48 기준 Perch ONNX inference가 가장 빠른 컴포넌트이므로 +2x shift TTA도 +30~60s 정도, 9h 한도 내 여유.
   - **pseudo-mix 5-fold weight ramp** (8% → 10~12%): trial_054에서 5-fold 자체는 동률이었으나 weight 인상은 미검증. EffNet 28% 예산 내 재배분으로 격리 가능.
3. **새 1차 모델 다른 백본 (학습 비용 있음, 보류)**:
   - sub_51 reflection rec #1 — **AudioMAE/BEATs probe**: Perch 외 사전학습 임베딩 frozen probe. ralph-x로는 학습 비용 처리 불가, 별도 세션에서 학습 후 컴포넌트로 추가.
4. **자동 submit 트리거 인프라 점검 (ralph 외부, 운영 작업)**:
   - kaggle CLI submit API 400 원인 조사 (코드 컴페티션 정상 호출 방법) — kernel push 후 "Submit to Competition" UI 액션의 API equivalent 찾기.
   - ralph-x 워크플로우에 kernel COMPLETE → 10분 timeout submission CSV 검증 단계 추가, 미생성 시 즉시 UI 알람·재시도.
5. **ralph it.12는 위 #2 중 Perch 멀티윈도우 TTA 우선** — kernel 변경 단순, 자동 submit 트리거 정상화 확인용으로도 활용 (ConvNeXt-free 베이스에서 자동 제출 재현되면 인프라 이슈 격리 확정).
