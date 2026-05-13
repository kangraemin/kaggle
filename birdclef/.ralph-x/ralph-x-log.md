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

## iter 10: trial_060 convnext_add
- 시작: 2026-05-13 KST (재개)
- 전략: sub_51 reflection it.10+ rec #2 실행 — ConvNeXt-Base XCL fold0를 6번째 컴포넌트로 추가. trial_059의 zero-effect distill 3% 슬롯에 1:1 치환. ConvNeXt = transformer-style 백본 + 9736-species XCL pretrain → EffNet/Perch와 완전히 다른 가족.
- 근거: trial_054~059 6연속 0.934 — EffNet 28% 예산 내 모든 재분배·손실함수 다양화 실패. 백본 가족 다양화는 미시도. trial_037에서 ConvNeXt 20% blend는 0.930 = 당시 baseline 동률(0 효과)이지만 현재는 5개 EffNet 변종 ensemble이라 ConvNeXt의 ortho 신호가 더 명확히 드러날 가능성.
- 구현: NEW cell 65에 transformers.ConvNextForImageClassification.from_pretrained(local_files_only=True) + 커스텀 head(1024→234) 로더+inference 추가. 입력은 mwf0/pmix와 같은 _mwf0_spec(n_fft=1024, n_mels=128, top_db=80, XCL norm). fusion cell 66: BLEND_DISTILL 0.03→0, BLEND_CONVNEXT 0→0.03. Perch 72% 그대로(distill swap이므로 share 불변).
- kernel v47 RUNNING (push 직후), 채점 PENDING.
- 위험: 1) transformers 라이브러리 의존(Kaggle base image 기본 포함), 2) ConvNeXt-Base CPU 추론 wall +10~20분 예상, 3) 과거 silent reject(4/30~5/1)는 플랫폼 이슈로 5/4 이후 정상화.
- kernel COMPLETE: 2026-05-13 18:28 KST (wall 445s = 7.4min, +185s vs trial_059. ConvNeXt fold0 추론 1.8min 정상). submission.csv 240×235, mean 0.0432, range 0.000~0.997. log: `ConvNeXt fold0 loaded (best_fold0.pth)`, `ConvNeXt fold0 inference done: 1.8min`, blend `Perch 72% + EffNet5fold 15% + fold0-B0 2% + pmix 8% + distill 0% + convnext 3%`. → ConvNeXt 컴포넌트 자체는 정상 통합.
- **TIMEOUT_90MIN**: kernel COMPLETE 후 90분 폴링(18:34~20:00 KST, 5분 간격 18회) 중 Kaggle 신규 submission 미등장. latest는 trial_059(0.934)에 고정. `kaggle competitions submit -k <kernel> -v 47` API 호출 → **400 Bad Request**. score 미수령으로 가설 검증 실패(보류).
- 추정 원인: 코드 컴페티션에서 kernel push만으론 채점 트리거 안 됨. "Submit to Competition" UI 클릭이 hidden test 마운트 후 재실행을 만들어야 publicScore 생성. ralph-x 자동화의 submit 단계가 누락된 듯. trial_058/059는 정상 작동했으나 trial_060만 미반영.
- 다음(it.11) 방향: ① 수동 "Submit to Competition" 트리거로 점수 회수 → ② 점수 +면 ConvNeXt 5fold 확장 / weight 0.03→0.05~0.10 ramp, 0이면 AudioMAE/BEATs probe 또는 Perch 멀티윈도우 TTA (sub_52 reflection 참조). ③ ralph-x 워크플로우에 submission 등장 확인 + 누락시 알람·재시도 단계 추가.

## iter 11: trial_061 convnext_5fold
- 시작: 2026-05-13 KST (delegated iter 3)
- 전략: trial_060 ConvNeXt fold0 single 3% (TIMEOUT, 점수 미확정) → fold0..4 5-fold 평균 (BLEND_CONVNEXT=0.03 동일). pmix fold-expansion 패턴(trial_051 2-fold neutral → trial_053 4-fold +0.001)을 ConvNeXt 컴포넌트에 그대로 적용 + trial_060 timeout 복구 시도.
- 근거: trial_060 kernel v47은 정상 완료(wall 445s, submission.csv 정상), Kaggle 신규 submission만 등장 안 함 — 일회성 API 400 추정. 새 push(v48)로 재시도하면서 동시에 single-fold noise 제거 효과 측정. 5개 ConvNeXt fold 모두 `ramkang/birdclef2026-convnext-5fold` 데이터셋 안에 이미 존재 (`kaggle datasets files` 확인, 각 351MB).
- 구현: cell 65 단일 fold 로더 → 5-fold 리스트 로더 (per-fold `ConvNextForImageClassification.from_pretrained + load_state_dict(strict=True)` — backbone reference 공유 방지). 추론은 pmix와 같은 `np.stack([m(_spec).numpy() for m in models]).mean(axis=0)` 패턴. cell 66 trial-label만 trial_060→trial_061. fusion 식·weights·prior_mask 전부 byte-identical. kernel-metadata.json 변경 없음.
- 검증: 양 cell `ast.parse` OK. dataset 5-fold 존재 확인. 메모리 ~2GB peak·wall 12-13min 예상 — 모두 한도 내.
- kernel v48 push 완료 ("Kernel version 48 successfully pushed"), 채점 PENDING.
- 위험: 1) trial_060 같은 timeout 재현 가능성 (단발 API 글리치였길 기대), 2) 5-fold avg가 noise뿐 아니라 약한 ortho 신호도 깎을 가능성, 3) backbone-family 다양화가 0.934 천장에서 무력하면 다음 iter는 다른 축 (Perch multi-window TTA 또는 Perch share 축소).
