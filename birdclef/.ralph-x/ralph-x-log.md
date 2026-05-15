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
- kernel COMPLETE 확인: KernelWorkerStatus.COMPLETE. 정상 종료.
- **TIMEOUT_90MIN (2연속)**: kernel COMPLETE 후 90분 폴링(20:29~21:55 KST, 5분 간격 20회) 동안 Kaggle 신규 submission 미등장. latest 행은 여전히 trial_059(2026-05-13 08:02, 0.934). trial_060 v47에 이어 v48도 동일 증상 — kernel 정상이지만 "Submit to Competition" 자동 트리거 누락. score 미수령으로 가설 검증 보류.
- 진단: trial_058 v45 / trial_059 v46는 정상 채점됐으므로 단순 API 글리치가 아닌 ConvNeXt 컴포넌트 추가 후 패턴화된 현상. 코드 컴페티션 hidden re-run 단계에서 ConvNeXt 데이터셋(`birdset-convnext-base-xcl` + `birdclef2026-convnext-5fold`) 마운트 거부 또는 wall 증가로 timeout 처리되어 publicScore 생성 단계 미도달 가능. 어느 쪽이든 ralph-x 자동 워크플로우로는 ConvNeXt-axis 검증 불가 확정.
- 결론: 2연속 TIMEOUT으로 ConvNeXt-axis 잠정 동결. ralph 비용 60분+ 낭비(2회 90분 폴링).
- 다음(it.12) 방향: ① ConvNeXt 컴포넌트 cell 65/66 제거, trial_058(0.934 best 동률) 베이스로 회귀 — 자동 submit 트리거 정상 작동 재확인. ② 그 위에서 새 축 시도 1순위 = Perch 멀티윈도우 TTA (sub_51 rec #3, kernel 변경 작고 자동 제출 트리거 정상화 확인용으로도 활용). ③ ralph-x 워크플로우 개선: kernel COMPLETE → 10분 timeout submission CSV 검증 단계 추가, 미생성 시 즉시 fallback. 자세한 가설은 sub_53 reflection.md.

## iter 12: trial_062 pmix_weight_up
- 시작: 2026-05-13 KST (delegated iter 4)
- 전략: sub_53 reflection it.12+ rec #1·#2 실행 — ConvNeXt-axis 잠정 동결(trial_060/061 2x TIMEOUT 회수) 후 freed 3pp 슬롯을 검증된 5-fold pseudo-mix 컴포넌트에 재배분. BLEND_PSEUDOMIX 0.08→0.11. mwf0 0.02 (trial_059 유지), distill·convnext 모두 0. Perch 72% 고정.
- 근거: trial_056 (mwf0 0.07→0.05 / pmix 0.06→0.08) 직선 방향을 한 단계 더 연장(=pmix weight up). pmix는 5-fold 검증된 컴포넌트, ConvNeXt는 검증 불가. 동시에 ConvNeXt 데이터셋 2개(`ramkang/birdclef2026-convnext-5fold`, `denden12/birdset-convnext-base-xcl`) 를 kernel-metadata 에서 제거 — sub_53 reflection 의 'ConvNeXt 데이터셋 마운트가 코드 컴페티션 hidden re-run에서 거부되어 auto-submit 트리거 누락' 가설 정면 테스트. 본 trial 채점이 정상 등장하면 가설 확정.
- 구현: Cell 62 `BLEND_PSEUDOMIX = 0.08` → `0.11` (코멘트 갱신). Cell 65 `BLEND_CONVNEXT = 0.03` → `0.0` 상단에서 조기 차단 — ConvNeXt 로드/추론 자체가 if 분기로 skip. Cell 66 헤더 코멘트+print label trial_061→trial_062 (수식은 byte-identical, BLEND_CONVNEXT*x 항은 0*x로 죽음). `kernel-metadata.json` ↔ `effnet-blend-kernel-metadata.json` 동기화 후 push (ConvNeXt 2개 dataset 제거).
- 검증: 변경 cell 3개 모두 `ast.parse` OK. Perch share = 1 - 0.15 - 0.02 - 0.11 - 0 - 0 = 0.72 ✓ (수식 직접 계산). 모델 가중치/prior_mask/postproc 전부 trial_058 byte-identical.
- kernel v49 push 완료 ("Kernel version 49 successfully pushed"). 채점 PENDING.
- 위험: 1) trial_056 → 0.08 변화 무효(trial_054 동률)였음 — 0.08 → 0.11도 동률 가능성. 2) pmix 11% 가 5-fold pmix 의 calibration 한계 초과시 -0.001 회귀 가능. 3) ConvNeXt 데이터셋 제거가 auto-submit 트리거 정상화에 영향 없으면(=무관 가설) 인프라 이슈가 여전히 미해결 — 그래도 본 trial 은 ConvNeXt 미사용으로 trial_058 v45/trial_059 v46 패턴(정상 채점)에 합류해야 함.

## iter 12: trial_062 pmix_weight_up 결과
- 제출: 2026-05-15 09:51:50 UTC, 채점: 2026-05-15 11:10 UTC (78분 소요 — Kaggle 서버 지연)
- score: **0.934** (best 동률)
- BLEND_PSEUDOMIX 0.08→0.11 (+3pp): 무효과. pmix weight 포화 확인.
- ConvNeXt dataset 제거 → auto-submit 정상화 확인 ✓ (trial_060/061 TIMEOUT 원인 확정)
- 다음(iter 13): trial_063 mwf0_zero_pmix13 (kernel v50 COMPLETE)

## iter 13 — trial_063 mwf0_zero_pmix13 (2026-05-15)

- 제출: 2026-05-15 11:12:27 UTC, 채점: 2026-05-15 12:28 UTC (76분 소요)
- score: **0.933** ❌ (-0.001 vs best 0.934)
- mwf0 완전 제거(0.02→0.00) + pmix 2pp up(0.11→0.13) → 역효과
- **mwf0 noisy 가설 기각**: mwf0이 실제 신호 기여 확인됨
- pmix 방향 완전 소진: 0.08→0.11→0.13 모두 0.934 이하
- 다음(iter 14): trial_064 EffNet weight 0.15→0.17, Perch 0.72→0.70 (mwf0/pmix 원복)

## iter 14 — trial_064 effnet_weight_up (2026-05-15)

- 제출: 2026-05-15 12:37:19 UTC, 채점: 2026-05-15 13:44 UTC (67분 소요)
- score: **0.934** ➖ (best 동률)
- EffNet 15%→17%, Perch 72%→70% → LB 무반응
- EffNet weight 방향 포화 가능성 확인
- 다음(iter 15): trial_065 EffNet 0.17→0.19, Perch 0.70→0.68 (방향 포화 최종 검증)

## iter 15 — trial_065 effnet_further_up (2026-05-15)

- 제출: 2026-05-15 13:51:11 UTC, 채점: 2026-05-15 14:58 UTC (67분 소요)
- score: **0.933** ❌ (-0.001 vs best 0.934)
- EffNet 17%→19%, Perch 70%→68% → 하락
- **EffNet weight↑ 방향 역효과 확정**: 0.15=baseline, 0.17=동률, 0.19=하락
- 다음(iter 16): trial_066 원복(EffNet 0.15, Perch 0.72) + mwf0 0.02→0.03 증량 시도
