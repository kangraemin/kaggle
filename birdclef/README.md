# BirdCLEF 2026

> 5초 오디오 클립에서 **새 종(bird species)의 존재 여부**를 예측하는 멀티라벨 분류 대회.
> 지표는 **macro-averaged ROC-AUC**(종마다 ROC-AUC를 따로 구해 평균, true positive가 없는 종은 스킵). 코드 대회(노트북을 hidden test 위에서 재실행해 채점, csv 직접 업로드 아님). 일일 제출 5회 제한.

**최종 성적**

| 항목 | 값 |
|---|---|
| 🏅 public 순위 | **398 / 4085 팀 (상위 9.7%)** |
| public best | **0.950** (공개 EoS.9 fork) |
| private best | **0.94238** (eos9-all, 다축 직교 조합) |
| 자체 파이프라인 천장 | public 0.938 / private 0.93608 |
| top1 (참고) | Nikita Babych 0.96720 |

---

## TL;DR

자체 Perch 임베딩 + 트리 모델로 로컬 검증(OOF) 0.97을 찍었지만 제출은 0.0/0.91로 무너지며 **"로컬 val이 곧 LB"라는 환상**부터 깨졌다. 검증된 0.912 공개 노트북을 **fork**해 첫 유효 제출을 만들고, ONNX 가속으로 코드 대회의 진짜 적인 타임아웃을 뚫고, mel-spec CNN(EffNet)을 작은 비중으로 섞어 **0.912 → 0.929 → 0.930**까지 올렸다. 거기서 distillation·pseudo-label·ConvNeXt·SoftAUC를 다 시도했지만 약한 보조 모델은 강한 Perch를 묽게 할 뿐이라 **0.930 천장**에 막혔고, EffNet 멀티윈도우와 pseudo-mix를 작게 누적해 **0.934**까지 짜냈으나 이후 weight를 8방향으로 흔든 11개 trial이 전부 0.934 동률이었다(자체 천장). 공개 0.947 ProtoSSM+SED 파이프라인으로 갈아타 **0.938**까지 갔지만 또 weight/scale/컴포넌트/추론 4축이 전부 포화했다.

**결정적 분기 ①** — "0.938은 대회의 한계가 아니라 우리 3-컴포넌트 접근의 한계"라고 인정하고, 파라미터 튜닝을 버리고 공개 메가앙상블 **EoS.9를 fork**해 단번에 **0.950(+0.012)** 돌파. **결정적 분기 ②** — public이 전부 0.950 동률이라 우열을 못 가리는 상황에서, 각각은 public 무효였지만 private에서 미세하게 +였던 **직교 보정 3축(sword·SED45·균등블렌드)을 한꺼번에 조합한 변형(eos9-all)을 final로 제출** → 마감 후 public 최고 픽(sword, 0.95087)이 private 최저(0.94138)로 진 반면, eos9-all이 **private 최고 0.94238**로 이겼다.

---

## 점수 진행 그래프 (best public score, 시간순)

```
0.950 ┤                                                          ●  ← EoS.9 fork (천장 돌파)
0.945 ┤
0.940 ┤
0.938 ┤                                              ●              ← ProtoSSM+SED (자체 파이프라인 천장)
0.935 ┤                                          ●
0.934 ┤                                  ●
0.933 ┤                              ●
0.932 ┤                          ●
0.930 ┤                  ●
0.929 ┤              ●
0.928 ┤      ●   ●
0.912 ┤  ●
      └──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─────────────
        912 928 929 930 932 933 934 935 938 950
```

| 단계 | best public | 무엇이 깼나 |
|---|---|---|
| 첫 유효 제출 | 0.912 | 0.912 공개노트북 fork |
| → 0.928 | 0.928 | 0.926 fork(우연한 재학습) / ONNX 타임아웃 해결 |
| → 0.929 | 0.929 | EffNet mel-spec CNN 8% 블렌드 (직교 신호) |
| → 0.930 | 0.930 | Perch→EffNet knowledge distillation |
| → 0.932 | 0.932 | CE fold0 EffNet 5% 추가 (3-way) |
| → 0.933 | 0.933 | pseudo-mix fold0 3% 추가 (4-way) |
| → 0.934 | 0.934 | pseudo-mix 4-fold 앙상블 (자체 weight 천장) |
| → 0.935 | 0.935 | 0.947 ref config weight 그대로 복사 |
| → 0.938 | 0.938 | SED 비중 18%→40% (자체 파이프라인 천장) |
| → 0.950 | **0.950** | **공개 EoS.9 메가앙상블 fork** |

---

## 대회 개요

- **Task**: 5초 단위 오디오 윈도우마다, 234개 후보 종 각각의 존재 확률을 출력하는 **멀티라벨 분류**. soundscape(자연 환경을 그대로 녹음한 장시간 음원)를 5초씩 잘라 추론한다.
- **지표**: **macro-averaged ROC-AUC** — 종별로 ROC-AUC를 따로 구해 평균. true positive가 한 건도 없는 종은 평균에서 제외(skip-empty). **rank 기반 지표**라, 모든 점수에 같은 상수를 더하거나 곱하는(monotonic/affine) 변환은 순위를 안 바꿔 점수에 영향이 없다.
- **코드 대회 특성**: csv를 직접 올리는 게 아니라 **노트북을 hidden test 위에서 재실행**해 채점한다. 그래서 "kernel COMPLETE = 채점 성공"이 아니며, 외부 dataset 마운트가 재실행 단계에서 거부되거나(silent reject), 제한 시간(~120분)을 넘으면 0점이다.
- **근본 난관 3가지**:
  1. **라벨이 적은 soundscape** — 학습 라벨 대부분이 단일 종 위주의 짧은 클립(focal recording)인데, test는 여러 종이 겹치는 현장 녹음(soundscape)이라 도메인 갭이 크다. 게다가 234종 중 28종은 train 샘플 0개(영구 0점), test는 사실상 브라질 Pantanal 단일 사이트였다.
  2. **로컬 val ↔ hidden test 괴리** — 검증셋 구성이 hidden test와 달라 OOF가 LB를 거의 예측하지 못했다(자체 OOF 0.97 → 제출 0.0/0.91). **검증은 오직 Kaggle 직접 제출**로만 가능했다.
  3. **추론 타임아웃** — hidden test가 대회 진행 중 점점 커져, 4/4 파일에서 성공한 코드가 4/5에서 타임아웃. ONNX 가속(TF 의존성 제거, 2x)이 이를 뚫은 결정적 한 수.

---

## Era별 실험사

## Era 1 — 임베딩 프로브에서 첫 fork까지 (sub 01–14, trial 001–025)

**한 줄 요약**: 자체 Perch 임베딩(오디오를 1536차원 벡터로 바꾸는 사전학습 모델) + 트리 모델로 OOF(검증셋 예측) 0.97을 찍었지만 제출은 0.0/0.91로 무너졌다 → "로컬 검증이 곧 LB 점수"라는 환상을 버리고 검증된 공개 노트북을 fork → ONNX 가속으로 타임아웃을 뚫고 → EffNet CNN을 8% 블렌드해 0.929까지. **시작 0.912 → 끝 0.929 (public)**. 핵심 교훈은 단 하나: 이 대회는 로컬 val이 LB와 거의 무관하고, 검증은 오직 Kaggle 직접 제출로만 가능하다.

> 용어 풀이
> - **OOF(out-of-fold)**: 교차검증에서 학습에 안 쓴 데이터로 만든 예측. 보통 LB의 대리 지표지만, 이 대회에선 검증셋 구성이 hidden test와 달라 거의 무의미했다.
> - **Perch**: Google의 새소리 임베딩 모델. 오디오 → 고차원 벡터(또는 14,795종 logit)로 변환.
> - **fork**: 남이 공개한 Kaggle 노트북을 복제해 내 계정에서 돌리는 것.
> - **ONNX**: 모델을 프레임워크 독립 포맷으로 바꿔 CPU 추론을 빠르게 하는 런타임.
> - **타임아웃**: 코드 대회는 hidden test 위에서 노트북을 재실행하며, 제한 시간(여기선 ~120분)을 넘으면 0점 처리.

### Trial별 상세

| trial | 왜 시도(가설/문제) | 무엇을 바꿈 | public | private | 결과·원인 |
|---|---|---|---|---|---|
| 001 perch_lgbm | "Perch 임베딩 + LightGBM이면 baseline 충분할 것" | Perch v2 임베딩 → LightGBM | val 0.8375 | - | baseline 확보. 제출 못 함(파이프라인 미완) |
| 002 with_soundscape | "soundscape(현장 녹음) 데이터를 더 넣으면 val이 오를 것" | soundscape 1478개 추가 | val 0.8731 | - | val +0.036. 학습데이터 양 효과 확인, 단 val 자체가 신뢰 불가 |
| 003 ensemble | "XGB가 LGBM보다 강하고 PCA로 차원 줄이면 더 좋을 것" | XGBoost, PCA 1536→512 | val 0.9709 | - | val 0.97 폭등 **but 제출 실패** — 이 0.97이 바로 함정(검증셋 누수성 과적합) |
| 004~006 probe들 | "PCA 차원·정규화(C)·트리 깊이를 튜닝하면 더 오를 것" | LR+PCA64, PCA sweep, XGB 튜닝 | val 0.954~0.975 | - | val은 다 0.95+. **로컬 최적이 LB로 이어진다는 보장 전혀 없음을 이때는 몰랐다** |
| **007 perch_probe_bayesian (sub 01)** | "자체 파이프라인은 너무 느리고 불안정 → 검증된 0.912 공개노트북을 fork하면 첫 유효 제출이 될 것" | 0.912 공개노트북 fork (Perch logit 직접 사용 + Bayesian site×hour prior + Gaussian smoothing) | **0.912** | - | ✅ **첫 유효 제출**. 자체 Perch CPU 추론은 90분 제한 초과·TF 2.20 의존성·row_id 포맷 문제로 v1~v13 전부 삽질. OOF 0.487(soundscape 59파일뿐이라 과소평가)이 실제 LB와 무관함이 드러남 |
| 008 post_processing (sub 02) | "2025 솔루션의 temperature/file-level/rank-aware 후처리를 얹으면 오를 것" | 후처리 3종 추가 (검증 없이) | 0.910 | - | ❌ **-0.002 악화**. 로컬에서 temperature만 보면 0.9754→0.9754 무변화라 "안전"하다 착각하고 제출. **OOF 검증 없이 제출한 실수 + 2026 데이터 특성이 2025와 달랐음** |
| 013 param_change_v18 (sub 03) | "로컬 trial_009에서 PCA96+C0.1이 best(0.9766)였으니 LB도 오를 것" | PCA96, C=0.1 | 0.904 | - | ❌ **-0.008 하락**. **간소화된 로컬 파이프라인(단순 LR)의 최적값이 Kaggle 복잡 파이프라인에선 역효과** — 파이프라인 불일치 = 무의미한 튜닝 |
| 014 full_upgrade_v19 (sub 03) | "MLP+TTA+후처리 전부 얹으면 강해질 것" | MLP + TTA 5x + 후처리 | - (타임아웃) | - | ❌ 타임아웃. test 600파일×12윈도우에 **TTA 5배 + per-class for-loop MLP** = 시간 초과. 벡터화 추론 필수 교훈 |
| **015 fork_926 (sub 04)** | "0.912가 천장 → 더 강한 0.926 공개노트북(ProtoSSM v5 + 벡터화 MLP)을 fork하면 베이스가 점프할 것" | yukiZ 0.926 fork 그대로 | **0.928** | - | ✅ **new best, +0.002**. 역설적 행운: **dataset 마운트 누락으로 pretrained weight를 못 불러와 노트북 안에서 처음부터 재학습** → seed/셔플 차이로 원본보다 좋은 모델이 나옴. multi-seed 앙상블 가능성 시사 |
| 016 fork_926_v4 (sub 05) | "multi-seed(5개) 앙상블 + epoch120 + isotonic으로 0.928을 넘을 것" | 5-seed + epoch120 + PCA192 + isotonic | - (채점 실패) | - | ❌ **빈 모델 학습 실패**. `train_proto_ssm_single`이 `ProtoSSM_PATH != None`이면 로드만 시도→파일 없으면 빈 모델 return → 학습 자체가 안 됨 |
| 017 fork_926_v7_multiseed (sub 06) | "웹에서 직접 ProtoSSM_PATH=None 강제하면 학습될 것" | 동일 multi-seed, 웹 수정 | - (점수 없음) | - | ❌ submission.csv 미생성 추정. API push든 웹 수정이든 ProtoSSM 학습 파이프라인 자체가 깨짐 |
| 018 seed_variant (sub 07) | "단순 seed만 42→1891로 바꾸면 안전할 것" | seed 변경 | - (Timeout) | - | ❌ **편집한 노트북이 016/017의 multi-seed 코드가 남은 오염 버전**. 5모델 학습+추론으로 2h 초과. dry-run(20파일) 22분 ≠ 실제 채점. "최신 버전이 이전 실패의 잔해일 수 있다" |
| 018b seed_variant_clean (sub 08) | "clean하게 seed1891만 + 예전 0.928 Version1 재제출하면 될 것" | clean seed + V1 재제출 | - (Timeout) | - | ❌ **0.928 나왔던 바로 그 V1도 타임아웃**. 진범 = **hidden test가 대회 진행 중 점점 커짐**. Perch CPU ~10초/파일 × 600파일 = ~100분 → 학습시간 더하면 120분 초과. 4/4 성공이 4/5 실패 |
| 019 tflite_speedup (미제출) | "TFLite INT8 양자화로 CPU 추론 가속" | TFLite INT8 | - (OOM) | - | ❌ **TFLite+SELECT_TF_OPS가 TF 전체 런타임을 메모리에 올려 16GB OOM**. 방향 폐기 |
| **020 onnx_perch (sub 09)** | "ONNX Runtime이면 TF 의존성 없이 추론 2배 빨라 타임아웃이 풀릴 것" | Perch SavedModel→ONNX (HF justinchuby/Perch-onnx), onnxruntime wheel을 dataset에 포함 | **0.928** | - | ✅ **타임아웃 해결 + best 유지**. SavedModel 0.64s→ONNX 0.31s/window(2x). **HF에 이미 변환본이 있어 직접 변환 불필요**. 이 ONNX 파이프라인이 이후 모든 실험의 토대 |
| 021 093_onnx_fork (sub 10) | "0.93 공개노트북(a4dc68)을 fork+ONNX하면 더 오를 것" | 다른 0.93 노트북 fork + ONNX | 0.925 | - | ❌ **-0.003**. **ONNX 변환이 Perch 출력에 미세 차이를 만들어**, 원본 모델 출력에 맞춰진 hardcoded per-class threshold가 ONNX 출력엔 안 맞음. "남의 threshold 복붙은 모델 출력이 다르면 무의미" |
| 022 full_upgrade (sub 11) | "V18 파라미터(d_model↑, layers↑) + audio feature(energy/ZCR/spectral)면 오를 것" | V18 CFG + audio FE + energy weighting | 0.928 | - | ➖ **무변화**. dry-run 20파일론 모델이 제대로 학습 안 됨. **audio feature는 Perch 임베딩이 이미 캡처해 redundant**. 파라미터 튜닝만으론 0.928 벽 못 넘음 |
| **023 effnet_blend (sub 12)** | "완전히 다른 feature space(mel-spectrogram CNN)를 섞으면 앙상블 다양성으로 오를 것" | EfficientNetV2-B0 1-fold + Perch 92:8 weighted avg | **0.929** | - | ✅ **new best, +0.001**. **mel-spec vs Perch-embedding = 직교 신호**라 1-fold만으로도 개선. 8% 비중이 적절(1-fold 품질 낮아 더 높이면 역효과). timm↔torchvision state_dict 키 구조 다름 주의 |
| 024 effnet5fold_lse (sub 13) | "5-fold + LSE pooling(시간축 log-sum-exp)으로 CNN을 더 강하게" | EffNet 5-fold + LSE 추론 + blend 0.10 | 0.922 | - | ❌ **-0.007**. **학습-추론 불일치**: head는 global-pool feature(1280-d)로 학습됐는데 추론은 forward_features→temporal(8×1280)을 먹임 → 분포가 달라 출력 붕괴. "LSE 효과 보려면 LSE로 처음부터 학습해야" |
| 025 effnet5fold_global (sub 14) | "LSE 제거하고 global pool로 복구하면 5-fold 단독 효과만 볼 수 있을 것" | LSE 제거, global pool, blend 0.08 | 0.929 | - | ➖ **best 동률, 0.922→0.929 회복**. **5-fold 단독은 public 무효과**(1-fold도 0.929). 분산 감소가 public엔 안 드러남. LSE가 -0.007의 진범이었음 재확인 |

(이 Era trial들은 모두 대회 종료 후 공개된 private 점수 맵의 "최근 50건" 범위(2026-05-04 이후) 밖이라 private 컬럼은 전부 "-". 비교 가능한 private은 Era 2 이후부터.)

### 이 Era에서 배운 핵심 교훈

- **로컬 val ≠ LB, 이 대회에선 OOF가 거짓말을 한다.** 자체 파이프라인 OOF는 0.97인데 제출은 0.0/0.91. 검증셋이 hidden test와 구성이 달라(soundscape 소수 파일 기준) 과적합·과소평가가 동시에 일어났다. trial_013의 -0.008이 결정타: **간소화된 로컬 파이프라인의 최적 하이퍼파라미터가 복잡한 Kaggle 파이프라인에선 역효과**. 결론 — 검증은 오직 Kaggle 직접 제출. 로컬 튜닝은 파이프라인이 100% 동일할 때만 의미.

- **"검증된 것에서 출발" > "처음부터 만들기".** 자체 Perch 추출은 속도·TF 버전·row_id 포맷으로 v1~v13 내내 실패. 0.912 fork 한 번으로 첫 유효 제출. 더 강한 0.926 fork로 점프(+행운의 재학습 +0.002). 바닥부터 짓는 비용 대비 검증된 baseline 위에서 차분(差分)만 바꾸는 게 압도적으로 효율적이었다.

- **코드 대회의 진짜 적은 정확도가 아니라 타임아웃, 그리고 hidden test는 대회 중 커진다.** sub 07/08의 4연속 타임아웃이 핵심 트라우마 — multi-seed 5모델 오염이 1차 원인, 더 근본적으론 **4/4에 0.928 성공한 바로 그 코드가 4/5엔 타임아웃**(test 파일 증가). dry-run(20파일) ≠ 채점(600+파일), 30배 차이. **ONNX Runtime(2x 가속, TF 의존성 제거)이 이 벽을 뚫은 결정적 한 수**였고, TFLite는 OOM으로 막다른 길.

- **앙상블 이득은 "직교하는 신호"에서만 나온다.** 파라미터 튜닝(trial_022)·audio feature·multi-seed·5-fold 같은 *같은 가족 내* 변형은 전부 0.928~0.929 무변화. 반면 **다른 feature space인 mel-spec CNN(EffNet)을 8% 섞자 +0.001**로 첫 돌파. 비중은 보조 모델 품질에 맞춰 작게(8%) — 1-fold CNN을 과신하면 강한 Perch를 묽게 해 역효과.

- **"이론상 같으니 괜찮다"는 추론-시점 트릭은 학습-추론 분포 불일치로 무너진다.** LSE pooling(trial_024)이 -0.007. head가 본 적 없는 feature 분포를 추론에서 먹이면 차원이 같아도 출력이 붕괴. 추론 구조를 바꾸려면 그 구조로 학습부터 다시 해야 한다.

---

## Era 2 — Knowledge Distillation·ConvNeXt·플랫폼 채점 장애 (sub 15~37, trial 028~044)

**한 줄 요약:** EffNet에 Perch 임베딩을 베껴 넣는 knowledge distillation(지식 증류, 강한 모델의 출력을 약한 모델이 모방하게 하는 학습)으로 0.930 best를 찍은 뒤, pseudo-label·HGNetV2·SoftAUC·ConvNeXt를 차례로 붙였지만 **모두 0.930 천장을 못 뚫었고**, 설상가상 4/30~5/4 Kaggle 채점 시스템 장애로 10연속 silent reject(채점 자체가 안 됨)를 맞았다. 이 Era는 "보조 모델 비중 키우기는 강한 Perch를 묽게 할 뿐"이라는 교훈과, ConvNeXt가 **추론 속도·데이터셋 마운트 거부**로 코드 대회에서 운영 불가라는 걸 비싸게 배운 구간이다. **시작 0.929 → 끝 0.930** (best는 갱신했으나 +0.001에 그침).

> 용어 빠른 정리: **public/private score** = 채점 지표 macro ROC-AUC (전 종 평균, 1.0이 만점). **silent reject** = 노트북은 COMPLETE인데 publicScore 칸이 비는 현상 = 채점 컨테이너가 hidden test 재실행에 실패. **blend** = 여러 모델 출력을 가중 평균. **OOF/CV** = 학습 데이터 내부 검증 점수(out-of-fold), LB와 다를 수 있음. **fold** = 데이터를 N등분해 교차검증한 모델들.

### Trial별 상세

| trial (sub) | 왜 시도 (직전 가설/문제) | 무엇을 바꿈 | public | private | 결과·왜 그 숫자가 나왔나 |
|---|---|---|---|---|---|
| **028 distill_5fold** (15) | "EffNet 5-fold 단독은 효과 없었다(0.929). Perch 임베딩을 EffNet이 베끼게 하면(distillation) CNN 표현력이 올라 blend가 개선될 것이다" | EffNet을 Perch 임베딩 L2-MSE로 재학습, BLEND 0.08 유지 | **0.930** | - | ✅ new best (+0.001). 같은 8% 비중에서 0.929→0.930 — distillation이 EffNet 품질을 실제로 높였다. 다만 강한 Perch가 92% 차지해 개선폭이 작음 |
| 029·030 blend_sweep (16·17) | "distill로 CNN이 강해졌으니 비중을 키우면 더 오를 것" | BLEND_EFFNET 0.08→0.10→0.15 | 0.930·0.930 | - | ➖ 동률. 0.08~0.15 전부 0.930 — Perch가 워낙 강해 EffNet 비중을 흔들어도 최종 점수에 영향 미미. **blend 미세조정은 무의미**로 확정 |
| **031 pseudo_5fold** (18) | "soundscape(현장 녹음) pseudo-label(모델이 자신있게 단 가짜 라벨)로 재학습하면 도메인 갭이 줄 것" | pseudo-label+distill 5-fold(23.8h 학습)로 교체, CV 0.9792 | 0.927 | - | ❌ -0.003. **CV 최고인데 LB 하락 = 전형적 CV-LB gap**. pseudo-label은 "기존 모델이 이미 맞히는 것"만 확신 라벨링 → 기존 bias를 self-reinforce(자기증폭). hidden test 일반화엔 역효과 |
| **032 protossm_attach** (19) | "sub15~18 내내 ProtoSSM weight가 못 붙어 fallback 0.5로 돌고 있었다. 실제 weight 붙이면 +0.005~0.01" | 작년(2026-04-04 학습) proto_ssm + residual_ssm weight 실제 로드 (residual mean_abs 0.45) | 0.929 | - | ❌ -0.001. weight가 붙긴 했으나 **작년 학습본이라 올해 test 도메인에 안 맞음** + residual이 max 1.06까지 overshoot(과보정). "로드 경고 사라짐 ≠ weight 품질 좋음" |
| **033 yusuf_baseline** (20) | "EffNet 기여도를 측정하려 외부 V18 노트북을 그대로 fork해 순수 baseline을 보자" | yusuf의 V18 Improvement 노트북 통째 fork (EffNet 없음) | (silent reject) | - | ❌ COMPLETE인데 publicScore 빈칸 — **이 Era 첫 silent reject**. 재실행 컨테이너에서 외부 노트북의 path/dataset 가정이 깨져 submission.csv 미생성 추정. "외부 노트북 통fork = 채점 재실행 리스크" |
| **034 hgnet_blend** (21) | "EffNet과 상관 낮은 HGNetV2를 앙상블에 더하면 다양성으로 오를 것" | HGNetV2-B0 4-fold 10% blend | 0.927 | - | ❌ -0.003. 삽질 끝(v2는 logit scale 안 맞춰 0.858 폭락 → z-score 정규화 후 회복) 결국 하락. **HGNet OOF 0.9657 < EffNet 0.9792 → 약한 모델은 noise로 작용**. blend는 강한 모델만 도움 |
| **035 softauc** (22) | "BirdCLEF 2025 1위가 쓴 SoftAUC(AUC 지표를 직접 최적화하는 ranking loss)로 EffNet을 학습하면 LB가 오를 것" | loss = 0.5·BCE + 0.5·SoftAUC, spec pre-compute로 6.5x 가속. OOF 0.9815(+0.0023) | 0.930 | - | ➖ 동률. OOF는 분명 올랐으나 **EffNet은 15% 비중뿐이라 자체 개선이 최종 LB에 안 묻음**. "OOF 개선 ≠ LB 개선, blend 작을수록 더 그렇다" |
| **036 epoch50_blend25** (23) | "SoftAUC 모델이 더 강하니 비중을 25%로 올리면 개선이 LB에 반영될 것" | epoch 30→50 + BLEND 0.15→0.25. OOF 0.9823 | 0.929 | - | ❌ -0.001. **BLEND 올리기 = EffNet 강화가 아니라 Perch 약화**. EffNet이 Perch보다 약한 한 비중 키우면 손해. (epoch와 blend를 동시에 바꿔 원인 분리도 실패) |
| **037 convnext_xcl** (24) | "EffNet과 상관 낮은 ConvNeXt-Base(9736종 XCL 사전학습)를 3번째 모델로 넣으면 다양성 이득" | ConvNeXt 추가, Perch 65%+EffNet 15%+ConvNeXt 20% (Fold 1만 완료해 제출) | 0.930 | - | ➖ 동률. Fold1 val 0.9895로 **모델 자체는 강함**. 1-fold만으론 앙상블 효과 제한. 삽질: spec 이중 squeeze, kernel-metadata 파일명 혼동(silent fallback), backbone 경로 분리 실패 |
| **038 prior_mask** (25) | "test에 안 나오는 67종(Tier C)에 prior 0.3을 곱하면 false positive 줄어 미세 ↑" | 후처리 한 줄: 비사이트 종 ×0.3 (모델 변경 0) | 0.930 | - | ➖ 동률. **macro AUC + skip-empty 특성상 test에 없는 종은 평균에 안 들어가 효과 없음**. 이 라운드 EDA로 핵심 발견: test=Pantanal(브라질) 단일 사이트(4주간 "미국 soundscape"로 잘못 가정), **234종 중 28종은 train 0샘플**(영구 0점) |
| **039 convnext_5fold** (26~30) | "Fold1만으로 동률이었으니 5-fold 전체로 ConvNeXt 다양성 극대화 (CV 0.9905)" | ConvNeXt fold0~4 전체, 이후 batch opt/ONNX INT8/3fold/1fold 5변형 제출 | ❌ silent ×5 | - | ❌ **4/30부터 채점 시스템 이상 시작 — 5회 제출 전부 COMPLETE인데 publicScore 없음**. INT8 변환·timeguard 버그수정도 무의미. 코드가 아닌 플랫폼 문제 |
| **040 effnet_multiwindow** (31·33) | "5초 윈도우 12개로 multiwindow 추론하는 EffNet fold0(val 0.9794)을 컴포넌트로 추가" | multiwindow EffNet (v1: dry-run 24행 버그, v8: 수정) | ❌ silent ×2 | - | ❌ 31은 dry-run fallback이 24행 CSV 제출, 33은 수정했지만 둘 다 silent reject. **플랫폼 이상 지속** |
| **032 diagnostic** (32) | "알려진 0.930 blend를 재제출해 플랫폼 이상인지 우리 코드 문제인지 가른다" | 검증된 0.930 blend 그대로 | PENDING | - | 🔍 진단용. 이 안정 구성도 점수가 안 나옴 → **코드가 아닌 플랫폼 채점 이상 확정** |
| **041 convnext_timeguard** (34) | "ConvNeXt가 채점 재실행에서 시간 초과로 죽나? deadline을 114분으로 강제하면" | ConvNeXt deadline 8h→114분 hard | ❌ silent | - | ❌ 114분 timeguard로도 silent. **경쟁 eval 환경은 더 느려 114분도 부족 + 플랫폼 이상 겹침 → ConvNeXt 3-way는 운영 불가 판단, 포기** |
| **042 gaussian_smoothing** (35) | "soundscape 시계열에 Gaussian temporal smoothing(σ=1.0)을 적용하면 노이즈 완화로 ↑" | prior mask 후 gaussian_filter1d(σ=1.0) per soundscape | ❌ silent | - | ❌ **sub26~35 10연속 silent reject** — 효과 검증 자체 불가. 플랫폼 이상의 한복판 |
| **043 convnext_remove** (36) | "ConvNeXt 제거해 timeout 위험 없애면 채점이 정상화되나" | ConvNeXt 완전 제거, Perch 85%+EffNet 15% 2-way, Gaussian 유지 | 0.928 | 0.92194 | ❌ -0.002. **드디어 채점 정상화 확인(ConvNeXt 마운트가 silent의 한 원인)**. 그런데 sub25(0.930)와 같은 구조인데 더 낮음 → Gaussian smoothing이 범인으로 지목 |
| **044 gaussian_remove** (37) | "sub36과 차이는 Gaussian뿐이니, 제거하면 0.930으로 복귀할 것" | Gaussian smoothing 14줄 제거, 나머지 동일 | **0.930** | 0.92634 | ➖ best 동률 복귀. sub36(0.928)↔sub37(0.930) 대조로 **Gaussian smoothing이 -0.002 역효과임을 인과 확정**. 기준선 안전 복귀 |

(private 점수: 마감 후 공개된 맵에 trial_043=0.92194, trial_044=0.92634만 매칭됨. sub15~37의 distill/pseudo/softauc/convnext 계열 대다수는 CLI가 최근 50건만 반환해 private 미공개라 "-")

### 이 Era의 핵심 교훈

- **약한 보조 모델의 비중을 키우면 강한 주력(Perch)을 묽게 할 뿐이다.** blend 0.08~0.25 전 구간(029·030·036), HGNetV2 10%(034), SoftAUC OOF +0.0023(035) — 모두 0.930에서 막혔다. 보조 모델의 OOF가 주력보다 낮으면(HGNet 0.9657) 그냥 noise다. **"OOF 개선 ≠ LB 개선"이며 blend 비중이 작을수록 자체 개선은 최종 점수에 안 묻는다.**

- **CV(내부 검증) 최고가 LB 최고가 아니다 — 특히 도메인 갭이 클 때.** pseudo-label(031)은 CV 0.9792로 최고였지만 LB는 -0.003. pseudo-label은 모델이 이미 맞히는 것만 확신 라벨링해 기존 bias를 자기증폭한다. 작년 ProtoSSM weight(032)도 "로드 성공"이 "품질"을 보장하지 못했다.

- **코드 대회의 silent reject는 코드가 아니라 환경·자원 문제다.** 4/30~5/4 채점 장애 + ConvNeXt 데이터셋 마운트 거부 + 추론 시간 초과가 겹쳤다. **COMPLETE = 점수 보장이 아니다.** 진단용 재제출(sub32)로 "알려진 0.930 구성도 채점 안 됨"을 확인한 게 플랫폼 이상 확정의 결정타였고, ConvNeXt 제거(sub36) 후 정상화로 원인 일부가 ConvNeXt 마운트였음을 좁혔다.

- **ConvNeXt는 정확도(val 0.9895)는 충분했지만 코드 대회에서 운영 불가였다.** 추론이 경쟁 eval에서 89분+ 걸려 timeout, 데이터셋 마운트가 hidden re-run에서 거부됨. **느린 강한 모델보다 빠른 안정 모델**이 코드 대회에선 실전 가치가 높다.

- **단순 후처리도 반드시 A/B로 인과를 확인하라.** Gaussian smoothing(σ=1.0)은 직관적으로 좋아 보였지만 -0.002 역효과였고, sub36(0.928)↔sub37(0.930) 단일 변수 대조로만 입증됐다. prior mask(038)도 macro+skip-empty 지표에선 효과 없음이 이론적으로 설명됐다 — **EDA로 지표 특성을 먼저 이해했으면 제출 한 번을 아꼈을 것.**

---

## Era 3 — EffNet 멀티윈도우와 pseudo-mix 블렌드 (sub 38~50 / trial 045~058)

**한 줄 요약:** Perch(구글이 공개한 새소리 사전학습 임베딩 모델) 단독 위에 EfficientNet(이미지 분류용 CNN 백본) 보조 컴포넌트를 작은 비중으로 계속 쌓아 0.930 → **0.934**까지 끌어올린 시대. standalone EffNet은 0.836으로 무력하지만, Perch와 블렌드하면 +0.002씩 먹힌다는 게 핵심 발견. 끝에서는 보조 컴포넌트 fold 앙상블이 4개에서 포화하고, fusion 방식을 rank-average로 바꾼 실험이 -0.005 역효과를 내며 "logit magnitude가 실제 신호"임을 확인하고 0.934 천장에 도달했다.

- **시작 점수:** public 0.930 (Era 2 마지막, trial_044 기준선)
- **끝 점수:** public **0.934** (trial_053에서 달성, trial_058에서 복원·확정) / private 약 0.933 (trial_053=0.93292)

> 용어 풀이
> - **Perch**: 구글이 Xeno-Canto(새소리 녹음 아카이브)로 사전학습한 오디오 임베딩 모델. 이 대회에서 가장 강한 단일 신호이며 블렌드의 backbone(주축)이다.
> - **EffNet (EfficientNetV2-B0)**: 이미지 분류용 CNN. 오디오를 멜 스펙트로그램(소리를 시간×주파수 이미지로 변환한 것)으로 바꿔 입력한다. 단독으로는 약하지만 Perch와 "오류 분포가 달라서" 보조로 쓰면 다양성(diversity) 이득을 준다.
> - **distill 5fold (지식 증류)**: EffNet이 Perch의 출력을 모방하도록 L2-MSE로 학습시킨 컴포넌트. 5-fold = 데이터를 5등분해 5개 모델 학습 후 평균.
> - **fold0 standalone (mwf0)**: 일반 cross-entropy(분류 정답을 직접 맞추는 손실)로 학습한 EffNet 단일 fold. distill과 학습 방식이 달라 오류 분포가 다름.
> - **pseudo-mix (pmix)**: 정답 라벨이 없는 soundscape(현장 녹음) 3만 개에 Perch가 매긴 soft 라벨(0~0.85 확률값)을 섞어 학습한 EffNet. 테스트 분포에 더 가까운 데이터로 학습해 또 다른 다양성을 줌.
> - **블렌드(blend)**: 여러 모델의 logit(시그모이드 직전 출력값)을 가중 합산하는 앙상블. `final = Perch×(1-Σw) + Σ(보조모델×weight)`.
> - **macro ROC-AUC**: 종마다 ROC-AUC를 따로 구해 평균. 순위(rank) 기반 지표라 상수 더하기/온도 스케일링 같은 단조 변환에는 점수가 안 변함.

### Trial별 상세

| trial | 왜 시도(직전 가설/문제) | 무엇을 바꿈 | public | private | 결과·왜 그 숫자가 나왔나 |
|---|---|---|---|---|---|
| 045 (sub38) | "standalone EffNet이 Val AUC 0.9794이니 LB에서도 쓸 만할 것이다" — Perch 없이 EffNet 단독으로 제출 시도 | EffNet 멀티윈도우 fold0 단독 제출 (Perch 제거) | 0.836 | 0.807 | ❌ -0.094 대폭락. Perch가 제공하는 사전학습 임베딩이 이 대회의 핵심 신호이고, EffNet 단독은 그것 없이 경쟁력이 없음. Val AUC가 높아도 LB는 무너짐(검증셋과 hidden test 분포 차이). EffNet은 **보조 역할로만** 유효함을 재확인 |
| 046 (sub39) | "CE로 학습한 fold0 standalone은 distill(Perch 모방)과 오류 분포가 달라 다양성을 줄 것이다" | Perch 80% + distill5fold 15% + **fold0 5%** 3-way 블렌드로 fold0 추가 | **0.932** | 0.930 | ✅ **new best (+0.002)**. 학습 방식이 다른(CE vs 증류) EffNet을 5%만 섞어도 오류 분포 차이로 다양성 이득. Perch를 85→80%로 낮춰도 점수 유지 — 2-way보다 3-way가 실질 개선 |
| 047 (sub40) | "fold0 비중을 5→8%로 올리면 더 개선될 것이다" | BLEND_MWF0 0.05→0.08, Perch 80→77% | 0.932 | 0.931 | ➖ 동률. fold0 다양성 기여는 5%에서 이미 포화 — 비중만 올려도 새 정보가 없어 LB 무반응. 단순 weight sweep은 효과 없음을 확인 |
| 048 (sub41) | "더 강한 5fold 컴포넌트로 교체하면 개선될 것이다" — epoch50 SoftAUC 대신 distill30 KD로 swap | 15% 슬롯을 epoch50 SoftAUC → distill30 KD로 교체 | 0.931 | 0.931 | ❌ -0.001. epoch50 SoftAUC(50 epoch, AUC 직접 최적화 손실)가 distill30 KD(30 epoch)보다 우수. epoch 수 차이 + SoftAUC가 macro-AUC에 더 직접적. epoch50을 best 5fold로 확정 |
| 050 (sub42) | "Perch soft 라벨로 학습한 pseudo-mix는 테스트 분포에 가까워 또 다른 다양성을 줄 것이다" (+ trial_048에서 distill30으로 잘못 바뀐 5fold를 epoch50으로 복원) | epoch50 복원 + **pmix fold0 3%** 추가 → 4-way 블렌드 (Perch 77%) | **0.933** | 0.931 | ✅ **new best (+0.001)**. soft pseudo-label EffNet이 distill/CE와 다른 오류 분포 → 3% 작은 비중으로 +0.001. "작은 weight의 다양한 EffNet을 계속 쌓는" 누적 전략이 0.930→0.932→0.933으로 작동 중 |
| 051 (sub43) | "pmix를 fold0+fold1 2-fold 앙상블로 만들면 분산이 줄어 약간 더 개선될 것이다" | pmix fold0 단일 → fold0+fold1 2-fold 평균 (weight 3% 고정) | 0.933 | 0.931 | ➖ 동률. **같은 종류 컴포넌트의 fold 추가는 한계효용 체감** — 3% weight가 작아서 2-fold의 분산 절감이 macro-AUC granularity(~0.001) 아래로 묻힘. submission mean 0.0577→0.0579 미세 변화가 사전 신호였음 |
| 052 (sub44) | "Perch 비중을 줄이고 EffNet 계열을 28%로 합치면 천장을 뚫을 것이다" | EffNet 4종 weight 재배분 (Perch 77→72%, mwf0 0.07/pmix 0.06) | 0.933 | 0.932 | ➖ 동률. Perch↓ 자체는 역효과 아님(회귀 없음)이지만 EffNet 28% 예산 내부 재배분만으로는 개선 없음. 단 이 재배분이 trial_053의 깔끔한 격리 실험 토대가 됨 |
| 053 (sub45) | "2-fold는 무효였지만 4-fold(분산 1/4)면 임계치를 넘을 것이다" | pmix를 fold0..3 **4-fold 앙상블**로 확장 (코드 변경 0, dataset만 교체, Perch 72%) | **0.934** | 0.933 | ✅ **new best (+0.001)**. 2-fold(분산 1/2)는 LB 해상도 아래였지만 4-fold에서 임계치를 넘어 반영. 추론 코드 변경 없이 dataset 버전만 올리는 가장 싼 개선. **자체 시대 private 최고권(0.93292)** |
| 054 (sub46) | "5-fold(분산 1/5)면 4-fold보다 더 개선될 것이다" | pmix fold0..4 **5-fold 앙상블**로 확장 | 0.934 | 0.933 | ➖ 동률, **한계효용 0**. 4→5-fold의 추가 분산 절감(1/4→1/5)이 macro-AUC granularity 아래. **보조 컴포넌트 fold 앙상블은 4개에서 포화** 확정. 같은 트릭으로는 0.935 불가 — 새 diversity 소스 필요 |
| 055 (sub47) | "15% 슬롯을 epoch50+distill 10ckpt 멀티로스 앙상블로 만들면 손실함수 다양성으로 개선될 것이다" | 15% EffNet 슬롯을 epoch50 단일 → epoch50+distill 10ckpt 평균으로 강화 | 0.933 | 0.933 | ❌ -0.001. distill을 평균에 섞어도 epoch50 단일보다 못함(trial_048에서 distill 단독 swap이 -0.001이었던 것과 일관). **멀티로스 앙상블 ≠ 다양성 이득** — 약한 컴포넌트를 평균에 넣으면 오히려 희석 |
| 056 (sub48) | trial_055 회귀를 복원하고, pmix를 8%로 올리면(노이즈 많은 mwf0에서 2pp 빼서) 개선될 것이다 | epoch50 단일 복원 + pmix 0.06→0.08, mwf0 0.07→0.05 (Perch 72% 고정) | 0.934 | 0.932 | ➖ 동률. EffNet 보조 컴포넌트들끼리 **서로 거의 교환 가능(interchangeable)** — 2pp를 어디에 주든 0.934 평형점 유지. EffNet 28% 예산 내부 재분배는 데드 엔드 확정. trial_055 회귀 복원이 진짜 성과 |
| 057 (sub49) | "logit 합에선 다이내믹 레인지 큰 컴포넌트가 명목 weight보다 과대 기여한다 → rank-average로 공정하게 만들면 ±0~+0.001" | 가중 logit 합 → 클래스별 percentile rank-average fusion으로 교체 (모델·weight 동일) | 0.929 | 0.923 | ❌ **-0.005 큰 폭 하락**. **logit magnitude가 실제 신호**였음 — Perch가 강하게 "present"라고 말하는 *강도*가 rank로 바뀌면 사라지고, EffNet 보조가 logit 합에서 명목보다 큰 effective weight로 +방향 기여하던 게 죽음. "fusion 연산자는 0.934를 만든 core 메커니즘"이라 바꾸면 크게 흔들림. private도 0.923으로 최저권 |
| 058 (sub50) | "[필수] rank-avg -0.005 회귀를 청소하고 trial_056 logit 블렌드를 복원하라" | cell 65를 trial_056 버전으로 되돌려 logit-space 가중 합 복원 (코드 byte-identical) | **0.934** | 0.933 | ✅ best 동률. rank-avg 회귀 청소 성공, logit-blend 복원 검증(submission mean 0.0425로 trial_056과 동일값). 0.934 천장 재확정 |

### 이 Era에서 배운 핵심 교훈

- **Perch가 압도적 backbone, EffNet은 보조 전용.** standalone EffNet은 Val AUC 0.9794여도 LB 0.836(-0.094). Perch 없이는 어떤 EffNet도 무력하고, 오직 작은 비중(3~5%)으로 블렌드할 때만 +0.001~+0.002의 다양성 이득을 준다. "검증셋 점수가 높다 ≠ LB에서 쓸 만하다"의 가장 극적인 사례.

- **다양성 이득은 "다른 학습 방식"에서 나오지, 같은 모델 더 쌓기에서 안 나온다.** CE fold0(+0.002) → pseudo-mix(+0.001)처럼 학습 *방식*이 다르면 작은 weight로도 먹히지만, 같은 종류의 fold를 늘리는 건 한계효용이 빠르게 0으로 수렴한다(pmix 2-fold=0, 4-fold=+0.001, 5-fold=0). **보조 컴포넌트 fold 앙상블은 4개에서 포화** — 이게 이 Era의 가장 재사용성 높은 정량 규칙.

- **앙상블 fusion 방식(logit 가중 합)은 건드리면 안 되는 core 메커니즘.** rank-average로 "공정하게" 바꾼 trial_057이 -0.005로 폭락하며, logit의 **magnitude 자체가 신호**임을 증명했다. Perch의 강한 confidence와 EffNet 보조의 다이내믹 레인지 차이(EffNet 헤드 logit이 더 큼 → 명목보다 큰 effective weight)가 0.934를 만든 숨은 메커니즘이었다. "weight·후처리는 marginal, fusion 연산자·컴포넌트 구성은 core"라는 위험 구분이 필요.

- **macro ROC-AUC는 순위 기반이라 단조 변환에 무감각하고, LB 해상도는 ~0.001.** 온도 스케일링·per-class 임계 같은 단조 후처리는 점수에 영향이 없고, submission mean 0.0577→0.0579 같은 미세 변화가 LB 결과의 사전 신호가 된다. EffNet 28% 예산 내부의 2pp 재배분은 전부 granularity 아래로 묻혀 무효.

- **가장 싼 개선부터 짜낸 뒤, 같은 축이 포화하면 깔끔한 회귀 복원이 곧 성과다.** "코드 변경 0, dataset 버전만 올리기"(pmix 4-fold)가 +0.001을 줬고, 잘못된 실험(distill30 swap, 멀티로스, rank-avg)은 -0.001~-0.005 회귀를 냈지만 즉시 복원해 0.934를 지켰다. private에서도 trial_053/054(0.933대)가 자체 시대 최고권 — public best와 private best가 이 Era에서는 일치했다(이후 Era에서 EoS fork가 천장을 0.950으로 돌파).

---

## Era 4 — ralph-x 자동루프와 0.934 weight 천장 (sub 51~63, trial 059~071)

**한 줄 요약:** ralph-x(가설→코드→Kaggle 제출→reflection을 사람 개입 없이 반복하는 자동 루프)로 6개 컴포넌트 블렌드(여러 모델 출력을 가중 합치는 앙상블)의 weight(비중)를 전 방향으로 ±2pp씩 흔들었으나, 11개 trial이 **전부 public 0.934 동률 또는 그 이하**로 수렴 — "weight space(비중 조합 공간)는 이미 소진됐다"는 것을 비싼 대가로 증명한 시대.
**시작/끝 점수:** public 0.934(trial_058 best 동률) → public 0.934(trial_071). 끝까지 best는 0.934, 단 1퍼밀(0.001)도 못 올림. private는 마감 후 공개 기준 0.930~0.933 구간.

> 용어: **blend weight** = 각 모델 출력에 곱하는 비중(다 더하면 1). **pp** = percentage point(0.02→0.04는 +2pp). **logit-space 합** = 확률로 바꾸기 전 raw 점수를 가중 합. **fold 앙상블** = 데이터를 N등분해 N번 학습한 모델들의 평균(노이즈↓). **ROC-AUC** = 점수의 순위(rank)만 보는 지표 → 모든 점수에 같은 단조변환(온도·상수배)을 가해도 값이 안 변함.

### 컴포넌트 구성 (이 시대 내내 거의 고정)
Perch(구글 사전학습 오디오 임베딩, 주력) ~72% + EffNet 5-fold(직접 학습한 EfficientNet 5겹 앙상블) 15% + mwf0(EfficientNet fold0 단일) 2% + pmix(pseudo-mix 5-fold) 8% — 합 약 0.97~1.0. 여기에 distill / ConvNeXt / SED를 5번째·6번째 컴포넌트로 붙여보는 게 이 시대의 거의 전부였다.

### trial별 상세

| trial | 왜 시도(가설/문제) | 무엇을 바꿈 | public | private | 결과·왜 그 숫자가 나왔나 |
|---|---|---|---|---|---|
| **059** distill_add (sub 51) | "손실함수가 다르면(SoftAUC vs KD-MSE) logit-space에서 다른 정보를 담아 +0.001 날 것이다" | distill 5-fold(지식증류로 학습)를 별도 5번째 컴포넌트로 추가, BLEND_DISTILL=0.03, mwf0 0.05→0.02 | 0.934 | 0.93095 | ➖ 동률. KD(지식증류) 출력이 SoftAUC·Perch와 logit-space에서 거의 직교(ortho) 정보를 안 담음. "별도 weight가 같은 슬롯 평균(trial_055 −0.001)보다 나을 것"이란 직관이 틀림 — EffNet 28% 예산 안의 어떤 재분배도 천장을 못 깸 |
| **060** convnext_add (sub 52) | "transformer 계열 백본 + 9736종 XCL 사전학습은 EffNet/Perch와 완전히 다른 가족이라 신호가 직교일 것이다" | distill 3% 슬롯을 ConvNeXt-Base fold0 3%로 1:1 치환 | **TIMEOUT** | - | ⏳ kernel은 정상 완료(추론 1.8분)했으나 **Kaggle 채점 행 자체가 90분 폴링 동안 미생성**. 코드 대회는 kernel push만으론 채점이 안 트리거되고 hidden-test 재실행이 필요한데, ConvNeXt **데이터셋 마운트가 hidden re-run에서 거부**돼 submission이 안 만들어짐 (`kaggle submit -k`도 400). 점수 검증 자체가 불가 |
| **061** convnext_5fold (sub 53) | "fold0 단일은 노이즈 큼 → pmix 패턴처럼 fold0..4 평균하면 ConvNeXt 신호가 LB에 잡힐 것이다" | ConvNeXt를 fold0 단일 → fold0..4 5-fold 평균으로 확장 (weight 3% 유지) | **TIMEOUT** | - | ⏳ trial_060과 **동일 증상 2연속**. 코드(5-fold 평균)는 정상 COMPLETE — 결함이 아니라 ConvNeXt 데이터셋이 채점 컨테이너에서 거부되는 인프라 문제. 90분씩 2번 = 폴링만 낭비. "kernel COMPLETE ≠ 채점 성공"을 ralph가 구분 못 한 게 비용 |
| **062** pmix_weight_up (sub 54) | "ConvNeXt 축은 동결하고, 검증된 5-fold pmix에 그 3pp를 몰아주면 오를 것이다" + "ConvNeXt 데이터셋을 metadata에서 빼면 채점이 정상화되는지 검증" | BLEND_PSEUDOMIX 0.08→0.11, ConvNeXt 데이터셋 2개 제거 | 0.93463 | 0.93075 | ➖ 동률. **ConvNeXt 데이터셋 제거 후 채점 정상 복귀(78분 소요) — TIMEOUT 원인이 그 데이터셋 마운트 거부였음 확정**. 단 pmix 증량은 무효(0.06→0.08도, 0.08→0.11도 동일) → 5-fold pmix는 8%에서 이미 포화 |
| **063** mwf0_zero_pmix13 (sub 55) | "mwf0는 fold0 단일이라 노이즈일 것 → 제거하고 clean한 pmix로 옮기면 순도↑" | mwf0 0.02→0.00(완전 제거), pmix 0.11→0.13 | 0.93336 | 0.92893 | ❌ −0.001. **가설 기각: mwf0는 노이즈가 아니라 실제 보완 신호였음** — 제거가 오히려 손실. 동시에 pmix 0.13은 0.11보다 하락 → pmix 방향 완전 소진 |
| **064** effnet_weight_up (sub 56) | "가장 검증된 보조 컴포넌트(EffNet 5-fold) 비중을 키워 Perch 단독 의존을 줄이면 천장 돌파" | EffNet 0.15→0.17, Perch 0.72→0.70 | 0.93420 | 0.93090 | ➖ 동률. EffNet weight↑ 방향 무반응 → 포화 가능성 |
| **065** effnet_further_up (sub 57) | "0.17이 동률이었으니 gradient 확인차 0.19까지 더 밀면 방향이 +인지 −인지 확정" | EffNet 0.17→0.19, Perch 0.70→0.68 | 0.93366 | 0.93097 | ❌ −0.001. **EffNet↑ 역효과 확정 → 최적 EffNet은 15~17% 사이**. 19%는 Perch를 너무 깎아 주력 신호 약화 |
| **066** mwf0_up (sub 58) | "trial_063에서 mwf0 제거가 −0.001이었으니 반대로 증량하면 +일 것이다" | EffNet 0.19→0.15 원복, mwf0 0.02→0.03, pmix 0.11→0.10 | 0.93490 | 0.93142 | ➖ 동률. mwf0 0.02→0.03 무반응 |
| **067** mwf0_further_up (sub 59) | "0.03이 동률 → 포화 경계 확정 위해 0.04까지 한 단계 더" | mwf0 0.03→0.04, pmix 0.10→0.09 | 0.93486 | 0.93197 | ➖ 동률. **mwf0 0.02→0.03→0.04 전부 0.934 → mwf0 증량 방향 완전 포화** |
| **068** perch_up (sub 60) | "보조 컴포넌트는 다 막혔으니 주력 Perch 자체 비중을 키우면?" | mwf0 0.02 원복, pmix→0.07, Perch 0.72→0.76(+4pp) | 0.93442 | 0.93027 | ➖ 동률. **Perch 4pp 증량도 무효 → weight space 8방향(pmix/mwf0/EffNet/Perch ×증감) 전부 0.934 천장 확정** |
| **069** prior_mask_relax (sub 61) | "weight가 다 막혔으니 후처리로: 비서식종 억제를 0.3→0.5로 완화하면 false-positive 줄어 오를 것" | CLASS_PRIOR_MASK 0.3→0.5 | 0.93442 | 0.93027 | ➖ 동률. prior mask 후처리도 LB 무반응 — 0.3 억제가 이미 적절했거나 이 축 자체가 무의미 |
| **070** temp_scale_off (sub 63) | "ROC-AUC는 rank-based이니 온도 스케일링(T_AVES)은 이론상 무의미할 것이다" | T_AVES 1.10→1.0 (온도 스케일링 제거) | 0.93430 | 0.93007 | ➖ 동률. **이론대로 확인** — 온도는 모든 점수에 같은 단조변환이라 순위 불변 → ROC-AUC 0 변화. T_AVES는 애초에 무의미한 파라미터였음 |
| **071** sed_blend (sub 62) | "SED(attention pooling 헤드)는 다른 정보 추출 → 5% 섞으면 diversity 확보" | Tucker Distilled-SED 5-fold ONNX 추가, BLEND_SED=0.05, pmix→0.06, Perch→0.72 | 0.93412 | 0.92988 | ➖ 동률. SED가 **Perch로 지식증류된 모델이라 예측이 Perch와 유사 → diversity 미확보**. 5%는 Perch 72%에 희석. **10연속 0.934 천장 확정** |

*(public 점수는 마감 후 공개된 정밀값. SUBMISSIONS.md는 모두 0.934로 반올림 표기.)*

### private 점수가 말해주는 것 (마감 후)
- 이 시대 trial들의 private는 **0.930~0.932** 구간으로, **public 0.934보다 약 1~4퍼밀 낮음** — 정상적인 약한 과적합. trial_057 rank-average(다음 시대로 가는 회귀)만 private 0.92326으로 뚝 떨어져 fusion 연산자 교체가 최악이었음을 재확인.
- 흥미로운 역전: **trial_055/053/054(이전 시대 끝자락)가 private 0.933 전후로 이 시대 어떤 trial보다 private가 높았다.** public이 같은 0.934여도 private는 미묘하게 달랐고, weight를 더 흔들수록 private는 오히려 살짝 내려가는 경향 → public 0.934 평지에서 weight 미세조정은 노이즈에 과적합하는 쪽이었음.
- 최종 대회 결과(참고): public 398/4085(상위 9.7%). 이 시대의 0.934 천장은 훗날 EoS 메가앙상블 포크(public 0.950)로만 깨졌고, public 최고 픽이 private 최저였던 과적합 반전도 거기서 나옴.

### 이 시대 핵심 교훈
- **블렌드 weight space는 한 점 근처에서 평평하다.** Perch/EffNet/mwf0/pmix를 ±2~4pp씩 8방향 전부 흔들어도 전부 public 0.934. ROC-AUC의 LB granularity(~0.001) 안에서 ±2pp weight 조정은 신호가 아니라 노이즈다. 평지를 확인하는 데 11 trial을 쓴 건 ralph-x 자동루프의 구조적 비용 — "다음 가설"을 계속 같은 축에서 뽑았기 때문.
- **"다른 모델을 섞는다"가 곧 diversity는 아니다.** distill·SED 둘 다 결국 Perch/EffNet의 지식증류·파생이라 logit-space에서 직교 정보를 거의 안 담았다 → 0 효과. 진짜 diversity는 *학습 데이터/임베딩 소스/백본 가족*이 근본적으로 달라야 나온다(다음 시대 EoS 포크가 그 증거).
- **코드 대회에서 "kernel COMPLETE"는 "채점 성공"이 아니다.** ConvNeXt 데이터셋 마운트가 hidden-test 재실행 단계에서 거부돼 submission 행 자체가 안 생기는데, ralph는 kernel 정상 완료만 보고 90분씩 2번 폴링을 낭비했다. 새 데이터셋 추가 시 hidden-test 호환성 사전 검증 + "10분 내 채점 행 미생성이면 즉시 fallback" 단계가 필수.
- **이론으로 미리 죽일 수 있는 trial이 있다.** T_AVES 온도 제거(trial_070)는 ROC-AUC가 rank-based인 이상 결과가 뻔했다(순위 불변 → 0). 단조변환류 후처리(온도·상수배·prior mask 배수)는 rank 지표에서 검증할 가치가 없다 — cross-file/cross-window/cross-class처럼 *순위를 실제로 바꾸는* 변환만 의미 있다.
- **자동루프는 "같은 축 소진"을 빨리 감지하지 못한다.** 0.934가 6연속, 8연속, 10연속 나올 때마다 reflection은 매번 "새 1차 모델 컴포넌트 필요"라고 옳게 진단했지만, 다음 trial은 여전히 weight 미세조정이었다. 멈추지 않는 루프의 대가는 "이미 답이 나온 진단을 실행으로 옮기는 결단의 지연"이다.

---

## Era 5 — ProtoSSM 파이프라인 전환과 0.938 천장 (sub 64~81, trial 072~088)

**한 줄 요약:** 0.934에서 막힌 자체 블렌드를 버리고, 공개 0.947 reference 파이프라인(ProtoSSM + SED + Perch)으로 완전 갈아탔다. 초기 blend 공식 버그로 6개 trial이 통째로 SKIP됐고, 버그를 잡은 뒤 **0.933 → 0.935 → 0.938**(자체 파이프라인 최고)까지 올렸지만, 이후 weight/scale/컴포넌트/추론 4축 + 자체재학습 + dual앙상블 **8개 trial이 전부 0.938 동률**로 천장을 확정했다. (시작 0.933 → 끝 0.938)

> **용어 풀이**
> - **ProtoSSM**: Perch 오디오 임베딩(미리 학습된 새소리 특징 벡터)을 입력으로 받는 prototype 기반 state-space 모델. 이 Era의 핵심 1차 모델(메인 예측기).
> - **SED (Sound Event Detection)**: EfficientNetB0 백본 + Perch 지식증류 + attention으로 만든 "소리 이벤트 검출" 모델. blend의 보조 신호.
> - **Perch**: Google이 공개한 새소리 임베딩/분류기. 여기선 prior-fused base 신호로 소량 섞임.
> - **blend weight**: 여러 모델 출력(logit, 확률 직전 점수)을 가중 합할 때 각 모델의 비율.
> - **ROC-AUC가 rank-based**: 채점이 점수의 절대값이 아니라 클래스 내 행(클립)들의 **순위**만 본다. 따라서 모든 클립에 같은 상수를 더하거나 곱하는(monotonic/affine) 변환은 순위를 안 바꿔 점수에 무영향.
> - **logit z-score 정규화**: 모델별로 점수 스케일이 다를 때 평균 0·표준편차 1로 맞춰 같은 weight가 같은 effective 영향을 갖게 하는 작업.
> - **TTA (Test-Time Augmentation)**: 추론 시 입력을 시간축으로 조금씩 밀어(shift) 여러 번 예측한 뒤 평균내 안정화하는 기법.
> - **silent reject**: 노트북은 COMPLETE인데 Kaggle이 publicScore를 안 매기는 채점 실패.

### Trial별 상세

| trial | 왜 시도(직전 가설/문제) | 무엇을 바꿈 | public | private | 결과·왜 그 숫자가 나왔나 |
|---|---|---|---|---|---|
| 072 sed_up | 자체 블렌드가 0.934에서 막힘 → 공개 0.947 ref 파이프라인(ProtoSSM+SED)으로 갈아타면 천장이 올라갈 것 | 새 파이프라인 도입, SED 비중 상향 (kernel v59-60) | - | - | **SKIP(버그).** `BLEND_EFFNET`/`effnet_logits` 미정의로 크래시. 신규 파이프라인 이식 첫 시도라 변수 연결이 깨짐 |
| 073 proto_blend | blend 공식대로 ProtoSSM+SED+Perch를 섞으면 ref 점수에 근접할 것 | BLEND_PROTO/SED/Perch 설정 (v63) | - | - | **SKIP(버그).** `BLEND_SED=0.0` override 버그로 실효 blend가 Proto 50%+Perch 43%+SED 0%. 의도와 다른 출력 |
| 074 proto_up / 075 sed_only / 076 effnet_sed | Proto·SED 비중을 키우면 신호가 강해질 것 | 각각 Proto↑ / SED 단독 / EffNet+SED 비중 변경 (v65~67) | - | - | **3건 전부 SKIP(같은 버그).** cell 68에 blend 공식이 빠지고 129자 주석만 남아, 무슨 weight를 줘도 **ProtoSSM 단독 출력**만 나옴. 설정 변경이 전부 무효 — 한 줄 누락이 여러 trial을 통째로 날린 케이스 |
| 077 birdnet_add | BirdNET(또다른 강한 오디오 분류기)을 추가하면 다양성으로 오를 것 | jarturo/birdnet dataset 추가 (v68) | - | - | **SKIP(silent reject).** BirdNET 추론은 로컬에서 정상 동작했으나, 코드 대회 hidden 재실행이 해당 외부 dataset 마운트를 거부 → publicScore 공란. "로컬 OK ≠ 채점 OK"의 전형 |
| 078 birdnet_remove | 077의 거부 원인(birdnet dataset)을 빼고 blend 공식을 고치면 드디어 유효 채점될 것 | BLEND_BIRDNET=0, Perch 23%+Proto 60%+SED 10%+mwf0 2%+pmix 5% (v69) | 0.933 | 0.92891 | **첫 유효 채점이지만 하락.** 버그는 잡혔으나 SED/mwf0/pmix를 잡다하게 소량씩 섞은 구성이 noise로 작동, 자체 시대 best 0.934에도 못 미침. 비중 배합이 ref와 달랐던 게 원인 |
| 079 ref_config | 내 임의 배합 말고 **0.947 공개 노트북의 weight를 그대로** 쓰면 정상 궤도에 오를 것 | Proto 72%+SED 18%+Perch 10%, mwf0/pmix 제거 (v71) | **0.935** | 0.92979 | **best 경신(+0.002).** "직접 튜닝"보다 검증된 ref config 통째 복사가 즉효. ProtoSSM을 메인(72%)으로 두는 구조가 옳았음 |
| 080 sed_up | SED 18%가 너무 낮다 — SED를 키우면 더 오를 것 | SED 18%→**40%**, Proto 72%→50%, Perch 10% (v72) | **0.938** | 0.93608 | **자체 파이프라인 최고(+0.003).** SED가 ProtoSSM과 다른 오류 패턴을 보완해 blend 순위를 실제로 개선. 이 Era의 핵심 발견 = **SED 비중이 메인 레버** |
| 081 proto_path_fix | ProtoSSM pretrained mount 경로가 틀려 fallback이 돌고 있다 → 경로 고치면 0.945+ 갈 것 | mount 경로 한 줄 수정, blend 동일 (v73) | 0.938 | 0.93614 | **동률(기대 빗나감).** 경로 수정 전후 출력 불변. 결론: 경로 수정 전에도 이미 정상 로드 중이었거나 fallback이 최종 blend에 영향 없었음. "0.945+" 가설 기각 |
| 082 sed_heavy | SED 40%가 좋았으니 **50%면 더 좋을 것**(선형 가정) | SED 40%→50%, Proto 50%→40% (v74) | 0.937 | 0.93760 | **-0.001 하락.** SED 40%가 **포화점**. 50%는 약한 SED를 과하게 키워 강한 Proto 기여를 깎음 (`blend-ratio-weak-model-hurts-ensemble` 재확인). 단일 축 격리라 원인 명확. (주목: private 0.93760은 080보다 높음 — public 최적≠private 최적) |
| 083 perch_up | SED 40 고정하고 Perch를 키우면 강한 base라 오를 것 | Proto 50→48, Perch 10→12, SED 40 고정 (v75) | 0.938 | 0.93608 | **동률.** Perch 10~12% 무차별. ±2pp weight 미세조정으로는 0.938 천장 못 뚫음 — Era 4의 0.934 패턴 재현 |
| 084 sed_znorm | SED logit 스케일이 달라서 명목 weight≠effective weight일 것 → z-score로 맞추면 오를 것 | SED를 proto 분포로 global z-score 매칭, 진단 print 추가 (v76) | 0.938 | 0.93620 | **동률.** 진단으로 셋 다 logit 공간 확인(proto mean −0.93 / **sed −7.26** / perch −1.15). 무효 이유: global affine의 상수항은 클래스 내 rank에 무영향, scale항은 SED weight↑(082 무효)와 동일. **ROC-AUC rank-based라 monotonic 변환 전멸** |
| 085 effnet_readd | 새 모델 가족(EffNetV2-B0)을 소량 넣으면 다양성으로 rank가 바뀔 것 | BLEND_EFFNET 0→5%, Proto 50→45 (v77) | 0.938 | 0.93825 | **동률.** EffNet 진단 mean −9.04. 5% 비중이 너무 작거나 Proto 50%에 묻힘. (private 0.93825는 080보다 높음) |
| 086 effnet_up | 085가 비중 부족이었나? 10%면 효과 날 것 | EffNet 5→10%, Proto 45→40 (v78) | 0.937 | 0.93960 | **-0.001 하락.** 5%=동률, 10%=하락 → **EffNet 기여 한계 확정.** 약한 모델 비중↑이 강한 Proto를 깎음. 모델 다양성 가설 기각. (private 0.93960 — 이 Era 자체 trial 중 private 최고. public 신호와 정반대) |
| 087 proto_tta | 메인 Proto(50%)의 추론 품질 자체를 올리면 천장이 오를 것 | ProtoSSM TTA shift 5→7개, best weight 복원 (v79) | 0.938 | 0.93620 | **동률.** V18에서 이미 5-shift라 추가 2개의 한계효용 0. 이로써 **weight/scale/컴포넌트/추론 4축 전부 0.938 포화 확정** |
| 088 proto_dual | 자체학습 ProtoSSM + 외부 pretrained를 평균하면 데이터 다양성으로 오를 것 | deepcopy로 두 모델 추론 평균 앙상블 (v81) | 0.938 | 0.93626 | **동률.** 같은 ProtoSSMv2 아키텍처 + 같은 Perch 입력이라 두 예측 상관이 높아 평균해도 rank 거의 불변. **모든 미탐색 레버 실측 소진 → 0.938 진짜 천장 확정** |

(참고: 자체학습 ProtoSSM 단독(v80)도 별도 제출 시 public 0.93836 / private 0.93703 — 외부 pretrained와 동률. OOF 0.65만 보고 "약하다"며 성급히 기각했던 판단이 실측으로 뒤집힘.)

### 이 Era에서 배운 핵심 교훈

- **검증된 ref config 통째 복사 > 내 임의 튜닝.** trial_078(임의 배합) 0.933 하락 → trial_079(0.947 노트북 weight 그대로) 0.935 경신. 새 파이프라인 이식 초기엔 직접 weight를 만지지 말고 reference를 byte 단위로 재현해 baseline부터 세우는 게 빠르다.
- **약한 보조 모델은 "포화점"이 있고, 넘으면 강한 메인을 깎는다.** SED는 18%→40%에서 +0.003이었지만 50%에선 −0.001. EffNet도 5%(동률)→10%(하락). 보조 신호 비중은 단조 증가가 아니라 sweet spot이 존재(`blend-ratio-weak-model-hurts-ensemble`).
- **ROC-AUC가 rank-based이므로 monotonic/affine 변환은 전부 무효.** logit z-score 정규화(084), 온도 스케일링 제거(Era 4 070), TTA 확대(087)가 전부 동률. 점수를 바꾸려면 클래스 내 **클립 순위를 실제로 뒤집는** 변경이어야 한다 — global 상수/스케일 조정은 효과 0.
- **같은 아키텍처끼리 앙상블은 다양성이 아니다.** 자체학습+외부 ProtoSSM dual 앙상블이 동률(088) — 둘 다 ProtoSSMv2 + Perch 입력이라 예측 상관이 높아 평균해도 rank 불변. 다양성은 "다른 데이터로 학습"이 아니라 "다른 오류 패턴(다른 아키텍처/입력)"에서 나온다.
- **blend 공식 한 줄 누락이 trial 6개를 통째로 날렸다(072~076).** cell 68에 blend 산술이 빠지고 주석만 남아, 어떤 weight를 줘도 ProtoSSM 단독 출력만 나왔다. 파이프라인 전환 직후엔 "설정이 실제로 출력에 반영되는지" 진단 print로 먼저 확인해야 헛제출(일일 5회 한도 소진)을 막는다.
- **public 천장 ≠ private 천장.** public은 080~088이 거의 0.938 평탄선이었지만, private에선 086 effnet_up이 **0.93960**으로 자체 trial 중 최고였고 public best였던 080(0.93608)보다 높았다. public 신호만 보고 고른 픽이 private에선 뒤집힐 수 있다는 과적합 경고 — 이 Era 끝에 자체 파이프라인을 포기하고 공개 fork로 전환(Era 6)한 결정의 복선.

---

## Era 6 — 공개 EoS.9 fork로 천장 돌파, 그리고 private 반전 (sub 82~91, trial 089~098)

**한 줄 요약**: 자체 파이프라인이 public 0.938에서 16+ trial 동안 천장에 막히자(리더보드 top 0.966과 0.028 격차), 공개 메가앙상블 노트북 **EoS.9**를 우리 계정으로 fork해 단번에 **0.950**(+0.012)으로 돌파. 그 위에서 5가지 변형을 튜닝했으나 public은 전부 0.950 동률. 마감 후 private이 공개되자 갈림 — public 최고 픽이 private 최저, **다축 조합 변형(eos9-all)이 private 최고 0.94238**. 최종 public 순위 **398/4085(상위 9.7%)**.

- **시작 점수**: public 0.938 (자체 파이프라인 천장, Era 5에서 넘어옴)
- **끝 점수**: public 0.950 / private 최고 0.94238 (fork 기반)

### 왜 fork였나 (Era 진입 배경)

직전 Era 5에서 자체 Perch+ProtoSSM+SED 3-컴포넌트 blend는 weight·scale·컴포넌트·추론 4축을 전부 소진하고도 0.938에서 멈췄다. blend 비율 미세조정(trial_082~088)은 전부 동률이거나 -0.001. 리더보드 1위(Nikita Babych)는 0.96720, 공개 노트북 EoS.9는 이미 0.950을 찍고 있었다. **0.028 격차는 파라미터 튜닝으로 메울 수 없는 "접근 자체의 한계"**라고 판단 → 더 강한 공개 솔루션으로 갈아탔다. (churn 대회에서 얻은 "로컬 한계에 막히면 공개 SOTA를 fork" 교훈 적용)

> 용어: **fork**(공개 노트북을 내 계정으로 복제해 그대로 제출하는 것), **EoS.9**(nina2025가 공개한 "Ensemble of Specialists v9" — Model_1~74를 블렌드한 대규모 앙상블 노트북), **ProtoSSM/SED**(자체 파이프라인의 보조 모델들 — 프로토타입 기반 분류기/Sound Event Detection), **macro-AUC**(종별 ROC-AUC를 평균한 대회 지표, rank-based라 logit에 상수를 더해도 점수 불변).

### Trial별 상세

| trial / sub | 왜 시도(직전 가설/문제) | 무엇을 바꿈 | public | private | 결과·왜 그 숫자가 나왔나 |
|---|---|---|---|---|---|
| **trial_089 eos9_fork** (sub 82) | "자체 0.938은 우리 접근의 한계지 대회의 한계가 아니다 — 더 강한 공개 솔루션으로 갈아타면 돌파될 것이다" | 자체 파이프라인 완전 폐기. 공개 EoS.9(nina2025) 메가앙상블을 fork. **코드/dataset(6개)/모델 전부 원본**, kernel id만 ramkang으로 변경 + id_no/docker_image 제거(원본 ID 충돌이 403 원인) + is_private | **0.950** | 0.94138 | ✅ **NEW BEST (+0.012)**. Model_1~74 대규모 블렌드라 우리 3-컴포넌트 blend로는 도달 불가능한 신호량. 천장 돌파의 핵심은 튜닝이 아니라 **솔루션 교체**였다 |
| **trial_090 yaroslav_fork** (sub 83) | "EoS.9가 참조한 원천 모델을 직접 fork하면 같거나 더 높을 것이다" | EoS의 참조모델 yaroslav-v221-tax(mtoshidesu) fork | 0.950 | 0.94175 | ➖ public 동률. EoS 계열은 결국 같은 0.950 천장. (단 private은 0.94175로 EoS.9 원본보다 약간 높음 — 후술) |
| **trial_091 pcen_fork** (sub 84) | "EoS.9의 개선판(OOF gating+PCEN 정규화)이면 더 높을 것이다" | EoS+OOF Gated PCEN fork(pilkwang). 제출은 dry-run 4행이었으나 hidden re-run에서 정상 채점 | 0.950 | 0.94141 | ➖ public 동률. PCEN(에너지 정규화) 개선이 public에 무반영 → EoS 계열 공개 천장이 0.950으로 확정 |
| **trial_092 raunak_fork** (sub 85) | "EoS와 다른 계열의 강한 노트북이면 0.950을 넘을 수도 있다" | 다른 계열 Multi-Model Ensemble fork(raunakdey07/v9) | 0.944 | 0.94168 | ❌ public -0.006. EoS 계열이 아닌 멀티모델 앙상블은 public이 더 약함. **단 private은 0.94168로 EoS.9 원본(0.94138)보다 높았다** — public/private 상관이 깨지는 첫 신호 |
| **trial_093 karnak_fork** (sub 86) | "계층적 분류(taxonomy) 후처리가 들어간 fork면 차별화될 것이다" | Hierarchical Taxonomy PP fork(karnakbaevarthur) | 0.950 | 0.94138 | ➖ public 동률. 공개 노트북 5개를 fork해 실측한 결과 EoS 계열은 전부 0.950 → **공개 천장 0.950 확정** |
| **trial_094 eos9_sword** (sub 87) | "EoS.9 위에 자체 swordsman proto_cont(0.77/0.14)를 얹으면 검증 점수가 오르니 hidden도 오를 것이다" | EoS.9 base blend에 proto_cont 컴포넌트 전체 적용 | 0.950 | **0.94138** | ➖ public 동률. **검증(OOF) 개선이 hidden에 무효**. public 0.95087로 가장 높았으나 → **private 최저 0.94138(EoS.9 원본과 동값)** = 전형적 public 과적합 |
| **trial_095 eos9_tax** (sub 88) | "taxonomy smoothing(같은 속·강끼리 확률을 부드럽게)을 강화하면 일반화가 좋아질 것이다" | TAX_SMOOTHING genus 0.20 / class 0.08로 강화 | 0.950 | 0.94177 | ➖ public 동률. taxonomy smoothing은 public 무반영이나 **private 0.94177로 원본보다 +0.0004** — 직교 축이 hidden에서 미세하게 유효 |
| **trial_096 eos9_mix** (sub 89) | "top-level blend가 M74에 0.967로 쏠려 있다 — M51/M74를 균등화하면 다양성이 살 것이다" | top-blend M51 0.021→0.35 / M74 0.967→0.638 균등화 | 0.950 | 0.94180 | ➖ public 동률. **private 0.94180** — 균등 블렌드가 hidden에서 미세 개선. 역시 직교 축 |
| **trial_097 eos9_sed** (sub 90) | "base blend의 SED 비중을 올리면(40→45%) Sound Event 신호가 더 잡힐 것이다" | base blend SED 40→45% (proto55/sed45) | 0.950 | 0.94169 | ➖ public 동률. SED 증량도 private에서 원본보다 +0.0003 |
| **trial_098 eos9_all** (sub 91) | "sword·SED45·균등블렌드 각각이 private에서 미세 +였다 — **세 직교 축을 한꺼번에 조합**하면 누적될 것이다" | sword + SED45 + 균등블렌드 조합 | 0.95034 | **0.94238 ← private 최고** | ➖ public 동률(오히려 0.95034로 변형 중 최저 public). 그러나 **private 0.94238 = 우리 전체 최고**. public 최고였던 sword(0.95087)보다 private +0.001. **여러 직교 효과를 조합한 변형이 hidden에서 누적 이득** = Era 핵심 반전 |

> 비고: 06-03 제출 한도 5/5를 sub 87~91에서 모두 소진. public이 전부 0.950 동률이라 "어느 변형이 진짜 좋은지" public만으로는 구별 불가능했고, private 공개 후에야 eos9-all이 최고로 드러났다.

### public vs private 반전 요약 (마감 후 공개)

| 변형 | public | private | 메모 |
|---|---|---|---|
| EoS.9 fork (원본) | 0.95087 | 0.94138 | private 최저권 |
| eos9-sword | 0.95087 | 0.94138 | **public 최고 동률 → private 최저** |
| eos9-sed | 0.95086 | 0.94169 | |
| eos9-tax | 0.95046 | 0.94177 | |
| eos9-mix | 0.95018 | 0.94180 | |
| **eos9-all** | 0.95034 | **0.94238** | **public 비최고 → private 최고** |
| yaroslav fork | 0.95000 | 0.94175 | |
| raunak v9 (다른 계열) | 0.94415 | 0.94168 | public 최저인데 private 중상위 |

- **public 최고 픽(sword/EoS.9 원본, 0.95087)이 private에선 최저(0.94138)**. public 최고를 final로 골랐다면 private에서 졌다.
- **private 폭은 0.94138~0.94238으로 단 0.001** — 변형들이 hidden에서 사실상 동급. 그 미세한 차이를 만든 건 단일 강한 변형이 아니라 **여러 직교 보정을 합친 조합(eos9-all)**.
- top1 = 0.96720. 우리 최고 fork(0.950)와도 0.017 격차 — 공개 노트북 fork만으로는 메달권까지 못 간다는 한계도 동시에 드러남.

### 이 Era의 핵심 교훈

- **"내 파이프라인으로 못 한다 ≠ 대회에서 못 한다."** 자체 0.938은 진짜 천장이었지만 그건 우리 3-컴포넌트 접근의 한계였을 뿐. 격차가 크면(여기선 0.028) 파라미터·재학습·앙상블 튜닝으로는 못 메우고, **더 강한 솔루션(공개 SOTA fork)으로 교체**해야 한다 — fork 하나로 +0.012 즉시 점프. (sub_82 reflection: 세션 내내 "0.938이 최종"이라며 종료하려 했으나 밀어붙여 fork까지 가서 돌파)

- **fork 절차 자체가 재사용 자산.** `kaggle kernels pull -m`으로 metadata 확보 → kernel id를 내 계정으로 변경 → **id_no/docker_image 제거**(원본 노트북 ID 충돌이 403의 원인) → is_private push. dataset이 전부 공개면 그대로 마운트. PCEN fork는 제출 시 dry-run 4행이었지만 코드 대회 hidden re-run에서 정상 채점됐다.

- **public 동률 ≠ 실력 동률, 그리고 public 최고 ≠ private 최고.** EoS 계열 10변형이 public 0.950으로 전부 동률이라 public만으로는 우열을 가릴 수 없었고, private 공개 후 public 최고(sword 0.95087)가 private 최저(0.94138), public 비최고(eos9-all 0.95034)가 private 최고(0.94238)로 **순위가 뒤집혔다**. **public 최고 픽은 public 노이즈에 과적합한 선택**일 수 있다.

- **직교(orthogonal) 보정은 단독으론 public 무효라도 private에서 누적된다.** sword(proto)·SED45·균등블렌드는 각각 public 0.950 동률(겉보기 무효)이었지만 private에선 전부 원본보다 +0.0003~+0.0004 미세 개선. **세 축을 한꺼번에 조합한 eos9-all이 private 최고** — 서로 다른 메커니즘의 작은 이득은 합치면 남는다. final 픽은 "public 최고 단일 변형"이 아니라 "직교 효과를 모은 조합"이어야 했다.

- **공개 fork는 천장이자 동시에 한계.** 0.950은 공개 노트북 5종을 실측해 확인한 확실한 천장이었지만, top1(0.96720)과는 여전히 0.017 격차. fork는 빠르게 상위권(9.7%)에 안착시켜 주지만 메달권은 자체 차별화(더 강한 단일모델/사적 데이터/독창적 후처리) 없이는 닿지 않는다.

---

## 마감 후: public vs private 반전 분석

대회 종료(2026-06-03 23:59 UTC) 후 공개된 private(hidden) 점수 맵의 핵심은 **EoS 계열 최종 변형들이 public에선 전부 0.950 동률인데 private에선 0.94138~0.94238로 갈렸다**는 것이다.

| 변형 | public | private | 결과 |
|---|---|---|---|
| eos9-sword (proto_cont) | **0.95087 ← public 최고** | 0.94138 ← private 최저 | **public 최고 픽이 패배** |
| EoS.9 fork (원본) | 0.95087 | 0.94138 | private 최저권 |
| EoS+PCEN fork | 0.95095 | 0.94141 | |
| yaroslav-v221 fork | 0.95000 | 0.94175 | |
| eos9-tax (taxonomy smoothing) | 0.95046 | 0.94177 | 직교 축 +0.0004 |
| eos9-sed (SED 45%) | 0.95086 | 0.94169 | 직교 축 +0.0003 |
| eos9-mix (M51/M74 균등) | 0.95018 | 0.94180 | 직교 축 +0.0004 |
| raunak v9 (다른 계열) | 0.94415 ← public 최저 | 0.94168 | public 최저인데 private 중상위 |
| **eos9-all (3축 조합)** | 0.95034 | **0.94238 ← private 최고** | **public 비최고가 private 최고** |

**왜 public 최고 픽(sword)이 졌나.** swordsman proto_cont 추가는 자체 OOF(검증)를 올렸고 public도 0.95087로 가장 높았다. 하지만 그건 **public 600여 파일에 과적합한 선택**이었다 — proto_cont의 신호가 public split에서만 우연히 잘 맞았고, hidden 전체에서는 EoS.9 원본과 똑같은 0.94138로 수렴했다. public 0.95087 - private 0.94138 = **0.0095 갭**은 "public 최고 = 가장 과적합된 픽"이라는 전형이다.

**왜 다축 조합(eos9-all)이 이겼나.** sword·SED45·M51M74 균등블렌드는 **서로 다른 메커니즘**(proto 컴포넌트 / Sound Event 비중 / top-blend 분산)이고, 각각 단독으로는 public 0.950 동률(겉보기 무효)이지만 private에서 전부 원본보다 +0.0003~+0.0004였다. 서로 직교하는 작은 이득이라 **합치면 상쇄되지 않고 누적**된다 → eos9-all이 private 0.94238로 우리 전체 최고. public이 오히려 가장 낮았던(0.95034) 변형이 hidden에서 이긴 것이다.

**final 선택 교훈.** 이 대회의 public-private 갭은 약 0.0095로 일정했고, EoS 변형 간 public 차이(~0.0009)는 그 갭보다 작아 **public 순위는 노이즈에 가까웠다.** 만약 final 2개를 "public 최고 단일 변형"으로 골랐다면 sword/원본(private 0.94138)에 묶여 졌다. 실제로 이긴 픽은 **"public이 안 떨어지는 한 직교 보정을 최대한 많이 합친 조합"** — public 신호를 신뢰하지 말고, 이론적으로 직교이고 private에서 잃지 않을 보정을 누적하는 게 정답이었다.

---

## 핵심 교훈 종합

대회 전체(98 trial, 91 submission)에서 뽑은 재사용 가능한 교훈:

1. **로컬 val ≠ LB, OOF가 거짓말을 한다.** 자체 OOF 0.97 → 제출 0.0/0.91. 검증셋이 hidden test와 구성이 달라 OOF는 LB를 거의 예측하지 못했다. 검증은 오직 Kaggle 직접 제출. 로컬 튜닝은 **파이프라인이 100% 동일할 때만** 의미가 있다(간소화된 로컬 LR 최적값이 복잡한 Kaggle 파이프라인에선 -0.008 역효과).

2. **macro ROC-AUC는 rank 기반이라 monotonic/affine 변환이 전부 무효다.** 온도 스케일링(T_AVES, trial_070), logit z-score 정규화(084), 상수 prior 배수(prior mask) — 모두 클래스 내 순위를 안 바꿔 0 효과. 점수를 바꾸려면 **클립 순위를 실제로 뒤집는** 변경(다른 모델 가족, cross-file/window 변환)이어야 한다. 이론으로 미리 죽일 수 있는 trial은 제출을 아껴라.

3. **약한 보조 모델의 비중을 키우면 강한 주력을 묽게 할 뿐이다(weak-model-hurts-ensemble).** EffNet/HGNet/SED/ConvNeXt 어느 것도 OOF가 주력(Perch/Proto)보다 낮으면 비중↑이 -방향. 보조 비중엔 **sweet spot**이 있다(SED 18→40% +0.003, 50% -0.001 / EffNet 5% 동률, 10% 하락). 단조 증가가 아니다.

4. **diversity는 "다른 모델"이 아니라 "다른 오류 패턴"에서 나온다.** distill·SED·dual-ProtoSSM은 결국 Perch/Proto의 증류·파생이라 logit-space에서 직교 정보가 없어 0 효과. 진짜 이득은 mel-spec CNN(EffNet, +0.001) / CE vs 증류 학습방식 차이(+0.002) / pseudo-mix soft label(+0.001)처럼 **임베딩 소스·학습 방식·데이터가 근본적으로 다를 때**만 나왔다.

5. **보조 컴포넌트 fold 앙상블은 4개에서 포화한다.** pmix 2-fold=0, 4-fold=+0.001, 5-fold=0. 작은 weight(3~8%)에선 4→5-fold의 추가 분산 절감이 LB granularity(~0.001) 아래로 묻힌다. "코드 변경 0, dataset 버전만 올리기"가 가장 싼 +0.001이지만 거기서 멈춰야 한다.

6. **blend weight space는 한 점 근처에서 평평하다.** ±2~4pp를 8방향 전부 흔들어도 public 동률(Era 4의 11 trial 0.934, Era 5의 8 trial 0.938). LB 해상도 안에서 weight 미세조정은 신호가 아니라 노이즈고, **private에선 오히려 더 흔들수록 살짝 내려가는 과적합** 경향이 있었다.

7. **앙상블 fusion 연산자(logit 가중 합)는 core 메커니즘이라 건드리면 크게 무너진다.** "공정한" rank-average로 바꾼 trial_057이 -0.005(private -0.009). logit의 **magnitude 자체가 신호**다 — Perch의 강한 confidence와 보조 모델의 다이내믹 레인지 차이가 effective weight를 만들어낸다. weight·후처리는 marginal, fusion·컴포넌트 구성은 core로 위험을 구분하라.

8. **코드 대회의 진짜 적은 정확도가 아니라 타임아웃이고, hidden test는 대회 중 커진다.** 4/4 성공한 코드가 4/5에서 타임아웃. dry-run(20파일) ≠ 채점(600+파일), 30배 차이. **ONNX Runtime(2x, TF 의존성 제거)**이 이 벽을 뚫었고 TFLite는 OOM으로 막다른 길이었다.

9. **"kernel COMPLETE"는 "채점 성공"이 아니다.** 외부 dataset 마운트가 hidden re-run에서 거부되면 submission 행 자체가 안 생긴다(silent reject). ConvNeXt·BirdNET·yusuf 통fork가 전부 이 함정. 새 dataset 추가 시 hidden-test 호환성을 먼저 검증하고, "10분 내 채점 행 미생성이면 즉시 fallback" 단계가 필요하다. COMPLETE 상태에서 90분씩 폴링을 낭비하지 마라.

10. **검증된 ref/공개 솔루션 통째 복사 > 내 임의 튜닝.** 자체 임의 배합(trial_078) 0.933 → ref config 그대로(079) 0.935. 새 파이프라인 이식 초기엔 reference를 byte 단위로 재현해 baseline부터 세워라. blend 공식 한 줄 누락(cell 68)이 trial 6개를 통째로 날린 것처럼, "설정이 실제 출력에 반영되는지" 진단 print로 먼저 확인하라.

11. **자체 천장을 인정하는 타이밍이 결정적이다.** 0.934(Era 4)·0.938(Era 5)에서 같은 축을 수십 trial 반복하며 reflection은 매번 "새 1차 모델 필요"라고 옳게 진단했지만 실행은 계속 weight 미세조정이었다. 격차가 크면(top과 0.028) 튜닝으로 못 메운다 — **"내 접근의 한계 ≠ 대회의 한계"**를 인정하고 더 강한 공개 SOTA로 교체하는 결단(fork +0.012)이 천장을 깬 유일한 수였다.

12. **public 최고 픽 ≠ private 최고. final은 직교 보정의 조합으로 골라라.** public-private 갭(~0.0095)이 변형 간 public 차이(~0.0009)보다 컸다 → public 순위는 노이즈. public 최고(sword 0.95087)가 private 최저(0.94138), 단독으론 public 무효지만 private에서 미세 +였던 **직교 3축을 합친 eos9-all이 private 최고(0.94238)**. public을 신뢰하지 말고 "잃지 않을 직교 보정"을 누적하는 게 final 전략이었다.

13. **fork는 상위권 입장권이자 동시에 한계다.** 공개 EoS.9 fork로 +0.012 점프해 상위 9.7%(398/4085)에 안착했지만, top1(0.96720)과는 0.017 격차가 남았다. 공개 노트북 fork만으로는 메달권에 못 가고, 그 위는 **자체 차별화(더 강한 단일모델/사적 데이터/독창적 후처리)** 없이는 닿지 않는다.

---

## 디렉토리 가이드

| 경로 | 내용 |
|---|---|
| `SUBMISSIONS.md` | 제출 인덱스 — sub 번호, public score, status(채점 성공/타임아웃/silent reject)를 시간순으로 기록한 마스터 로그 |
| `TRIALS.md` | trial 인덱스 — 각 trial의 가설(왜 시도), key change, val/public 점수, status를 정리한 실험 대장 |
| `submissions/sub_NN/` | 제출 단위 작업 폴더. 각 폴더에 노트북(.ipynb)/kernel-metadata.json/`reflection.md`(가설·결과·교훈·다음 가설). base 노트북이 다르면 새 sub 폴더로 분리 |
| `submissions/sub_NN/reflection.md` | 그 제출의 정성적 기록 — 왜 그 가설을 세웠고, 점수가 왜 그렇게 나왔고, 무엇을 버리고 다음에 뭘 할지 |
| `notebooks/` | 공통 노트북·파이프라인 코드(Perch ONNX 추론, EffNet 학습, ProtoSSM/SED blend, fork base 노트북) |

---

*BirdCLEF 2026 (2026-03 ~ 2026-06-03 마감). 최종 public 398/4085 (상위 9.7%), private best 0.94238. 98 trial / 91 submission. 이 README는 SUBMISSIONS.md·TRIALS.md·각 sub의 reflection.md를 종합해 작성됨.*