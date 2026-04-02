# Kaggle 실험 레포

| 폴더 | 대회 | Best Public | 상태 |
|------|------|-------------|------|
| `churn/` | Playground S6E3 — 고객 이탈 예측 | 0.91707 (private 0.91815) | 84+ trial, 15 제출, 대회 종료 |
| `irrigation/` | Playground S6E4 — 관개 수준 분류 | 0.9609 | 2 trial, 2 제출, 진행 중 |
| `birdclef/` | BirdCLEF+ 2026 — 새소리 종 분류 | 0.912 | 진행 중 (post-processing 실험) |
| `ts-forecasting/` | Hedge Fund — 시계열 예측 | 0.1499 | 4번 제출, 3번 0점 |
| `march-mania/` | March Mania 2026 — NCAA 농구 예측 | 미제출 | 마감 놓침 |

---

## churn (Playground S6E3)

**한 줄 요약**: 통신사 고객이 이탈할지 예측. AUC-ROC (높을수록 좋음, 최대 1.0). 4,142팀 참가.
**핵심 난관**: 상위권이 전부 0.914~0.917에 몰려있어서 0.001이 수십 등수 차이. 로컬 GBDT 한계를 Kaggle 노트북 fork(RealMLP)로 돌파.
**최종 성적**: Best public **0.91707**, Best private **0.91815**. 15회 제출, 84+ trials.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 LightGBM | 데이터 파악 겸 baseline | val 0.9161, **public 0.9138**. 기본으로도 상위권 | target encoding + 하이퍼파라미터 튜닝 |
| 002~004 피처+튜닝 | ChargeGap(요금 증가율), Optuna 50회 탐색 | val 0.9166 → **public 0.9139**. 단일 모델 한계 체감 | 여러 모델 앙상블 |
| 005~008 앙상블 | LightGBM + XGBoost + CatBoost 조합 | val 0.9167 → **public 0.9140**. +0.001 | 비중 최적화 |
| 009~013 실패 모음 | 외부 데이터(Telco 7천행), "No internet"→"No" 통합, 10-Fold | **전부 val 하락**. 외부 데이터는 분포가 달라서 독 | 외부 데이터 포기, 모델 다양성에 집중 |
| 014~017 최적 blend | 5모델 OOF grid search. XGBoost가 0.4로 주도 | val 0.9168 → **public 0.91404**. 여기가 GBDT 천장 | multi-seed, 다른 모델 |
| 018~024 총동원 | groupby 83변수, multi-seed(7시드×5폴드=35모델), 상대적 위치 피처 | val 0.9169(최고) → **public 0.9140 미돌파**. 과적합 확인 | GBDT 계열로는 한계. NN 필요 |
| 025~059 대탐색 | MLP, RealMLP(로컬), Ridge, HistGBM, DART, pseudo-labeling, WoE, calibration, depth=1... **총 35개 trial** | **전부 돌파 실패**. 로컬 NN은 M1 Mac 한계 | Kaggle 노트북에서 NN 돌리기 |
| 060 RealMLP fork | Kaggle 노트북에서 RealMLP 20-fold 실행 | val 0.9194 → **public 0.91683**. **+0.003 돌파!** | RealMLP + XGBoost blend |
| 061 RealMLP+XGB | RealMLP×0.85 + XGBoost×0.15 | val 0.9195 → **public 0.91686** | distribution features로 추가 개선 시도 |
| 067 distribution | pctrank, zscore, 숫자 자릿수 피처 등 | val 0.9185 → **public 0.91571**. 과적합 | feature 과잉 → 정규화 필요 |
| 074 TE std enriched | TE mean+std + 120 combo + digit features. distribution 없이 clean | val **0.91762** — 로컬 GBDT best | Ridge ensemble에 기여 |
| 075~084 Ridge ensemble | 55개 OOF를 Ridge(alpha=100)로 앙상블. 약한 모델도 필터 없이 전부 투입 | val 0.9196 → **public 0.91707** → **private 0.91815** 🏆 | 대회 종료 |

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 001 baseline | 0.91377 | raw features만으로도 상위권 |
| 02 | 004 튜닝 | 0.91393 | Optuna로 소폭 개선 |
| 03 | 014 앙상블 | **0.91404** | 5모델 blend. GBDT 최고점 |
| 04~06 | 017~028 | 0.91395~0.91404 | 여러 조합 시도했지만 0.914 벽 못 넘음 |
| 07 | 060 RealMLP | **0.91683** | Kaggle 노트북 fork. NN이 GBDT 벽을 넘음 |
| 08 | 061 blend | **0.91686** | RealMLP + XGBoost blend |
| 09 | 067 distribution | 0.91571 | 피처 과잉으로 과적합 |
| 10 | Ridge 55 OOFs | **0.91707** | 55개 OOF Ridge ensemble. **best public** |
| 11~15 | 078~084 | 0.91689~0.91704 | Ridge 변형 + blend 조합. **private 0.91815 🏆** |

---

## irrigation (Playground S6E4)

**한 줄 요약**: 토양/기상/작물 정보로 관개(물 공급) 수준을 3단계(Low/Medium/High)로 분류. accuracy (높을수록 좋음).
**핵심 난관**: baseline이 이미 98.4%라 개선 여지가 매우 작음. Val-Public gap이 -0.025로 꽤 커서 일반화가 관건.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 LightGBM baseline | 데이터 파악 겸 baseline. label encoding + 5-fold | val 0.9844, **public 0.9589**. fold std 0.0002로 안정적이지만 gap -0.025 큼 | domain FE로 일반화 개선 |
| 002 FE + 앙상블 | "증발산(ET_proxy), 수분수지(water_balance) 등 도메인 피처를 넣으면 일반화될 것" | val 0.9853, **public 0.9609**. public 개선폭(+0.002)이 val(+0.0009)보다 큼 → FE가 일반화에 효과적 | CatBoost가 0.9824로 가장 약함 — 제외하거나 튜닝 필요 |

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 001 baseline | 0.9589 | raw features + LightGBM. gap -0.025로 train/test 분포 차이 시사 |
| 02 | 002 FE+앙상블 | **0.9609** | ET_proxy, water_balance 등 11개 FE + LGBM(4)+XGB(5)+CAT(1) 앙상블. gap -0.024로 소폭 개선 |

---

## birdclef (BirdCLEF+ 2026)

**한 줄 요약**: 60초 야외 녹음을 5초씩 잘라서 234종의 새/개구리/곤충을 맞추는 multi-label 분류. macro-averaged ROC-AUC.
**핵심 난관**: Code Competition이라 Kaggle 노트북에서만 제출 가능. CPU 90분 제한. 자체 파이프라인이 전부 실패하고 공개노트북 fork로 첫 제출 성공.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001~003 자체 파이프라인 | Perch v2 임베딩(1536차원)을 뽑아서 LightGBM/XGBoost에 넣으면 될 것 | 로컬 val AUC 0.97까지 올랐지만 **Kaggle 제출 16번 연속 실패** (경로, GPU 제한, TF 버전, timeout 등) | 자체 파이프라인 포기. 검증된 공개 노트북 fork |
| 004 LR+PCA64 | Discussion에서 LR이 XGBoost보다 좋다는 걸 발견 | val **0.9754** (XGBoost 0.9580보다 좋음). 41초 완료 | 공개 노트북도 LR 사용 확인 |
| 005 PCA sweep | PCA 차원(64~1536)에 따른 XGBoost 성능 비교 | no PCA(0.9580)가 best이지만 LR+PCA64(0.9754)가 전부 이김 | XGBoost는 버리고 LR로 통일 |
| 007 공개노트북 fork | 0.912 공개노트북(Perch logits 직접 매핑 + Bayesian prior + LR probe + Gaussian smoothing) fork | **Public 0.912** — 첫 유효 제출! | post-processing 추가로 0.916+ 노려볼 것 |
| 008 post-processing | temperature scaling(새=1.10, 개구리/곤충=0.95) + file-level/rank-aware scaling 추가 | **Public 0.910** — 오히려 악화(-0.002) | OOF 검증 없이 제출한 실수. 원본으로 되돌리고 probe 튜닝만 시도 |

### 16번의 노트북 삽질 (v1 → v16)

| 문제 유형 | 버전 | 뭐가 터졌나 |
|----------|------|-----------|
| 경로 | v1~v4 | 인터넷 차단(URL 불가), 대회 데이터 경로(`/competitions/` 필요), 모델 마운트 경로 |
| 코드 | v5~v7 | 변수명 불일치, try-except가 에러 삼킴 |
| 성능 | v8~v9 | CPU XGBoost 230종 학습 timeout, `gpu_hist` 폐지 |
| 환경 | v10~v12 | test 파일 0개(commit 단계), GPU 제출 불가(GPU max=0분) |
| TF | v13~v14 | TFLite 변환 불안정, Perch v2_cpu가 TF 2.20 필요(기본 2.19) |
| 의존성 | v15 | TF wheel이 dataset 아닌 kernel_sources로 마운트 |
| **성공** | **v16** | **0.912 fork + TF 2.20 wheel + perch-meta 캐시 = 성공** |

**교훈**: Code Competition에서 삽질하지 않으려면 **검증된 공개 노트북을 fork**하고, 거기에 개선점을 얹어라. 처음부터 만들면 환경 차이만으로 일주일 날린다.

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 007 fork | **0.912** | Perch logits(14,795종→234종 매핑) + site×hour Bayesian prior + LR probe. 공개 노트북과 동일 |
| 02 | 008 post-processing | 0.910 | temperature/file-level/rank-aware scaling 추가. OOF 미검증 상태로 제출 → 악화 |

---

## ts-forecasting (Hedge Fund)

**한 줄 요약**: 금융 시계열 36,923개 예측. 시험 데이터의 89%가 처음 보는 새 시리즈. weighted RMSE score.
**핵심 난관**: 새 시리즈는 아무 정보 없이 예측해야 하는데, 특정 시리즈 하나(중요도 13조)를 틀리면 점수가 0이 됨.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 기본 모델 | raw feature만으로 LightGBM | **public 0.1499**. 유일하게 0점 아닌 제출 | 직전값(lag)이 AR(1)=0.86으로 강력 → 추가 |
| 002~010 직전값 | 직전값을 넣으면 val 0.89까지 올라감 | **제출하니 0.0000**. 시험엔 직전값이 없었음 | val에서 정답지 보고 채점한 꼴. val을 시험처럼 재구성해야 |
| 015~021 cold-start val | "처음 보는 시리즈" 상황을 val에서 재현 | val 0.64. **또 0.0000** | 83EG83KQ(중요도 13조, 실제값≈0)에 6.37 예측 → 분자 폭발 |
| 054~058 weight 압축 | weight를 0.05제곱으로 압축해서 균등 학습 | val 0.59. **또 0.0000** | 그룹 평균(-0.67)이 새 시리즈(실제값≈0)에 독. "0보다 나쁜 예측" |

### 0점 3번의 교훈

1. **val이 시험을 반영 못 함** (sub_02) — 시험엔 직전값이 없는데 val에선 있었음. val 0.89에 속아서 제출 → 0점
2. **고가중치 시리즈 폭발** (sub_03) — 중요도 13조인 시리즈에 6.37 예측. 실제값은 0.000009 → 오차 5.6×10¹⁴
3. **그룹 평균이 독** (sub_04) — 훈련 평균(-0.67)을 새 시리즈에 쓰면 0으로 찍는 것보다 나쁨

**핵심 발견**: 대회 호스트가 "public 0.5+ 점수는 미래 데이터 사용(반칙) 의심"이라고 발언. 정직한 ceiling이 0.3~0.5.

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 001 raw | **0.1499** | raw feature만 → 안전하지만 약함 |
| 02 | 010 lag | 0.0000 | 시험에 lag가 없어서 전부 NaN → 0 |
| 03 | 021 cold-start | 0.0000 | 중요도 13조 시리즈 예측 폭발 |
| 04 | 058 weight압축 | 0.0000 | 그룹 평균(-0.67)이 실제값(≈0)보다 나쁜 예측 |

---

## march-mania (March Machine Learning Mania 2026)

**한 줄 요약**: NCAA 농구 토너먼트 승리 확률 예측. Brier Score (낮을수록 좋음).
**핵심 난관**: 제출 마감(3/19)을 확인 안 하고 시작해서 제출 못 함.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 시드 차이 | 기본 승률로 예측 | Brier 0.259. 전부 0.5로 찍는 것(0.250)보다 나쁨 | 전문가 랭킹 추가 |
| 002 랭킹 | 197개 전문가 랭킹 평균 | Brier 0.240. 개선 | Elo + 상세 스탯 |
| 003~005 앙상블 | Elo + FG% + 리바운드 + 최근 10경기 폼 | **남자 0.161, 여자 0.132**. discussion 상위권 도달 | **마감 지남** 😭 |

**삽질**: "9 days to go"가 제출 마감이 아니라 토너먼트 결과 확정까지 남은 시간이었음.

---

## 전체 교훈

1. **val score를 맹신하면 안 됨** — val이 시험 상황을 반영하는지 항상 확인 (ts-forecasting 3번 0점)
2. **제출 전 예측값 분포 확인** — 점수 0이 나오는 건 예측값 폭발/편향이 원인
3. **로컬 한계를 느끼면 Kaggle 노트북 fork** — churn에서 GBDT 벽을 RealMLP fork로 돌파(+0.003), BirdCLEF에서도 공개 노트북 fork가 정답
4. **약한 모델도 버리지 마라** — Ridge ensemble에 55개 OOF 전부 넣었더니 best. Chris Deotte도 "필터링하지 마라" 조언
5. **Code Competition은 환경 삽질이 반** — 처음부터 만들지 말고 검증된 노트북에 얹어라
6. **대회 마감일 먼저 확인** — "X days to go"가 제출 마감이 아닐 수 있음
7. **OOF 검증 없이 제출하지 말 것** — post-processing도 반드시 로컬 val 먼저
8. **Val 올리는 방향이 맞다** — churn에서 Val-Public gap은 음수였지만 Private이 Public보다 높게 나옴. Val이 더 정확한 지표

---

자세한 규칙: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)
