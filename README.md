# Kaggle 실험 레포

| 폴더 | 대회 | Best Public | 상태 |
|------|------|-------------|------|
| `churn/` | Playground S6E3 — 고객 이탈 예측 | 0.91707 (private 0.91815) | 84+ trial, 15 제출, 대회 종료 |
| `irrigation/` | Playground S6E4 — 관개 수준 분류 | 0.9721 | 9 trial, 6 제출, 진행 중 |
| `birdclef/` | BirdCLEF+ 2026 — 새소리 종 분류 | 0.928 | 17 trial, 6 제출, 진행 중 |
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
**핵심 난관**: 메트릭이 accuracy가 아니라 balanced_accuracy. 3클래스(Low/Medium/High) 중 High가 minority. threshold 최적화와 pairwise feature engineering이 핵심.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001 LightGBM baseline | 데이터 파악 겸 baseline | val 0.9844(acc), **public 0.9589** | 메트릭을 bal_acc로 수정 |
| 002 FE + 앙상블 | 도메인 피처(ET_proxy, water_balance) + 3모델 앙상블 | val 0.9853(acc), **public 0.9609** | bal_acc 기준 재평가 필요 |
| 003 balanced blend | bal_acc 메트릭 수정 + class_weight + threshold opt(High×2.6) | val **0.9711**(bal_acc), **public 0.9691** | pairwise TE 확장 |
| 004~006 TE 탐색 | target encoding 변형 (28 pair, ext data, full pairwise 171) | val 0.9692~0.9699 | pairwise는 factorize만, TE는 원본 cat에만 |
| 007 stacking | Ridge meta-learner + bias tuning + CatBoost 주도 | val 0.9707 | sklearn TE로 전환 |
| 008 sklearn TE | multiclass TE(cv=5) 265 cols + bias tuning | val 0.9712, **public 0.9692** | TE를 cat_cols에만 한정 |
| 008b fullpair | 171 pairwise factorize + cat_cols TE(24) + threshold(High×3.7) | val **0.9738**, **public 0.9721** 🏆 | stat group features |
| 009 stat group | 88 pairs × 4 stats = 352 cols + orig TE | val 0.9710 | 008b보다 낮음, 과적합 |

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 001 baseline | 0.9589 | raw features + LightGBM. gap -0.025 |
| 02 | 002 FE+앙상블 | 0.9609 | 도메인 FE + 3모델 앙상블 |
| 03 | 003 balanced | 0.9691 | bal_acc 메트릭 수정 + threshold opt. **+0.008 점프** |
| 04 | 008b fullpair | **0.9721** | 171 pairwise factorize + multiclass TE + threshold(High×3.7). **best** |
| 06 | 006 pairwise | 0.9668 | full pairwise 171 + XGB only |
| 08 | 008 sklearn TE | 0.9692 | 265 TE cols → 008b보다 noise 많아 낮음 |

---

## birdclef (BirdCLEF+ 2026)

**한 줄 요약**: 60초 야외 녹음을 5초씩 잘라서 234종의 새/개구리/곤충을 맞추는 multi-label 분류. macro-averaged ROC-AUC.
**핵심 난관**: Code Competition이라 Kaggle 노트북에서만 제출 가능. CPU 90분 제한. 자체 파이프라인 16번 연속 실패 후 공개노트북 fork로 전환.

### 실험 흐름

| Trial | 왜 시도했나 | 결과 | 다음엔 |
|-------|------------|------|--------|
| 001~006 자체 파이프라인 | Perch v2 임베딩 + LightGBM/XGBoost/LR | 로컬 val 0.97이지만 **Kaggle 16번 연속 실패** (경로/GPU/TF 버전/timeout) | 공개 노트북 fork |
| 007 0.912 fork | Perch logits + Bayesian prior + LR probe | **Public 0.912** — 첫 유효 제출 | post-processing |
| 008 post-processing | temperature + file-level + rank-aware scaling | **0.910** — 악화. OOF 미검증 실수 | fork 전략 변경 |
| 009~014 개선 시도 | PCA sweep, pseudo-label, CNN, 파라미터 변경 | 013: 0.904 하락, 014: 타임아웃 | 더 높은 점수 fork |
| 015 0.926 fork | yukiZ 0.926 노트북 fork. dataset 누락→재학습 | **Public 0.928** 🏆 (+0.002 보너스) | multi-seed 앙상블 |
| 016~017 multi-seed | 5-seed ProtoSSM 앙상블 (API push / 웹 수정) | **둘 다 점수 없음** — submission 생성 실패 | 노트북 output 로그 확인 필요 |

### 제출 기록

| sub | trial | public | 왜 이 결과가 나왔나 |
|-----|-------|--------|---------------------|
| 01 | 007 fork | 0.912 | 0.912 공개노트북 fork. 첫 유효 제출 |
| 02 | 008 post-processing | 0.910 | OOF 미검증 상태로 후처리 추가 → 악화 |
| 03 | 013 param change | 0.904 | PCA96+C0.1 → 하락 |
| 04 | 015 fork_926 | **0.928** | yukiZ 0.926 fork. dataset 누락→재학습으로 +0.002. **best** |
| 05 | 016 API push | - | 빈 모델 학습 실패 (ProtoSSM_PATH 문제) |
| 06 | 017 5-seed | - | 웹 수정 multi-seed. submission 생성 실패 |

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
