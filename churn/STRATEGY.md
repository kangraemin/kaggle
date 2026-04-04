# Next Strategy — 2026-03-31 (v5)

## 회고 요약
- 지금까지 best local CV: **0.91850** (trial_067, distribution_digit) — LB 0.91571로 overfit 확인
- best LB: **0.91686** (trial_061, RealMLP×0.85 + XGB×0.15 blend)
- best "safe" CV: **0.91695** (trial_036, 9-model grid search blend)
- 총 70 trials 진행, trial_071/074 코드 작성됨 (미실행)
- v4에서 제안한 trial_068~070 모두 실패 (feature 추가 포화 확인)

### LB 현황 (2026-03-31)
| 순위 | 이름 | Public LB | 비고 |
|---|---|---|---|
| 1 | Chris Deotte | 0.91771 | 3-model GPU blend (LogReg+XGB+MLP) |
| 2 | Optimistix | 0.91757 | 비공개 |
| 3 | W-Horacio | 0.91754 | 비공개 |
| 4 | Traiko Dinev | 0.91752 | Single XGB + TE std + all combos |
| ... | 우리 best | 0.91686 | trial_061 (RealMLP+XGB blend) |

**Gap to top**: 0.91771 - 0.91686 = **0.00085**

### 효과있었던 것 (누적)
- Multi-seed averaging (7 seeds): 안정적
- XGBoost: 이 데이터셋에서 일관 우세
- RealMLP: LB 최고 (0.91686) — tree와 다른 패턴을 잡아 blend 효과적
- 2-way categorical combos + pseudo labels: trial_066에서 CV 0.91710 달성

### 실패한 것 (반복 금지)
- Feature 추가 포화: combo+dist 합체(068), KNN(069), N-gram(070) — 피처 수 늘리는 건 효과 없음
- Distribution features: CV 0.91850이지만 LB 0.91571 (overfit)
- 단순 regularization 변경 (037~040)
- Non-tree 단독 (041, 042, 047, 049): 대폭 하락
- Shallow trees (058, 059): 대폭 하락
- Feature selection/제거 (045, 064): 정보 손실
- 단순 Ridge→XGB (065): 구현 부실

### 핵심 교훈
- **Feature 수를 늘리는 건 포화.** TE std 같은 질적 변환만 유효할 가능성
- **Distribution features는 overfit.** Synthetic data에서 분포 기반 feature는 train에만 맞음
- **Diverse 모델 blend가 LB에서 가장 효과적.** Top LB 대부분 submission-level blending
- **20-fold CV**: BlamerX, RealMLP 모두 20-fold 사용. 우리는 대부분 5-fold

## Discussion/노트북 인사이트 (v5 업데이트)

### 1. BlamerX Ridge→XGB (CV 0.91927) — 재분석 완료 🔍
최신 코드 확인 결과:
- **20-fold outer CV**, 5-fold inner TE
- Ridge (alpha=10) → XGB two-stage (Ridge OOF를 XGB feature로 추가)
- **Inner-fold TE stats**: `std`, `min`, `max` (mean이 아닌!) — 모든 categorical + num-as-cat에 적용
- **N-gram TE**: top 6 categoricals의 bigram/trigram 조합 → TE mean으로 변환
- XGB params: **lr=0.0063**, colsample=0.32, subsample=0.81, reg_alpha=3.5, reg_lambda=1.29
- Distribution features 포함 (orig data 사용)
- **핵심 차이점**: 매우 낮은 lr + 좁은 colsample + TE std/min/max + Ridge hint + N-gram TE

### 2. RealMLP (Vladimir Demidov) — 2026-03-31 업데이트 🆕
최신 노트북 확인:
- **20-fold** StratifiedKFold
- **Heavy binning**: TotalCharges → KBinsDiscretizer (5000 quantile bins, 400 quantile/kmeans bins)
- **Digit features**: TotalCharges d-3 (1000의 자리), MonthlyCharges decimal
- **Mod features**: TotalCharges mod100, mod1000, is_multiple_10
- **Numericals as categorical**: tenure, MonthlyCharges, TotalCharges → factorize
- **3-way combo**: Contract × InternetService × PaymentMethod → TE
- RealMLP params: hidden [512,256,128], n_ens=8, lr=0.075, embedding_size=6
- **시사점**: 수치형을 대량의 카테고리 bin으로 변환하면 RealMLP에서 더 효과적

### 3. Chris Deotte — LB #1 (0.91771) 🆕
- 3-model blend: LogReg(TE pair) + XGB + Custom MLP
- GPU 필수 (cuML LogReg, CUDA MLP)
- LB 1위지만 단일 모델 CV는 0.9178로 낮음 → **diversity가 핵심**

### 4. saamhm — Undersampling Diversity 🆕
- 다수 클래스를 5등분 → 각 등분+다른 부분 샘플+전체 소수 클래스로 5개 diverse training set
- LGBM + CatBoost × 5 sets
- **시사점**: 데이터 레벨 diversity도 효과적 (우리는 seed diversity만 사용)

### 5. 상위 LB 공통 패턴 (변동 없음)
- **Submission-level blending이 지배적**: top 10 대부분 여러 공개 노트북 출력을 blend
- 이는 단일 파이프라인 최적화의 한계를 시사

## 갭 분석: 우리 vs 상위

### CV 개선 가능 요소
| 요소 | 영향도 | 현재 상태 | 근거 |
|---|---|---|---|
| **TE std/min/max inner-fold** | ★★★★★ | 미적용. trial_074 코드 있음 | BlamerX/Traiko 모두 사용 |
| **20-fold CV** | ★★★★☆ | 5-fold 사용 중 | BlamerX/RealMLP 모두 20-fold |
| **Ridge→XGB two-stage** (재구현) | ★★★★☆ | trial_065 실패. 재구현 필요 | BlamerX 핵심 구조 |
| **Low lr + narrow colsample** | ★★★★☆ | lr=0.05, colsample=0.8 | BlamerX: lr=0.0063, colsample=0.32 |
| Heavy numerical binning | ★★★☆☆ | 미적용 | RealMLP에서 효과적 |
| Data-level diversity (undersampling) | ★★☆☆☆ | 미적용 | saamhm 노트북 |

### LB 개선 가능 요소
| 요소 | 영향도 | 현재 상태 |
|---|---|---|
| Distribution features 제거 | ★★★★★ | trial_067 기반 모델은 LB에서 overfit |
| 3-model diversity blend | ★★★★☆ | RealMLP+XGB만 시도 |
| 20-fold + clean features | ★★★★☆ | 미적용 |

## 다음 3 trials 전략

### 1. trial_074_te_std_enriched (기존 코드 있음, 최우선 실행)
**Inner-fold TE std/min/max features + clean feature set — BlamerX/Traiko 검증 방식**

이전 전략에서도 제안했으나 미실행. **가장 높은 기대값**.

구현:
- **Distribution features 제외** (overfit 방지)
- Charge features + ORIG stats(mean+std) + digit features 유지
- Inner 5-fold TE pipeline:
  - raw cats (15개) + 2-way combos (상위 6개 조합)
  - TE stats: **mean + std + min + max** (BlamerX 방식 확대)
  - Leakage 방지: outer fold train에서만 inner fold 구성
- XGB 7 seeds × 5-fold
- **근거**: BlamerX(0.91927), Traiko(0.91789) 모두 TE std 사용. 우리 trial_066(0.91710)에 TE std만 추가하면 improvement 기대
- **목표**: CV 0.917+ (distribution 없이), LB에서 안정 (overfit gap < 0.002)

### 2. trial_076_blamerx_pipeline (🆕 — BlamerX 핵심 재현)
**Ridge→XGB two-stage + 20-fold + low lr + N-gram TE — BlamerX 전체 파이프라인 충실 재현**

trial_065에서 Ridge→XGB를 시도했으나 구현이 부실해서 실패. BlamerX 코드를 참조해서 정확히 재현.

구현:
- **20-fold outer CV**, 5-fold inner TE
- Inner-fold TE stats: `std`, `min`, `max` (모든 cats + num-as-cat)
- N-gram TE: top 6 categoricals (Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling)의 bigram(C(6,2)=15) + trigram(C(6,3)=20) → TE mean
- Ridge (alpha=10) OOF → XGB의 추가 feature로 사용
- XGB params 대폭 변경: **lr=0.006, colsample=0.32, subsample=0.81, reg_alpha=3.5, reg_lambda=1.3**
- **Distribution features 포함하되** 원본 데이터 기반 (BlamerX와 동일)
- 7 seeds × 20-fold
- **근거**: BlamerX가 CV 0.91927 달성한 정확한 파이프라인. distribution features를 포함하지만, Ridge hint + low lr + narrow colsample이 regularization 역할을 해서 overfit 완화 가능
- **위험**: distribution features 포함 → LB에서 떨어질 수 있음. 단, BlamerX는 이 방식으로 LB에서도 상위에 있음
- **목표**: CV 0.919+

### 3. trial_077_diverse_blend_20fold (🆕 — 최종 diversity blend)
**trial_074 XGB + trial_076 XGB + RealMLP OOF — 3-source diversity blend, 20-fold**

구현:
- **Source 1**: trial_074 XGB OOF (clean features + TE std)
- **Source 2**: trial_076 XGB OOF (Ridge→XGB pipeline)
- **Source 3**: RealMLP OOF (trial_060/061 방식, 20-fold 재학습 또는 기존 OOF 활용)
- Grid search로 최적 blend 가중치 탐색
- Rank averaging도 시도
- **근거**: Chris Deotte가 LB #1 (0.91771)인 이유는 완전히 다른 3개 모델 타입의 blend. 우리도 (clean XGB + Ridge→XGB + RealMLP) 3가지 diverse source를 blend
- **의존성**: trial_074 + trial_076 완료 후 실행
- **목표**: LB 0.917+ (우리 best LB 0.91686 대비 +0.001+)

## 우선순위 및 의존성

```
trial_074 (TE std, clean features) ─── 최우선 실행 (코드 있음)
        │
trial_076 (BlamerX pipeline 재현) ─── 독립 실행 가능
        │
        ▼
trial_077 (3-source diversity blend) ─── 074 + 076 OOF 필요
```

**실행 순서**: trial_074 → trial_076 → trial_077

### trial_071 (Optuna deep) 재배치
- 기존 코드가 있으나, distribution features 포함 → **overfit 위험 높음**
- trial_076이 BlamerX의 low lr + narrow colsample 접근을 이미 포함하므로 Optuna보다 검증된 파라미터 직접 사용이 더 효율적
- trial_071은 trial_076 이후에 clean feature set으로 Optuna를 돌리는 trial_078로 변경 검토
- **Priority: 하향** (trial_077 이후로 이동)

## 위험 요소
- **trial_076 distribution features**: BlamerX는 LB에서도 상위이므로 Ridge hint + low lr이 overfit을 충분히 완화할 가능성. 그래도 distribution 포함/미포함 두 버전 비교 필요
- **20-fold 실행 시간**: 5-fold 대비 4배. 7 seeds × 20-fold = 140 models. CPU에서 수 시간
- **TE inner-fold 구현 복잡도**: leakage 방지 위해 careful한 fold 구조 필요
- **Submission 전략**: trial_074(clean, LB-safe) + trial_076(aggressive) 각각 submit해서 LB 비교

## v4 대비 변경사항
- ✅ trial_074 유지 (최우선으로 승격)
- ❌ trial_071 하향 (distribution features overfit 위험 + Optuna보다 검증된 params 직접 사용이 효율적)
- ❌ trial_075 (3-model blend) → trial_077 (3-source blend)로 변경: MLP 단독 학습 대신 기존 RealMLP OOF 활용이 더 효율적
- 🆕 trial_076 (BlamerX 파이프라인 재현): 최신 노트북 코드 분석 결과 반영
- 🆕 LB 현황 업데이트: top 0.91771 (Chris Deotte)
