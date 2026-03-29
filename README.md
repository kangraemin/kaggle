# Kaggle 실험 레포

대회별 trial/submission 관리 레포. 모든 대회는 `TRIAL_GUIDE.md` 규칙을 따른다.

## 대회 목록

| 폴더 | 대회 | 상태 | Best Public |
|------|------|------|-------------|
| `ts-forecasting/` | Hedge fund - Time series forecasting | 진행중 | 0.1499 (sub_01) |
| `churn/` | Playground S6E3 - Binary Classification | 진행중 | TBD |

---

## ts-forecasting

### Submissions

| sub | Best Trial | Val | Public | 왜 이걸 골랐나 | 결과/교훈 |
|-----|-----------|-----|--------|----------------|-----------|
| 01 | trial_001 (raw features) | 0.1163 | **0.1499** | 첫 baseline — lag feature 없이 raw만 | public > val. raw feature가 생각보다 잘 일반화됨 |
| 02 | trial_010 (target encoding) | 0.8923 | **0.0000** | val score가 높아서 선택 | lag feature가 test에서 NaN → 전부 0 예측. val이 test 상황 반영 못함 |
| 03 | trial_058 (weight^0.05) | 0.5912 | TBD | weight 압축으로 균형있는 학습 + 자동 danger 패치 | pending |

### Trial 히스토리

#### sub_01 (2026-03-26)
| trial | val | 왜 시도했나 | 결과 |
|-------|-----|-------------|------|
| 001 lgbm_baseline | 0.1163 | 가장 단순한 baseline — raw feature + LightGBM | ✅ public 0.1499 제출 |

#### sub_02 (2026-03-27) — lag feature 탐색기
| trial | val | 왜 시도했나 | 결과 |
|-------|-----|-------------|------|
| 002 lgbm_lags | 0.8432 | AR(1)=0.86 발견 → lag 추가 | val 폭등, 하지만 test에서 lag=NaN |
| 003 more_lags | 0.8422 | lag 범위 확장 | 개선 없음 |
| 004 cross_horizon | 0.8913 | 다른 horizon y_target을 feature로 | val inflated (test엔 y_target 없음) |
| 005 ewm | 0.8433 | 지수 가중 이동평균 | 소폭 개선 |
| 006 per_horizon | 0.8441 | horizon별 모델 분리 | cross_horizon signal 잃어서 오히려 하락 |
| 007 all_features | 0.8912 | 모든 feature 조합 | cross_horizon이 top importance 독점 |
| 009 hparam_tuning | 0.8913 | hyperparameter 최적화 | 거의 동일 |
| 010 target_enc | 0.8923 | series_mean/std 추가 | ✅ **제출 → public 0.0000** |

**reflection**: lag feature가 test에서 NaN → 예측 0 → 점수 0. val split이 완전히 잘못됨. val의 y_target이 있어서 lag 동작 → val score는 거짓말.

#### sub_02 이후 실험 (제출 대기중)
| trial | val | 왜 시도했나 | 결과 |
|-------|-----|-------------|------|
| 012~014 | 0.30 | lag 없이 raw+target enc, recursive, hierarchical enc | test 89% 신규 시리즈 → series stats NaN |
| 015~021 | 0.64 (cold-start) | cold-start val 설계 (시리즈 holdout) | cold-start val이 진짜 test 반영, 그러나 고가중치 시리즈 예측 실패로 public 0.0000 |
| 031~033 | 0.20 | ts_index val 복귀 + known series last_y feature | last_y가 핵심 signal 발견 (+0.085 jump) |
| 034~041 | 0.47 | rolling stats, mixed val, subcode stats | mixed val(15% cold-start) 최적 |
| 042~045 | 0.49 | num_leaves 511, lr 조정, weight clipping | weight 압축이 핵심 발견 |
| 054~058 | **0.59** | sqrt/^0.25/^0.1/^0.05 weight 변환 | **weight^0.05 = 0.5912 달성** |

### 핵심 인사이트

**데이터 구조:**
- 36,923 시리즈, 각 평균 144 step
- test 시리즈 **89%가 신규** (training에 없는 KEY 조합)
- AR(1)=0.86 (강력), 하지만 test에는 lag 없음
- 고가중치 시리즈 (83EG83KQ, weight=13조): y_target≈0, 잘못 예측하면 score=0

**무엇이 효과 있었나:**
1. **known series의 last_y** (마지막 training y_target) → +0.085 jump
2. **mixed val** (ts_index split + cold-start 15% holdout) → unknown series 반영
3. **weight^0.05 변환** → 극단 weight 압축, 균형있는 학습 → 0.59 달성
4. **자동 danger 패치** (save_submission 내 danger_ratio 검사) → 0점 방지

**무엇이 안 됐나:**
- lag/rolling features → test에서 NaN
- series_mean (train stats) → test 89% NaN
- pseudo-labeling, target normalization, clustering → val 0점
- AR 예측값 feature → 중복 정보라 노이즈

**제출 전 필수 체크:**
```python
# utils.py의 save_submission이 자동 실행
# danger_ratio > 0.1 코드는 train mean으로 자동 패치됨
```

---

## churn (Playground S6E3)

| sub | Best Trial | Val | Public | 왜 이걸 골랐나 | 결과/교훈 |
|-----|-----------|-----|--------|----------------|-----------|
| - | - | - | - | 진행중 | - |

---

## 공통 규칙

- 자세한 규칙: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)
- 실험(Trial)과 제출(Submission) 분리
- Public score 수령 후 reflection 작성 + library 기록 의무
- **제출 전 danger_ratio 체크 필수** (0점 방지)
