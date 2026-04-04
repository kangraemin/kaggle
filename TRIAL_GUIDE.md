# Kaggle Trial & Submission Guide

실험(Trial)과 제출(Submission)을 구분해서 관리한다.
모든 Kaggle 대회 폴더에 동일한 규칙을 적용한다.

---

## 개념 정의

| 개념 | 설명 |
|------|------|
| **Trial** | 단일 실험. 가설 하나를 검증하는 스크립트 + val 결과 |
| **Submission** | 제출 단위. 여러 trial 중 best를 골라 Kaggle에 제출 |
| **Reflection** | Public score를 받은 후 원인-결과 분석 |

> Trial은 Claude가 가설을 세우고 실행한다.
> Submission은 사용자가 best trial을 선택 또는 Claude가 추천해서 제출한다.

---

## 디렉토리 구조

```
<competition>/
  SUBMISSIONS.md              ← 제출 이력 + 점수 인덱스
  TRIALS.md                   ← trial 전체 인덱스
  utils.py
  submissions/
    sub_NN/
      meta.json               ← 포함된 trials, best trial 선택 이유
      reflection.md           ← Public score 받은 후 원인-결과 분석
      trial_NNN_<name>/
        trial_NNN_<name>.py   ← 실험 스크립트
        trial_NNN_<name>.log
        trial_NNN_<name>.csv  ← 예측 파일
        meta.json             ← 가설, 근거, 변경사항 (실험 전 작성)
        results.json          ← val 점수, feature importance (실험 후 작성)
```

---

## Trial 생명주기

### 1. 계획
Claude가 가설을 세우고 `meta.json` 작성. 실험 실행 전에 반드시 완료.

### 2. 실행
스크립트 실행 → log 파일 저장.

### 3. 결과 기록
`results.json` 작성 → `TRIALS.md` 업데이트.

### 4. Library 기록 (유의미한 인사이트가 있을 때만)
feature X가 효과 없음, Y 조합이 시너지 등 **재사용 가능한 인사이트**만 library에 기록.
단순 수치 기록은 TRIALS.md로 충분.

---

## Sub 경계 규칙

**Kaggle 제출 1회 = sub 1개. 예외 없음.**

같은 base 노트북이라도 제출할 때마다 새 sub. 하나의 sub에 여러 제출을 몰아넣지 않는다.

예시:
- 0.912 fork에서 PCA 바꿔서 제출 → sub_03
- 같은 0.912 fork에서 MLP 넣어서 제출 → **sub_04** (같은 base라도 새 sub)
- 0.926 fork로 전환해서 제출 → **sub_05**
- 0.926 fork에서 5-seed 넣어서 제출 → **sub_06**

---

## Submission 생명주기

### 1. 후보 선정
현재까지의 trials 중 val score 기준 후보 추출.
Claude가 best trial 추천 + 이유 설명.

### 2. 제출 준비
`sub_NN/meta.json` 작성 → 후보 trial CSV 파일들을 `sub_NN/`에 모음 → Kaggle 제출.

### 3. Reflection (점수 수령 후 필수)
Public score를 받으면 즉시 `sub_NN/reflection.md` 작성.

```markdown
## Submission NN Reflection

### 결과
- Val score: 0.XXXX
- Public score: 0.XXXX
- Val-Public gap: +/- 0.XXXX

### Gap 원인 분석
- Val-Public gap이 발생한 구체적 원인 (overfitting? val/test 분포 차이? data leakage?)
- 어떤 feature/변경이 실제로 기여했는가? (val과 public 모두에서)
- 어떤 것이 val에서만 효과가 있었고 public에서는 아니었는가?
- 예상과 달랐던 것은 무엇인가?

### 버려야 할 것
- 이 결과로 검증된 효과 없는 접근법

### 유지해야 할 것
- 실제로 기여한 것

### 다음 가설
- 이 결과를 바탕으로 다음에 시도할 구체적 가설 (막연한 "개선" 금지)
```

### 4. Library 기록 (의무)
Reflection 완료 후 배운 것을 반드시 library에 기록한다.

---

## meta.json 스키마

### Trial meta.json
```json
{
  "id": "003",
  "name": "more_lags",
  "base_trial": "002",
  "created_at": "2026-03-27",
  "hypothesis": "lag_1이 압도적으로 중요했으므로 더 긴 lag(50, 100)와 EWM이 장기 패턴을 포착할 것이다.",
  "changes": [
    "lag 추가: 50, 100",
    "EWM 추가: span 5, 20, 50"
  ],
  "rationale": "trial_002 feature importance: lag_1 >> lag_2 >> roll_mean_5. 더 긴 기억이 필요한지 검증.",
  "expected_impact": "medium"
}
```

### Submission meta.json
```json
{
  "id": "01",
  "date": "2026-03-28",
  "best_trial": "007",
  "candidate_trials": ["003", "005", "007"],
  "selection_reason": "trial_007(val 0.891)이 가장 높았고, cross_horizon feature가 핵심 신호임을 확인.",
  "submission_file": "submissions/sub_01/trial_007_all_features.csv"
}
```

---

## results.json 스키마

```json
{
  "id": "003",
  "status": "done",
  "val_score": 0.8422,
  "best_iteration": 112,
  "top_features": ["lag_1", "lag_2", "roll_mean_5"],
  "notes": "lag_50, lag_100은 feature importance에서 낮았음. EWM ≈ rolling mean.",
  "conclusion": "longer lags와 EWM은 효과 없음."
}
```

---

## SUBMISSIONS.md 형식

```markdown
| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-03-27 | trial_001 | 0.116 | 0.1499 | - | +0.034 | ✅ |
| 02 | 2026-03-28 | trial_007 | 0.891 | TBD | - | - | 📋 |
```

**Gap** = Public - Val. 양수면 val이 보수적, 음수면 overfitting 의심.

---

## TRIALS.md 형식

```markdown
| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.1163 | 0.1499 | raw features only | ✅ submitted |
| 002 | lgbm_lags | 0.8432 | - | lag 1~20 + rolling | ✅ done |
```

---

## .gitignore

```gitignore
*.parquet
*.feather
trials/*.log
submissions/**/*.csv
*.pkl
*.joblib
*.h5
```

**커밋**: 스크립트, meta.json, results.json, reflection.md, TRIALS.md, SUBMISSIONS.md
**미커밋**: 데이터, CSV, 로그, 모델 바이너리

---

## Git 커밋 규칙

submission 준비와 reflection 완료 시에만 커밋한다.

```
feat(sub-01): trial_007 제출 준비
chore(sub-01): public 0.XXX 기록 + reflection 작성
```

---

## 체크리스트

### Trial
- [ ] `meta.json` 작성 (가설 + 근거)
- [ ] 스크립트 실행
- [ ] `results.json` 작성 + `TRIALS.md` 업데이트
- [ ] 유의미한 인사이트 있으면 library 기록

### Submission
- [ ] 후보 trial 비교 + best 선택
- [ ] `sub_NN/meta.json` 작성
- [ ] 후보 CSV 파일들을 `sub_NN/`에 모음
- [ ] Kaggle 제출
- [ ] git commit

### Reflection (점수 수령 후 즉시)
- [ ] `sub_NN/reflection.md` 작성 (gap 원인 분석, 버릴 것, 유지할 것, 다음 가설)
- [ ] `SUBMISSIONS.md` public score 업데이트
- [ ] library에 배운 것 기록
- [ ] git commit
