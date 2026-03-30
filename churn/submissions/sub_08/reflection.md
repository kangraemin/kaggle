## Submission 08 Reflection

### 결과
- Val score: 0.91945 (trial_061 RealMLP×0.85 + XGB×0.15)
- Public score: 0.91686
- Val-Public gap: -0.00259
- 등수: 394 / 39,867 (상위 0.99%)

### Gap 원인 분석
- gap -0.00259는 이전 sub들(-0.0027~-0.0029)보다 줄었음
- RealMLP가 GBDT보다 일반화가 잘 되는 것 같음
- 20-Fold로 OOF 분산 감소 효과

### 핵심 발견
- **RealMLP가 게임 체인저**: 로컬 GBDT val 0.91694 → RealMLP val 0.91938 (+0.0024)
- **Kaggle에서 fork해서 돌리는 게 핵심**: pytabkit CUDA 호환성 문제 → 원본 노트북 fork로 해결
- **NN + GBDT 앙상블**: RealMLP 단독(0.91683) → +XGB 15%(0.91686). 미미하지만 개선

### 버려야 할 것
- 로컬에서 NN 돌리기 — 환경 문제로 제대로 안 됨
- depth=1 모델 — synthetic 데이터에선 안 맞음

### 유지해야 할 것
- Kaggle notebook fork 전략
- RealMLP 20-fold
- NN + GBDT 앙상블

### 다음 가설
- TabM fork (discussion에서 CV 0.91898, LB 0.91682)
- CV 0.91930 XGB+CB 노트북 fork
- RealMLP + TabM + GBDT Hill Climbing
- 1위(0.91762)까지 0.00076 차이 → 모델 다양성 확보가 핵심
