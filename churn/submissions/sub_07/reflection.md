## Submission 07 Reflection

### 결과
- Val best: 0.91695 (trial_036 ultimate blend)
- Public: 미제출 — val 기준 기존 best 못 넘음
- 총 22개 trial(036~059) 실행, 돌파 실패

### 시도한 것들 (전부 기존 best 못 넘음)
- **정규화 강화** (lambda 5→10): 한계 도달 (0.91681)
- **다른 모델**: MLP(0.91327), RealMLP 로컬(0.91247), Ridge(0.90341), HistGBM(0.91606), DART(0.91482), ExtraTrees(0.91258), LR one-hot(0.91078)
- **XGB 100 trials 넓은 범위**: 0.91693 (동일)
- **Rank averaging, Geometric mean blend**: 0.91694 (동일)
- **Feature selection (top 15)**: 0.91515 (하락)
- **Pseudo-labeling**: 0.91634 (하락)
- **WoE encoding**: 0.91685 (동일)
- **RepeatedStratifiedKFold 15-fold**: 0.91682 (동일)
- **Platt calibration**: 0.91693 (동일)
- **depth=1 (Chris Deotte EDA)**: 0.91375 (하락 — synthetic 데이터에선 안 맞음)
- **Kaggle RealMLP CPU (n_ens=1)**: 0.91363 (하락)

### 핵심 교훈
1. **로컬 GBDT 한계**: val 0.91694가 천장. 피처/파라미터/인코딩/앙상블 다 해봤으나 돌파 불가
2. **depth=1은 원본(7천 행)에서만 최적**: synthetic(59만 행)에선 더 깊은 트리가 나음
3. **pytabkit Kaggle GPU CUDA 호환성 문제**: T4에서 CUDA kernel image 에러 → CPU fallback 또는 노트북 fork로 해결
4. **Kaggle notebook fork가 답**: 원본 노트북 fork하면 환경 호환성 해결됨

### 버려야 할 것
- 같은 GBDT 계열로 계속 파는 것
- 로컬에서 NN 돌리기
- depth=1 모델

### 유지해야 할 것
- Kaggle notebook fork 전략
- 강한 정규화 (lambda=2.0)
- multi-seed 앙상블

### 다음 가설
- Kaggle에서 RealMLP 제대로 돌리기 → sub_08에서 0.91683 달성
- TabM, XGB+CB 노트북 fork
