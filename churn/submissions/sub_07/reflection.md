## Submission 07 Reflection

### 결과
- Val best: 0.91695 (trial_036 ultimate blend)
- Public: 미제출 — val 기준으로 기존 best(0.91694) 대비 개선 없음
- 17개 trial 실행했으나 돌파 실패

### 시도한 것들 (전부 기존 best 못 넘음)
- **정규화 강화** (lambda 2→5→10): val 거의 동일 (0.91683→0.91681). 한계 도달
- **다른 모델**: MLP(0.91327), RealMLP(0.91247), Ridge(0.90341), HistGBM(0.91606), DART(0.91482), ExtraTrees(0.91258), LR one-hot(0.91078) — 전부 GBDT보다 낮음
- **XGB 100 trials 넓은 범위**: 0.91693 → 기존과 동일
- **Rank averaging blend**: 0.91694 → 기존과 동일
- **Feature selection (top 15)**: 0.91515 → 하락
- **Pseudo-labeling**: 0.91634 → 하락

### 핵심 교훈
1. **로컬 환경 한계**: 상위권(0.917+)은 Kaggle GPU에서 RealMLP, TabM 등을 n_ens=32로 돌림. 로컬 M1 Max에서는 RealMLP가 0.912대밖에 안 나옴
2. **single model 한계**: XGB 100 trials 해도 0.9169가 천장. 상위권은 0.9189. 피처셋이 아니라 모델 구현/환경 차이
3. **앙상블 다양성 한계**: GBDT 계열(LGBM/XGB/CatBoost)만으로는 앙상블 다양성 부족. NN이 0.913대로 약해서 기여 못 함
4. **val=0.91694가 로컬 환경의 천장**

### 버려야 할 것
- 같은 GBDT 계열로 더 파는 것 — 한계 도달
- 로컬에서 NN 모델 돌리기 — 성능 부족

### 다음 방향
- **Kaggle Notebook 환경**에서 GPU 가속으로 RealMLP/TabM 돌리기
- 또는 **이 대회는 여기서 마무리**하고 다른 대회로 넘어가기
