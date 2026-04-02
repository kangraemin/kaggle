# Trials — playground-series-s6e4

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | lgbm_baseline | 0.9844 | 0.9589 | LightGBM default, label encoding, 5-fold CV | done |
| 002 | fe_catboost | 0.9853 | 0.9609 | FE(ET_proxy,water_balance,interactions) + LGBM+XGB+CAT ensemble(4:5:1) | done |

## 메트릭
- Task: classification (3-class: Low / Medium / High)
- Metric: accuracy
- Direction: higher_is_better
