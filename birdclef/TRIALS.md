# Trials — birdclef-2026

| # | Name | Val Score | Public Score | Key Changes | Status |
|---|------|-----------|--------------|-------------|--------|
| 001 | perch_lgbm | 0.8375 | - | Perch v2 임베딩 + LightGBM baseline | ✅ |
| 002 | with_soundscape | 0.8731 | - | soundscape 데이터 1478개 추가 | ✅ |
| 003 | ensemble | 0.9709 | - | XGBoost >> LightGBM. PCA 1536→512 | ✅ (제출 실패: re-run 에러) |
| 004 | logreg_pca64 | 0.9754 | - | LR + PCA 64. XGBoost보다 빠르고 좋음 | ✅ |
| 005 | pca_sweep | 0.9580 (best) | - | PCA 64~1536 비교. no PCA가 best지만 LR이 전부 이김 | ✅ |
| 006 | xgb_tuned | 0.9559 (best) | - | XGB n_est/depth 튜닝. 미미한 개선 | ✅ |
| 007 | perch_probe_bayesian | OOF 0.487 | 0.912 | 0.912 공개노트북 fork. Perch logits + Bayesian prior + LR probe | ✅ 첫 유효 제출 |
| 008 | post_processing | OOF 미검증 | TBD | temperature scaling + file-level/rank-aware scaling + gaussian smoothing | ⏳ re-run 중 |

## 메트릭
- Task: multi-label classification (5초 오디오에서 새 종 존재 여부 예측)
- Metric: macro-averaged ROC-AUC (true positive 없는 종은 스킵)
- Direction: higher is better
