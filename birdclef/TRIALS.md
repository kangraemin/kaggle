# Trials — birdclef-2026

| # | Name | Sub | Val Score | Public Score | Key Changes | Status |
|---|------|-----|-----------|--------------|-------------|--------|
| 001 | perch_lgbm | 01 | 0.8375 | - | Perch v2 임베딩 + LightGBM baseline | ✅ |
| 002 | with_soundscape | 01 | 0.8731 | - | soundscape 데이터 1478개 추가 | ✅ |
| 003 | ensemble | 01 | 0.9709 | - | XGBoost >> LightGBM. PCA 1536→512 | ✅ (제출 실패) |
| 004 | logreg_pca64 | 01 | 0.9754 | - | LR + PCA 64 | ✅ |
| 005 | pca_sweep | 01 | 0.9580 | - | PCA 64~1536 비교 | ✅ |
| 006 | xgb_tuned | 01 | 0.9559 | - | XGB n_est/depth 튜닝 | ✅ |
| 007 | perch_probe_bayesian | 01 | OOF 0.487 | **0.912** | 0.912 공개노트북 fork | ✅ 첫 유효 제출 |
| 008 | post_processing | 02 | 미검증 | 0.910 | temperature/file-level/rank-aware | ❌ 악화 |
| 009 | probe_sweep | 03 | 0.9766 | - | PCA dim×C sweep. PCA96+C0.1 local best | ✅ |
| 010 | pseudo_label | 03 | - | - | soundscape pseudo-labeling | 🔄 미완 |
| 011 | local_val | 03 | 0.768 | - | Kaggle 파이프라인 복제 로컬 검증. 파이프라인 불일치 | ❌ |
| 011b | local_val_sweep | 03 | 0.768 | - | prior_weight/sigma/PCA 스윕 | ✅ |
| 012 | cnn_efficientnet | - | 완료 | - | EfficientNetV2-B0 mel CNN. 로컬 M1 학습 완료 (25MB) | ✅ |
| 013 | param_change_v18 | 03 | - | 0.904 | PCA96, C=0.1. 0.912 대비 하락 | ❌ (-0.008) |
| 014 | full_upgrade_v19 | 03 | - | - | MLP+TTA+후처리. 타임아웃 | ❌ 실패 |
| 015 | fork_926 | **04** | - | **0.928** | yukiZ 0.926 fork. dataset 누락→재학습 +0.002 | ✅ **new best** |
| 016 | fork_926_v4 | 05 | - | - | API push 5-seed. 빈 모델 학습 실패 (ProtoSSM_PATH 문제) | ❌ 실패 |
| 017 | fork_926_v7_multiseed | **06** | - | - | 웹 수정 5-seed + epoch120 + PCA192 + isotonic | ❌ 점수 없음 |
| 018 | seed_variant | **07** | - | - | seed 42→1891 but multi-seed 오염 → Timeout | ❌ Timeout |
| 018b | seed_variant_clean | **08** | - | - | clean API push seed1891 + V1 재제출 → hidden test 증가로 전부 Timeout | ❌ Timeout |
| 019 | tflite_speedup | - | - | - | TFLite INT8 Kaggle OOM. 제출 안 함 | ❌ OOM |
| 020 | onnx_perch | **09** | - | **0.928** | Perch ONNX Runtime 변환. 추론 2x 가속. 타임아웃 해결 | ✅ best 유지 |
| 021 | 093_onnx_fork | **10** | - | 0.925 | 0.93 노트북(a4dc68) fork + ONNX. best 대비 -0.003 | ❌ 하락 |
| 022 | full_upgrade | **11** | - | 0.928 | V18 파라미터 + audio features. 효과 없음 (동일) | ➖ 변화없음 |
| 023 | effnet_blend | **12** | - | **0.929** | EfficientNetV2 1-fold + Perch 블렌딩(92:8) | ✅ **new best** |
| 024 | effnet5fold_lse | **13** | - | TBD | EffNet 5-fold + LSE inference (forward_features→LSE pool) + Perch 블렌딩(90:10) | 🔄 실행 중 |

## 메트릭
- Task: multi-label classification (5초 오디오에서 새 종 존재 여부 예측)
- Metric: macro-averaged ROC-AUC (true positive 없는 종은 스킵)
- Direction: higher is better
