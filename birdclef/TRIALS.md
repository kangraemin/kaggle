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
| 024 | effnet5fold_lse | **13** | - | 0.922 | EffNet 5-fold + LSE inference (forward_features→LSE pool) + Perch 블렌딩(90:10). LSE 역효과 | ❌ 하락 |
| 025 | effnet5fold_global | **14** | - | 0.929 | LSE 제거, global pool 복구, BLEND 0.08, 5-fold | ➖ best 동일 |
| 028 | distill_5fold | **15** | - | **0.930** | Knowledge distillation (EffNet→Perch L2-MSE) 5-fold, BLEND 0.08 | ✅ **new best** |
| 029 | blend_sweep | **16** | - | 0.930 | BLEND_EFFNET 0.08→0.10 | ➖ best 동일 |
| 030 | blend_sweep | **17** | - | 0.930 | BLEND_EFFNET 0.15 | ➖ best 동일 |
| 031 | pseudo_5fold | **18** | CV AUC 0.9792 | 0.927 | pseudo-label 학습(23.8h) EffNet 5-fold 교체, BLEND 0.15 유지 | ❌ 하락 (-0.003, CV-LB gap) |
| 032 | protossm_attach | **19** | - | 0.929 | hideyukizushi sgkfk dataset 추가 → proto_ssm + residual_ssm 실제 로드 (residual mean_abs=0.4487) | ❌ 약간 하락 (-0.001, 작년 weight 일반화 실패) |
| 033 | yusuf_baseline | **20** | - | (silent reject) | yusuf "Improvement" V18 그대로 fork 진단 시도 | ❌ Kaggle 채점 publicScore 빈칸. 진단 실패 |
| 034 | hgnet_blend | **21** | OOF 0.9657 | 0.927 | HGNetV2-B0 4-fold 학습 후 10% blend (z-score 정규화). raw logit scale 불일치 삽질(v2→0.858) 후 수정 | ❌ -0.003, OOF 낮아 noise 효과 |
| 035 | softauc_loss | **22** | OOF 0.9815 | 0.930 | SoftAUC hybrid loss (0.5*BCE+0.5*SoftAUC) + spec pre-compute (6.5x 속도향상). OOF +0.0023 vs baseline | ➖ best 동률 |
| 036 | epoch50_blend25 | **23** | OOF 0.9823 | 0.929 | N_EPOCHS 30→50 + BLEND 0.15→0.25. OOF +0.0008 but BLEND 올려 Perch 비중 감소 → LB 하락 | ❌ -0.001 |
| 037 | convnext_xcl | **24** | Fold1 0.9895 | 0.930 | ConvNeXt-Base XCL 3-way blend (Perch 65%+EffNet 15%+ConvNeXt 20%). Fold 1 only 제출. 5-fold 진행 중 | ➖ best 동률 |
| 038 | prior_mask | **25** | (post-hoc) | 0.930 | 도메인 prior mask 후처리 한 줄 (Tier C 67종 ×0.3). 학습/모델 변경 0. EDA 발견 통합 첫 라운드 (sub_25 v1) | ➖ best 동률 |
| 039 | convnext_5fold | **26~30** | CV 0.9905 | ❌ silent | ConvNeXt-Base XCL fold0~4 5-fold. 4/30~5/1 총 5회 제출 변형(v24 batch, v25 ONNX INT8, v26 3fold, v27 1-fold). 전부 silent reject | ❌ 플랫폼 이상으로 채점 불가 |
| 040 | effnet_multiwindow | **31,33** | Val AUC 0.9794 | ❌ silent | EfficientNetV2-B0 fold0 ep16. 5초 multiwindow 12개. 31: dry-run 24행 버그, 33: 수정 후 재제출. 둘 다 silent reject | ❌ 플랫폼 이상 |
| 041 | convnext_timeguard | **34** | - | ❌ silent | ConvNeXt deadline 8h→114분. 경쟁 eval에서도 silent reject. ConvNeXt 3-way blend 운영 불가 판단 | ❌ 포기 |
| 042 | gaussian_smoothing | **35** | - | ❌ silent | scipy gaussian_filter1d sigma=1.0 per soundscape, prior mask 후 적용. v31 push. 플랫폼 채점 이상으로 결과 미확인 | ❌ 플랫폼 이상 |
| 043 | convnext_remove | **36** | - | ❌ silent | ConvNeXt 완전 제거. Perch 85%+EffNet 15% 2-way blend. Gaussian smoothing 유지. v32 push. 30분+ PENDING → 플랫폼 이상 지속 | ❌ 플랫폼 이상 |

## 메트릭
- Task: multi-label classification (5초 오디오에서 새 종 존재 여부 예측)
- Metric: macro-averaged ROC-AUC (true positive 없는 종은 스킵)
- Direction: higher is better
