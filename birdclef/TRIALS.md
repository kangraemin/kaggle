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
| 043 | convnext_remove | **36** | - | 0.928 | ConvNeXt 완전 제거. Perch 85%+EffNet 15% 2-way blend. Gaussian smoothing 유지. v32 push. 채점 정상화 확인 | ❌ -0.002 (best 0.930 대비) |
| 044 | gaussian_remove | **37** | - | **0.930** | Gaussian smoothing(σ=1.0) 제거. Perch 85%+EffNet 15%+prior_mask. v33 push | ➖ best 동률 |
| 045 | effnet_mw_retry | **38** | Val AUC 0.9794 | 0.836 | EffNet multiwindow standalone 재시도. enable_gpu=false 수정 + IS_DRY_RUN v8. kernel v9 push | ❌ -0.094 (Perch 없이 standalone EffNet 한계) |
| 046 | effnet_mw_blend | **39** | - | **0.932** | EffNetF0 fold0 standalone 3번째 컴포넌트 추가. Perch 80%+distill5fold 15%+fold0 5%. kernel v34 push | ✅ **new best** (+0.002) |
| 047 | effnet_mw_blend | **40** | - | 0.932 | BLEND_MWF0 0.05→0.08 상향. Perch 77%+distill5fold 15%+fold0 8%. kernel v35 push | ➖ best 동률 (비중 조정 효과 없음) |
| 048 | distill30_swap | **41** | - | 0.931 | 5fold 컴포넌트 epoch50 SoftAUC → distill30 KD 교체. Perch 80%+distill30 15%+fold0 5%. kernel v36 push | ❌ -0.001 (epoch50 SoftAUC가 distill30 KD보다 우수) |
| 050 | pseudo_mix_blend | **42** | - | **0.933** | epoch50 5fold 복원 + EffNet pseudo-mix fold0 (30k pseudo-labeled ss10k 혼합 학습, EffNetMixup arch) 4번째 컴포넌트 추가. Perch 77%+epoch50-5fold 15%+mwf0 5%+pmix 3%. 새 Kaggle dataset birdclef2026-effnet-pseudo-mix. kernel v37 | ✅ **new best** (+0.001, ralph it.1) |
| 051 | pmix_2fold_blend | **43** | - | 0.933 | pseudo-mix fold0 → fold0+fold1 2-fold 앙상블로 교체 (fold1 로컬 학습 완료, val AUC ~0.981). BLEND_PSEUDOMIX 0.03 고정해 fold1 추가 효과만 격리. dataset birdclef2026-effnet-pseudo-mix v2. kernel v38 | ➖ best 동률 (ralph it.2, 2-fold 확장 무효과) |
| 052 | effnet_reweight | **44** | - | 0.933 | EffNet 계열 weight 재배분: BLEND_MWF0 0.05→0.07, BLEND_PSEUDOMIX 0.03→0.06 (Perch 77→72%, EffNet 합산 23→28%). 5fold(0.15)·prior_mask·모델 변경 없음. kernel v39. submission mean 0.0395 | ➖ best 동률 (ralph it.3, weight 재배분 무효과) |
| 053 | pmix_4fold_blend | **45** | - | **0.934** | pseudo-mix 컴포넌트 fold0+fold1 2-fold → fold0..3 4-fold 앙상블 확장 (dataset v3). weight·다른 컴포넌트·prior_mask 모두 trial_052 고정 (Perch 72%). 코드 변경 0 (loader가 best_fold*.pth glob+mean). kernel v40. submission mean 0.0396 | ✅ **new best** (+0.001, ralph it.3 재개. 2-fold(trial_051)는 동률이었으나 4-fold에서 LB 반영) |
| 054 | pmix_5fold_blend | **46** | - | 0.934 | pseudo-mix 컴포넌트 fold0..3 4-fold → fold0..4 5-fold 앙상블 확장 (dataset v4, best_fold4.pth 추가). weight·다른 컴포넌트·prior_mask 모두 trial_053 고정 (Perch 72%). 코드 변경 0 (loader glob+mean). kernel v41. wall 216s, submission mean 0.0396 | ➖ best 동률 (ralph it.4, 4→5-fold 한계효용 0. 2→4-fold는 +0.001이었으나 보조 컴포넌트 fold 앙상블은 4개에서 포화) |
| 055 | effnet5fold_multiloss | **47** | - | 0.933 | 15% EffNet5fold 슬롯을 epoch50 SoftAUC 5ckpt 단일 → epoch50+distill KD 10ckpt mean-pool 앙상블로 강화 (multi-loss diversity). 동일 _BirdEffNet arch·_EffSpec(n_fft=2048/n_mels=256) — 로컬 strict load 검증. distill 단일 swap은 trial_048에서 -0.001이었으나 epoch50과 평균하면 손실함수 diversity로 ±0~+0.001 기대. weight·다른 컴포넌트·prior_mask 전부 trial_054 고정 (Perch 72%). kernel v42, wall 254.5s (+38s), submission mean 0.0400 | ❌ -0.001 (ralph it.5. distill을 평균에 섞어도 epoch50 단일보다 못함 — multi-loss 앙상블 ≠ diversity 이득) |
| 056 | pmix_weight_up | **48** | - | 0.934 | trial_055 회귀 → 15% 슬롯 epoch50 SoftAUC 5ckpt 단일로 복원(trial_054 0.934 구성). 그 위에서 sub_45 reflection 제안대로 BLEND_PSEUDOMIX 0.06→0.08, 2pp는 가장 노이즈 많은 mwf0 fold0에서 빼옴(BLEND_MWF0 0.07→0.05). Perch 72% 고정(trial_036 Perch↓ 역효과 회피). EffNet 예산 28% 안에서 단일 fold0 → 5-fold 앙상블 pmix 쪽으로 재배분 격리. kernel v43, wall 203.6s, submission mean 0.0425 | ➖ best 동률 (ralph it.6, EffNet 내부 2pp 재배분 LB 무효과 — pmix 5-fold가 8% weight에서도 mwf0 fold0와 차이 없음. trial_054 0.934 복귀 확인) |

## 메트릭
- Task: multi-label classification (5초 오디오에서 새 종 존재 여부 예측)
- Metric: macro-averaged ROC-AUC (true positive 없는 종은 스킵)
- Direction: higher is better
