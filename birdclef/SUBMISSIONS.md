# Submissions — birdclef-2026

| # | Date | Best Trial | Base | Public | Status |
|---|------|------------|------|--------|--------|
| 01 | 2026-03-31 | trial_007 | 0.912 fork | 0.912 | ✅ 첫 유효 제출 |
| 02 | 2026-03-31 | trial_008 | 0.912 fork | 0.910 | ❌ 후처리 악화 |
| 03 | 2026-04-03 | trial_013 | 0.912 fork | 0.904 | ❌ PCA96+C0.1 하락, trial_014 타임아웃 |
| 04 | 2026-04-04 | trial_015 | **0.926 fork** | **0.928** | ✅ **best** |
| 05 | 2026-04-04 | trial_016 | 0.926 fork | - | ❌ API push 빈 모델 학습 실패 |
| 06 | 2026-04-05 | trial_017 | 0.926 fork | - | ❌ 점수 없음 (submission 생성 실패 추정) |
| 07 | 2026-04-05 | trial_018 | 0.926 fork | - | ❌ Notebook Timeout (multi-seed 오염) |
| 08 | 2026-04-05 | trial_018 clean + V1 재제출 | 0.926 fork | - | ❌ Notebook Timeout (hidden test 증가) |
| 09 | 2026-04-06 | trial_020 | ONNX Perch | **0.928** | ✅ **best**, 타임아웃 해결 |
| 10 | 2026-04-06 | trial_021 | 0.93 fork + ONNX | 0.925 | ❌ best 대비 -0.003 |
| 11 | 2026-04-06 | trial_022 | ONNX + V18 + audio FE | 0.928 | ➖ best 동일, 효과 없음 |
| 12 | 2026-04-07 | trial_023 | Perch + EffNet blend | **0.929** | ✅ **new best** |
| 13 | 2026-04-10 | trial_024 | Perch + EffNet 5fold + LSE | 0.922 | ❌ LSE 역효과 (-0.007) |
| 14 | 2026-04-11 | trial_025 | Perch + EffNet 5fold global pool | 0.929 | ➖ best 동일, 5-fold 효과 없음 |
| 15 | 2026-04-14 | trial_028 | Perch + EffNet distill 5fold | **0.930** | ✅ **new best** (+0.001) |
| 16 | 2026-04-15 | trial_029 | distill 5fold BLEND=0.10 | 0.930 | ➖ best 동일, BLEND 0.10=0.08 |
| 17 | 2026-04-15 | trial_030 | distill 5fold BLEND=0.15 | 0.930 | ➖ best 동일, BLEND 0.15=0.08 |
| 18 | 2026-04-19 | trial_031 | pseudo 5fold (CV 0.9792) | 0.927 | ❌ -0.003 하락, CV-LB gap (CV 0.9792 → LB 0.927) |
| 19 | 2026-04-19 | trial_032 | distill + proto_ssm attach | 0.929 | ❌ -0.001, proto_ssm(작년 학습) 일반화 실패 |
| 20 | 2026-04-19 | trial_033 | yusuf Improvement V18 baseline fork | (silent reject) | ❌ COMPLETE인데 publicScore 빈칸. 채점 컨테이너 재실행 실패로 추정 |
| 21 | 2026-04-20 | trial_034 | distill 5fold + HGNetV2 4fold 10% blend | 0.927 | ❌ -0.003, HGNetV2 OOF 낮아 noise |
| 22 | 2026-04-25 | trial_035 | distill 5fold weights → SoftAUC 5fold weights (BLEND=0.15 유지) | 0.930 | ➖ best 동률, SoftAUC OOF +0.0023 but LB 미반영 |
| 23 | 2026-04-26 | trial_036 | SoftAUC 5fold 50 epochs + BLEND=0.25 | 0.929 | ❌ -0.001, BLEND 올려 Perch 비중 감소 역효과 |
| 24 | 2026-04-27 | trial_037 | ConvNeXt-Base XCL 3-way blend (Fold 1 only) | 0.930 | ➖ best 동률, 5-fold 완성 후 재제출 예정 |
| 25 | 2026-04-28 | trial_038 | ConvNeXt fold2 + 도메인 prior mask 후처리 | 0.930 | ➖ best 동률. prior mask 효과 미미 (macro AUC skip-empty 특성) |
| 26 | 2026-04-30 | trial_039 | ConvNeXt 5fold 전체 앙상블 (fold0~4) | ❌ silent | ❌ COMPLETE, publicScore 없음. 4월 30일부터 채점 시스템 이상 시작 |
| 27 | 2026-04-30 | trial_039 v24 | ConvNeXt 5fold batch opt | ❌ silent | ❌ COMPLETE, publicScore 없음. batch_size 최적화 시도 |
| 28 | 2026-05-01 | trial_039 v25 | ConvNeXt ONNX 5fold INT8 | ❌ silent | ❌ COMPLETE, publicScore 없음. ONNX INT8 변환 시도 |
| 29 | 2026-05-01 | trial_039 v26 | ConvNeXt ONNX 3fold INT8 + timeguard | ❌ silent | ❌ COMPLETE, publicScore 없음. 3fold 축소 + 7.5h timeguard |
| 30 | 2026-05-01 | trial_039 v27 | ConvNeXt 1-fold fold2 + timeguard | ❌ silent | ❌ COMPLETE, publicScore 없음. timeguard 기준점 버그 수정 후에도 silent |
| 31 | 2026-05-03 | trial_040 v1 | EffNet multiwindow dry-run (24행 CSV) | ❌ format | ❌ COMPLETE, publicScore 없음. dry-run fallback으로 24행 CSV 제출 |
| 32 | 2026-05-03 | blend v28 diagnostic | 알려진 0.930 blend 재제출 | PENDING | 🔍 채점 시스템 진단용. v25~31 모두 silent이므로 플랫폼 이상 확인 중 |
| 33 | 2026-05-03 | trial_040 v8 | EffNet multiwindow dry-run fix | ❌ silent | ❌ COMPLETE, publicScore 없음. dry-run 수정 후에도 채점 안 됨 → 플랫폼 이상 |
| 34 | 2026-05-03 | trial_041 | ConvNeXt timeguard 114min (v30) | ❌ silent | ❌ COMPLETE, publicScore 없음. 114분 deadline도 경쟁 eval에서 부족. ConvNeXt 포기 |
| 35 | 2026-05-04 | trial_042 | Gaussian smoothing σ=1.0 (v31) | ❌ silent | ❌ COMPLETE, publicScore 없음. 플랫폼 채점 이상 지속 |
| 36 | 2026-05-04 | trial_043 | ConvNeXt 제거 2-way blend (v32) | 0.928 | ❌ -0.002. 채점 정상화 확인. Gaussian smoothing 효과 미미 (0.930→0.928) |
| 37 | 2026-05-04 | trial_044 | Gaussian smoothing 제거 (v33) | **0.930** | ➖ best 동률. Gaussian smoothing 역효과 확인, 기준선 복귀 |
| 38 | 2026-05-04 | trial_045 | EffNet multiwindow standalone retry (kernel v9) | 0.836 | ❌ -0.094. Perch 없는 standalone EffNet 한계 확인 |
| 39 | 2026-05-04 | trial_046 | EffNet fold0 3-way blend (Perch 80%+distill5fold 15%+fold0 5%) | **0.932** | ✅ **new best** (+0.002) |
