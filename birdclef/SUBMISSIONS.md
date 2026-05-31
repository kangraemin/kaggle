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
| 40 | 2026-05-05 | trial_047 | EffNet fold0 3-way blend (Perch 77%+distill5fold 15%+fold0 8%) | 0.932 | ➖ best 동률 (0.05→0.08 효과 없음) |
| 41 | 2026-05-05 | trial_048 | distill30 KD swap (epoch50 SoftAUC → distill30 KD) | 0.931 | ❌ -0.001 (epoch50 SoftAUC가 더 우수) |
| 42 | 2026-05-12 | trial_050 | EffNet pseudo-mix fold0 4-way blend (epoch50 복원 + pmix 3%) | **0.933** | ✅ **new best** (+0.001, kernel v37, ralph it.1) |
| 43 | 2026-05-12 | trial_051 | EffNet pseudo-mix fold0+fold1 2-fold 앙상블 (BLEND_PSEUDOMIX 0.03 고정) | 0.933 | ➖ best 동률 (kernel v38, ralph it.2, 2-fold 확장 LB 무효과) |
| 44 | 2026-05-12 | trial_052 | EffNet 계열 weight 재배분 (Perch 72% + epoch50-5fold 15% + mwf0 7% + pmix 6%) | 0.933 | ➖ best 동률 (kernel v39, ralph it.3, weight 재배분 무효과) |
| 45 | 2026-05-12 | trial_053 | pseudo-mix 4-fold 앙상블 (fold0..3) — weights = trial_052 (Perch 72%) | **0.934** | ✅ **new best** (+0.001, kernel v40, ralph it.3 재개) |
| 46 | 2026-05-13 | trial_054 | pseudo-mix 5-fold 앙상블 (fold0..4) — weights = trial_053 (Perch 72%) | 0.934 | ➖ best 동률 (kernel v41, ralph it.4. 4→5-fold 한계효용 0 — 보조 컴포넌트 fold 앙상블 4개에서 포화) |
| 47 | 2026-05-13 | trial_055 | EffNet 15% 슬롯 multi-loss 앙상블 (epoch50 SoftAUC 5ckpt + distill KD 5ckpt = 10ckpt avg) — 다른 컴포넌트·weight 전부 trial_054 고정 (Perch 72%) | 0.933 | ❌ -0.001 (kernel v42, ralph it.5. distill 섞으면 epoch50 단일보다 못함) |
| 48 | 2026-05-13 | trial_056 | 15% 슬롯 epoch50 5ckpt 복원 + EffNet weight 재배분 mwf0 0.07→0.05 / pmix 0.06→0.08 (Perch 72% 고정) | 0.934 | ➖ best 동률 (kernel v43, ralph it.6. trial_054 0.934 복귀 + EffNet 내부 2pp 재배분 LB 무효과) |
| 49 | 2026-05-13 | trial_057 | 앙상블 fusion 연산자 변경: 가중 logit 합 → 클래스별 percentile rank-average (모델·weight·후처리 전부 trial_056 동일) | 0.929 | ❌ -0.005 (kernel v44, ralph it.7. logit 공간 magnitude가 실제 신호 — rank-avg는 Perch confidence를 묽게 하고 후퇴. 다음: logit 블렌드 복원) |
| 50 | 2026-05-13 | trial_058 | trial_056 logit-space 가중 합 fusion 복원 (rank-avg 제거, Perch 72%+EffNet5fold 15%+mwf0 5%+pmix 8%; 모델·weight·후처리 = trial_056) | **0.934** | ✅ best 동률 (kernel v45, ralph it.8. rank-avg -0.005 회귀 청소 성공, logit-blend 복원 검증) |
| 51 | 2026-05-13 | trial_059 | distill_5fold 5th component 추가 (BLEND_DISTILL=0.03, mwf0 0.05→0.02, Perch 72% 고정, kernel v46) | 0.934 | ➖ best 동률 (ralph it.9, loss-fn diversity 가설 검증 실패 — KD 별도 컴포넌트로도 0.934 천장 못 깸) |
| 52 | 2026-05-13 | trial_060 | ConvNeXt-Base XCL fold0 6th component 추가 (BLEND_CONVNEXT=0.03, BLEND_DISTILL 0.03→0 1:1 치환, Perch 72% 그대로, kernel v47) | TIMEOUT | ⏳ kernel v47 COMPLETE (wall 445s, submission.csv 240×235, mean 0.0432, range 0.000~0.997, ConvNeXt fold0 1.8min 정상 로드/추론) **but Kaggle 채점 미생성** — 90분 폴링 중 신규 submission 미등장. "Submit to Competition" 트리거 필요 (API 400). ralph it.10 score 미확정 |
| 53 | 2026-05-13 | trial_061 | ConvNeXt 컴포넌트 fold0 single → fold0..4 5-fold 평균 (BLEND_CONVNEXT=0.03 그대로, Perch 72%·EffNet5fold 15%·mwf0 2%·pmix 8%·distill 0% 전부 trial_060 동일, kernel v48) | TIMEOUT | ⏳ kernel v48 COMPLETE. **90분 폴링(20:29~21:55 KST, 5분 간격 20회) 동안 Kaggle 신규 submission 미등장** — trial_060과 동일 증상. "Submit to Competition" 자동 트리거 누락 의심. ralph it.11 score 미확정. 2연속 ConvNeXt-axis TIMEOUT → 다음 iter는 trial_058(0.934) 베이스에서 ConvNeXt-free 축 재탐색 권장 |
| 63 | 2026-05-16 | trial_070 | T_AVES 1.10→1.0 제거. blend weights 동일 (Perch 76%/EffNet 15%/mwf0 2%/pmix 7%). kernel v57 | 0.934 | ➖ best 동률 (ralph it.20. T_AVES 제거 무효과 — ROC-AUC rank-based 이론 확인) |
| 62 | 2026-05-16 | trial_071 | Tucker Distilled-SED 5-fold ONNX 추가 (BLEND_SED=0.05). pmix 0.07→0.06, Perch 0.76→0.72. kernel v58 | 0.934 | ➖ best 동률 (ralph it.21. SED 5% 무반응 — 10연속 0.934 천장 확정) |
| 61 | 2026-05-16 | trial_069 | prior mask 0.3→0.5 완화. blend weights 동일 (Perch 76%/EffNet 15%/mwf0 2%/pmix 7%). kernel v56 | 0.934 | ➖ best 동률 (ralph it.19. prior mask 완화 무효과) |
| 62 | 2026-05-16 | trial_071 | Distilled-SED 5fold ONNX 추가. BLEND_SED=0.05, Perch 72%. kernel v58 | 0.934 | ➖ best 동률 (SED 5% 무반응) |
| 63 | 2026-05-16 | trial_070 | T_AVES 1.10→1.0 (온도 스케일링 제거). blend 동일. kernel v57 | 0.934 | ➖ best 동률 (ROC-AUC rank-based, 온도 무효) |
| 64 | 2026-05-17 | trial_072 | sed_up. BLEND_EFFNET/effnet_logits 미정의 크래시. kernel v59-60 | - | ❌ SKIP (bug, trial_073 대체) |
| 65 | 2026-05-17 | trial_073 | proto_blend. BLEND_SED=0 override 버그 (실효 Proto 50%+Perch 43%). kernel v63 | - | ❌ SKIP (bug) |
| 66 | 2026-05-17 | trial_074 | proto_up. blend 공식 버그로 ProtoSSM 단독 출력. kernel v65 | - | ❌ SKIP (bug) |
| 67 | - | trial_074 | proto_heavy. trial_073 결과 대기용 placeholder. 미제출 | - | ⏸ 미제출 (NOT_YET) |
| 68 | 2026-05-17 | trial_075 | sed_only. blend 공식 버그로 ProtoSSM 단독 출력. kernel v66 | - | ❌ SKIP (bug) |
| 69 | 2026-05-17 | trial_076 | effnet_sed. blend 공식 버그로 ProtoSSM 단독 출력. kernel v67 | - | ❌ SKIP (bug) |
| 70 | 2026-05-17 | trial_077 | birdnet_add. jarturo/birdnet dataset hidden re-run 거부. kernel v68 | - | ❌ SKIP (silent reject) |
| 71 | 2026-05-18 | trial_078 | birdnet_remove. BLEND_BIRDNET=0, Perch 23%+Proto 60%+SED 10%+mwf0 2%+pmix 5%. blend 버그 수정 후 첫 유효 채점. kernel v69 | 0.933 | ❌ 하락 (blend 추가 역효과) |
| 72 | 2026-05-20 | trial_079 | ref_config. 0.947 reference config 적용. Proto 72%+SED 18%+Perch 10%. kernel v71 | 0.935 | ✅ best 경신 (0.933→0.935) |
| 73 | 2026-05-29 | trial_080 | sed_up. SED 기여도 테스트. SED 40%+Proto 50%+Perch 10%. kernel v72 | **0.938** | ✅ **new best** (0.935→0.938, SED 40% 효과) |
| 74 | 2026-05-31 | trial_081 | proto_path_fix. ProtoSSM pretrained 경로 수정. blend 동일 (Proto 50%+SED 40%+Perch 10%). kernel v73 | 0.938 | ➖ best 동률 (경로 수정 채점 무변경) |
| 75 | 2026-05-31 | trial_082 | sed_heavy. SED 0.40→0.50 (+10pp), Proto 0.50→0.40 (-10pp), Perch 0.10 고정. kernel v74 | 0.937 | ❌ -0.001 (SED 40%가 포화점, 50% 과다 역효과) |
| 76 | 2026-05-31 | trial_083 | perch_up. SED 0.40 고정, Proto 0.50→0.48 (-2pp), Perch 0.10→0.12 (+2pp). kernel v75 | 0.938 | ➖ best 동률 (Perch 증량 무효과, weight 미세조정 천장) |
| 77 | 2026-05-31 | trial_084 | sed_znorm. SED logit을 proto 분포로 z-score 정규화 + 진단 print. weight SED40/Proto50/Perch10. kernel v76 | 0.938 | ➖ best 동률 (logit affine 정규화 무효과, ROC-AUC rank-based) |
| 60 | 2026-05-16 | trial_068 | mwf0 0.04→0.02 원복 (-2pp), pmix 0.09→0.07 (-2pp), Perch 0.72→0.76 (+4pp). EffNet 0.15 고정. kernel v55 | 0.934 | ➖ best 동률 (ralph it.18. Perch 증량도 포화 — weight space 전 방향 소진 확정) |
| 59 | 2026-05-16 | trial_067 | mwf0 0.03→0.04 (+1pp), pmix 0.10→0.09. EffNet/Perch 고정(0.15/0.72). kernel v54 | 0.934 | ➖ best 동률 (ralph it.17. mwf0 0.04도 동률 — 증량 방향 포화 확정) |
| 58 | 2026-05-15 | trial_066 | EffNet 0.19→0.15 원복, mwf0 0.02→0.03 (+1pp), pmix 0.11→0.10, Perch 0.72. kernel v53 | 0.934 | ➖ best 동률 (ralph it.16. mwf0 증량 LB 무효과) |
| 57 | 2026-05-15 | trial_065 | EffNet 0.17→0.19 (+2pp), Perch 0.70→0.68. mwf0/pmix 고정. kernel v52 | 0.933 | ❌ -0.001 (ralph it.15. EffNet↑ 역효과 확정) |
| 56 | 2026-05-15 | trial_064 | EffNet 0.15→0.17 (+2pp), Perch 0.72→0.70. mwf0 0.02 원복, pmix 0.11 원복. kernel v51 | 0.934 | ➖ best 동률 (ralph it.14. EffNet weight 증량 LB 무효과) |
| 55 | 2026-05-15 | trial_063 | mwf0 완전 제거(0.02→0.00) + pmix 0.11→0.13 (2pp 재배분). Perch 72% + EffNet5fold 15% + pmix 13%. kernel v50 | 0.933 | ❌ -0.001 (ralph it.13. mwf0 제거 역효과 — mwf0 기여 확인됨) |
| 54 | 2026-05-15 | trial_062 | ConvNeXt-axis 동결 + BLEND_PSEUDOMIX 0.08→0.11 (3pp ConvNeXt 슬롯 → 검증된 5-fold pmix 컴포넌트). Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 11% + distill 0 + convnext 0. ConvNeXt 데이터셋 2개 kernel-metadata 에서 제거 (auto-submit 트리거 가설 검증). kernel v49 | 0.934 | ➖ best 동률 (ralph it.12. pmix 0.08→0.11 무효과. ConvNeXt dataset 제거 후 채점 정상화 확인 ✓) |
