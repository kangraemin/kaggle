# Sub 42 Reflection — trial_050 pseudo_mix_blend

**Base**: trial_046/047 best 0.932 (Perch 80%/77% + epoch50-5fold 15% + mwf0-fold0 5%/8%)
**Trial**: trial_050 kernel v37 (ralph-x iteration 1/10)

## 결과
- Public: **0.933** ✅ **new best** (+0.001)

## 변경사항 (현재 노트북 = trial_048 distill30 상태 대비)
- **5fold 컴포넌트 복원**: distill30 KD → epoch50 SoftAUC (trial_048에서 0.931로 하락했던 것을 best 기준선으로 회귀)
- **EffNet pseudo-mix fold0 추가**: `models/effnet_pseudo_mix/best_fold0.pth` (로컬 학습, 5/9 완료, 30 epochs)
  - 아키텍처: EffNetMixup = stem_conv(1→3) + tf_efficientnetv2_b0(in_chans=3) + Linear(1280, 234) — 기존 EffNetF0(mwf0)와 동일
  - 학습 데이터: focal 28k~ + **30k pseudo-labeled ss10k 윈도우 (soft Perch label, 범위 [0, 0.85])** 혼합 → BCEWithLogitsLoss
  - spec: cache_v2/cache_ss10k와 동일 (MelSpec n_fft=1024 hop=320 n_mels=128 fmin=50 fmax=14000, TOP_DB=80, norm mean=-4.268 std=4.569) → 노트북 `_mwf0_spec` 그대로 재사용
- 새 Kaggle dataset: `ramkang/birdclef2026-effnet-pseudo-mix`
- 4-way blend: **Perch 77% + epoch50-5fold 15% + mwf0-fold0 5% + pmix-fold0 3%** (BLEND_MWF0 0.08→0.05로도 복원)
- pmix 추론은 mwf0 추론 루프에 합쳐 오디오 재읽기 없음 (`_mwf0_spec(chunks)` 한 번 → 두 모델 forward)

## 가설 (검증됨)
pseudo-label로 학습한 fold0 (테스트 분포인 soundscape에 더 가까운 데이터)이 distill/CE 학습 컴포넌트들과 다른 오류 분포를 가져 diversity 기여 → 3% blend로 +0.001 확인.

## 교훈
- **soft Perch pseudo-label을 섞어 학습한 EffNet은 작은 weight(3%)로도 blend에 +0.001** — diversity source로 유효
- mwf0 fold0(CE, +0.002) → pmix fold0(pseudo-mix, +0.001): 작은 weight의 다양한 EffNet 컴포넌트를 계속 쌓는 전략이 누적적으로 작동 중 (0.930 → 0.932 → 0.933)
- 노트북이 trial_048(0.931, distill30) 상태에 멈춰 있던 것을 발견 → epoch50 복원이 새 실험의 전제였음. 다음 이터레이션 시작 시 노트북이 best 상태인지 항상 확인할 것
- pseudo-mix는 동일 아키텍처(EffNetF0)·동일 spec이라 추론 코드 추가 비용 거의 0

## 유지해야 할 것
- Perch ONNX (backbone): 압도적 기여
- epoch50-5fold 15%: 안정적 (distill30보다 우수)
- mwf0-fold0 5% + pmix-fold0 3%: 누적 diversity
- prior_mask (Tier C ×0.3): neutral~ 약간 도움

## 다음 가설 (ralph it.2+)
- **pseudo-mix fold1~4 추가**: 현재 로컬에서 fold2 학습 중 (fold0/1 완료 추정). 완성 시 pmix 5-fold 앙상블 → 더 강한 컴포넌트
- BLEND_PSEUDOMIX sweep (3% → 5%, mwf0와 합쳐 8~10%)
- EffNetV2-S + Xeno pretrain (trial_049, kernel v12 학습 중) 완성 시 BLEND_EFFNET_S 활성화
- pseudo-label 품질 개선: ss10k subset 선정 기준 (현재 30k) 조정
