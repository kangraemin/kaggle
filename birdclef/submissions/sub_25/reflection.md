# Sub 25 Reflection — trial_038_prior_mask (sub_25 v1)

**Base**: sub_24 trial_037 ConvNeXt 3-way blend (0.930) — Kaggle kernel v18 fold0 / v22 fold2 single
**Trial**: trial_038
**Hypothesis**: Pantanal/labeled soundscape에 등장 안 하는 67종(C tier)에 prior 0.3 곱하면 false positive 감소로 LB 미세 ↑. 단 macro AUC + skip-empty species metric 특성상 효과 제한적 예상.

## 결과
- Public: **0.930** (2026-04-28 16:08 제출, 채점 완료)
  - best 동률. prior mask 효과 없음 (macro AUC + skip-empty 특성상 예상된 결과)
- 노트북 v22 출력 검증 OK:
  - `[sub_25 v1] prior mask: A+B=167 C=67 mean=0.800` (cell 17)
  - `[sub_25 v1] applying prior mask...` (cell 64, threshold sharpening 직후)
  - Wall time 211s (dry-run, sub_24와 동일 수준)
  - submission.csv 정상 생성 (NaN 없음, 234 컬럼 일치)

## 변경사항 (sub_24 대비)
- 노트북 cell 17 추가: `_compute_class_prior_mask(PRIMARY_LABELS, BASE)` — train.csv + train_soundscapes_labels에서 in-notebook 계산
  - Pantanal box: 위도 -21.6~-16.5, 경도 -57.6~-55.9 (recording_location.txt)
  - Tier A (75) + B (92) → 1.0 / Tier C (67) → 0.3
- 노트북 cell 64 수정: `apply_per_class_thresholds` 직후 `probs = probs * CLASS_PRIOR_MASK[None,:]` 한 줄
- 외부 dataset 추가 0 (mask는 competition data로 즉시 계산)
- 모델 weight 변경 0 (sub_24와 동일 — ConvNeXt fold2 single + EffNet 5fold + Perch ONNX)

## EDA 발견 (sub_25 라운드 핵심)
- **test = single site S05 (Pantanal, Brazil), 새벽 1시** — sub_18 reflection의 "미국 자연 soundscape" 가정은 4주간 잘못된 도메인 모델링. `recording_location.txt` 직접 확인으로 발견.
- **taxonomy 234 클래스 중 28개는 train.csv 0샘플** (25 Insect sonotype + 3 기타). 이전 모든 학습이 이 28개 영구 0점 (uniform 0.0042). labeled soundscape 66 files만이 학습 소스.
- **labeled soundscape (1478 segments × 5초 multi-label)** 은 test 도메인과 매칭됨에도 학습/val에 미사용. val=focal XCL holdout이라 LB 갭이 큼 (val 0.99 vs LB 0.93).
- **train.csv `secondary_labels`** 컬럼 4372 row (12%)에 161종 secondary 정보 있음. 단일 single-hot 학습으로 무시되어 옴.
- **BirdNET pseudo-label**: cross-model이지만 6522 종 모두 Aves (곤충 0%), labeled subset에서 recall 19% 측정 → ROI 낮음 결론.

## 교훈
- **macro AUC + skip-empty species metric 함의**: test에 등장 안 하는 종은 metric 영향 없음. Prior masking으로 false positive 줄여도 macro 평균 자체에는 거의 영향 없음. **zero-data 클래스(28개)를 학습 가능하게 만드는 게 ROI 훨씬 큼** (25/234 × Δ_per_class).
- **데이터 location/메타파일 1차 출처 확인 필수**: 자기 reflection의 도메인 가정 신뢰 금지. 4주간 잘못 가정한 사례가 sub_18에 있음.
- **첫 5초만 cache되는 함정** (`scripts/train_convnext_5fold.py:110-111`): random crop 코드가 캐시 사용 분기에서 무용지물. 매 epoch 동일한 첫 5초 spec 반복 학습 → 시간축 다양성 0.
- **macOS arm64 + tflite-runtime wheel 부재** → ai-edge-litert + sys.modules shim으로 birdnetlib 동작 (1차 삽질).

## 버려야 할 것
- BirdNET pseudo-label on unlabeled soundscape (recall 19% + 곤충 0% — task #9 보류)
- "test = 미국 soundscape" 가정 (실제 Pantanal Brazil)
- prior mask C tier 0.3 weight: macro AUC + skip-empty 분석상 효과 미미 — 결과 보고 후 다른 weight (0.5, 0.7) 시도 또는 폐기

## 유지해야 할 것
- sub_24 ConvNeXt 3-way blend 골격 (Perch 65% + EffNet 15% + ConvNeXt 20%)
- in-notebook prior 계산 패턴 (외부 dataset 의존성 0)
- labels_v2.npz 통합 라벨 (`data/v2/labels_v2.npz`) — 다음 학습 라운드 정답
- multi-window cache 빌더 (`build_multi_window_cache.py`) — 첫 5초 함정 해결책

## 다음 가설 (Phase B — v2 학습)
1. **trial_039 multi_window_v2_train (HIGH)**: ConvNeXt v2 학습 (multi-window cache + labels_v2 + secondary 0.3 + soundscape 1478 통합). fold 5 끝나면 (~24h 후) 시작. 25 sonotype 클래스 학습 신호 0 → 첫 학습 가능. 예상 LB +0.01~0.02.
2. **trial_040 stacker_perch_lgbm (MED)**: Perch embedding × labeled soundscape 1478 → LightGBM 234. 노트북 추론 layer 추가 → CNN+Perch+stacker 4-way blend. 코드 작성 완료, 학습 진행 중.
3. **trial_041 prior_mask_tuning (LOW)**: PENDING 결과 본 후 weight 튜닝 (0.3→0.5/0.7). 또는 폐기.
4. **synthetic soundscape mixing**: train_soundscapes 빈 구간 → focal mixup background. domain gap 직격. v2 학습에 추가.

## 진행 중 자원 (병렬)
- ConvNeXt fold 4 학습 epoch 11/30 (best 0.9908 @ epoch 4 fold3). fold 5까지 ~24h.
- Multi-window cache build 24% (37027 중 9000+). ETA 47분.
- Perch stacker 학습 (1478 × 1536 → LightGBM 234).
- Kaggle v22 채점 PENDING (>9h).
