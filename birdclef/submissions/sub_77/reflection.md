## Submission 77 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_084 sed_znorm

### 결과
- Public: **0.938** (best 동률)
- 제출: 2026-05-31 15:59 → COMPLETE

### 변경사항 (이전 sub 대비)
- blend 산술 앞에 진단 print 3종(proto/sed/perch mean·std·range) 추가
- SED raw logit을 proto 분포로 global z-score 매칭 (`sed=(sed-mean)/std*proto.std+proto.mean`)
- weight best 복원 (SED40/Proto50/Perch10), kernel v75→v76

### 진단 결과 (핵심 자산 — 다음 trial 근거)
| 컴포넌트 | mean | std | range |
|---|---|---|---|
| proto | -0.93 | 2.42 | [-9.9, 10.2] |
| **sed** | **-7.26** | 1.82 | [-13.1, 3.6] |
| perch | -1.15 | 4.13 | [-13.6, 12.8] |
- **셋 다 logit 공간** (내 "확률 0~1" 가정은 틀림). SED만 mean이 -7.26으로 크게 치우침.

### 교훈
- **logit global z-score 정규화는 ROC-AUC에서 무효과**. 이유: ① global affine `a*sed+b`의 상수항 b는 모든 행/클래스에 동일 → 클래스별 row-rank 무영향 (ROC-AUC는 rank-based), ② scale항 a(=2.42/1.82≈1.33)는 SED effective weight를 1.33배 올린 것과 동일 → trial_082(SED weight↑) 무효과와 일관.
- T_AVES 온도스케일링 제거(trial_070)도 무효과였던 것과 같은 원리 — rank 보존 변환은 ROC-AUC에 안 통한다.

### 버려야 할 것
- logit 스케일/정규화/온도 조정 계열 전부. ROC-AUC rank-based라 monotonic 변환은 무의미.
- weight 미세조정 (trial_082/083에서 이미 포화 확인).

### 유지해야 할 것
- **trial_080 best 0.938** (SED40/Proto50/Perch10).
- 진단 print (다음에도 컴포넌트 분포 확인용으로 유용).

### 다음 가설
weight·scale 축 모두 포화. **클래스별 rank를 실제로 바꾸는 변경**만 유효:
1. **EffNet5fold 재도입** — 검증된 0.15 컴포넌트(과거 best 흐름). 현재 Perch/Proto/SED 3종에서 빠짐. SED40/Proto45/Perch10/EffNet5 식으로 모델 가족 다양성 추가 → 새 신호로 rank 변화. **가장 유망**.
2. **per-class rank 변경 후처리** — 클래스별 calibration (분포가 다른 클래스 보정). row-rank를 클래스마다 다르게 조정.
3. **ProtoSSM TTA shift 확대** — proto가 50% 최대 비중, TTA shift 늘려 proto 품질 자체 향상.
