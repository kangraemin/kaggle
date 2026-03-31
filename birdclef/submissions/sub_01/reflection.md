## Submission 01 Reflection

### 결과
- Val score: OOF 0.487 (soundscape 59파일 기준, 과소평가)
- Public score: 0.912
- Val-Public gap: 해당없음 (OOF가 soundscape 데이터만 사용, 실제 성능과 무관)

### 경과
- trial_001~003: 자체 파이프라인 (Perch 임베딩 → PCA → XGBoost/LightGBM)
- v1~v13: Kaggle 노트북 제출 삽질 (경로, 변수명, GPU 제한, TFLite, test 파일 부재, TF 버전)
- trial_007: 0.912 공개노트북 fork → 첫 유효 제출 성공

### 왜 자체 파이프라인이 실패했나
1. **Perch CPU 추론이 90분 제한에 걸림** — TFLite 변환도 불안정
2. **row_id 형식** — `seg * 5` vs `(seg+1) * 5` 차이 가능성
3. **TF 버전** — Perch v2_cpu가 TF 2.20 필요, 기본 환경은 2.19

### 왜 fork가 성공했나
1. **검증된 파이프라인** — 이미 수백명이 제출 성공한 코드
2. **Perch logits 직접 사용** — 14,795종 → 234종 매핑. 별도 모델 불필요
3. **사전 캐시** — train embeddings를 데이터셋으로 미리 계산, 추론 시간 절약
4. **TF 2.20 wheel** — kernel_sources로 설치

### 버려야 할 것
- 자체 Perch 임베딩 추출 파이프라인 (너무 느리고 불안정)
- XGBoost per-species 모델 (LR보다 느리고 나쁨)

### 유지해야 할 것
- 0.912 fork 코드 기반
- Bayesian prior fusion (site × hour)
- PCA 32 + LogisticRegression probes
- Gaussian temporal smoothing

### 다음 가설
- probe 하이퍼파라미터 튜닝 (PCA dim, C, alpha)
- 0.916 공개노트북 참고 (tuko_tanzwoo)
- SED 모델 앙상블 추가 (EfficientNet)
- pseudo-labeling on unlabeled soundscapes
