## Submission 02 Reflection

### 결과
- Val score: OOF 미검증 (로컬 검증 불가능한 구조)
- Public score: 0.910
- 이전 대비: -0.002 (sub_01 0.912 → 0.910)

### Gap 원인 분석
- post-processing(temperature scaling, file-level confidence, rank-aware scaling)이 **오히려 악화**
- temperature scaling: Aves=1.10, Texture=0.95로 설정했지만 최적값이 아닐 수 있음
- file-level/rank-aware scaling: BirdCLEF 2025 솔루션에서 가져왔지만 2026 데이터 특성이 다를 수 있음
- **OOF 검증 없이 제출한 실수** — 로컬에서 temperature만 테스트했을 때 변화 없었음 (0.9754→0.9754)

### 버려야 할 것
- 검증 없는 post-processing 추가
- file-level confidence scaling, rank-aware scaling (효과 미검증)
- "discussion에서 좋다고 했으니까" 식의 무비판적 적용

### 유지해야 할 것
- 원본 0.912 fork 코드 (sub_01 그대로)
- Gaussian smoothing (원본에 이미 포함)

### 다음 가설
- probe 하이퍼파라미터만 변경 (안전한 방향)
  - 로컬 trial_009에서 PCA 96 + C=0.1이 best (0.9766 > 기존 PCA 32 + C=0.25)
  - 단, 로컬 OOF와 Kaggle 실제 점수는 직접 비교 불가
- MLP probe 실험 결과 기다린 후 결정
- soundscape 전체 임베딩 추출 완료되면 pseudo-labeling으로 데이터 증가
