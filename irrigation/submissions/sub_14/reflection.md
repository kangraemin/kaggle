# Sub 14 Reflection — trial_015_pseudo_label

## 결과
- trial_015: Val **0.9796** / Public **0.97771**
- 이전 best (013): Public 0.97833 → **-0.00062 하락**
- Gap: -0.0019 (013 대비 불안정해짐)

## trial_015 핵심 변경점 (trial_013 대비)
1. **pseudo-labeling** — conf>0.95 test samples 249K 추가 (Low:144K, Med:96K, High:8K)
2. **trial_011 arch 사용** (trial_013 arch 아님) — single XGB, lr=0.01, 4000 rounds
3. **threshold** — (Low×0.75, Med×0.6, High×3.6)

## 교훈

### pseudo-label은 역효과
- val OOF 동일(0.9796)하지만 public 하락 → 테스트셋 분포 노출로 오히려 overfitting
- 249K 추가 중 High class가 8K밖에 없음 — 클래스 불균형 그대로 증폭
- trial_011 arch(단순)로 돌아간 것도 regression 원인 — trial_013보다 약한 arch

### pseudo-label 적용 시 주의사항
- trial_013 arch(3-seed, coord descent bias)와 결합해야 fair 비교 가능
- confidence threshold 0.95도 너무 관대할 수 있음 (249K는 과다)
- High class pseudo-label이 부족하면 오히려 균형이 더 나빠짐

## 버려야 할 것
- trial_011 arch 재사용 — trial_013보다 약함, regression 위험
- conf>0.95 단순 pseudo-label — 분포 편향 확인 없이 적용 금지

## 유지해야 할 것
- trial_013 구조 (3-seed XGB, coord descent bias, orig append w=0.35)
- best public = 0.97833 (trial_013)

## 다음 가설
- trial_013 arch + pseudo-label (더 높은 conf threshold, 예: 0.99)
- 5-seed XGB (3→5 seed 확장)
- Chris Deotte magic formula features (logit anchor)
