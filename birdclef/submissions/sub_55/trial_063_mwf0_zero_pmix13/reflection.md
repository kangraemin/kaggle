# trial_063 reflection — mwf0_zero_pmix13

- score: **0.933** ❌ (-0.001 vs best 0.934)
- scored_at: 2026-05-15 12:28 UTC (76 min PENDING)

## 결과 분석

mwf0 완전 제거(0.02→0.00) + pmix 2pp 추가(0.11→0.13) → -0.001 역효과.

**가설 기각: mwf0은 noisy하지 않았다.**
- trial_063 이전 판단: mwf0은 single fold0 기반 → noisy → 제거하면 블렌드 순도 향상
- 실제 결과: mwf0 제거가 오히려 0.001 손실
- 결론: mwf0(fold0 B0 + fold0 S 앙상블)이 실제 신호를 담고 있었음. "single fold"라도 보완적 정보 기여.

**pmix 포화 재확인:**
- trial_062: pmix 0.08→0.11 (3pp up) → 0.934 동률
- trial_063: pmix 0.11→0.13 (2pp up) → 0.933 하락
- pmix 방향 완전 소진. 추가 pmix weight 증가는 역효과.

## 시사점

현재 블렌드(Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 11%)가 optimal에 가까울 수 있음.
breakthrough를 위해 **다른 슬롯** 탐색 필요:

1. **EffNet weight 증량**: 0.15→0.17 (+2pp), Perch 0.72→0.70. 아직 미시험 방향.
2. mwf0 원복(0.02 유지)는 당연. trial_062 weights가 현재 best.

## 다음 trial_064

BLEND_EFFNET 0.15→0.17 (+2pp), BLEND_PERCH 0.72→0.70 (-2pp).
mwf0 0.02, pmix 0.11 유지. distill/convnext 0.
Perch ↔ EffNet 교환 — EffNet이 더 강력한지 검증.
