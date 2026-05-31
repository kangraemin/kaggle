## Submission 75 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_082 sed_heavy

### 결과
- Public: **0.937** (best 0.938 대비 **-0.001 하락**)
- 제출: 2026-05-31 10:51 → COMPLETE

### 변경사항 (이전 sub 대비)
- `BLEND_SED` 0.40 → 0.50 (+10pp)
- `BLEND_PROTO` 0.50 → 0.40 (-10pp)
- Perch 0.10 고정, 그 외 컴포넌트 0 유지
- kernel v73 → v74

### 교훈
- **SED 40%가 포화점/최적**. trial_079(18%)→080(40%)에서 +0.003 올랐던 SED는 50%에서 처음으로 역효과(-0.001).
- 📚 library `blend-ratio-weak-model-hurts-ensemble` 재확인: 약한 모델(SED) 비중을 최적 너머로 올리면 강한 모델(ProtoSSM 40%) 기여가 줄어 앙상블 전체가 하락. SED↔Proto trade-off에서 SED 40/Proto 50이 sweet spot.
- 단일 축 재배분 격리 실험이라 원인이 명확 — SED 비중만 바뀌었고 결과는 하락. 다른 변수 오염 없음.

### 버려야 할 것
- SED 비중 50% 이상 push. SED 40%가 상한.
- "SED를 더 올리면 더 오른다"는 선형 가정 (포화 확인됨).

### 유지해야 할 것
- **trial_080 구성 (SED 40% + Proto 50% + Perch 10%, 0.938)** = 현재 best, 다음 base.
- SED↔Proto 축에서 40/50 비율.

### 다음 가설
1. **SED fold 앙상블 품질 강화** — SED 비중은 40%가 상한이니, 비중 대신 SED 컴포넌트 자체 품질을 올린다. 현재 5-fold mean인데 fold별 temperature/weight 튜닝 또는 SED logit 정규화(z-score) 검토.
2. **Perch 비중 미세 조정** — SED 40/Proto 50 고정하고 Perch 10%→8%/12% 미세 탐색 (지금까지 Perch 10% 고정만 테스트).
3. **ProtoSSM fold 앙상블** — Proto 50%가 큰 비중이므로 Proto 단일→fold 앙상블로 신호 안정화 (pmix에서 검증된 fold 확장 패턴).
4. **세 컴포넌트 미세 그리드** — SED 0.40 고정, Proto/Perch를 (48/12),(52/8) 등 ±2pp 그리드로 0.938 천장 돌파 시도.
