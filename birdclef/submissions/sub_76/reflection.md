## Submission 76 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_083 perch_up

### 결과
- Public: **0.938** (best 동률)
- 제출: 2026-05-31 12:51 → COMPLETE

### 변경사항 (이전 sub 대비)
- `BLEND_PROTO` 0.50 → 0.48 (-2pp), `BLEND_SED` 0.50→0.40 복원(trial_082 회귀 청소)
- Perch 0.10 → 0.12 (+2pp)
- SED 0.40 고정 (trial_082에서 확인한 포화점)
- kernel v74 → v75

### 교훈
- SED 40% 고정 상태에서 Perch 10%→12% 미세 재배분은 **LB 무변화**(0.938 동률).
- Perch는 prior-fused 강한 base지만 10~12% 범위에서 무차별 — Proto 48~50% / Perch 10~12% 어디든 0.938.
- **weight 미세조정(±2pp)으로는 0.938 천장을 못 뚫는다.** trial_062~071에서 0.934 천장을 weight로 못 뚫었던 것과 동일 패턴이 0.938에서 재현.

### 버려야 할 것
- weight ±2pp 미세 그리드 추가 탐색. SED40/Proto48~50/Perch10~12 plateau 확인됨 — 더 쪼개도 무의미.
- "Perch 비중 키우면 강한 base라 오른다"는 가정 (이 범위에선 무효).

### 유지해야 할 것
- **trial_080 = best 0.938** (SED40/Proto50/Perch10), trial_083도 동률이므로 둘 다 제출 후보.
- SED 40% 포화점.

### 다음 가설
weight space는 0.938에서 포화. **컴포넌트 자체 품질/구성을 바꿔야 천장 돌파 가능**:
1. **SED logit z-score 정규화** — SED와 Proto/Perch의 logit 스케일이 다르면 명목 weight≠effective weight. library `cnn-logit-scale-mismatch-breaks-blend` 참조, SED logit 정규화 후 동일 weight 재시도.
2. **ProtoSSM fold 앙상블** — Proto 48~50%가 가장 큰 비중. 단일 Proto → fold 앙상블(pmix 패턴)로 신호 안정화, 같은 weight에서 품질↑.
3. **새 1차 컴포넌트 추가** — 현재 Perch/Proto/SED 3종. EffNet5fold(검증된 0.15 슬롯)를 SED40/Proto40/Perch10/EffNet10 식으로 재도입해 모델 다양성 확보.
