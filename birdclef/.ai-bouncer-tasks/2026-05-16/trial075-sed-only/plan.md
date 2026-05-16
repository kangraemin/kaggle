# Plan: trial_075_sed_only — ProtoSSM 제거, SED 15% 단독 격리 테스트 (kernel v66)

## Context

trial_074 (kernel v65, COMPLETE): ProtoSSM 65% + SED 15% + Perch 13%. 아직 미제출 (UTC 00:00 대기).

**trial_075 목적**: ProtoSSM 없이 SED 15% 단독으로 0.934 천장 돌파 여부 격리 검증.
- `BLEND_PROTO = 0.65 → 0.0` (ProtoSSM 완전 제거)
- Perch = 1 - 0 - 0.02 - 0 - 0.05 - 0 - 0 - 0.15 - 0.0 = **0.78** (78%)
- SED 15% 유지, mwf0 2%, pmix 5%

**진단 가치**:
- 075 = 0.934: SED 중립, ProtoSSM도 중립 → EffNet5fold 제거가 천장 원인
- 075 > 0.934: SED 15%가 단독으로 개선 → ProtoSSM이 오히려 노이즈였음
- 075 < 0.934: Perch 78% (EffNet 없음)이 기존 Perch 72% + EffNet 15%보다 열위

## 변경 파일

### `birdclef2026-effnet-5fold-blend.ipynb`

**변경 1: cells[62] — BLEND_PROTO 0.65→0.0**

Before:
```python
BLEND_PROTO = 0.65  # trial_074: ProtoSSM up 50→65% (073>=0.940 branch). Perch 28%→13%.
```

After:
```python
BLEND_PROTO = 0.0  # trial_075: ProtoSSM 제거. SED 15% 단독 격리. Perch 65%→78%.
```

**변경 2: cells[67] — 헤더 코멘트 trial_074→trial_075**

Before:
```python
# === blend: Perch 13% + ProtoSSM 65% + mwf0 2% + pmix 5% + SED 15% (trial_074 — ProtoSSM up 50%→65%, BLEND_SED 0.0 bug fix) ===
```

After:
```python
# === blend: Perch 78% + ProtoSSM 0% + mwf0 2% + pmix 5% + SED 15% (trial_075 — ProtoSSM 제거, SED 15% 격리 테스트) ===
```

## 검증 명령어

```bash
python3 -c "
import json
nb = json.load(open('/Users/ram/programming/vibecoding/kaggle/birdclef/notebooks/birdclef2026-effnet-5fold-blend.ipynb'))
s62 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_PROTO = 0.0' in s62, 'TC-1 fail: BLEND_PROTO not 0.0'
assert 'BLEND_SED = 0.15' in s62, 'TC-2 regression: BLEND_SED must stay 0.15'
assert 'BLEND_EFFNET = 0.0' in s62, 'TC-3 regression: BLEND_EFFNET'
assert '_perch_scores_raw = test_base_scores.copy()' in s62, 'TC-4 regression: _perch_scores_raw'
assert 'distill_logits = np.zeros' in s62, 'TC-5 regression: distill_logits'
s67 = ''.join(nb['cells'][67]['source'])
assert 'trial_075' in s67, 'TC-6 fail: cells[67] not updated'
assert 'ProtoSSM 0%' in s67 or 'ProtoSSM 0' in s67, 'TC-7 fail: cells[67] wrong ProtoSSM weight'
print('PASS')
"
```

## 개발 Phase 계획

### Phase 1: trial075-sed-only
**목표**: cells[62] BLEND_PROTO 0.65→0.0, cells[67] 헤더 갱신
- Step 1: cells[62]/cells[67] 수정 + 검증 PASS
