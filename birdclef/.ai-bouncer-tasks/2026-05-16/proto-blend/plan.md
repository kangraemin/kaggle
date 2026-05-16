# Plan: trial_073 ProtoSSM blend 추가

## Context
ProtoSSM v4가 cell 61에서 실행되어 `final_test_scores` (ProtoSSM+MLP 50:50 blend)를 출력하지만,
cell 68에서 `final_test_scores`가 Perch+EffNet 등 blend로 덮어써져 **ProtoSSM 출력이 제출에 반영 안 됨**.
0.947 LB cluster (482팀)는 ProtoSSM 기반 → 이를 blend에 추가해 0.947+ 달성 목표.

변경: cell 61 끝에 `proto_logits` 저장 → cell 63에 `BLEND_PROTO=0.50` 추가 → cell 68 blend에 포함.
최종 weights: ProtoSSM 50% + Perch 13% + EffNet 15% + mwf0 2% + pmix 5% + SED 15% = 100%

## 변경 파일

### `notebooks/birdclef2026-effnet-5fold-blend.ipynb`

**변경 1: cell 61 끝 (cells[60]) — proto_logits 저장**

Before (cell 61 마지막 줄):
```python
LOGS["test_inference"] = test_logs
```

After:
```python
LOGS["test_inference"] = test_logs
proto_logits = final_test_scores.copy()  # trial_073: save before cell 68 overwrites
print(f"proto_logits saved: {proto_logits.shape}, range [{proto_logits.min():.3f}, {proto_logits.max():.3f}]")
```

**변경 2: cell 63 (cells[62]) — BLEND_PROTO 상수 추가**

Before:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: 0.06->0.05 (-1pp). SED 0.05->0.15 (+10pp), pmix -1pp + Perch -9pp. Perch = 1-0.15-0.02-0.05-0.15=0.63.
```

After:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: 0.06->0.05 (-1pp). SED 0.05->0.15 (+10pp), pmix -1pp + Perch -9pp. Perch = 1-0.15-0.02-0.05-0.15=0.63.
BLEND_PROTO = 0.50  # trial_073: ProtoSSM v4+MLP OOF-blend. Perch 63%→13%.
```

**변경 3: cell 68 (cells[67]) — blend formula에 ProtoSSM 추가**

Before:
```python
# === blend: Perch 63% + EffNet 5fold 15% + mwf0 2% + pmix 5% + SED 5-fold 15% (trial_072 — SED ONNX 추가, T_AVES=1.0, prior mask 0.5) ===
BLEND_DISTILL = 0.0  # trial_060/061: distill zero-effect, dropped. trial_062: stays 0.
# BLEND_SED is defined in cell 62
final_test_scores = (
    _perch_scores_raw * (1 - BLEND_EFFNET - BLEND_MWF0 - BLEND_EFFNET_S - BLEND_PSEUDOMIX - BLEND_DISTILL - BLEND_CONVNEXT - BLEND_SED)
    + BLEND_EFFNET * effnet_logits
    + BLEND_MWF0 * mwf0_logits
    + BLEND_EFFNET_S * effnets_logits
    + BLEND_PSEUDOMIX * pmix_logits
    + BLEND_DISTILL * distill_logits
    + BLEND_CONVNEXT * convnext_logits
    + BLEND_SED * sed_logits
)
_p = 1 - BLEND_EFFNET - BLEND_MWF0 - BLEND_EFFNET_S - BLEND_PSEUDOMIX - BLEND_DISTILL - BLEND_CONVNEXT - BLEND_SED
print(f"blend (logit-space weighted sum, trial_072): Perch {_p:.0%} + EffNet5fold {BLEND_EFFNET:.0%} "
      f"+ fold0-B0 {BLEND_MWF0:.0%} + fold0-S {BLEND_EFFNET_S:.0%} + pmix {BLEND_PSEUDOMIX:.0%} + distill {BLEND_DISTILL:.0%} + convnext5fold {BLEND_CONVNEXT:.0%}")
print(f"final_test_scores (logit) range: [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}], mean {final_test_scores.mean():.3f}")
```

After:
```python
# === blend: Perch 13% + ProtoSSM 50% + EffNet 5fold 15% + mwf0 2% + pmix 5% + SED 5-fold 15% (trial_073 — ProtoSSM v4 추가) ===
BLEND_DISTILL = 0.0  # trial_060/061: distill zero-effect, dropped. trial_062: stays 0.
# BLEND_SED is defined in cell 62 (BLEND_PROTO also)
final_test_scores = (
    _perch_scores_raw * (1 - BLEND_EFFNET - BLEND_MWF0 - BLEND_EFFNET_S - BLEND_PSEUDOMIX - BLEND_DISTILL - BLEND_CONVNEXT - BLEND_SED - BLEND_PROTO)
    + BLEND_EFFNET * effnet_logits
    + BLEND_MWF0 * mwf0_logits
    + BLEND_EFFNET_S * effnets_logits
    + BLEND_PSEUDOMIX * pmix_logits
    + BLEND_DISTILL * distill_logits
    + BLEND_CONVNEXT * convnext_logits
    + BLEND_SED * sed_logits
    + BLEND_PROTO * proto_logits
)
_p = 1 - BLEND_EFFNET - BLEND_MWF0 - BLEND_EFFNET_S - BLEND_PSEUDOMIX - BLEND_DISTILL - BLEND_CONVNEXT - BLEND_SED - BLEND_PROTO
print(f"blend (logit-space weighted sum, trial_073): Perch {_p:.0%} + ProtoSSM {BLEND_PROTO:.0%} + EffNet5fold {BLEND_EFFNET:.0%} "
      f"+ fold0-B0 {BLEND_MWF0:.0%} + fold0-S {BLEND_EFFNET_S:.0%} + pmix {BLEND_PSEUDOMIX:.0%} + SED {BLEND_SED:.0%}")
print(f"final_test_scores (logit) range: [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}], mean {final_test_scores.mean():.3f}")
```

## 검증
```bash
python3 -c "
import json
nb = json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb'))
# cell 61: proto_logits 저장 확인
src61 = ''.join(nb['cells'][60]['source'])
assert 'proto_logits = final_test_scores.copy()' in src61
# cell 63: BLEND_PROTO 확인
src63 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_PROTO = 0.50' in src63
# cell 68: blend formula 확인
src68 = ''.join(nb['cells'][67]['source'])
assert 'BLEND_PROTO * proto_logits' in src68
assert 'BLEND_SED - BLEND_PROTO' in src68
print('PASS')
"
```

## E2E 테스트
```bash
cd /Users/ram/programming/vibecoding/kaggle/birdclef
python3 -c "
import json
nb = json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb'))
src61 = ''.join(nb['cells'][60]['source'])
assert 'proto_logits = final_test_scores.copy()' in src61, 'TC-1 fail'
src63 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_PROTO = 0.50' in src63, 'TC-2 fail'
src68 = ''.join(nb['cells'][67]['source'])
assert 'BLEND_PROTO * proto_logits' in src68, 'TC-3 fail'
assert 'BLEND_SED - BLEND_PROTO' in src68, 'TC-4 fail'
print('PASS: all 4 TCs')
" && echo "✅ TC 통과"
```

## 개발 Phase 계획

### Phase 1: ProtoSSM blend 통합
**목표**: 3개 cell 수정으로 ProtoSSM v4 출력을 최종 blend에 반영
- Step 1: cell 61 끝에 `proto_logits` 저장 라인 추가 + cell 63에 `BLEND_PROTO` 상수 추가 + cell 68 blend formula 업데이트 — 검증 명령 통과 확인
