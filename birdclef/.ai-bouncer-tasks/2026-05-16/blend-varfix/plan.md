# Plan: blend-varfix — BLEND_EFFNET/BLEND_MWF0/effnet_logits 미정의 버그 수정

## Context
**기존 proto-blend 작업(완료)**: trial_073 proto_logits/BLEND_PROTO 추가는 완료됨.

**신규 버그 (kernel v61 crash 예상)**: commit 9573514에서 SED 추가 시 구 cell 63(EffNet 5fold 추론)을 제거하면서
해당 셀이 정의하던 `BLEND_EFFNET=0.15`, `BLEND_MWF0=0.02`, `effnet_logits` 변수가 사라짐.
현재 cell 68(blend formula)이 이 세 변수를 사용하므로 NameError 크래시 발생 예상.

**수정**: EffNet 5fold 추론은 복원하지 않고 cell 63에 상수로 추가:
- `BLEND_EFFNET = 0.0` (EffNet 5fold 추론 없으므로)
- `BLEND_MWF0 = 0.02` (mwf0 fold0 추론은 cell 65에서 여전히 실행)
- `effnet_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)` (Python이 BLEND_EFFNET=0이어도 변수 평가하므로 placeholder 필요)

결과 weights (trial_073 실제): ProtoSSM 50% + Perch 28% + mwf0 2% + pmix 5% + SED 15% = 100%

## 변경 파일

### `notebooks/birdclef2026-effnet-5fold-blend.ipynb`

**변경 1: cell 63 (cells[62]) — 누락 변수 추가**

Before:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: 0.06->0.05 ...
BLEND_PROTO = 0.50  # trial_073: ProtoSSM v4+MLP OOF-blend. Perch 63%→13%.
```

After:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: 0.06->0.05 ...
BLEND_PROTO = 0.50  # trial_073: ProtoSSM v4+MLP OOF-blend. Perch 63%→13%.
BLEND_EFFNET = 0.0  # trial_073: EffNet 5fold cell removed in SED integration. Perch absorbs slot.
BLEND_MWF0 = 0.02   # trial_073: mwf0 fold0 weight (cell 65 inference still runs).
effnet_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)  # placeholder (BLEND_EFFNET=0)
```

**변경 2: cell 68 (cells[67]) — 주석 Perch% 수정**

Before:
```python
# === blend: Perch 13% + ProtoSSM 50% + EffNet 5fold 15% + mwf0 2% + pmix 5% + SED 5-fold 15% (trial_073 — ProtoSSM v4 추가) ===
```

After:
```python
# === blend: Perch 28% + ProtoSSM 50% + mwf0 2% + pmix 5% + SED 5-fold 15% (trial_073 — ProtoSSM v4 추가, EffNet slot → Perch) ===
```

---
*(아래는 이전 proto-blend 작업의 Before/After — 이미 완료됨, 참고용)*

**[완료] 변경: cell 61 끝 (cells[60]) — proto_logits 저장**

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

## 검증 명령어
```bash
python3 -c "
import json
nb = json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb'))
src63 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_EFFNET = 0.0' in src63, 'TC-1 fail'
assert 'BLEND_MWF0 = 0.02' in src63, 'TC-2 fail'
assert 'effnet_logits = np.zeros' in src63, 'TC-3 fail'
src67 = ''.join(nb['cells'][67]['source'])
assert 'BLEND_PROTO * proto_logits' in src67, 'TC-4 regression fail'
print('PASS')
"
```

## 개발 Phase 계획

### Phase 1: blend-varfix
**목표**: cell 63에 누락 변수 추가로 kernel v61 crash 방지
- Step 1: cell 63 수정 + cell 68 주석 업데이트 — 검증 PASS 확인
