# Plan: trial_078_birdnet_remove — jarturo/birdnet 제거 + blend 공식 유지 (kernel v69)

## Context

trial_077 (kernel v68) silent reject 원인: `jarturo/birdnet` 데이터셋이 Kaggle 코드 컴페티션 hidden re-run 환경에서 거부됨.
동일 패턴: trial_060/061 ConvNeXt 데이터셋 거부 → trial_062에서 제거 후 채점 정상화.
커널 로그에서 submission.csv 정상 생성 확인 (BirdNET 157 mappings, mean 0.1003, shape 240×235).
But publicScore 공란 = auto-submit 트리거 미발생.

**trial_078 목적**:
1. `jarturo/birdnet` dataset_sources에서 제거 → silent reject 해결
2. `BLEND_BIRDNET=0.0` → Perch 13%→23% 자동 증가
3. 나머지 blend 구성 유지 (ProtoSSM 60% + SED 10% + mwf0 2% + pmix 5%)
→ 이것이 **blend 공식 수정 효과만 격리 검증하는 첫 번째 유효 제출**

## 변경 파일

### 1. `notebooks/kernel-metadata.json`

Before:
```json
"dataset_sources": [..., "tuckerarrants/bc2026-distilled-sed-public", "jarturo/birdnet"]
```

After:
```json
"dataset_sources": [..., "tuckerarrants/bc2026-distilled-sed-public"]
```
(jarturo/birdnet 제거)

### 2. `notebooks/birdclef2026-effnet-5fold-blend.ipynb` — cells[62]

Before:
```python
BLEND_PSEUDOMIX = 0.05  # trial_077: Perch = 1-0.60-0-0.02-0.05-0.10-0.10=0.13.
BLEND_PROTO = 0.60  # trial_077: ProtoSSM 복원 (blend 버그 수정). 0.947 72% 근사.
BLEND_EFFNET = 0.0  # trial_077: EffNet5fold 0 유지 (BirdNET 격리 테스트).
BLEND_MWF0 = 0.02   # trial_073: mwf0 fold0 weight.
BLEND_SED = 0.10  # trial_077: SED 10% (BirdNET 추가로 15%→10%).
BLEND_BIRDNET = 0.10  # trial_077: BirdNET v2.4 TFLite 신규.
```

After:
```python
BLEND_PSEUDOMIX = 0.05  # trial_078: Perch = 1-0.60-0-0.02-0.05-0.10-0=0.23.
BLEND_PROTO = 0.60  # trial_078: ProtoSSM 60% (blend 공식 격리 검증).
BLEND_EFFNET = 0.0  # trial_078: EffNet5fold 0 유지.
BLEND_MWF0 = 0.02   # trial_073: mwf0 fold0 weight.
BLEND_SED = 0.10  # trial_078: SED 10% 유지.
BLEND_BIRDNET = 0.0  # trial_078: BirdNET 제거 (jarturo/birdnet silent reject 수정).
```

### 3. `notebooks/birdclef2026-effnet-5fold-blend.ipynb` — cells[69] 헤더 첫 줄

Before:
```python
# === blend: Perch 13% + ProtoSSM 60% + BirdNET 10% + SED 10% + mwf0 2% + pmix 5% (trial_077 — blend formula 버그 수정 + BirdNET 신규) ===
```

After:
```python
# === blend: Perch 23% + ProtoSSM 60% + SED 10% + mwf0 2% + pmix 5% (trial_078 — jarturo/birdnet 제거, blend 공식 격리 검증) ===
```

### 4. 신규: `submissions/sub_71/trial_078_birdnet_remove/meta.json`

```json
{
  "trial_id": "trial_078",
  "name": "birdnet_remove",
  "sub_id": "sub_71",
  "iter": 28,
  "kernel": "ramkang/birdclef2026-effnet-5fold-blend",
  "date": "2026-05-18",
  "base_trial": "trial_077 birdnet_add (kernel v68, silent reject)",
  "strategy": "jarturo/birdnet dataset 제거. BLEND_BIRDNET=0, Perch 13%→23%. ProtoSSM 60% + SED 10% + mwf0 2% + pmix 5% 유지. blend 공식 수정 효과 첫 번째 유효 검증.",
  "weights": {
    "perch": 0.23,
    "proto_ssm": 0.60,
    "sed_5fold": 0.10,
    "mwf0_fold0": 0.02,
    "pmix_fold0_to_4": 0.05,
    "effnet5fold": 0.0,
    "birdnet_v2.4": 0.0
  },
  "expected": "0.934+ (blend 공식 수정 + ProtoSSM 60% 첫 유효 제출)",
  "score": "PENDING",
  "kernel_version": 69,
  "submitted_at": "PENDING",
  "submit_command": "kaggle competitions submit -c birdclef-2026 -k ramkang/birdclef2026-effnet-5fold-blend -v 69 -f submission.csv -m 'trial_078 birdnet_remove: Perch 23%+ProtoSSM 60%+SED 10%+mwf0 2%+pmix 5% kernel v69 (jarturo/birdnet 제거, blend 공식 격리)'"
}
```

## 검증 (6-TC)

```python
import json
nb = json.load(open('notebooks/birdclef2026-effnet-5fold-blend.ipynb'))
s62 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_BIRDNET = 0.0' in s62, 'TC-1'
assert 'BLEND_PROTO = 0.60' in s62, 'TC-2'
assert 'BLEND_SED = 0.10' in s62, 'TC-3'
assert 'trial_078' in s62, 'TC-4'
s69 = ''.join(nb['cells'][69]['source'])
assert 'trial_078' in s69, 'TC-5'
km = json.load(open('notebooks/kernel-metadata.json'))
assert 'jarturo/birdnet' not in km['dataset_sources'], 'TC-6'
print('PASS (6 TCs)')
```

## 개발 Phase 계획

### Phase 1: trial078-birdnet-remove
**목표**: 3개 파일 수정 + meta.json 생성 + 6-TC PASS + kernel push
- Step 1: kernel-metadata.json jarturo/birdnet 제거 + cells[62] BLEND_BIRDNET=0 + cells[69] 헤더 수정 + meta.json 생성 + 검증 PASS
