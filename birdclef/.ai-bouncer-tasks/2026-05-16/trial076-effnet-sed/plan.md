# Plan: trial_076_effnet_sed — EffNet5fold 복원 + SED 15% 유지 (kernel v67)

## Context

trial_075 (kernel v66): SED 15% + Perch 78% (ProtoSSM 0%). 제출 대기 중.

**trial_076 목적**: EffNet5fold(epoch50 SoftAUC 5-fold) 복원 + SED 15% 동시 유지.
- trial_071부터 EffNet5fold 제거됨 (BLEND_EFFNET=0)
- 현재 0.934 천장이 "EffNet5fold 제거" 때문인지 격리 검증
- BLEND_EFFNET 0→0.15, Perch 78%→63%

**진단 가치**:
- 076 > 0.934: EffNet5fold 복원 효과 확인 → EffNet5fold 제거가 천장 원인
- 076 = 0.934: EffNet5fold+SED 중 하나가 Perch와 중복
- 076 < 0.934: SED+EffNet이 상충 또는 Perch 63%가 부족

## 현재 노트북 구조 (trial_075 기준)

총 72 cells. 관련 cells:
- cells[62]: blend constants (BLEND_EFFNET=0.0, effnet_logits placeholder)
- cells[63]: EffNetF0 class 정의 (mwf0 fold0 standalone — 다른 아키텍처)
- cells[64]: mwf0/pmix fold0 inference
- cells[65]: ConvNeXt inference (BLEND_CONVNEXT=0, 영향 없음)
- cells[66]: SED 5-fold ONNX inference
- cells[67]: blend formula

## 변경 파일

### `birdclef2026-effnet-5fold-blend.ipynb`

**변경 1: cells[62] — BLEND_EFFNET 0.0→0.15, effnet_logits placeholder 제거**

Before:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: pmix -1pp. trial_074: Perch = 1-0.15-0.02-0.05-0.65=0.13.
BLEND_PROTO = 0.0  # trial_075: ProtoSSM 제거. SED 15% 단독 격리. Perch 65%→78%.
BLEND_EFFNET = 0.0  # trial_073: EffNet 5fold cell removed in SED integration. Perch absorbs slot.
BLEND_MWF0 = 0.02   # trial_073: mwf0 fold0 weight (cell 65 inference still runs).
BLEND_SED = 0.15  # trial_072: SED 5-fold 15%. Was missing from cell62 (cells[66] had 0.0 override bug).
effnet_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)  # placeholder (BLEND_EFFNET=0)
distill_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)  # placeholder (BLEND_DISTILL=0)
_perch_scores_raw = test_base_scores.copy()  # trial_073: pure Perch (prior-fused)
```

After:
```python
BLEND_PSEUDOMIX = 0.05  # trial_072: pmix -1pp. trial_076: Perch = 1-0.15-0.02-0.05-0-0.15=0.63.
BLEND_PROTO = 0.0  # trial_075: ProtoSSM 제거. trial_076: 유지(0).
BLEND_EFFNET = 0.15  # trial_076: EffNet5fold 복원 + SED 유지. Perch 78%→63%.
BLEND_MWF0 = 0.02   # trial_073: mwf0 fold0 weight (cell 65 inference still runs).
BLEND_SED = 0.15  # trial_072: SED 5-fold 15%.
distill_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)  # placeholder (BLEND_DISTILL=0)
_perch_scores_raw = test_base_scores.copy()  # trial_073: pure Perch (prior-fused)
```
(effnet_logits = np.zeros 라인 제거 — 새 inference cell에서 실제 값 생성)

**변경 2: cells[63]에 새 셀 삽입 — EffNet5fold epoch50 5-fold 추론**

기존 cells[63]~cells[67]은 cells[64]~cells[68]로 shift.

새 cells[63] 전체 내용:
```python
# === EffNet 5-Fold Global Pool Inference (trial_076: restored, EffNet5fold+SED dual-component) ===
import timm
import torchaudio
import torchvision

_wall_effnet = time.time()

class _EffSpec(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=2048, hop_length=512,
            n_mels=256, f_min=20, f_max=16000,
            mel_scale="htk", pad_mode="reflect", power=2.0, norm="slaney", center=True)
        self.resize = torchvision.transforms.Resize(size=(256, 256))
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mel = self.mel_transform(x)
        mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
        mel = mel.unsqueeze(1)
        mel = self.resize(mel)
        flat = mel.view(mel.shape[0], 1, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel = (mel - mins) / (maxs - mins + 1e-7)
        return mel

class _BirdEffNet(nn.Module):
    """EffNet global avg pool inference (same as training)."""
    def __init__(self, num_labels=234):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_b0", pretrained=False, in_chans=1, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_labels)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

import glob as _glob
_fold_paths = []
for _slug in ["birdclef2026-effnet-5fold-epoch50"]:
    _d1 = Path(f"/kaggle/input/datasets/ramkang/{_slug}")
    _d2 = Path(f"/kaggle/input/{_slug}")
    _d = _d1 if _d1.exists() else (_d2 if _d2.exists() else None)
    if _d is not None:
        _fold_paths += sorted(_glob.glob(str(_d / "best_fold*.pth")))
if not _fold_paths:
    _fold_paths = sorted(_glob.glob("/kaggle/input/birdclef2026-effnet-5fold-epoch50/best_fold*.pth"))
_effnet_models = []
_effnet_spec = _EffSpec()
for fp in _fold_paths:
    _state = torch.load(fp, map_location="cpu", weights_only=False)
    _model_state = {k: v for k, v in _state.items() if not k.startswith("spec.") and not k.startswith("mixup.")}
    m = _BirdEffNet()
    m.load_state_dict(_model_state, strict=True)
    m.eval()
    _effnet_models.append(m)
print(f"Loaded {len(_effnet_models)} EffNet folds (global pool, epoch50 SoftAUC)")

effnet_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)
if len(_effnet_models) > 0:
    _row = 0
    for fi, fpath in enumerate(test_paths):
        try:
            audio, _ = sf.read(fpath, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception:
            _row += N_WINDOWS
            continue
        chunks = np.zeros((N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for w in range(N_WINDOWS):
            s = w * WINDOW_SAMPLES
            chunk = audio[s:s+WINDOW_SAMPLES]
            if len(chunk) < WINDOW_SAMPLES:
                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
            chunks[w] = chunk
        with torch.no_grad():
            batch = torch.from_numpy(chunks)
            spec_batch = _effnet_spec(batch)
            fold_preds = np.stack([m(spec_batch).numpy() for m in _effnet_models])
            effnet_logits[_row:_row+N_WINDOWS] = fold_preds.mean(axis=0)
        _row += N_WINDOWS
        if (fi+1) % 10 == 0:
            print(f"  EffNet [{fi+1}/{len(test_paths)}] {(time.time()-_wall_effnet)/60:.1f}min")
    print(f"EffNet 5-fold done: {(time.time()-_wall_effnet)/60:.1f}min")
print(f"Wall after EffNet: {(time.time()-_WALL_START)/60:.1f}min")
```

**변경 3: cells[68] (insert 후 old cells[67]) — 헤더 코멘트 trial_075→trial_076**

Before:
```python
# === blend: Perch 78% + ProtoSSM 0% + mwf0 2% + pmix 5% + SED 15% (trial_075 — ProtoSSM 제거, SED 15% 격리 테스트) ===
```

After:
```python
# === blend: Perch 63% + EffNet5fold 15% + mwf0 2% + pmix 5% + SED 15% (trial_076 — EffNet5fold 복원+SED유지, EffNet제거 천장 가설 검증) ===
```

**변경 4: `submissions/sub_69/trial_076_effnet_sed/meta.json` 신규 생성**

```json
{
  "trial_id": "trial_076",
  "name": "effnet_sed",
  "sub_id": "sub_69",
  "iter": 26,
  "ralph_iter": 26,
  "kernel": "ramkang/birdclef2026-effnet-5fold-blend",
  "date": "2026-05-17",
  "base_trial": "trial_075 sed_only (kernel v66, ProtoSSM=0, SED 15% 격리)",
  "strategy": "EffNet5fold 복원(BLEND_EFFNET 0→0.15) + SED 15% 유지. Perch 78%→63%. EffNet5fold 제거(trial_071~)가 0.934 천장 원인인지 격리 검증.",
  "weights": {
    "perch": 0.63,
    "effnet5fold_epoch50": 0.15,
    "mwf0_fold0": 0.02,
    "pmix_fold0_to_4": 0.05,
    "sed_5fold": 0.15,
    "proto_ssm": 0.0,
    "distill": 0.0,
    "convnext": 0.0
  },
  "files_changed": [
    "birdclef2026-effnet-5fold-blend.ipynb cells[62] (BLEND_EFFNET 0→0.15, effnet_logits placeholder 제거)",
    "birdclef2026-effnet-5fold-blend.ipynb cells[63] (EffNet5fold epoch50 inference cell 신규 삽입)",
    "birdclef2026-effnet-5fold-blend.ipynb cells[68] (blend label trial_075→trial_076, Perch 63%, EffNet5fold 15%)"
  ],
  "expected": "0.934+ (EffNet5fold 복원이 천장 원인이었다면 개선). 동률이면 SED가 EffNet diversity를 이미 커버.",
  "score": "PENDING",
  "kernel_version": 67,
  "submitted_at": "PENDING",
  "submit_command": "kaggle competitions submit -c birdclef-2026 -k ramkang/birdclef2026-effnet-5fold-blend -v 67 -f submission.csv -m 'trial_076 effnet_sed: Perch 63% + EffNet5fold 15% + SED 15% + mwf0 2% + pmix 5% kernel v67 (EffNet5fold 복원+SED유지)'"
}
```

## 검증 명령어

```bash
git show HEAD:birdclef/notebooks/birdclef2026-effnet-5fold-blend.ipynb | python3 -c "
import json, sys
nb = json.loads(sys.stdin.read())
print('Total cells:', len(nb['cells']))
s62 = ''.join(nb['cells'][62]['source'])
assert 'BLEND_EFFNET = 0.15' in s62, 'TC-1: BLEND_EFFNET not 0.15'
assert 'effnet_logits = np.zeros' not in s62, 'TC-2: placeholder not removed'
assert 'BLEND_SED = 0.15' in s62, 'TC-3: BLEND_SED regression'
assert 'distill_logits = np.zeros' in s62, 'TC-4: distill placeholder regression'
assert '_perch_scores_raw = test_base_scores.copy()' in s62, 'TC-5: perch_scores_raw regression'
s63 = ''.join(nb['cells'][63]['source'])
assert '_BirdEffNet' in s63, 'TC-6: EffNet model class missing in cells[63]'
assert 'effnet_logits' in s63, 'TC-7: effnet_logits not computed in cells[63]'
assert 'birdclef2026-effnet-5fold-epoch50' in s63, 'TC-8: dataset path missing'
s68 = ''.join(nb['cells'][68]['source'])
assert 'trial_076' in s68, 'TC-9: blend header not updated to trial_076'
assert 'EffNet5fold 15%' in s68, 'TC-10: EffNet5fold 15% not in header'
print('PASS')
"
```

## 개발 Phase 계획

### Phase 1: trial076-effnet-sed
**목표**: cells[62] BLEND_EFFNET 0→0.15 + effnet_logits placeholder 제거, cells[63]에 EffNet5fold 추론 셀 삽입, cells[68] 헤더 갱신, meta.json 생성
- Step 1: 노트북 수정 (3개 변경 + cell 삽입) + meta.json 생성 + 10-TC 검증 PASS
