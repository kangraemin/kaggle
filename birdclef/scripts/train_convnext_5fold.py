"""ConvNeXt-Base XCL 5-fold training — BirdCLEF 2026.

Changes over train_effnet_5fold_softauc.py:
- Backbone: ConvNeXt-Base (denden12/birdset-convnext-base-xcl, 9736-species pretrained)
- Spectrogram: n_mels=128, hop=320, n_fft=1024, XCL norm (mean=-4.268, std=4.569)
- Loss: 0.5*BCE + 0.5*SoftAUC (no distillation)
- Output dim: 1024 (ConvNeXt-Base pooler_output)
- spec_indices mmap pattern for memory efficiency
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path

DATA = Path(__file__).parent.parent / 'data'
CONVNEXT_PATH = Path(__file__).parent.parent / 'models' / 'convnext_xcl'
CONVNEXT_SPEC_CACHE = Path(__file__).parent.parent / 'models' / 'convnext_spec_cache.npy'
OUT = Path(__file__).parent.parent / 'models' / 'convnext_5fold'
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 3e-4
CONVNEXT_DIM = 1024


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_convnext_weights():
    if (CONVNEXT_PATH / 'model.safetensors').exists():
        print(f'ConvNeXt XCL weights found: {CONVNEXT_PATH}')
        return
    print('Downloading ConvNeXt XCL weights from Kaggle...')
    CONVNEXT_PATH.mkdir(parents=True, exist_ok=True)
    ret = os.system(
        f'kaggle datasets download denden12/birdset-convnext-base-xcl'
        f' -p {CONVNEXT_PATH} --unzip'
    )
    if ret != 0:
        raise RuntimeError('Failed to download ConvNeXt XCL weights')
    print(f'Downloaded to {CONVNEXT_PATH}')


class ConvNeXtSpec(nn.Module):
    """MelSpectrogram matching ConvNeXt XCL training preprocessing."""
    def __init__(self, sr=32000, n_fft=1024, n_mels=128, hop_length=320,
                 f_min=0, f_max=None, top_db=80.0):
        super().__init__()
        self.top_db = top_db
        self.mean = -4.268
        self.std = 4.569
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
            power=2.0, center=True,
        )

    def forward(self, wav):  # (T,) or (B, T)
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True
        mel = self.mel_transform(wav)                              # (B, 128, T)
        db = 10.0 * torch.log10(mel.clamp(min=1e-10))             # dB scale
        max_val = db.flatten(-2).max(dim=-1).values[..., None, None]
        db = torch.maximum(db, max_val - self.top_db)             # top_db clip
        db = (db - self.mean) / self.std                          # XCL normalize
        db = db.unsqueeze(1)                                       # (B, 1, 128, T)
        if squeeze:
            db = db.squeeze(0)                                     # (1, 128, T)
        return db


def precompute_convnext_specs(paths):
    if CONVNEXT_SPEC_CACHE.exists():
        print(f'ConvNeXt spec cache found: {CONVNEXT_SPEC_CACHE}')
        arr = np.load(CONVNEXT_SPEC_CACHE, mmap_mode='r')
        print(f'  shape={arr.shape}, dtype={arr.dtype}')
        return arr

    print(f'Pre-computing ConvNeXt spectrograms for {len(paths)} files...')
    spec_fn = ConvNeXtSpec()
    dur = 5 * 32000
    specs = []

    for i, p in enumerate(paths):
        wav, _ = sf.read(p, dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if len(wav) >= dur:
            wav = wav[:dur]
        else:
            wav = np.pad(wav, (0, dur - len(wav)))
        with torch.no_grad():
            s = spec_fn(torch.from_numpy(wav)).squeeze(0).numpy()  # (1, 128, T)
        specs.append(s.astype(np.float16))
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(paths)}', flush=True)

    specs = np.stack(specs)  # (N, 1, 128, T)
    np.save(CONVNEXT_SPEC_CACHE, specs)
    print(f'Saved: {CONVNEXT_SPEC_CACHE} ({specs.nbytes/1e9:.2f} GB)', flush=True)
    return np.load(CONVNEXT_SPEC_CACHE, mmap_mode='r')


class SoftAUCLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        diff = logits.unsqueeze(1) - logits.unsqueeze(0)   # (B, B, C)
        pos_w = targets.unsqueeze(1)                        # (B, 1, C)
        neg_w = (1 - targets).unsqueeze(0)                 # (1, B, C)
        pair_w = pos_w * neg_w                             # (B, B, C)
        loss = -(torch.log(torch.sigmoid(self.gamma * diff) + 1e-8) * pair_w).sum()
        return loss / (pair_w.sum() + 1e-8)


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=20, time_mask_param=40,
                 n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq = n_freq_masks
        self.n_time = n_time_masks

    def forward(self, x):
        for _ in range(self.n_freq):
            x = self.freq_masking(x)
        for _ in range(self.n_time):
            x = self.time_masking(x)
        return x


class Mixup(nn.Module):
    def __init__(self, alpha=1.0, theta=0.8):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y):
        if not self.training or random.random() > self.theta:
            return x, y
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


class BirdDataset(Dataset):
    """Dataset returning (spec, label). Uses spec_indices mmap pattern."""
    def __init__(self, paths, labels, sr=32000, is_train=True,
                 specs=None, spec_indices=None):
        self.paths = paths
        self.labels = labels
        self.specs = specs            # full mmap array (N_total, 1, 128, T)
        self.spec_indices = spec_indices  # int array mapping dataset idx → mmap idx
        self.sr = sr
        self.dur = 5 * sr
        self.is_train = is_train

    def load_sound(self, filepath):
        wav, _ = sf.read(filepath, dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)
        l = len(wav)
        if l < self.dur:
            wav2 = torch.zeros(self.dur)
            s = np.random.randint(max(1, self.dur - l))
            wav2[s:s + l] = wav
            wav = wav2
        else:
            if self.is_train:
                s = random.randint(0, l - self.dur)
                wav = wav[s:s + self.dur]
            else:
                wav = wav[:self.dur]
        return wav

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.specs is not None:
            real_idx = int(self.spec_indices[idx])
            # integer indexing → mmap view → copy only 1 sample (~128KB)
            x = torch.from_numpy(np.array(self.specs[real_idx], dtype=np.float32))
        else:
            x = self.load_sound(self.paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, label


class BirdConvNeXt(nn.Module):
    def __init__(self, convnext_path, num_labels=234):
        super().__init__()
        from transformers import ConvNextForImageClassification
        _clf = ConvNextForImageClassification.from_pretrained(
            str(convnext_path), ignore_mismatched_sizes=True
        )
        self.backbone = _clf.convnext      # ConvNextModel, output: pooler_output (B, 1024)
        self.head = nn.Linear(CONVNEXT_DIM, num_labels)
        self.spec = ConvNeXtSpec()
        self.spec_aug = SpecAugment()
        self.mixup = Mixup(alpha=1.0, theta=0.8)

    def forward(self, x, y=None):
        if x.dim() == 2:                   # (B, T) raw audio
            x = self.spec(x)               # (B, 1, 128, T)
        if self.training:
            x = self.spec_aug(x)
            if y is not None:
                x, y = self.mixup(x, y)
        outputs = self.backbone(pixel_values=x)
        feat = outputs.pooler_output       # (B, 1024)
        logits = self.head(feat)
        return logits, y


def train_fold(fold, paths, labels, df, all_specs_mmap=None):
    set_seed(42 + fold)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(paths, df['primary_label'].fillna('none')))[fold]

    train_ds = BirdDataset(
        [paths[i] for i in train_idx],
        labels[train_idx],
        is_train=True,
        specs=all_specs_mmap,
        spec_indices=train_idx,
    )
    val_ds = BirdDataset(
        [paths[i] for i in val_idx],
        labels[val_idx],
        is_train=False,
        specs=all_specs_mmap,
        spec_indices=val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'\n{"="*50}')
    print(f'Fold {fold+1}/{N_FOLDS}: Train={len(train_ds)}, Val={len(val_ds)}')
    print(f'Loss: 0.5*BCE + 0.5*SoftAUC')
    print(f'{"="*50}')

    model = BirdConvNeXt(CONVNEXT_PATH).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_softauc = SoftAUCLoss(gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

    ckpt_path = OUT / f'checkpoint_fold{fold}.pth'
    start_epoch = 0
    best_auc = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt['best_auc']
        print(f'  Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}')

    for epoch in range(start_epoch, N_EPOCHS):
        t0 = time.time()

        model.train()
        train_loss = train_bce = train_sauc = 0
        for spec, target in train_loader:
            spec = spec.to(DEVICE)
            target = target.to(DEVICE)

            logits, target_mixed = model(spec, target)

            loss_bce = criterion_bce(logits, target_mixed)
            loss_softauc = criterion_softauc(logits, target_mixed)
            loss = 0.5 * loss_bce + 0.5 * loss_softauc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bce += loss_bce.item()
            train_sauc += loss_softauc.item()

        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for spec, target in val_loader:
                spec = spec.to(DEVICE)
                logits, _ = model(spec)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(target.numpy())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        aucs = []
        for j in range(labels.shape[1]):
            if targets[:, j].sum() > 0:
                aucs.append(roc_auc_score(targets[:, j], preds[:, j]))
        auc = np.mean(aucs)

        elapsed = time.time() - t0
        n = len(train_loader)
        print(f'  Epoch {epoch+1}/{N_EPOCHS}: '
              f'loss={train_loss/n:.4f} '
              f'(bce={train_bce/n:.4f}, sauc={train_sauc/n:.4f}), '
              f'auc={auc:.4f} ({elapsed:.0f}s)', flush=True)

        if auc > best_auc:
            best_auc = auc
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith('spec.') and not k.startswith('spec_aug.')}
            torch.save(state, OUT / f'best_fold{fold}.pth')
            print(f'    -> New best: {best_auc:.4f}', flush=True)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_auc': best_auc,
        }, ckpt_path)

    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f'Fold {fold+1} best AUC: {best_auc:.4f}')
    return best_auc


def main():
    set_seed(42)

    download_convnext_weights()

    df = pd.read_csv(DATA / 'train.csv')
    tax = pd.read_csv(DATA / 'taxonomy.csv')
    LABELS = sorted(tax.primary_label.dropna().unique())
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    n_classes = len(LABELS)

    paths = [str(DATA / 'train_audio' / f) for f in df['filename'].values]
    labels = np.zeros((len(df), n_classes), dtype=np.float32)
    for i, row in df.iterrows():
        if row['primary_label'] in label_to_idx:
            labels[i, label_to_idx[row['primary_label']]] = 1.0

    print(f'Classes: {n_classes}, Samples: {len(df)}')
    print(f'Loss: 0.5*BCE + 0.5*SoftAUC, Epochs: {N_EPOCHS}, LR: {LR}')

    all_specs_mmap = precompute_convnext_specs(paths)

    fold_aucs = []
    t_start = time.time()
    for fold in range(N_FOLDS):
        best_path = OUT / f'best_fold{fold}.pth'
        ckpt_path = OUT / f'checkpoint_fold{fold}.pth'
        if best_path.exists() and not ckpt_path.exists():
            print(f'\nFold {fold+1}: already completed, skipping')
            fold_aucs.append(-1)
            continue
        auc = train_fold(fold, paths, labels, df, all_specs_mmap=all_specs_mmap)
        fold_aucs.append(auc)

    print(f'\n{"="*50}')
    print(f'All folds complete!')
    for i, auc in enumerate(fold_aucs):
        if auc > 0:
            print(f'  Fold {i+1}: {auc:.4f}')
    completed = [a for a in fold_aucs if a > 0]
    if completed:
        print(f'  Mean: {np.mean(completed):.4f}')
    print(f'  Total time: {(time.time()-t_start)/60:.1f} min')
    print(f'  Models saved: {OUT}')


if __name__ == '__main__':
    main()
