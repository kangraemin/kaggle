"""EfficientNetV2-B0 5-fold training — 50 epochs (up from 30) + BLEND 0.25.

Changes over train_effnet_5fold_softauc.py:
- N_EPOCHS 30 → 50 (spec precompute로 epoch당 ~170s, total ~2.4h 추가)
- DataLoader num_workers=4, persistent_workers=True (MPS 환경 최적화)
- OUT → models/effnet_5fold_epoch50/
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import timm
import time
from pathlib import Path

DATA = Path(__file__).parent.parent / 'data'
EMB_DIR = DATA / 'perch_embeddings' / 'perch_embeddings'
OUT = Path(__file__).parent.parent / 'models' / 'effnet_5fold_epoch50'
OUT.mkdir(parents=True, exist_ok=True)
SPEC_CACHE = Path(__file__).parent.parent / 'models' / 'spec_cache.npy'

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-4
DISTILL_ALPHA = 0.3
PERCH_DIM = 1536
EFFNET_DIM = 1280


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def precompute_specs(paths):
    """한 번만 실행. 전체 오디오 → 스펙트로그램 → float16 npy 저장."""
    if SPEC_CACHE.exists():
        print(f'Spec cache found: {SPEC_CACHE}')
        return np.load(SPEC_CACHE)
    print(f'Pre-computing spectrograms for {len(paths)} files...')
    spec_fn = Spectrogram()
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
            s = spec_fn(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        specs.append(s.astype(np.float16))
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(paths)}', flush=True)
    specs = np.stack(specs)  # (N, 1, 256, 256)
    np.save(SPEC_CACHE, specs)
    print(f'Saved: {SPEC_CACHE} ({specs.nbytes/1e6:.0f} MB)', flush=True)
    return specs


class SoftAUCLoss(nn.Module):
    """Pairwise ranking loss that directly approximates per-class AUC."""
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B, C) in [0, 1]
        diff = logits.unsqueeze(1) - logits.unsqueeze(0)  # (B, B, C)
        pos_w = targets.unsqueeze(1)          # (B, 1, C)
        neg_w = (1 - targets).unsqueeze(0)   # (1, B, C)
        pair_w = pos_w * neg_w               # (B, B, C)
        loss = -(torch.log(torch.sigmoid(self.gamma * diff) + 1e-8) * pair_w).sum()
        return loss / (pair_w.sum() + 1e-8)


class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512,
                 f_min=20, f_max=16000, target_size=(256, 256), top_db=80.0):
        super().__init__()
        self.top_db = top_db
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
            mel_scale="htk", pad_mode="reflect", power=2.0, norm="slaney", center=True,
        )
        self.resize = torchvision.transforms.Resize(size=target_size)

    def power_to_db(self, S):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(S))
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        mel_spec = self.mel_transform(x)
        mel_spec = self.power_to_db(mel_spec)
        mel_spec = mel_spec.unsqueeze(1).repeat(1, 1, 1, 1)
        mel_spec = self.resize(mel_spec)
        B, C = mel_spec.shape[:2]
        flat = mel_spec.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)
        if squeeze:
            mel_spec = mel_spec.squeeze(0)
        return mel_spec


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=30, time_mask_param=40, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        for _ in range(self.n_freq_masks):
            x = self.freq_masking(x)
        for _ in range(self.n_time_masks):
            x = self.time_masking(x)
        return x


class BirdDataset(Dataset):
    def __init__(self, paths, labels, perch_embs, sr=32000, is_train=True, specs=None):
        self.paths = paths
        self.labels = labels
        self.perch_embs = perch_embs
        self.specs = specs  # (N, 1, 256, 256) float16 or None
        self.sr = sr
        self.dur = 5 * sr
        self.is_train = is_train

    def load_sound(self, filepath):
        wav, sr = sf.read(filepath, dtype='float32')
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
            x = torch.from_numpy(self.specs[idx].astype(np.float32))  # (1, 256, 256)
        else:
            x = self.load_sound(self.paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        perch_emb = torch.tensor(self.perch_embs[idx], dtype=torch.float32)
        return x, label, perch_emb


class Mixup(nn.Module):
    def __init__(self, alpha=1.0, theta=0.8):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y, e):
        if not self.training or random.random() > self.theta:
            return x, y, e
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        e = lam * e + (1 - lam) * e[idx]
        return x, y, e


class BirdModelDistill(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_b0", num_labels=234,
                 pretrained=True, perch_dim=1536, effnet_dim=1280):
        super().__init__()
        self.spec = Spectrogram()
        self.spec_aug = SpecAugment(freq_mask_param=30, time_mask_param=40,
                                    n_freq_masks=2, n_time_masks=2)
        self.mixup = Mixup(alpha=1.0, theta=0.8)
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           in_chans=1, num_classes=0)
        self.head = nn.Linear(effnet_dim, num_labels)
        self.proj = nn.Sequential(
            nn.Linear(effnet_dim, effnet_dim),
            nn.ReLU(),
            nn.Linear(effnet_dim, perch_dim),
        )

    def forward(self, x, y=None, perch_emb=None):
        if x.dim() == 2:  # (B, T) raw audio
            x = self.spec(x)
        # else (B, 1, 256, 256) pre-computed spec — skip
        if self.training:
            x = self.spec_aug(x)
            if y is not None and perch_emb is not None:
                x, y, perch_emb = self.mixup(x, y, perch_emb)
        feat = self.backbone(x)
        logits = self.head(feat)
        proj = self.proj(feat)
        return logits, y, proj, perch_emb


def train_fold(fold, paths, labels, perch_embs, df, all_specs=None):
    set_seed(42 + fold)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(paths, df['primary_label'].fillna('none')))[fold]

    train_ds = BirdDataset(
        [paths[i] for i in train_idx],
        labels[train_idx],
        perch_embs[train_idx],
        is_train=True,
        specs=all_specs[train_idx] if all_specs is not None else None,
    )
    val_ds = BirdDataset(
        [paths[i] for i in val_idx],
        labels[val_idx],
        perch_embs[val_idx],
        is_train=False,
        specs=all_specs[val_idx] if all_specs is not None else None,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=False, persistent_workers=True,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, persistent_workers=True)

    print(f'\n{"="*50}')
    print(f'Fold {fold+1}/{N_FOLDS}: Train={len(train_ds)}, Val={len(val_ds)}')
    print(f'Loss: 0.5*BCE + 0.5*SoftAUC + {DISTILL_ALPHA}*Distill')
    print(f'{"="*50}')

    model = BirdModelDistill(pretrained=True).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_softauc = SoftAUCLoss(gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

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
        train_loss = train_cls_loss = train_distill_loss = 0
        for audio, target, perch_emb in train_loader:
            audio = audio.to(DEVICE)
            target = target.to(DEVICE)
            perch_emb = perch_emb.to(DEVICE)

            logits, target_mixed, proj, perch_emb_mixed = model(audio, target, perch_emb)

            loss_bce = criterion_bce(logits, target_mixed)
            loss_softauc = criterion_softauc(logits, target_mixed)
            loss_cls = 0.5 * loss_bce + 0.5 * loss_softauc

            proj_n = F.normalize(proj, dim=-1)
            emb_n = F.normalize(perch_emb_mixed, dim=-1)
            loss_distill = F.mse_loss(proj_n, emb_n)

            loss = loss_cls + DISTILL_ALPHA * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_cls_loss += loss_cls.item()
            train_distill_loss += loss_distill.item()

        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for audio, target, _ in val_loader:
                audio = audio.to(DEVICE)
                logits, _, _, _ = model(audio)
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
              f'(cls={train_cls_loss/n:.4f}, distill={train_distill_loss/n:.4f}), '
              f'auc={auc:.4f} ({elapsed:.0f}s)', flush=True)

        if auc > best_auc:
            best_auc = auc
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith('proj.')}
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

    df = pd.read_csv(DATA / 'train.csv')
    tax = pd.read_csv(DATA / 'taxonomy.csv')
    LABELS = sorted(tax.primary_label.dropna().unique())
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    n_classes = len(LABELS)

    emb_meta = pd.read_csv(EMB_DIR / 'audio_metadata.csv')
    emb_array = np.load(EMB_DIR / 'audio_embeddings.npy').astype(np.float32)
    fn_to_emb = {row['filename']: emb_array[i] for i, row in emb_meta.iterrows()}

    paths = [str(DATA / 'train_audio' / f) for f in df['filename'].values]
    labels = np.zeros((len(df), n_classes), dtype=np.float32)
    perch_embs = np.zeros((len(df), PERCH_DIM), dtype=np.float32)

    for i, row in df.iterrows():
        if row['primary_label'] in label_to_idx:
            labels[i, label_to_idx[row['primary_label']]] = 1.0
        if row['filename'] in fn_to_emb:
            perch_embs[i] = fn_to_emb[row['filename']]

    matched = np.abs(perch_embs).sum(axis=1) > 0
    print(f'Classes: {n_classes}, Samples: {len(df)}, Perch emb matched: {matched.sum()}')
    print(f'Loss: 0.5*BCE + 0.5*SoftAUC + {DISTILL_ALPHA}*Distill')
    print(f'Epochs: {N_EPOCHS}, LR: {LR}')

    all_specs = precompute_specs(paths)

    fold_aucs = []
    t_start = time.time()
    for fold in range(N_FOLDS):
        best_path = OUT / f'best_fold{fold}.pth'
        ckpt_path = OUT / f'checkpoint_fold{fold}.pth'
        if best_path.exists() and not ckpt_path.exists():
            print(f'\nFold {fold+1}: already completed, skipping')
            fold_aucs.append(-1)
            continue
        auc = train_fold(fold, paths, labels, perch_embs, df, all_specs=all_specs)
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
