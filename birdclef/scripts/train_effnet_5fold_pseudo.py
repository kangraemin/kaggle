"""EfficientNetV2-B0 5-fold training with Perch distillation + pseudo-labeled soundscapes.

Additions over train_effnet_5fold_distill.py:
- Pseudo-labeled soundscape segments added to training data
- Confidence filtering: max_score > PSEUDO_CONF_THR
- BirdDataset extended with end_secs for soundscape segment slicing
"""
import os
import sys
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
OUT = Path(__file__).parent.parent / 'models' / 'effnet_5fold_pseudo'
OUT.mkdir(parents=True, exist_ok=True)

PSEUDO_DIR = DATA / 'ss10k_pseudo'
SS_DIR = DATA / 'train_soundscapes'
SS_EMB_FILE = DATA / 'soundscape_extracted' / 'all_ss_embeddings.npy'
SS_META_FILE = DATA / 'soundscape_extracted' / 'all_ss_meta.csv'
PSEUDO_CONF_THR = 0.7
PSEUDO_MAX_SAMPLES = 10000   # top-N by confidence; None = use all

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

N_FOLDS = 5
N_EPOCHS = 30
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
    def __init__(self, paths, labels, perch_embs, sr=32000, is_train=True, end_secs=None):
        self.paths = paths
        self.labels = labels
        self.perch_embs = perch_embs  # (N, 1536) float32
        self.sr = sr
        self.dur = 5 * sr
        self.is_train = is_train
        self.end_secs = end_secs  # list of int or None per sample; None = train_audio (random crop)

    def load_sound(self, filepath, end_sec=None):
        wav, sr = sf.read(filepath, dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)
        l = len(wav)

        if end_sec is not None:
            # soundscape segment: extract fixed 5s window ending at end_sec
            end_sample = int(end_sec * self.sr)
            start_sample = max(0, end_sample - self.dur)
            wav = wav[start_sample:end_sample]
            l = len(wav)

        if l < self.dur:
            wav2 = torch.zeros(self.dur)
            s = np.random.randint(max(1, self.dur - l))
            wav2[s:s + l] = wav
            wav = wav2
        else:
            if self.is_train and end_sec is None:
                # random crop for train_audio only
                s = random.randint(0, l - self.dur)
                wav = wav[s:s + self.dur]
            else:
                wav = wav[:self.dur]
        return wav

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        end_sec = self.end_secs[idx] if self.end_secs is not None else None
        audio = self.load_sound(self.paths[idx], end_sec=end_sec)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        perch_emb = torch.tensor(self.perch_embs[idx], dtype=torch.float32)
        return audio, label, perch_emb


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
        x = self.spec(x)
        if self.training:
            x = self.spec_aug(x)
            if y is not None and perch_emb is not None:
                x, y, perch_emb = self.mixup(x, y, perch_emb)
        feat = self.backbone(x)
        logits = self.head(feat)
        proj = self.proj(feat)
        return logits, y, proj, perch_emb


def train_fold(fold, paths, labels, perch_embs, df, end_secs=None):
    set_seed(42 + fold)

    # StratifiedKFold on first n_train samples (train_audio only) to keep fold logic stable
    n_train = len(df)
    train_paths = paths[:n_train]
    train_labels = labels[:n_train]
    train_perch = perch_embs[:n_train]
    train_end_secs = end_secs[:n_train] if end_secs else None

    pseudo_paths = paths[n_train:]
    pseudo_labels = labels[n_train:]
    pseudo_perch = perch_embs[n_train:]
    pseudo_end_secs = end_secs[n_train:] if end_secs else None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(train_paths, df['primary_label'].fillna('none')))[fold]

    # Combine: train split + all pseudo
    combined_paths = [train_paths[i] for i in train_idx] + list(pseudo_paths)
    combined_labels = np.concatenate([train_labels[train_idx], pseudo_labels], axis=0)
    combined_perch = np.concatenate([train_perch[train_idx], pseudo_perch], axis=0)
    combined_end_secs = (
        ([train_end_secs[i] for i in train_idx] if train_end_secs else [None] * len(train_idx))
        + (list(pseudo_end_secs) if pseudo_end_secs else [None] * len(pseudo_paths))
    )

    train_ds = BirdDataset(
        combined_paths, combined_labels, combined_perch,
        is_train=True, end_secs=combined_end_secs,
    )
    val_ds = BirdDataset(
        [train_paths[i] for i in val_idx],
        train_labels[val_idx],
        train_perch[val_idx],
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f'\n{"="*50}')
    print(f'Fold {fold+1}/{N_FOLDS}: Train={len(train_ds)} (orig={len(train_idx)}, pseudo={len(pseudo_paths)}), Val={len(val_ds)}')
    print(f'DISTILL_ALPHA={DISTILL_ALPHA}')
    print(f'{"="*50}')

    model = BirdModelDistill(pretrained=True).to(DEVICE)
    criterion_cls = nn.BCEWithLogitsLoss()
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

            loss_cls = criterion_cls(logits, target_mixed)
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
        for j in range(train_labels.shape[1]):
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


def parse_row_id(row_id):
    """BC2026_Train_0001_S08_20250606_030007_10 → ('BC2026_Train_0001_S08_20250606_030007.ogg', 10)"""
    parts = row_id.rsplit('_', 1)
    return parts[0] + '.ogg', int(parts[1])


def load_pseudo_data(n_classes):
    print("Loading pseudo-label soundscape data...")

    pseudo_meta = pd.read_csv(PSEUDO_DIR / 'meta.csv')
    pseudo_scores = np.load(PSEUDO_DIR / 'scores.npy').astype(np.float32)
    pseudo_labels_raw = np.load(PSEUDO_DIR / 'labels.npy').astype(np.float32)

    # confidence filter
    max_scores = pseudo_scores.max(axis=1)
    conf_mask = max_scores > PSEUDO_CONF_THR
    print(f"  Confidence filter (>{PSEUDO_CONF_THR}): {conf_mask.sum():,} / {len(conf_mask):,} kept")

    pseudo_meta_f = pseudo_meta[conf_mask].reset_index(drop=True)
    pseudo_labels_f = pseudo_labels_raw[conf_mask]

    # subsample top-N by confidence
    if PSEUDO_MAX_SAMPLES and len(pseudo_meta_f) > PSEUDO_MAX_SAMPLES:
        top_idx = max_scores[conf_mask].argsort()[::-1][:PSEUDO_MAX_SAMPLES]
        pseudo_meta_f = pseudo_meta_f.iloc[top_idx].reset_index(drop=True)
        pseudo_labels_f = pseudo_labels_f[top_idx]
        print(f"  Subsampled to top-{PSEUDO_MAX_SAMPLES} by confidence")

    # match Perch embeddings
    ss_meta = pd.read_csv(SS_META_FILE)
    ss_emb = np.load(SS_EMB_FILE)
    rid_to_idx = {rid: i for i, rid in enumerate(ss_meta['row_id'])}

    pseudo_embs = np.zeros((len(pseudo_meta_f), PERCH_DIM), dtype=np.float32)
    matched = 0
    for i, rid in enumerate(pseudo_meta_f['row_id']):
        if rid in rid_to_idx:
            pseudo_embs[i] = ss_emb[rid_to_idx[rid]]
            matched += 1
    print(f"  Perch emb matched: {matched:,}/{len(pseudo_meta_f):,} ({matched/len(pseudo_meta_f)*100:.1f}%)")

    # audio paths + end_secs
    pseudo_paths = []
    pseudo_end_secs = []
    for rid in pseudo_meta_f['row_id']:
        fname, esec = parse_row_id(rid)
        pseudo_paths.append(str(SS_DIR / fname))
        pseudo_end_secs.append(esec)

    return pseudo_paths, pseudo_labels_f, pseudo_embs, pseudo_end_secs


def main():
    dry_run = '--dry-run' in sys.argv
    set_seed(42)

    df = pd.read_csv(DATA / 'train.csv')
    tax = pd.read_csv(DATA / 'taxonomy.csv')
    LABELS = sorted(tax.primary_label.dropna().unique())
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    n_classes = len(LABELS)

    # Load train_audio Perch embeddings
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

    matched_orig = (np.abs(perch_embs).sum(axis=1) > 0).sum()
    print(f'Classes: {n_classes}, Train audio: {len(df)}, Perch emb matched: {matched_orig}')

    # Load pseudo-labeled soundscape data
    pseudo_paths, pseudo_labels, pseudo_embs, pseudo_end_secs = load_pseudo_data(n_classes)

    # Combine: train_audio first, pseudo after (train_fold uses len(df) as split point)
    all_paths    = paths + pseudo_paths
    all_labels   = np.concatenate([labels, pseudo_labels], axis=0)
    all_perch    = np.concatenate([perch_embs, pseudo_embs], axis=0)
    all_end_secs = [None] * len(paths) + pseudo_end_secs

    print(f'Combined: {len(paths):,} train_audio + {len(pseudo_paths):,} pseudo = {len(all_paths):,} total')
    print(f'Distillation alpha: {DISTILL_ALPHA}, Epochs: {N_EPOCHS}, LR: {LR}')

    if dry_run:
        print('\n[dry-run] Data loading OK. Exiting without training.')
        print(f'OUT = {OUT}')
        return

    fold_aucs = []
    t_start = time.time()
    for fold in range(N_FOLDS):
        best_path = OUT / f'best_fold{fold}.pth'
        ckpt_path = OUT / f'checkpoint_fold{fold}.pth'
        if best_path.exists() and not ckpt_path.exists():
            print(f'\nFold {fold+1}: already completed, skipping')
            fold_aucs.append(-1)
            continue
        auc = train_fold(fold, all_paths, all_labels, all_perch, df, end_secs=all_end_secs)
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
