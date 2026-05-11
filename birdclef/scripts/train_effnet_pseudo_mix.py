"""EfficientNetV2-B0 fold0 training: focal cache + pseudo-labeled ss10k.

Differences from train_effnet_multiwindow.py:
- ADDS cache_ss10k.npy (30k ss windows) + ss10k_subset.npz soft labels to training
- PseudoMixDataset handles two separate caches (cache_v2 focal + cache_ss ss)
- Val set: focal_val only (same as baseline)
- TRAIN_FOLDS = [0], output: models/effnet_pseudo_mix/
"""
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]  # birdclef/
DATA_V2 = ROOT / 'data' / 'v2'
CACHE_PATH = DATA_V2 / 'cache_v2.npy'
CACHE_META = DATA_V2 / 'cache_v2_meta.npz'
LABELS_PATH = DATA_V2 / 'labels_v2.npz'
SS_CACHE_PATH = DATA_V2 / 'cache_ss10k.npy'
SS_SUBSET_PATH = DATA_V2 / 'ss10k_subset.npz'
WORK_DIR = ROOT / 'models' / 'effnet_pseudo_mix'

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 5e-4
EFFNET_DIM = 1280


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        return x, y


class PseudoMixDataset(Dataset):
    """Combined dataset: focal samples (multi-window) + ss10k pseudo (single-window each)."""

    def __init__(self, focal_sample_indices, focal_labels, focal_window_lists, cache_v2,
                 ss_indices, ss_labels, cache_ss, is_train=True):
        self.focal_sample_indices = focal_sample_indices
        self.focal_labels = focal_labels
        self.focal_window_lists = focal_window_lists
        self.cache_v2 = cache_v2
        self.ss_indices = ss_indices      # indices into cache_ss (and ss_labels)
        self.ss_labels = ss_labels
        self.cache_ss = cache_ss
        self.is_train = is_train
        self.n_focal = len(focal_sample_indices)

    def __len__(self):
        return self.n_focal + len(self.ss_indices)

    def __getitem__(self, idx):
        if idx < self.n_focal:
            windows = self.focal_window_lists[idx]
            if self.is_train and len(windows) > 1:
                w = random.choice(windows)
            else:
                w = windows[0]
            spec = torch.from_numpy(self.cache_v2[w].astype(np.float32))
            label = torch.from_numpy(np.asarray(self.focal_labels[idx], dtype=np.float32))
        else:
            ss_i = idx - self.n_focal
            w = self.ss_indices[ss_i]
            spec = torch.from_numpy(self.cache_ss[w].astype(np.float32))
            label = torch.from_numpy(np.asarray(self.ss_labels[ss_i], dtype=np.float32))
        return spec, label


class FocalCacheDataset(Dataset):
    """Validation: focal-only samples."""

    def __init__(self, sample_indices, labels, window_lists, cache):
        self.sample_indices = sample_indices
        self.labels = labels
        self.window_lists = window_lists
        self.cache = cache

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        windows = self.window_lists[idx]
        w = windows[0]
        spec = torch.from_numpy(self.cache[w].astype(np.float32))
        label = torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32))
        return spec, label


class EffNetMixup(nn.Module):
    """1ch spec → stem_conv(1→3) → EffNetV2-B0 backbone → linear head."""

    def __init__(self, n_classes=234, backbone='tf_efficientnetv2_b0',
                 pretrained=True, effnet_dim=EFFNET_DIM):
        super().__init__()
        self.spec_aug = SpecAugment(freq_mask_param=30, time_mask_param=40,
                                    n_freq_masks=2, n_time_masks=2)
        self.mixup = Mixup(alpha=1.0, theta=0.8)
        self.stem_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          in_chans=3, num_classes=0)
        self.head = nn.Linear(effnet_dim, n_classes)

    def forward(self, x, targets=None):
        if self.training:
            x = self.spec_aug(x)
            if targets is not None:
                x, targets = self.mixup(x, targets)
        x = self.stem_conv(x)
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits, targets


def train_fold(fold, labels_data, cache_v2, meta, ss_labels, cache_ss):
    set_seed(42 + fold)

    n = len(labels_data['paths'])
    window_lists = [[] for _ in range(n)]
    sample_idx = meta['sample_idx']
    for w in range(len(sample_idx)):
        window_lists[int(sample_idx[w])].append(w)

    labels = labels_data['labels']
    source = labels_data['source']
    primary_strat = labels_data['primary_strat']

    # Use only focal for stratified k-fold split
    focal_idx = np.where(source == 'focal')[0]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    focal_train_i, focal_val_i = list(
        skf.split(focal_idx, primary_strat[focal_idx])
    )[fold]

    train_focal = focal_idx[focal_train_i]
    val_focal = focal_idx[focal_val_i]
    ss10k_all = np.arange(len(ss_labels))  # all 30k pseudo-labeled ss windows

    train_ds = PseudoMixDataset(
        train_focal, labels[train_focal],
        [window_lists[i] for i in train_focal], cache_v2,
        ss10k_all, ss_labels, cache_ss, is_train=True,
    )
    val_ds = FocalCacheDataset(
        val_focal, labels[val_focal],
        [window_lists[i] for i in val_focal], cache_v2,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'\n{"=" * 50}')
    print(f'Fold {fold + 1}/{N_FOLDS}: '
          f'Train={len(train_ds)} (focal={len(train_focal)}, ss10k={len(ss10k_all)}), '
          f'Val={len(val_ds)}')
    print(f'{"=" * 50}')

    n_classes = labels.shape[1]
    model = EffNetMixup(n_classes=n_classes).to(DEVICE)
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WORK_DIR / f'checkpoint_fold{fold}.pth'
    start_epoch = 0
    best_auc = 0.0
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
        train_loss = 0.0
        for audio, target in train_loader:
            audio = audio.to(DEVICE)
            target = target.to(DEVICE)

            logits, target_mixed = model(audio, target)
            loss = criterion_cls(logits, target_mixed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for audio, target in val_loader:
                audio = audio.to(DEVICE)
                logits, _ = model(audio)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(target.numpy())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        aucs = []
        for j in range(n_classes):
            t = (targets[:, j] > 0.5).astype(int)
            if t.sum() > 0:
                aucs.append(roc_auc_score(t, preds[:, j]))
        auc = float(np.mean(aucs)) if aucs else 0.0

        elapsed = time.time() - t0
        n_batches = max(1, len(train_loader))
        print(f'  Epoch {epoch + 1}/{N_EPOCHS}: '
              f'loss={train_loss / n_batches:.4f}, '
              f'auc={auc:.4f} ({elapsed:.0f}s)', flush=True)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), WORK_DIR / f'best_fold{fold}.pth')
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
    print(f'Fold {fold + 1} best AUC: {best_auc:.4f}')
    return best_auc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    dry_run = args.dry_run
    train_folds = args.folds
    set_seed(42)

    print(f'Device: {DEVICE}')
    print(f'Loading focal cache: {CACHE_PATH}')
    cache_v2 = np.load(CACHE_PATH, mmap_mode='r')
    print(f'  cache_v2 shape: {cache_v2.shape}, dtype: {cache_v2.dtype}')

    print(f'Loading meta: {CACHE_META}')
    meta = np.load(CACHE_META, allow_pickle=True)

    print(f'Loading focal labels: {LABELS_PATH}')
    labels_npz = np.load(LABELS_PATH, allow_pickle=True)
    labels_data = {
        'paths': labels_npz['paths'],
        'labels': labels_npz['labels'].astype(np.float32),
        'source': np.array([str(s) for s in labels_npz['source']]),
        'primary_strat': np.array([str(s) for s in labels_npz['primary_strat']]),
    }
    n_focal = int((labels_data['source'] == 'focal').sum())
    print(f'  paths={len(labels_data["paths"])}, focal={n_focal}')

    print(f'Loading ss10k cache: {SS_CACHE_PATH}')
    cache_ss = np.load(SS_CACHE_PATH, mmap_mode='r')
    print(f'  cache_ss shape: {cache_ss.shape}, dtype: {cache_ss.dtype}')

    print(f'Loading ss10k subset: {SS_SUBSET_PATH}')
    ss_npz = np.load(SS_SUBSET_PATH, allow_pickle=True)
    ss_labels = ss_npz['labels'].astype(np.float32)  # (30000, 234) soft labels [0, 0.85]
    print(f'  ss10k samples={len(ss_labels)}, label range=[{ss_labels.min():.3f}, {ss_labels.max():.3f}]')

    print(f'Pseudo-SS added: {len(ss_labels)} windows (soft labels from Perch)')

    if dry_run:
        print(f'\n[dry-run] Data loading OK. Device={DEVICE} WORK_DIR={WORK_DIR}')
        print(f'train_folds={train_folds}')
        return

    fold_aucs = []
    t_start = time.time()
    for fold in train_folds:
        best_path = WORK_DIR / f'best_fold{fold}.pth'
        ckpt_path = WORK_DIR / f'checkpoint_fold{fold}.pth'
        if best_path.exists() and not ckpt_path.exists():
            print(f'\nFold {fold + 1}: already completed, skipping')
            fold_aucs.append(-1)
            continue
        auc = train_fold(fold, labels_data, cache_v2, meta, ss_labels, cache_ss)
        fold_aucs.append(auc)

    print(f'\n{"=" * 50}')
    print('Training complete!')
    for i, auc in enumerate(fold_aucs):
        if auc > 0:
            print(f'  Fold {train_folds[i] + 1}: {auc:.4f}')
    completed = [a for a in fold_aucs if a > 0]
    if completed:
        print(f'  Mean: {np.mean(completed):.4f}')
    print(f'  Total time: {(time.time() - t_start) / 60:.1f} min')
    print(f'  Models saved: {WORK_DIR}')


if __name__ == '__main__':
    main()
