"""EfficientNetV2-S fold0 training on multi-window spec cache (local MPS).

Differences from train_effnet_multiwindow.py:
- backbone = tf_efficientnetv2_s (EFFNET_DIM=1280 same as B0)
- Xeno pretrain loading if available (torch.load .ckpt)
- Augmentation moved to cpu_augment() outside model (MPS compatibility)
- N_EPOCHS = 20, TRAIN_FOLDS = [0]
- Output: models/effnetv2s_mw/
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]  # birdclef/
DATA_V2 = ROOT / 'data' / 'v2'
CACHE_PATH = DATA_V2 / 'cache_v2.npy'
CACHE_META = DATA_V2 / 'cache_v2_meta.npz'
LABELS_PATH = DATA_V2 / 'labels_v2.npz'
XENO_CKPT_PATH = (ROOT / 'data' / 'bird-clef-2025-all-pretrained-models'
                  / 'models_2025'
                  / 'tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt')
WORK_DIR = ROOT / 'models' / 'effnetv2s_mw'

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

N_FOLDS = 5
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 5e-4
EFFNET_DIM = 1280
TRAIN_FOLDS = [0]


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_xeno_pretrain(backbone, ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    new_sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
    model_sd = backbone.state_dict()
    filtered = {k: v for k, v in new_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    missing, _ = backbone.load_state_dict(filtered, strict=False)
    print(f'[XenoPretrain] loaded={len(filtered)}/{len(new_sd)} missing={len(missing)}')


def cpu_augment(x, y, freq_mask=30, time_mask=40, mixup_theta=0.8, mixup_alpha=1.0):
    """SpecAugment + Mixup on CPU tensors (avoids MPS compatibility issues)."""
    F, T = x.shape[-2], x.shape[-1]
    for _ in range(2):
        f0 = random.randint(0, max(0, F - freq_mask))
        x[..., f0:f0 + freq_mask, :] = 0
    for _ in range(2):
        t0 = random.randint(0, max(0, T - time_mask))
        x[..., :, t0:t0 + time_mask] = 0
    if random.random() < mixup_theta:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
    return x, y


class BirdCacheDataset(Dataset):
    """매 epoch random window from precomputed cache."""

    def __init__(self, sample_indices, labels, window_lists, cache, is_train=True):
        self.sample_indices = sample_indices
        self.labels = labels
        self.window_lists = window_lists
        self.cache = cache
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        windows = self.window_lists[idx]
        if self.is_train and len(windows) > 1:
            w = random.choice(windows)
        else:
            w = windows[0]
        spec = torch.from_numpy(self.cache[w].astype(np.float32))  # (1, 128, T)
        label = torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32))
        return spec, label


class EffNetS(nn.Module):
    """EffNetV2-S: 1ch spec → stem_conv(1→3) → tf_efficientnetv2_s → linear head."""

    def __init__(self, n_classes=234):
        super().__init__()
        self.stem_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False,
                                          in_chans=3, num_classes=0)
        if XENO_CKPT_PATH.exists():
            load_xeno_pretrain(self.backbone, XENO_CKPT_PATH)
        else:
            print(f'[WARN] Xeno ckpt not found at {XENO_CKPT_PATH.name}, using random init')
        self.head = nn.Linear(EFFNET_DIM, n_classes)

    def forward(self, x):
        return self.head(self.backbone(self.stem_conv(x)))


def train_fold(fold, labels_data, cache, meta):
    set_seed(42 + fold)

    n = len(labels_data['paths'])
    window_lists = [[] for _ in range(n)]
    sample_idx = meta['sample_idx']
    for w in range(len(sample_idx)):
        window_lists[int(sample_idx[w])].append(w)

    labels = labels_data['labels']
    source = labels_data['source']
    primary_strat = labels_data['primary_strat']

    focal_idx = np.where(source == 'focal')[0]
    ss_idx = np.where(source != 'focal')[0]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    focal_train_i, focal_val_i = list(
        skf.split(focal_idx, primary_strat[focal_idx])
    )[fold]

    train_indices = np.concatenate([focal_idx[focal_train_i], ss_idx])
    val_indices = focal_idx[focal_val_i]

    train_ds = BirdCacheDataset(
        train_indices, labels[train_indices],
        [window_lists[i] for i in train_indices], cache, is_train=True,
    )
    val_ds = BirdCacheDataset(
        val_indices, labels[val_indices],
        [window_lists[i] for i in val_indices], cache, is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f'\n{"=" * 50}')
    print(f'Fold {fold + 1}/{N_FOLDS}: '
          f'Train={len(train_ds)} (focal={len(focal_train_i)}, ss={len(ss_idx)}), '
          f'Val={len(val_ds)}')
    print(f'{"=" * 50}')

    n_classes = labels.shape[1]
    model = EffNetS(n_classes=n_classes).to(DEVICE)
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
            audio, target = cpu_augment(audio, target)
            audio = audio.to(DEVICE)
            target = target.to(DEVICE)

            logits = model(audio)
            loss = criterion_cls(logits, target)

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
                logits = model(audio)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(target.numpy())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        aucs = []
        for j in range(n_classes):
            if targets[:, j].sum() > 0:
                aucs.append(roc_auc_score(targets[:, j], preds[:, j]))
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
    dry_run = '--dry-run' in sys.argv
    set_seed(42)

    print(f'Device: {DEVICE}')
    print(f'Loading cache memmap: {CACHE_PATH}')
    cache = np.load(CACHE_PATH, mmap_mode='r')
    print(f'  cache shape: {cache.shape}, dtype: {cache.dtype}')

    print(f'Loading meta: {CACHE_META}')
    meta = np.load(CACHE_META, allow_pickle=True)

    print(f'Loading labels: {LABELS_PATH}')
    labels_npz = np.load(LABELS_PATH, allow_pickle=True)
    labels_data = {
        'paths': labels_npz['paths'],
        'labels': labels_npz['labels'].astype(np.float32),
        'source': np.array([str(s) for s in labels_npz['source']]),
        'primary_strat': np.array([str(s) for s in labels_npz['primary_strat']]),
    }
    print(f'  paths={len(labels_data["paths"])}, labels={labels_data["labels"].shape}')

    if dry_run:
        print(f'\nBuilding EffNetS model (Xeno pretrain check)...')
        n_classes = labels_data['labels'].shape[1]
        _ = EffNetS(n_classes=n_classes)
        print(f'\n[dry-run] Data loading OK. Device={DEVICE} WORK_DIR={WORK_DIR}')
        print(f'train_folds={TRAIN_FOLDS}')
        return

    fold_aucs = []
    t_start = time.time()
    for fold in TRAIN_FOLDS:
        best_path = WORK_DIR / f'best_fold{fold}.pth'
        ckpt_path = WORK_DIR / f'checkpoint_fold{fold}.pth'
        if best_path.exists() and not ckpt_path.exists():
            print(f'\nFold {fold + 1}: already completed, skipping')
            fold_aucs.append(-1)
            continue
        auc = train_fold(fold, labels_data, cache, meta)
        fold_aucs.append(auc)

    print(f'\n{"=" * 50}')
    print('Training complete!')
    for i, auc in enumerate(fold_aucs):
        if auc > 0:
            print(f'  Fold {TRAIN_FOLDS[i] + 1}: {auc:.4f}')
    completed = [a for a in fold_aucs if a > 0]
    if completed:
        print(f'  Mean: {np.mean(completed):.4f}')
    print(f'  Total time: {(time.time() - t_start) / 60:.1f} min')
    print(f'  Models saved: {WORK_DIR}')


if __name__ == '__main__':
    main()
