"""ConvNeXt 5-fold v2 — sub_25 도메인 정제 통합 학습.

기존 train_convnext_5fold.py 대비 변경:

1. **라벨 = data/v2/labels_v2.npz** (focal 35549 primary + 0.3 secondary, soundscape 1478 segments multi-hot)
2. **캐시 = data/v2/cache_v2.npy** (multi-window per sample, RMS top-K)
3. **Dataset.__getitem__**: 샘플의 윈도우 풀에서 매 epoch 랜덤 1개 추출 → 시간축 다양성 확보
4. **출력 = models/convnext_5fold_v2/** (v1과 분리)
5. **Mixup은 자동으로 focal ↔ soundscape 섞음** (배치 안에서 랜덤 페어링 → 도메인 mix augmentation 효과)

전제: build_labels_v2.py + build_multi_window_cache.py 선행 실행.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA_V2 = DATA / "v2"
OUT = ROOT / "models" / "convnext_5fold_v2"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
SR = 32000
DUR = 5 * SR
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 5e-5
CONVNEXT_DIM = 1024
CONVNEXT_PATH = ROOT / "models" / "convnext_base_xcl"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=20, time_mask_param=40, n_freq_masks=2, n_time_masks=2):
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


class SoftAUCLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        diff = logits.unsqueeze(1) - logits.unsqueeze(0)
        pos_w = targets.unsqueeze(1)
        neg_w = (1 - targets).unsqueeze(0)
        pair_w = pos_w * neg_w
        loss = -(torch.log(torch.sigmoid(self.gamma * diff) + 1e-8) * pair_w).sum()
        return loss / (pair_w.sum() + 1e-8)


class BirdConvNeXt(nn.Module):
    def __init__(self, convnext_path, num_labels: int):
        super().__init__()
        from transformers import ConvNextForImageClassification

        self.backbone = ConvNextForImageClassification.from_pretrained(
            str(convnext_path), num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.head = nn.Linear(CONVNEXT_DIM, num_labels)
        self.spec_aug = SpecAugment()
        self.mixup = Mixup()

    def forward(self, spec, target=None):
        # spec: (B, 1, 128, T) → 1ch → ConvNeXt expects 3ch input
        if self.training and target is not None:
            spec = self.spec_aug(spec)
        spec3 = spec.repeat(1, 3, 1, 1)
        if self.training and target is not None:
            spec3, target = self.mixup(spec3, target)
        out = self.backbone(spec3)
        return out.logits, target


class MultiWindowDataset(Dataset):
    """샘플마다 윈도우 풀(focal=K개, soundscape=1개) 중 랜덤 1개 추출."""

    def __init__(self, sample_idx_keep: np.ndarray, labels: np.ndarray,
                 cache: np.ndarray, meta_sample_idx: np.ndarray, is_train: bool):
        self.sample_idx_keep = sample_idx_keep
        self.labels = labels
        self.cache = cache
        self.meta_sample_idx = meta_sample_idx
        self.is_train = is_train

        # sample_idx → 그 샘플의 windows (cache 인덱스 list)
        self.windows_per_sample: dict[int, list[int]] = {}
        for w_idx, s_idx in enumerate(meta_sample_idx):
            self.windows_per_sample.setdefault(int(s_idx), []).append(w_idx)

    def __len__(self):
        return len(self.sample_idx_keep)

    def __getitem__(self, idx):
        s_idx = int(self.sample_idx_keep[idx])
        win_pool = self.windows_per_sample[s_idx]
        if self.is_train and len(win_pool) > 1:
            w = random.choice(win_pool)
        else:
            w = win_pool[0]  # val: 가장 강한(첫 번째) 윈도우 고정
        spec = torch.from_numpy(np.array(self.cache[w], dtype=np.float32))
        label = torch.tensor(self.labels[s_idx], dtype=torch.float32)
        return spec, label


def train_fold(fold: int, sample_indices: np.ndarray, labels: np.ndarray,
               strat: np.ndarray, cache, meta_sample_idx):
    set_seed(42 + fold)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_pos, val_pos = list(skf.split(sample_indices, strat))[fold]
    train_idx = sample_indices[train_pos]
    val_idx = sample_indices[val_pos]

    train_ds = MultiWindowDataset(train_idx, labels, cache, meta_sample_idx, is_train=True)
    val_ds = MultiWindowDataset(val_idx, labels, cache, meta_sample_idx, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}: Train={len(train_ds)}, Val={len(val_ds)}")
    print(f"Loss: 0.5*BCE + 0.5*SoftAUC | Cache: multi-window v2")
    print(f"{'='*50}")

    num_labels = labels.shape[1]
    model = BirdConvNeXt(CONVNEXT_PATH, num_labels=num_labels).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_softauc = SoftAUCLoss(gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    ckpt_path = OUT / f"checkpoint_fold{fold}.pth"
    start_epoch = 0
    best_auc = 0.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["best_auc"]
        print(f"  Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    for epoch in range(start_epoch, N_EPOCHS):
        t0 = time.time()

        model.train()
        tl = bce = sauc = 0.0
        for spec, target in train_loader:
            spec = spec.to(DEVICE)
            target = target.to(DEVICE)
            logits, tgt_mixed = model(spec, target)
            l_bce = criterion_bce(logits, tgt_mixed)
            l_sauc = criterion_softauc(logits, tgt_mixed)
            loss = 0.5 * l_bce + 0.5 * l_sauc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item()
            bce += l_bce.item()
            sauc += l_sauc.item()

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
        aucs = [roc_auc_score(targets[:, j], preds[:, j])
                for j in range(num_labels) if targets[:, j].sum() > 0]
        auc = float(np.mean(aucs))

        elapsed = time.time() - t0
        n = max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: "
              f"loss={tl/n:.4f} (bce={bce/n:.4f}, sauc={sauc/n:.4f}), "
              f"auc={auc:.4f} ({elapsed:.0f}s)", flush=True)

        if auc > best_auc:
            best_auc = auc
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("spec_aug.") and not k.startswith("mixup.")}
            torch.save(state, OUT / f"best_fold{fold}.pth")
            # OOF predictions for this val fold (for stacker downstream)
            np.save(OUT / f"oof_preds_fold{fold}.npy", preds)
            np.save(OUT / f"oof_idx_fold{fold}.npy", val_idx)
            print(f"    -> New best: {best_auc:.4f}", flush=True)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
        }, ckpt_path)

    print(f"Fold {fold+1} best AUC: {best_auc:.4f}")
    return best_auc


def main():
    print("Loading v2 labels + cache...")
    npz = np.load(DATA_V2 / "labels_v2.npz", allow_pickle=True)
    labels = npz["labels"]
    strat_full = npz["primary_strat"]
    sample_indices = np.arange(len(labels))

    cache = np.load(DATA_V2 / "cache_v2.npy", mmap_mode="r")
    meta = np.load(DATA_V2 / "cache_v2_meta.npz", allow_pickle=True)
    meta_sample_idx = meta["sample_idx"]

    print(f"samples = {len(labels)} (focal+soundscape), cache windows = {len(meta_sample_idx)}")
    print(f"label C = {labels.shape[1]}")

    # 캐시에 등장하는 샘플만 학습 (실패 케이스 제외)
    valid_samples = np.array(sorted({int(x) for x in meta_sample_idx}))
    print(f"valid samples (in cache): {len(valid_samples)}/{len(labels)}")

    valid_strat = np.array([str(strat_full[i]) for i in valid_samples])

    t_start = time.time()
    fold_aucs = []
    for fold in range(N_FOLDS):
        auc = train_fold(fold, valid_samples, labels, valid_strat, cache, meta_sample_idx)
        fold_aucs.append(auc)
        print(f"  → fold {fold+1} best AUC: {auc:.4f}")

    print(f"\n=== ConvNeXt 5-fold v2 done ===")
    print(f"Mean OOF AUC: {np.mean(fold_aucs):.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
