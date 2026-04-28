"""ConvNeXt 5-fold v2 — sub_25 도메인 정제 통합 학습.

기존 train_convnext_5fold.py 대비 변경:

1. **라벨 = data/v2/labels_v3.npz** (focal 35549 + soundscape 1478 + ss10k pseudo 30000 = 67027)
   - focal: primary 1.0 + secondary 0.3
   - soundscape: multi-hot 1.0 (segment 단위)
   - pseudo (Perch ss10k confident): soft 0.85 (cross-model, 25 sonotype 제외)
2. **두 캐시**: data/v2/cache_v2.npy (focal+ss multi-window) + cache_ss10k.npy (pseudo single window)
3. **Dataset.__getitem__**: 샘플의 윈도우 풀에서 매 epoch 랜덤 1개 → 시간축 다양성
4. **Synthetic soundscape mixing**: focal 학습 시 RMS 낮은 ss/pseudo background spec과 합성 (도메인 갭 직접 해결)
5. **출력 = models/convnext_5fold_v2/** (v1과 분리)
6. **Mixup은 자동으로 focal ↔ soundscape ↔ pseudo 섞음** (배치 내 랜덤 페어링 보너스)

전제: build_labels_v2 + build_multi_window_cache + extract_ss10k_pseudo + build_ss10k_cache + combine_labels_v3 선행.
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


class MultiCacheDataset(Dataset):
    """labels_v3.npz 기반 — focal+ss(multi-window) + ss10k(single window) 통합.

    cache_offset == "main" → cache_v2.npy 사용 (multi-window pool)
    cache_offset == "ss10k" → cache_ss10k.npy 사용 (single window)

    Synthetic soundscape mixing (옵션):
      focal 샘플은 일정 확률로 random ss/pseudo background spec과 mixup → 도메인 갭 줄임.
    """

    def __init__(self, sample_idx_keep: np.ndarray, labels: np.ndarray,
                 source: np.ndarray, cache_offset: np.ndarray,
                 cache_main, meta_main_sample_idx: np.ndarray,
                 cache_ss10k, n_main_samples: int, is_train: bool,
                 soundscape_mix_prob: float = 0.4, soundscape_mix_lam: float = 0.5):
        self.sample_idx_keep = sample_idx_keep
        self.labels = labels
        self.source = source
        self.cache_offset = cache_offset
        self.cache_main = cache_main
        self.cache_ss10k = cache_ss10k
        self.n_main_samples = n_main_samples
        self.is_train = is_train
        self.soundscape_mix_prob = soundscape_mix_prob
        self.soundscape_mix_lam = soundscape_mix_lam

        # main cache: sample_idx (0~n_main_samples-1) → window 인덱스 풀
        self.main_windows_per_sample: dict[int, list[int]] = {}
        for w_idx, s_idx in enumerate(meta_main_sample_idx):
            self.main_windows_per_sample.setdefault(int(s_idx), []).append(w_idx)

        # background pool: ss + ss10k 샘플 (focal 아닌 것)
        self.bg_pool: list[int] = []
        for i in range(len(labels)):
            src = str(source[i])
            if src in ("soundscape", "pseudo_ss10k"):
                self.bg_pool.append(i)

    def _fetch_spec(self, sample_idx: int) -> np.ndarray:
        co = str(self.cache_offset[sample_idx])
        if co == "main":
            wins = self.main_windows_per_sample.get(sample_idx, [])
            if not wins:
                return np.zeros_like(self.cache_main[0])  # 캐시 누락 안전장치
            w = random.choice(wins) if (self.is_train and len(wins) > 1) else wins[0]
            return np.array(self.cache_main[w], dtype=np.float32)
        ss10k_idx = sample_idx - self.n_main_samples
        return np.array(self.cache_ss10k[ss10k_idx], dtype=np.float32)

    def __len__(self):
        return len(self.sample_idx_keep)

    def __getitem__(self, idx):
        s_idx = int(self.sample_idx_keep[idx])
        spec = self._fetch_spec(s_idx)
        label = self.labels[s_idx].copy()

        if self.is_train and str(self.source[s_idx]) == "focal" and self.bg_pool:
            if random.random() < self.soundscape_mix_prob:
                bg_idx = random.choice(self.bg_pool)
                bg_spec = self._fetch_spec(bg_idx)
                lam = self.soundscape_mix_lam
                spec = lam * spec + (1 - lam) * bg_spec
                bg_label = self.labels[bg_idx]
                label = np.maximum(label, bg_label * 0.5)  # focal label 우선, bg는 약하게 흡수

        return torch.from_numpy(spec), torch.tensor(label, dtype=torch.float32)


def train_fold(fold: int, sample_indices: np.ndarray, labels: np.ndarray,
               strat: np.ndarray, source: np.ndarray, cache_offset: np.ndarray,
               cache_main, meta_main_sample_idx: np.ndarray,
               cache_ss10k, n_main_samples: int):
    set_seed(42 + fold)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_pos, val_pos = list(skf.split(sample_indices, strat))[fold]
    train_idx = sample_indices[train_pos]
    val_idx = sample_indices[val_pos]

    train_ds = MultiCacheDataset(
        train_idx, labels, source, cache_offset,
        cache_main, meta_main_sample_idx, cache_ss10k, n_main_samples,
        is_train=True, soundscape_mix_prob=0.4, soundscape_mix_lam=0.5,
    )
    val_ds = MultiCacheDataset(
        val_idx, labels, source, cache_offset,
        cache_main, meta_main_sample_idx, cache_ss10k, n_main_samples,
        is_train=False, soundscape_mix_prob=0.0,
    )

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
    print("Loading v3 labels + caches (main + ss10k)...")
    npz = np.load(DATA_V2 / "labels_v3.npz", allow_pickle=True)
    labels = npz["labels"]
    source = npz["source"]
    cache_offset = npz["cache_offset"]
    strat_full = npz["primary_strat"]

    cache_main = np.load(DATA_V2 / "cache_v2.npy", mmap_mode="r")
    meta_main = np.load(DATA_V2 / "cache_v2_meta.npz", allow_pickle=True)
    meta_main_sample_idx = meta_main["sample_idx"]

    cache_ss10k = np.load(DATA_V2 / "cache_ss10k.npy", mmap_mode="r")

    n_main_samples = int((cache_offset == "main").sum())
    print(f"samples = {len(labels)} (main {n_main_samples} + ss10k {len(labels)-n_main_samples})")
    print(f"  source dist: focal={int((source=='focal').sum())}, ss={int((source=='soundscape').sum())}, pseudo={int((source=='pseudo_ss10k').sum())}")
    print(f"  cache_main windows = {len(meta_main_sample_idx)}, cache_ss10k samples = {len(cache_ss10k)}")
    print(f"  label C = {labels.shape[1]}")

    # main cache에 windows 있는 sample만 + ss10k 전부 학습 가능
    main_valid = {int(x) for x in meta_main_sample_idx}
    valid_samples_list = []
    for i in range(len(labels)):
        co = str(cache_offset[i])
        if co == "main" and i in main_valid:
            valid_samples_list.append(i)
        elif co == "ss10k":
            valid_samples_list.append(i)
    valid_samples = np.array(valid_samples_list)
    print(f"valid samples (in caches): {len(valid_samples)}/{len(labels)}")

    valid_strat = np.array([str(strat_full[i]) for i in valid_samples])

    t_start = time.time()
    fold_aucs = []
    for fold in range(N_FOLDS):
        auc = train_fold(
            fold, valid_samples, labels, valid_strat, source, cache_offset,
            cache_main, meta_main_sample_idx, cache_ss10k, n_main_samples,
        )
        fold_aucs.append(auc)
        print(f"  → fold {fold+1} best AUC: {auc:.4f}")

    print(f"\n=== ConvNeXt 5-fold v2 done ===")
    print(f"Mean OOF AUC: {np.mean(fold_aucs):.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
