"""sub_25 — ss10k_subset.npz의 30000 segment에 대한 single-window spec cache.

각 segment는 라벨된 정확한 5초 위치 1개씩 (focal의 multi-window와 다름).

산출:
  data/v2/cache_ss10k.npy        (30000, 1, 128, T) float16
  data/v2/cache_ss10k_meta.npz   sample_idx (= ss10k_subset 행 인덱스), rms

후속:
  train_convnext_v2.py 가 cache_v2 (focal+labeled ss) + cache_ss10k 를 concat 인덱스로 사용.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[3]
DATA_V2 = ROOT / "data" / "v2"

SR = 32000
DUR = 5 * SR
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
F_MIN = 50.0
F_MAX = 14000.0
TOP_DB = 80.0
NORM_MEAN = -4.268
NORM_STD = 4.569


def make_mel():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        f_min=F_MIN, f_max=F_MAX,
    )


def wav_to_spec(mel_module, wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mel = mel_module(wav)
    db = 10.0 * torch.log10(mel.clamp(min=1e-10))
    max_val = db.flatten(-2).max(dim=-1).values[..., None, None]
    db = torch.maximum(db, max_val - TOP_DB)
    db = (db - NORM_MEAN) / NORM_STD
    return db


def main():
    sub = np.load(DATA_V2 / "ss10k_subset.npz", allow_pickle=True)
    paths = sub["paths"]
    seg_starts = sub["seg_start"]
    n = len(paths)
    T_frames = DUR // HOP_LENGTH + 1
    print(f"ss10k_subset: {n} segments, T={T_frames}")
    cache_path = DATA_V2 / "cache_ss10k.npy"
    out = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16,
        shape=(n, 1, N_MELS, T_frames),
    )
    rms = np.empty(n, dtype=np.float32)
    sample_idx = np.arange(n, dtype=np.int32)

    mel = make_mel()
    t0 = time.time()
    n_err = 0
    for i, (p, s_sec) in enumerate(zip(paths, seg_starts)):
        try:
            wav, sr = sf.read(str(p), dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != SR:
                import resampy
                wav = resampy.resample(wav, sr, SR)
            s = int(float(s_sec) * SR)
            if s + DUR > len(wav):
                wav = np.pad(wav, (0, s + DUR - len(wav)))
            seg = wav[s : s + DUR]
            spec = wav_to_spec(mel, torch.from_numpy(seg))
            out[i] = spec.numpy().astype(np.float16)
            rms[i] = float(np.sqrt((seg * seg).mean() + 1e-12))
        except Exception as e:
            n_err += 1
            if n_err < 5:
                print(f"  ERR {i} {p}: {e}")
            continue
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n} ({el/60:.1f}m, ETA {eta/60:.1f}m, errs={n_err})", flush=True)
    out.flush()
    np.savez_compressed(
        DATA_V2 / "cache_ss10k_meta.npz",
        sample_idx=sample_idx, rms=rms,
    )
    print(f"Saved cache_ss10k.npy ({cache_path.stat().st_size/1e9:.1f} GB), {n_err} errors")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
