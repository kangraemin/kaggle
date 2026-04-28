"""sub_25 — Multi-window spec cache (CPU only).

문제:
  기존 캐시 `convnext_spec_cache.npy`는 녹음의 **첫 5초**만 저장.
  매 epoch 같은 첫 5초만 보여줘서 시간축 augmentation = 0.
  첫 5초는 보통 침묵/녹음자 멘트 비율 큼 → 새 울음 못 배움.

해법:
  녹음마다 **RMS 상위 K개 5초 윈도우** 미리 저장 (non-overlapping).
  학습 시 매 epoch 그 중 1개 랜덤 추출 → 시간축 다양성 + 새 울음 보장.
  Soundscape segment는 라벨된 정확한 5초 위치 1개씩.

Pipeline:
  1) data/v2/labels_v2.npz 읽기 (paths, source, seg_start, seg_dur)
  2) focal: 녹음 전체 → 5초 sliding RMS → top-K non-overlapping 윈도우
  3) soundscape: seg_start ~ seg_start+5sec 정확 윈도우
  4) Mel-spec 계산 (mel=128, hop=320, fft=1024, sr=32000, top_db=80, XCL norm)
  5) float16 저장 (mmap-friendly)

산출:
  data/v2/cache_v2.npy        — (N_windows_total, 1, 128, T) float16
  data/v2/cache_v2_meta.npz   — sample_idx, k_idx, rms, source

학습 시 사용:
  meta['sample_idx'][w] == sample i인 윈도우 인덱스 집합 = i의 후보 풀.
  매 epoch 그 풀에서 랜덤 1개 → cache_v2[w]를 읽음.

CPU 전용 (torchaudio MelSpectrogram도 기본 CPU). MPS 학습과 충돌 없음.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[3]
DATA_V2 = ROOT / "data" / "v2"

SR = 32000
DUR_SEC = 5.0
DUR = int(DUR_SEC * SR)  # 160000
HOP_AUDIO = SR  # 1초 hop으로 sliding RMS
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
F_MIN = 50.0
F_MAX = 14000.0
TOP_DB = 80.0
# XCL 통계 (학습 코드와 동일)
NORM_MEAN = -4.268
NORM_STD = 4.569

K_FOCAL = 5  # focal 녹음당 윈도우 수


def make_mel_module():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
    )


def wav_to_spec(mel_module, wav: torch.Tensor) -> torch.Tensor:
    """wav (1D float32, len=DUR) → (1, 128, T) float16 spec."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, DUR)
    mel = mel_module(wav)  # (1, 128, T)
    db = 10.0 * torch.log10(mel.clamp(min=1e-10))
    max_val = db.flatten(-2).max(dim=-1).values[..., None, None]
    db = torch.maximum(db, max_val - TOP_DB)
    db = (db - NORM_MEAN) / NORM_STD
    return db  # (1, 128, T)


def topk_windows_by_rms(wav: np.ndarray, k: int = K_FOCAL) -> list[int]:
    """녹음 전체에서 RMS 상위 K개 non-overlapping 5초 윈도우 시작 인덱스.

    1초 hop sliding으로 RMS 계산 → greedy non-overlap top-K.
    """
    L = len(wav)
    if L < DUR:
        return [0]  # 짧으면 zero-pad용 1개만
    # 1초 hop sliding RMS
    starts = np.arange(0, L - DUR + 1, HOP_AUDIO)
    if len(starts) == 0:
        return [0]
    rms = np.empty(len(starts), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = wav[s : s + DUR]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    order = np.argsort(-rms)  # high→low
    chosen: list[int] = []
    for idx in order:
        s = int(starts[idx])
        if all(abs(s - c) >= DUR for c in chosen):
            chosen.append(s)
            if len(chosen) >= k:
                break
    return chosen


def load_audio(path: str, target_len: int | None = None) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        # rare — soundfile이 원본 SR 그대로 줌. resample은 librosa 또는 soxr.
        import resampy

        wav = resampy.resample(wav, sr, SR)
    if target_len is not None and len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav.astype(np.float32)


def process_focal(path: str, mel_module) -> tuple[np.ndarray, np.ndarray]:
    """returns (specs (K, 1, 128, T) f16, rms (K,) f32). pad-fill if recording short."""
    wav = load_audio(path)
    if len(wav) < DUR:
        wav = np.pad(wav, (0, DUR - len(wav)))
        specs = wav_to_spec(mel_module, torch.from_numpy(wav))
        rms_val = float(np.sqrt(np.mean(wav * wav) + 1e-12))
        return specs.unsqueeze(0).numpy().astype(np.float16), np.array([rms_val], np.float32)
    starts = topk_windows_by_rms(wav, K_FOCAL)
    specs_list = []
    rms_list = []
    for s in starts:
        seg = wav[s : s + DUR]
        spec = wav_to_spec(mel_module, torch.from_numpy(seg))  # (1, 128, T)
        specs_list.append(spec.numpy().astype(np.float16))
        rms_list.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    return np.stack(specs_list, axis=0), np.array(rms_list, np.float32)


def process_soundscape(path: str, seg_start: float, mel_module) -> tuple[np.ndarray, float]:
    """소수 segment 정확 위치 1개. returns (1, 1, 128, T) f16."""
    wav = load_audio(path)
    s = int(seg_start * SR)
    if s + DUR > len(wav):
        wav = np.pad(wav, (0, s + DUR - len(wav)))
    seg = wav[s : s + DUR]
    spec = wav_to_spec(mel_module, torch.from_numpy(seg))
    rms_val = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return spec.unsqueeze(0).numpy().astype(np.float16), rms_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처음 N개만 (디버깅)")
    parser.add_argument("--workers", type=int, default=1, help="현재 single-process (multi-proc은 librosa GIL 이슈 있어서)")
    args = parser.parse_args()

    labels_npz = np.load(DATA_V2 / "labels_v2.npz", allow_pickle=True)
    paths = labels_npz["paths"]
    source = labels_npz["source"]
    seg_start = labels_npz["seg_start"]
    n = len(paths) if args.limit is None else min(args.limit, len(paths))

    # 첫 패스: focal은 K개, soundscape은 1개 → 총 윈도우 수 미리 산출
    n_focal = int((source[:n] == "focal").sum())
    n_ss = int((source[:n] == "soundscape").sum())
    n_windows_max = n_focal * K_FOCAL + n_ss
    # T 차원 (mel frame)
    T_frames = DUR // HOP_LENGTH + 1  # 500 + 1
    print(f"sample={n} (focal={n_focal}, soundscape={n_ss})")
    print(f"max windows = {n_windows_max}, shape per window = (1, {N_MELS}, {T_frames})")
    bytes_total = n_windows_max * 1 * N_MELS * T_frames * 2
    print(f"max disk size: {bytes_total/1e9:.1f} GB float16")

    # mmap 출력 파일 생성 (memmap으로 직접 쓰기 — 실제 윈도우 수 < max)
    cache_path = DATA_V2 / "cache_v2.npy"
    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_windows_max, 1, N_MELS, T_frames),
    )

    sample_idx = np.empty(n_windows_max, dtype=np.int32)
    k_idx = np.empty(n_windows_max, dtype=np.int8)
    rms_arr = np.empty(n_windows_max, dtype=np.float32)
    src_arr = np.empty(n_windows_max, dtype=object)

    mel_module = make_mel_module()

    w = 0  # 윈도우 카운터
    t0 = time.time()
    n_short_focal = 0
    for i in range(n):
        p = str(paths[i])
        src = str(source[i])
        try:
            if src == "focal":
                specs, rms_vals = process_focal(p, mel_module)
                if specs.shape[0] < K_FOCAL:
                    n_short_focal += 1
                for k, (spec, rv) in enumerate(zip(specs, rms_vals)):
                    out[w] = spec
                    sample_idx[w] = i
                    k_idx[w] = k
                    rms_arr[w] = rv
                    src_arr[w] = "focal"
                    w += 1
            else:
                spec, rv = process_soundscape(p, float(seg_start[i]), mel_module)
                out[w] = spec[0]
                sample_idx[w] = i
                k_idx[w] = 0
                rms_arr[w] = rv
                src_arr[w] = "soundscape"
                w += 1
        except Exception as e:
            print(f"  ERR {i} {p}: {e}")
            continue
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n} ({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m, windows={w})", flush=True)

    out.flush()

    # 실제 사용된 windows 수만큼 trim한 cache 만들기
    actual_w = w
    print(f"\nactual windows = {actual_w} (focal short: {n_short_focal})")

    # cache mmap을 actual_w 만큼 trim — 새 파일에 저장 후 원본 삭제
    cache_trim_path = DATA_V2 / "cache_v2_trimmed.npy"
    trimmed = np.lib.format.open_memmap(
        cache_trim_path,
        mode="w+",
        dtype=np.float16,
        shape=(actual_w, 1, N_MELS, T_frames),
    )
    trimmed[:] = out[:actual_w]
    trimmed.flush()
    del trimmed, out

    # rename
    cache_path.unlink()
    cache_trim_path.rename(cache_path)

    # meta 저장
    np.savez_compressed(
        DATA_V2 / "cache_v2_meta.npz",
        sample_idx=sample_idx[:actual_w],
        k_idx=k_idx[:actual_w],
        rms=rms_arr[:actual_w],
        source=src_arr[:actual_w].astype(object),
    )
    final_size = cache_path.stat().st_size / 1e9
    print(f"Saved: {cache_path} ({final_size:.1f} GB)")
    print(f"Saved: {DATA_V2 / 'cache_v2_meta.npz'}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
