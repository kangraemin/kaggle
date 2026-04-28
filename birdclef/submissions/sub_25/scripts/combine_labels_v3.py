"""sub_25 — labels_v2 + ss10k_subset → labels_v3 통합.

labels_v3.npz 구조:
  paths           — (N,) 오디오 경로
  labels          — (N, 234) multi-hot (focal: 1.0/0.3, ss: 1.0, pseudo: 0.85)
  source          — (N,) "focal" | "soundscape" | "pseudo_ss10k"
  seg_start       — (N,) sec (focal: -1, ss/pseudo: 정확 위치)
  seg_dur         — (N,) sec (focal: -1, ss/pseudo: 5)
  primary_strat   — (N,) StratifiedKFold용
  cache_offset    — (N,) "main" 또는 "ss10k" (학습 시 cache 라우팅)

학습 시:
  - cache_offset == "main": cache_v2.npy (multi-window per sample)
  - cache_offset == "ss10k": cache_ss10k.npy (single window per sample)
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_V2 = ROOT / "data" / "v2"


def main():
    v2 = np.load(DATA_V2 / "labels_v2.npz", allow_pickle=True)
    ss = np.load(DATA_V2 / "ss10k_subset.npz", allow_pickle=True)

    n2 = len(v2["paths"])
    n_ss = len(ss["paths"])
    print(f"v2: {n2}, ss10k: {n_ss}, combined: {n2 + n_ss}")

    paths = np.concatenate([v2["paths"], ss["paths"]])
    labels = np.vstack([v2["labels"], ss["labels"]])
    source = np.concatenate([v2["source"], ss["source"]])
    seg_start = np.concatenate([v2["seg_start"], ss["seg_start"]])
    seg_dur = np.concatenate([v2["seg_dur"], ss["seg_dur"]])
    strat = np.concatenate([v2["primary_strat"], ss["primary_strat"]])
    cache_offset = np.concatenate([
        np.full(n2, "main", dtype=object),
        np.full(n_ss, "ss10k", dtype=object),
    ])

    out = DATA_V2 / "labels_v3.npz"
    np.savez_compressed(
        out,
        paths=paths, labels=labels, source=source, seg_start=seg_start, seg_dur=seg_dur,
        primary_strat=strat, cache_offset=cache_offset,
        classes=v2["classes"],
    )
    print(f"Saved: {out}")
    print(f"  source dist: focal={np.sum(source=='focal')}, soundscape={np.sum(source=='soundscape')}, pseudo={np.sum(source=='pseudo_ss10k')}")
    print(f"  total label sum: {labels.sum():.0f}")
    print(f"  classes covered (label > 0): {(labels.sum(0) > 0).sum()}/234")


if __name__ == "__main__":
    main()
