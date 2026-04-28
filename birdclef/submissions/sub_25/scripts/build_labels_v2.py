"""sub_25 — 통합 학습 라벨 생성.

기존 학습 코드는 train.csv primary_label만 single-hot 사용 (학습 클래스 234 중 28개 영구 0점).

이 스크립트는 다음 정제를 적용해서 multi-hot 라벨을 만든다:

1. **train.csv primary** = 1.0 (그대로)
2. **train.csv secondary_labels** = 0.3 (weak label, 4372 row × 161 종 cover)
3. **labeled soundscape (66 files, 1478 segments)** = 1.0 (이게 25 Insect sonotype 학습의 유일한 소스)
   - segment offset/duration 메타도 함께 저장 → 학습 시 해당 시간 윈도우만 자른다.

산출:
- data/v2/labels_v2.npz
    - paths: (N,) str — 오디오 파일 절대경로
    - labels: (N, 234) float32 multi-hot
    - source: (N,) str — 'focal' | 'soundscape'
    - seg_start: (N,) float32 — soundscape segment 시작 (sec). focal=−1 (전체 사용)
    - seg_dur: (N,) float32 — segment 길이. focal=−1
    - primary_strat: (N,) str — StratifiedKFold용 stratify key (focal=primary, soundscape=primary[0])

234 클래스 = sorted(taxonomy.csv.primary_label).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = ROOT / "data" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

SECONDARY_WEIGHT = 0.3

train = pd.read_csv(DATA / "train.csv")
tax = pd.read_csv(DATA / "taxonomy.csv")
ts = pd.read_csv(DATA / "train_soundscapes_labels.csv")

LABELS = sorted(tax["primary_label"].dropna().astype(str).unique())
label_to_idx = {l: i for i, l in enumerate(LABELS)}
C = len(LABELS)
print(f"Target classes: {C}")

# 1) Focal (train.csv) — primary + secondary multi-hot
focal_paths = []
focal_labels = []
focal_strat = []
n_secondary_added = 0
for _, row in train.iterrows():
    p = str(DATA / "train_audio" / row["filename"])
    vec = np.zeros(C, dtype=np.float32)
    prim = str(row["primary_label"])
    if prim in label_to_idx:
        vec[label_to_idx[prim]] = 1.0
    sec_raw = row.get("secondary_labels", "[]")
    try:
        sec = ast.literal_eval(sec_raw) if pd.notna(sec_raw) else []
    except (ValueError, SyntaxError):
        sec = []
    for s in sec:
        if s in label_to_idx and vec[label_to_idx[s]] < SECONDARY_WEIGHT:
            vec[label_to_idx[s]] = SECONDARY_WEIGHT
            n_secondary_added += 1
    focal_paths.append(p)
    focal_labels.append(vec)
    focal_strat.append(prim)

print(f"focal: {len(focal_paths)} samples, secondary signals added: {n_secondary_added}")

# 2) Soundscape labeled — segment 단위 multi-hot
def parse_time(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


ss_dir = DATA / "train_soundscapes"
ss_paths = []
ss_labels = []
ss_starts = []
ss_durs = []
ss_strat = []
n_unknown = 0
for _, row in ts.iterrows():
    p = ss_dir / row["filename"]
    if not p.exists():
        n_unknown += 1
        continue
    start = parse_time(row["start"])
    end = parse_time(row["end"])
    vec = np.zeros(C, dtype=np.float32)
    species = str(row["primary_label"]).split(";")
    valid = []
    for s in species:
        if s in label_to_idx:
            vec[label_to_idx[s]] = 1.0
            valid.append(s)
    if not valid:
        continue
    ss_paths.append(str(p))
    ss_labels.append(vec)
    ss_starts.append(float(start))
    ss_durs.append(float(end - start))
    ss_strat.append(valid[0])

print(f"soundscape: {len(ss_paths)} segments (skipped missing files: {n_unknown})")

# 3) 통합
all_paths = np.array(focal_paths + ss_paths, dtype=object)
all_labels = np.stack(focal_labels + ss_labels, axis=0)
all_source = np.array(["focal"] * len(focal_paths) + ["soundscape"] * len(ss_paths), dtype=object)
all_starts = np.array([-1.0] * len(focal_paths) + ss_starts, dtype=np.float32)
all_durs = np.array([-1.0] * len(focal_paths) + ss_durs, dtype=np.float32)
all_strat = np.array(focal_strat + ss_strat, dtype=object)

print(f"total: {len(all_paths)}")
print(f"label sum per class top10: {pd.Series(all_labels.sum(0)).sort_values(ascending=False).head(10).to_dict()}")
print(f"label sum per class bottom10 (학습 신호 거의 없음):")
zero_classes = pd.Series(all_labels.sum(0)).sort_values()
print(zero_classes.head(10).to_dict())
print(f"학습 신호 0인 클래스: {(all_labels.sum(0) == 0).sum()}")

out = OUT / "labels_v2.npz"
np.savez_compressed(
    out,
    paths=all_paths,
    labels=all_labels,
    source=all_source,
    seg_start=all_starts,
    seg_dur=all_durs,
    primary_strat=all_strat,
    classes=np.array(LABELS, dtype=object),
)
print(f"Saved: {out} ({out.stat().st_size/1e6:.1f} MB)")
