"""sub_25 — BirdNET cross-model 추론.

목적:
1. **S05 9파일** 추론 → test prior (test가 정확히 같은 사이트, BirdNET이 본 종 분포)
2. **labeled 66파일** 추론 → BirdNET ↔ ground truth 정합성 검증 (false positive rate 추정)
3. (선택) unlabeled 10592파일 → cross-model pseudo-label

BirdNET label 형식: `{scientific_name}_{common_name}` → scientific_name으로 taxonomy.csv 매칭.

산출:
- data/v2/birdnet_s05.csv          — S05 9파일 detection
- data/v2/birdnet_labeled.csv      — labeled 66파일 detection (검증용)
- data/v2/birdnet_label_map.csv    — BirdNET → taxonomy.csv 매핑 테이블
- data/v2/birdnet_pseudo_unlabel.csv (--full 옵션 시) — 10592 unlabeled

실행:
  python birdnet_infer.py            # S05 + labeled (수 분)
  python birdnet_infer.py --full     # + unlabeled (수 시간)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# litert→tflite_runtime shim (tflite-runtime이 macOS arm64 wheel 없음)
import ai_edge_litert.interpreter as litert  # noqa: E402

mod = type(sys)("tflite_runtime")
mod.interpreter = litert
sys.modules["tflite_runtime"] = mod
sys.modules["tflite_runtime.interpreter"] = litert

from birdnetlib import Recording  # noqa: E402
from birdnetlib.analyzer import Analyzer  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = ROOT / "data" / "v2"
EDA = ROOT / "submissions" / "sub_25" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

# Pantanal site coordinates (recording_location.txt: 위도 -16.5~-21.6, 경도 -55.9~-57.6)
PANTANAL_LAT = -19.0
PANTANAL_LON = -56.7

MIN_CONF = 0.10  # BirdNET confidence threshold


def build_label_map(analyzer: Analyzer) -> pd.DataFrame:
    """BirdNET 6522 labels → taxonomy.csv 234 클래스 매핑.

    BirdNET label: 'Genus species_Common Name' → scientific_name 추출 후 매칭.
    """
    tax = pd.read_csv(DATA / "taxonomy.csv")
    tax_sci = tax.set_index("scientific_name")["primary_label"].astype(str).to_dict()

    rows = []
    for label in analyzer.labels:
        sci, _, _ = label.partition("_")
        prim = tax_sci.get(sci, None)
        rows.append({"birdnet_label": label, "scientific_name": sci, "primary_label": prim})
    df = pd.DataFrame(rows)
    matched = df.dropna(subset=["primary_label"])
    print(f"BirdNET → taxonomy 매핑: {len(matched)} / {len(df)}")
    print(f"taxonomy 234 클래스 중 BirdNET cover: {matched['primary_label'].nunique()}")
    df.to_csv(OUT / "birdnet_label_map.csv", index=False)
    return df


def parse_filename_datetime(name: str) -> datetime | None:
    """BC2026_*_SXX_YYYYMMDD_HHMMSS.ogg → datetime."""
    parts = name.replace(".ogg", "").split("_")
    if len(parts) < 5:
        return None
    try:
        date = parts[-2]  # YYYYMMDD
        time_str = parts[-1]  # HHMMSS
        return datetime.strptime(f"{date}_{time_str}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def analyze_files(analyzer: Analyzer, files: list[Path], min_conf: float = MIN_CONF) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(files):
        dt = parse_filename_datetime(f.name) or datetime(2025, 2, 27, 1, 0, 0)
        rec = Recording(
            analyzer,
            str(f),
            lat=PANTANAL_LAT,
            lon=PANTANAL_LON,
            date=dt,
            min_conf=min_conf,
        )
        try:
            rec.analyze()
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
            continue
        for d in rec.detections:
            rows.append(
                {
                    "filename": f.name,
                    "datetime": dt.isoformat(),
                    "start_sec": d["start_time"],
                    "end_sec": d["end_time"],
                    "scientific_name": d["scientific_name"],
                    "common_name": d["common_name"],
                    "confidence": d["confidence"],
                }
            )
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(files)} ({rows[-1]['filename'] if rows else 'no det'})")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run on all 10592 unlabeled soundscapes")
    parser.add_argument("--min-conf", type=float, default=MIN_CONF)
    args = parser.parse_args()

    print("Loading BirdNET...")
    analyzer = Analyzer()
    print(f"BirdNET labels: {len(analyzer.labels)}")

    label_map = build_label_map(analyzer)

    # S05 — test와 같은 사이트
    s05_files = pd.read_csv(EDA / "s05_files.csv")["filename"].tolist()
    s05_paths = [DATA / "train_soundscapes" / f for f in s05_files]
    s05_paths = [p for p in s05_paths if p.exists()]
    print(f"\n=== S05: {len(s05_paths)} files ===")
    t0 = time.time()
    s05_det = analyze_files(analyzer, s05_paths, args.min_conf)
    print(f"S05 detections: {len(s05_det)} ({time.time()-t0:.1f}s)")
    s05_det.to_csv(OUT / "birdnet_s05.csv", index=False)

    # labeled 66 — 검증용
    ts = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labeled_files = sorted(ts["filename"].unique())
    labeled_paths = [DATA / "train_soundscapes" / f for f in labeled_files]
    labeled_paths = [p for p in labeled_paths if p.exists()]
    print(f"\n=== labeled: {len(labeled_paths)} files ===")
    t0 = time.time()
    lab_det = analyze_files(analyzer, labeled_paths, args.min_conf)
    print(f"labeled detections: {len(lab_det)} ({time.time()-t0:.1f}s)")
    lab_det.to_csv(OUT / "birdnet_labeled.csv", index=False)

    if args.full:
        all_files = sorted((DATA / "train_soundscapes").glob("*.ogg"))
        labeled_set = set(labeled_files)
        s05_set = set(s05_files)
        unlabeled = [f for f in all_files if f.name not in labeled_set and f.name not in s05_set]
        print(f"\n=== unlabeled: {len(unlabeled)} files (this takes hours) ===")
        t0 = time.time()
        unlab_det = analyze_files(analyzer, unlabeled, args.min_conf)
        print(f"unlabeled detections: {len(unlab_det)} ({time.time()-t0:.1f}s)")
        unlab_det.to_csv(OUT / "birdnet_pseudo_unlabel.csv", index=False)


if __name__ == "__main__":
    main()
