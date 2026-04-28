"""sub_25 EDA — Pantanal/S05 domain analysis.

Outputs to sub_25/eda/:
- domain_stats.md     — 인사이트 요약
- class_prior.csv     — 235 클래스별 prior tier (high/mid/low)
- pantanal_classes.csv — train.csv lat/lon이 Pantanal 박스 안인 종 분포
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = ROOT / "submissions" / "sub_25" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

# Pantanal recorder box (recording_location.txt)
LAT_MIN, LAT_MAX = -21.6, -16.5
LON_MIN, LON_MAX = -57.6, -55.9

train = pd.read_csv(DATA / "train.csv")
tax = pd.read_csv(DATA / "taxonomy.csv")
ts = pd.read_csv(DATA / "train_soundscapes_labels.csv")

# 1) 클래스 카탈로그
all_classes = tax["primary_label"].astype(str).tolist()
train_classes = set(train["primary_label"].astype(str).unique())
ss_classes: set[str] = set()
for labs in ts["primary_label"]:
    ss_classes.update(str(labs).split(";"))

never_in_train = set(all_classes) - train_classes  # 학습 데이터 0
in_ss = ss_classes & set(all_classes)
in_train_not_ss = train_classes - ss_classes

# 2) Pantanal 박스 안인 학습 녹음
in_box = (
    (train["latitude"].between(LAT_MIN, LAT_MAX))
    & (train["longitude"].between(LON_MIN, LON_MAX))
)
pantanal_train = train[in_box]
pantanal_classes = set(pantanal_train["primary_label"].astype(str).unique())

# 3) 클래스별 prior tier
prior_records = []
for cls in all_classes:
    in_train_n = int((train["primary_label"].astype(str) == cls).sum())
    in_pantanal_n = int((pantanal_train["primary_label"].astype(str) == cls).sum())
    in_ss_seg = sum(cls in str(x).split(";") for x in ts["primary_label"])
    if cls in in_ss:
        tier = "A_in_soundscape"  # 라벨된 soundscape에 직접 등장 = 가장 높은 신뢰
    elif cls in pantanal_classes:
        tier = "B_in_pantanal_train"  # Pantanal 박스 안 학습 데이터에 등장
    elif in_train_n > 0:
        tier = "C_train_only"  # 학습 데이터 있으나 Pantanal·soundscape 모두 미등장
    else:
        tier = "D_no_train_data"  # 학습 데이터 0개 (25 Insect sonotype)
    prior_records.append(
        {
            "primary_label": cls,
            "scientific_name": tax.loc[tax["primary_label"] == cls, "scientific_name"].iloc[0],
            "class_name": tax.loc[tax["primary_label"] == cls, "class_name"].iloc[0],
            "n_train": in_train_n,
            "n_pantanal_train": in_pantanal_n,
            "n_soundscape_seg": in_ss_seg,
            "tier": tier,
        }
    )
prior_df = pd.DataFrame(prior_records)
prior_df.to_csv(OUT / "class_prior.csv", index=False)

tier_counts = prior_df["tier"].value_counts().sort_index()

# 4) train_soundscapes 폴더 메타 (사이트/시각/날짜)
ss_dir = DATA / "train_soundscapes"
files = sorted(p.name for p in ss_dir.glob("*.ogg"))
fpat = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.ogg")
rows = []
for f in files:
    m = fpat.match(f)
    if not m:
        continue
    rows.append(
        {
            "filename": f,
            "site": m.group(2),
            "year": int(m.group(3)),
            "month": int(m.group(4)),
            "day": int(m.group(5)),
            "hour": int(m.group(6)),
        }
    )
sf = pd.DataFrame(rows)
site_counts = sf["site"].value_counts().sort_index()
s05 = sf[sf["site"] == "S05"].copy()
s05.to_csv(OUT / "s05_files.csv", index=False)

# 5) 마크다운 요약
md = f"""# sub_25 — Domain EDA Summary

## 클래스 카탈로그
- **taxonomy.csv 클래스: {len(all_classes)}** (submission target)
- **train.csv 클래스: {len(train_classes)}**
- **labeled soundscape 클래스: {len(ss_classes)}** (in target: {len(in_ss)})
- **학습 데이터 0인 클래스: {len(never_in_train)}** ← 현재 모델 영구 0점

## Prior Tier 분포 (235 클래스)
| tier | 의미 | 수 |
|---|---|---|
| A_in_soundscape | labeled soundscape 등장 | {tier_counts.get('A_in_soundscape', 0)} |
| B_in_pantanal_train | Pantanal 박스 안 학습 데이터에 등장 | {tier_counts.get('B_in_pantanal_train', 0)} |
| C_train_only | 학습 데이터 있으나 Pantanal·soundscape 미등장 | {tier_counts.get('C_train_only', 0)} |
| D_no_train_data | 학습 데이터 0 | {tier_counts.get('D_no_train_data', 0)} |

→ 후처리: A=1.0, B=1.0, C=0.3, D=0.5 같은 prior mask 적용 검토.
   D는 25 Insect sonotype = labeled soundscape에서만 학습 가능.

## train_soundscapes 폴더 메타 ({len(sf)} files)
- 사이트: {len(site_counts)}개
- 사이트 분포 top: {site_counts.head(5).to_dict()}
- **S05 파일: {len(s05)}개** (test와 같은 사이트)
- S05 시각 분포: {s05['hour'].value_counts().sort_index().to_dict()}
- S05 월 분포: {s05['month'].value_counts().sort_index().to_dict()}
- S05 연도 분포: {s05['year'].value_counts().sort_index().to_dict()}

## Pantanal 박스 안 학습 녹음
- 전체 train.csv: {len(train)}
- Pantanal 박스 안: {len(pantanal_train)} ({len(pantanal_train)/len(train)*100:.1f}%)
- Pantanal에서 녹음된 종 수: {len(pantanal_classes)}

## 시사점
1. **A+B = {tier_counts.get('A_in_soundscape', 0) + tier_counts.get('B_in_pantanal_train', 0)}** 종이 test에 진짜 나타날 가능성 큼. 나머지 ~{tier_counts.get('C_train_only', 0)} 종은 false positive 위험.
2. **D 25개 (Insect sonotype)** 는 labeled soundscape (66 files)만이 학습 소스. 무조건 활용 필수.
3. **S05 {len(s05)}개 파일** 은 test와 정확히 같은 사이트. BirdNET pseudo-label 우선순위.
"""
(OUT / "domain_stats.md").write_text(md)
print(md)
print(f"\nWrote: {OUT / 'class_prior.csv'} (tier counts: {tier_counts.to_dict()})")
print(f"Wrote: {OUT / 's05_files.csv'} ({len(s05)} S05 files)")
print(f"Wrote: {OUT / 'domain_stats.md'}")
