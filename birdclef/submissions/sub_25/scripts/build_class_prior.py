"""sub_25 — 234 클래스별 prior mask 생성.

후처리에서 prediction에 곱해서 false positive 줄이기:
  pred_calibrated = pred * prior_mask

Tier별 가중치 (보수적 시작값):
- A_in_soundscape       (75 종) → 1.00  (test에 진짜 등장하는 강한 증거)
- B_in_pantanal_train   (92 종) → 1.00  (Pantanal 박스 안 학습 데이터 있음)
- C_train_only          (67 종) → 0.30  (Pantanal·soundscape 모두 미등장)

산출:
- data/v2/class_prior_mask.npy : (234,) float32
- data/v2/class_prior_mask.csv : tier 정보 포함 디버깅용
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
EDA = ROOT / "submissions" / "sub_25" / "eda"
OUT = ROOT / "data" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

TIER_WEIGHTS = {
    "A_in_soundscape": 1.00,
    "B_in_pantanal_train": 1.00,
    "C_train_only": 0.30,
    "D_no_train_data": 0.50,  # 안 나오지만 안전장치
}

tax = pd.read_csv(DATA / "taxonomy.csv")
prior = pd.read_csv(EDA / "class_prior.csv")

LABELS = sorted(tax["primary_label"].dropna().astype(str).unique())
prior = prior.set_index("primary_label").reindex(LABELS).reset_index()

prior["weight"] = prior["tier"].map(TIER_WEIGHTS).fillna(1.0)
mask = prior["weight"].astype(np.float32).values
assert mask.shape[0] == len(LABELS) == 234, f"shape {mask.shape}"

np.save(OUT / "class_prior_mask.npy", mask)
prior.to_csv(OUT / "class_prior_mask.csv", index=False)

print(f"234 클래스 prior mask 저장: {OUT / 'class_prior_mask.npy'}")
print(f"\ntier 분포 + 가중치:")
print(prior.groupby("tier")["weight"].agg(["count", "mean"]))
print(f"\n전체 mask 통계: mean={mask.mean():.3f} min={mask.min()} max={mask.max()}")
