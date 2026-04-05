"""
Trial 009: Submission Voting Ensemble
- Top submissions의 label-level majority voting
- Nina 방식: if A==B return A, else return C
- 다양한 voting 조합 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys; sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent
SUB_DIR = Path(__file__).resolve().parents[2]

# Load all submissions
subs = {}
paths = {
    "008b": SUB_DIR / "sub_04/trial_008_multiclass_te_fullpair/submission.csv",  # public 0.9721
    "008": SUB_DIR / "sub_08/trial_008_sklearn_multiclass_te/submission.csv",     # public 0.9692
    "003": SUB_DIR / "sub_03/trial_003_balanced_blend/submission.csv",            # public 0.9691
    "007": SUB_DIR / "sub_07/trial_007_bias_tuned_stacking/submission.csv",       # public ?
    "006": SUB_DIR / "sub_06/trial_006_full_pairwise_ensemble/submission.csv",    # public 0.9668
}

for name, path in paths.items():
    if path.exists():
        subs[name] = pd.read_csv(path)
        print(f"Loaded {name}: {path.name} ({len(subs[name])} rows)")
    else:
        print(f"SKIP {name}: {path} not found")

print(f"\nLoaded {len(subs)} submissions")

# Agreement analysis
if len(subs) >= 2:
    keys = list(subs.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            agree = (subs[a]["Irrigation_Need"] == subs[b]["Irrigation_Need"]).mean()
            print(f"  {a} vs {b}: {agree:.4f} agreement")

# ============ Voting Strategies ============

results = {}

# Strategy 1: Simple majority voting (top 3)
if len(subs) >= 3:
    top3 = ["008b", "008", "003"]
    top3_available = [k for k in top3 if k in subs]
    if len(top3_available) >= 3:
        votes = pd.DataFrame({k: subs[k]["Irrigation_Need"] for k in top3_available})
        majority = votes.mode(axis=1)[0]
        sub_vote3 = subs["008b"].copy()
        sub_vote3["Irrigation_Need"] = majority
        sub_vote3.to_csv(OUT_DIR / "submission_vote3.csv", index=False)
        agree_with_best = (majority == subs["008b"]["Irrigation_Need"]).mean()
        print(f"\nVote3 ({', '.join(top3_available)}): {agree_with_best:.4f} agreement with 008b")
        results["vote3"] = {"subs": top3_available, "agree_with_best": round(agree_with_best, 4)}

# Strategy 2: if A==B return A, else return C (Nina style)
if all(k in subs for k in ["008b", "008", "003"]):
    a, b, c = subs["008b"], subs["008"], subs["003"]
    merged = a.copy()
    mask_agree = a["Irrigation_Need"] == b["Irrigation_Need"]
    merged.loc[~mask_agree, "Irrigation_Need"] = c.loc[~mask_agree, "Irrigation_Need"]
    merged.to_csv(OUT_DIR / "submission_nina_abc.csv", index=False)
    changed = (~mask_agree).sum()
    print(f"\nNina (if 008b==008 → 008b, else → 003): {changed} rows changed ({changed/len(merged)*100:.2f}%)")
    results["nina_abc"] = {"changed": int(changed), "pct": round(changed/len(merged)*100, 2)}

# Strategy 3: if A==B return A, else return A (= just A, baseline)
# Skip, this is just 008b

# Strategy 4: All 5 majority voting
if len(subs) >= 5:
    votes = pd.DataFrame({k: subs[k]["Irrigation_Need"] for k in subs.keys()})
    majority = votes.mode(axis=1)[0]
    sub_vote_all = subs["008b"].copy()
    sub_vote_all["Irrigation_Need"] = majority
    sub_vote_all.to_csv(OUT_DIR / "submission_vote_all.csv", index=False)
    agree_with_best = (majority == subs["008b"]["Irrigation_Need"]).mean()
    print(f"\nVote ALL ({len(subs)} subs): {agree_with_best:.4f} agreement with 008b")
    results["vote_all"] = {"n_subs": len(subs), "agree_with_best": round(agree_with_best, 4)}

# Strategy 5: Weighted voting (proportional to public score)
public_scores = {"008b": 0.9721, "008": 0.9692, "003": 0.9691, "006": 0.9668, "007": 0.9700}
available_scored = {k: v for k, v in public_scores.items() if k in subs}
if len(available_scored) >= 3:
    target_map = {"Low": 0, "Medium": 1, "High": 2}
    target_inv = {v: k for k, v in target_map.items()}

    # Convert to numeric
    n_rows = len(subs["008b"])
    weighted_votes = np.zeros((n_rows, 3))
    for name, score in available_scored.items():
        labels = subs[name]["Irrigation_Need"].map(target_map).values
        for cls in range(3):
            weighted_votes[:, cls] += (labels == cls).astype(float) * score

    weighted_preds = weighted_votes.argmax(axis=1)
    sub_weighted = subs["008b"].copy()
    sub_weighted["Irrigation_Need"] = [target_inv[p] for p in weighted_preds]
    sub_weighted.to_csv(OUT_DIR / "submission_weighted.csv", index=False)
    agree_with_best = (sub_weighted["Irrigation_Need"] == subs["008b"]["Irrigation_Need"]).mean()
    print(f"\nWeighted vote: {agree_with_best:.4f} agreement with 008b")
    results["weighted"] = {"agree_with_best": round(agree_with_best, 4)}

# Distribution check
print("\n=== Prediction Distributions ===")
for name, sub in subs.items():
    dist = sub["Irrigation_Need"].value_counts().to_dict()
    print(f"  {name}: {dist}")

import json
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
