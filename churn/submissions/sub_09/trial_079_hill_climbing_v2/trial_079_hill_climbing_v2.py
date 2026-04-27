"""Hill Climbing v2 — trial_074 포함, distribution/overfit 모델 제외"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from collections import Counter
import glob, json, os

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_079_hill_climbing_v2"

# Models to exclude (distribution features → overfit, or self)
EXCLUDE = {"trial_067_distribution_digit", "trial_068_dist_combos_20fold",
           "trial_069_knn_stacking", "trial_070_ngram_direct",
           "trial_075_hill_climbing", "trial_079_hill_climbing_v2"}

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
y = (train["Churn"] == "Yes").astype(int).values

# Load all OOF/test pairs
oof_files = sorted(glob.glob(f"{DATA_DIR}/submissions/**/oof_preds.npy", recursive=True))
models = []
for oof_path in oof_files:
    test_path = oof_path.replace("oof_preds.npy", "test_preds.npy")
    if not os.path.exists(test_path):
        continue
    trial_name = os.path.basename(os.path.dirname(oof_path))
    if trial_name in EXCLUDE:
        continue
    oof = np.load(oof_path)
    tst = np.load(test_path)
    if len(oof) != len(y):
        continue
    auc = roc_auc_score(y, oof)
    models.append({"name": trial_name, "oof": oof, "test": tst, "auc": auc})

models.sort(key=lambda m: m["auc"], reverse=True)
print(f"Loaded {len(models)} models (excluded: {', '.join(EXCLUDE)})")
for i, m in enumerate(models[:20]):
    print(f"  {i+1:2d}. {m['name']:45s} AUC={m['auc']:.5f}")

# Rank transform
oof_ranked = np.column_stack([rankdata(m["oof"]) for m in models])
n_models = len(models)

# Hill climbing
def hill_climb(oof_ranked, y, n_rounds=500):
    best_single, best_auc = -1, 0
    for i in range(oof_ranked.shape[1]):
        auc = roc_auc_score(y, oof_ranked[:, i])
        if auc > best_auc:
            best_auc, best_single = auc, i

    selected = [best_single]
    blend = oof_ranked[:, best_single].copy().astype(float)
    current_auc = best_auc
    print(f"\nStart: {models[best_single]['name']}, AUC={current_auc:.5f}")

    no_improve = 0
    for rnd in range(n_rounds):
        best_cand, best_cand_auc = -1, current_auc - 1e-7
        for i in range(oof_ranked.shape[1]):
            trial = blend + oof_ranked[:, i]
            auc = roc_auc_score(y, trial)
            if auc > best_cand_auc:
                best_cand_auc, best_cand = auc, i

        if best_cand == -1 or best_cand_auc <= current_auc - 1e-7:
            no_improve += 1
            if no_improve >= 10:
                print(f"  Stopped at round {rnd+1} (no improvement)")
                break
            continue

        no_improve = 0
        selected.append(best_cand)
        blend += oof_ranked[:, best_cand]
        current_auc = best_cand_auc
        if rnd < 30 or (rnd+1) % 10 == 0:
            print(f"  Round {rnd+1:3d}: +{models[best_cand]['name']:45s} AUC={current_auc:.5f}")

    counts = Counter(selected)
    total = len(selected)
    weights = {idx: cnt/total for idx, cnt in counts.items()}
    return weights, selected, current_auc

print("\n=== Hill Climbing ===")
weights, selected, final_auc = hill_climb(oof_ranked, y, n_rounds=300)

# Results
print(f"\nFinal ensemble AUC: {final_auc:.5f}")
print(f"Models selected: {len(set(selected))} unique")
weight_info = []
for idx in sorted(weights, key=weights.get, reverse=True):
    w = weights[idx]
    print(f"  {models[idx]['name']:45s} weight={w:.4f} (solo={models[idx]['auc']:.5f})")
    weight_info.append({"name": models[idx]["name"], "weight": float(w), "auc": float(models[idx]["auc"])})

# Generate predictions
oof_final = np.zeros(len(y), dtype=float)
test_final = np.zeros(len(models[0]["test"]), dtype=float)
for idx, w in weights.items():
    oof_final += w * rankdata(models[idx]["oof"])
    test_final += w * rankdata(models[idx]["test"])

oof_auc = roc_auc_score(y, oof_final)
print(f"\nFinal OOF AUC (rank-weighted): {oof_auc:.5f}")

# Save
np.save(f"{OUT_DIR}/oof_preds.npy", oof_final)
np.save(f"{OUT_DIR}/test_preds.npy", test_final)
sub = pd.DataFrame({"id": test_df["id"], "Churn": test_final})
sub.to_csv(f"{OUT_DIR}/submission.csv", index=False)

json.dump({
    "id": "079", "name": "hill_climbing_v2",
    "status": "done", "val_score": float(oof_auc),
    "n_models_total": n_models, "n_models_selected": len(set(selected)),
    "n_rounds": len(selected), "weights": weight_info,
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done!")
