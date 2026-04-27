"""Hill Climbing Ensemble — Chris Deotte 방식
모든 OOF를 rankdata 변환 후 Hill Climbing으로 최적 가중치 탐색.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import glob, json, os

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_075_hill_climbing"

# --- Load ground truth ---
train = pd.read_csv(f"{DATA_DIR}/train.csv")
y = (train["Churn"] == "Yes").astype(int).values

# --- Load all OOF and test predictions ---
oof_files = sorted(glob.glob(f"{DATA_DIR}/submissions/**/oof_preds.npy", recursive=True))
test_files = sorted(glob.glob(f"{DATA_DIR}/submissions/**/test_preds.npy", recursive=True))

# Match OOF with test (same directory)
models = []
for oof_path in oof_files:
    test_path = oof_path.replace("oof_preds.npy", "test_preds.npy")
    if os.path.exists(test_path):
        trial_name = os.path.basename(os.path.dirname(oof_path))
        # Exclude self to avoid circular reference
        if "trial_075" in trial_name:
            continue
        oof = np.load(oof_path)
        test = np.load(test_path)
        auc = roc_auc_score(y, oof)
        models.append({
            "name": trial_name,
            "oof": oof,
            "test": test,
            "auc": auc,
        })

print(f"Loaded {len(models)} models with both OOF and test preds")
print()

# --- Show individual model AUCs ---
models.sort(key=lambda m: m["auc"], reverse=True)
print("Top 15 individual models:")
for i, m in enumerate(models[:15]):
    print(f"  {i+1:2d}. {m['name']:40s} AUC={m['auc']:.5f}")
print()

# --- Rank transform all predictions (Chris Deotte tip) ---
oof_ranked = np.column_stack([rankdata(m["oof"]) for m in models])
test_ranked = np.column_stack([rankdata(m["test"]) for m in models])
n_models = len(models)

# --- Hill Climbing ---
def hill_climb(oof_ranked, y, n_rounds=500):
    """Greedy hill climbing: each round, pick the model that improves AUC the most."""
    n_models = oof_ranked.shape[1]

    # Start with the best single model
    best_single = -1
    best_auc = 0
    for i in range(n_models):
        auc = roc_auc_score(y, oof_ranked[:, i])
        if auc > best_auc:
            best_auc = auc
            best_single = i

    selected = [best_single]
    blend = oof_ranked[:, best_single].copy().astype(float)
    current_auc = best_auc

    print(f"Start: model={models[best_single]['name']}, AUC={current_auc:.5f}")

    no_improve_count = 0
    for round_idx in range(n_rounds):
        best_candidate = -1
        best_candidate_auc = current_auc - 1e-7  # allow tiny tolerance

        for i in range(n_models):
            # Try adding model i
            trial_blend = blend + oof_ranked[:, i]
            trial_auc = roc_auc_score(y, trial_blend)
            if trial_auc > best_candidate_auc:
                best_candidate_auc = trial_auc
                best_candidate = i

        if best_candidate == -1 or best_candidate_auc <= current_auc - 1e-7:
            no_improve_count += 1
            if no_improve_count >= 10:
                print(f"  No improvement for 10 rounds, stopping at round {round_idx+1}")
                break
            continue

        no_improve_count = 0
        selected.append(best_candidate)
        blend += oof_ranked[:, best_candidate]
        current_auc = best_candidate_auc

        if (round_idx + 1) % 10 == 0 or round_idx < 20:
            print(f"  Round {round_idx+1:3d}: +{models[best_candidate]['name']:40s} AUC={current_auc:.5f}")

    # Compute weights from selection counts
    from collections import Counter
    counts = Counter(selected)
    total = len(selected)
    weights = np.zeros(n_models)
    for idx, cnt in counts.items():
        weights[idx] = cnt / total

    return weights, selected, current_auc

print("\n=== Hill Climbing ===")
weights, selected, final_auc = hill_climb(oof_ranked, y, n_rounds=300)

# --- Show final weights ---
print(f"\nFinal ensemble AUC: {final_auc:.5f}")
print(f"Models selected: {len(set(selected))} unique out of {n_models}")
print()
print("Final weights (non-zero):")
weight_info = []
for i in range(n_models):
    if weights[i] > 0:
        print(f"  {models[i]['name']:40s} weight={weights[i]:.4f} (solo AUC={models[i]['auc']:.5f})")
        weight_info.append({"name": models[i]["name"], "weight": float(weights[i]), "auc": float(models[i]["auc"])})

# --- Generate final OOF and test predictions ---
oof_final = np.zeros(len(y), dtype=float)
test_final = np.zeros(len(np.load(models[0]["test"].__class__.__name__) if False else models[0]["test"]), dtype=float)

for i in range(n_models):
    if weights[i] > 0:
        oof_final += weights[i] * rankdata(models[i]["oof"])
        test_final += weights[i] * rankdata(models[i]["test"])

oof_auc = roc_auc_score(y, oof_final)
print(f"\nFinal OOF AUC (rank-weighted): {oof_auc:.5f}")

# --- Save ---
np.save(f"{OUT_DIR}/oof_preds.npy", oof_final)
np.save(f"{OUT_DIR}/test_preds.npy", test_final)

# --- Create submission ---
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sample_sub["Churn"] = test_final
sample_sub.to_csv(f"{OUT_DIR}/submission.csv", index=False)

# --- Save results ---
results = {
    "id": "075",
    "name": "hill_climbing",
    "status": "done",
    "val_score": float(oof_auc),
    "n_models_total": n_models,
    "n_models_selected": len(set(selected)),
    "n_rounds": len(selected),
    "weights": weight_info,
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT_DIR}/")
print("Done!")
