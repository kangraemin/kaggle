"""Ridge Ensemble v2 — all clean OOFs + trial_074, Ridge stacking with CV"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
import glob, json, os

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_080_ridge_ensemble_v2"

# Exclude overfit/distribution models and self-references
EXCLUDE = {"trial_067_distribution_digit", "trial_068_dist_combos_20fold",
           "trial_069_knn_stacking", "trial_070_ngram_direct",
           "trial_075_hill_climbing", "trial_079_hill_climbing_v2",
           "trial_078_realmlp_te_std_blend",  # derived from 060+074, avoid double counting
           "trial_080_ridge_ensemble_v2"}

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
print(f"Loaded {len(models)} models")
for i, m in enumerate(models[:20]):
    print(f"  {i+1:2d}. {m['name']:45s} AUC={m['auc']:.5f}")

# Stack OOF predictions as features
X_oof = np.column_stack([m["oof"] for m in models])
X_test = np.column_stack([m["test"] for m in models])
print(f"\nOOF matrix: {X_oof.shape}, Test matrix: {X_test.shape}")

# Ridge stacking with OOF
best_alpha = None
best_auc = 0

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y)):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_oof[tr_idx], y[tr_idx])
        oof_preds[val_idx] = ridge.predict(X_oof[val_idx])

    auc = roc_auc_score(y, oof_preds)
    print(f"  Alpha={alpha:>6.1f}: AUC={auc:.5f}")
    if auc > best_auc:
        best_auc = auc
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}, Best AUC: {best_auc:.5f}")

# Generate final predictions with best alpha
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_oof = np.zeros(len(y))
final_test = np.zeros(X_test.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y)):
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_oof[tr_idx], y[tr_idx])
    final_oof[val_idx] = ridge.predict(X_oof[val_idx])
    final_test += ridge.predict(X_test) / 5

final_auc = roc_auc_score(y, final_oof)
print(f"\nFinal OOF AUC: {final_auc:.5f}")

# Show Ridge coefficients (from full data fit)
ridge_full = Ridge(alpha=best_alpha)
ridge_full.fit(X_oof, y)
coefs = ridge_full.coef_
print("\nTop 10 coefficients:")
sorted_idx = np.argsort(np.abs(coefs))[::-1]
for i in sorted_idx[:10]:
    print(f"  {models[i]['name']:45s} coef={coefs[i]:.4f} (solo={models[i]['auc']:.5f})")

# Save
np.save(f"{OUT_DIR}/oof_preds.npy", final_oof)
np.save(f"{OUT_DIR}/test_preds.npy", final_test)
sub = pd.DataFrame({"id": test_df["id"], "Churn": final_test})
sub.to_csv(f"{OUT_DIR}/submission.csv", index=False)

json.dump({
    "id": "080", "name": "ridge_ensemble_v2",
    "status": "done", "val_score": float(final_auc),
    "best_alpha": float(best_alpha),
    "n_models": len(models),
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done!")
