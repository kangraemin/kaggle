"""
Trial 014: Ridge/LR Stacking on Multiple Trial OOF Probabilities
- Collect OOF proba from trials 008b, 010_cat, 010_xgb, 011, 012
- Stack as features (N_trials × 3 classes = 15 features)
- Ridge/LR meta-learner with CV
- Threshold optimization
- churn에서 55-OOF Ridge가 best였던 접근 재현
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import sys; sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent
SUB_DIR = Path(__file__).resolve().parents[2]

# ============ Load Labels ============
train = pd.read_csv(DATA_DIR / "train.csv")
target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values
n_train = len(y)  # 630000

print(f"Train size: {n_train}")

# ============ Load OOF + Test Predictions ============

oofs = {}
tests = {}

# Helper: load and reshape to (n_samples, 3) per model
def load_preds(name, oof_path, test_path):
    oof = np.load(oof_path)
    test = np.load(test_path)

    # Handle different shapes
    if oof.ndim == 3:
        # (n_models, n_samples, 3) → average across models
        # Only use 630K rows
        oof_avg = oof.mean(axis=0)
        test_avg = test.mean(axis=0)
        if oof_avg.shape[0] != n_train:
            print(f"  SKIP {name}: oof shape {oof_avg.shape[0]} != {n_train}")
            return
        oofs[name] = oof_avg
        tests[name] = test_avg
    elif oof.ndim == 2 and oof.shape == (n_train, 3):
        oofs[name] = oof
        tests[name] = test
    else:
        print(f"  SKIP {name}: unexpected shape {oof.shape}")
        return

    print(f"  Loaded {name}: oof {oofs[name].shape}, test {tests[name].shape}")

trials = {
    "008b_xgb": ("sub_04/trial_008_multiclass_te_fullpair", None),
    "008b_lgbm": ("sub_04/trial_008_multiclass_te_fullpair", None),
    "010_cat": ("sub_05/trial_010_multiseed_cat", None),
    "010_xgb": ("sub_10/trial_010_multiseed_xgb", None),
    "011": ("sub_11/trial_011_slow_xgb_deeper_trees", None),
    "012": ("sub_12/trial_012_extend_rounds", None),
    "008_sklearn": ("sub_08/trial_008_sklearn_multiclass_te", None),
    "007": ("sub_07/trial_007_bias_tuned_stacking", None),
    "006": ("sub_06/trial_006_full_pairwise_ensemble", None),
}

for name, (sub_path, _) in trials.items():
    oof_path = SUB_DIR / sub_path / "oof_preds.npy"
    test_path = SUB_DIR / sub_path / "test_preds.npy"
    if oof_path.exists() and test_path.exists():
        load_preds(name, oof_path, test_path)
    else:
        print(f"  SKIP {name}: files not found")

print(f"\nLoaded {len(oofs)} OOF predictions")

# ============ Build Stacking Features ============
# Each OOF is (630000, 3) → flatten to 3 features per trial

oof_names = sorted(oofs.keys())
X_stack = np.hstack([oofs[name] for name in oof_names])  # (630000, n_trials*3)
X_test_stack = np.hstack([tests[name] for name in oof_names])

print(f"Stacking features: {X_stack.shape[1]} ({len(oof_names)} trials × 3 classes)")
print(f"Trials used: {oof_names}")

# ============ Meta-Learner CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Try multiple meta-learners
results_meta = {}

for meta_name, meta_cls in [
    ("Ridge_1", RidgeClassifier(alpha=1.0)),
    ("Ridge_10", RidgeClassifier(alpha=10.0)),
    ("Ridge_100", RidgeClassifier(alpha=100.0)),
    ("LR_C1", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ("LR_C10", LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs")),
]:
    oof_meta = np.zeros(n_train)
    oof_meta_proba = np.zeros((n_train, 3)) if hasattr(meta_cls, "predict_proba") else None
    test_meta_proba = np.zeros((len(X_test_stack), 3)) if hasattr(meta_cls, "predict_proba") else None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_stack, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_stack[tr_idx])
        X_val = scaler.transform(X_stack[val_idx])
        X_te = scaler.transform(X_test_stack)

        meta_cls.fit(X_tr, y[tr_idx])
        oof_meta[val_idx] = meta_cls.predict(X_val)

        if oof_meta_proba is not None:
            oof_meta_proba[val_idx] = meta_cls.predict_proba(X_val)
            test_meta_proba += meta_cls.predict_proba(X_te) / N_FOLDS

    bal_acc = balanced_accuracy_score(y, oof_meta)
    results_meta[meta_name] = {
        "bal_acc": bal_acc,
        "oof_proba": oof_meta_proba,
        "test_proba": test_meta_proba,
    }
    print(f"  {meta_name}: bal_acc = {bal_acc:.6f}")

# ============ Best Meta-Learner ============

best_meta = max(results_meta, key=lambda k: results_meta[k]["bal_acc"])
best_bal_acc = results_meta[best_meta]["bal_acc"]
print(f"\nBest meta-learner: {best_meta} ({best_bal_acc:.6f})")

# ============ Threshold Optimization ============

# Use proba if available, else use best single trial for threshold
if results_meta[best_meta]["oof_proba"] is not None:
    oof_ens = results_meta[best_meta]["oof_proba"]
    test_ens = results_meta[best_meta]["test_proba"]
else:
    # Ridge doesn't have predict_proba, use simple average
    oof_ens = X_stack.reshape(n_train, len(oof_names), 3).mean(axis=1)
    test_ens = X_test_stack.reshape(-1, len(oof_names), 3).mean(axis=1)

# Also try simple proba average + threshold
oof_avg = X_stack.reshape(n_train, len(oof_names), 3).mean(axis=1)
test_avg = X_test_stack.reshape(-1, len(oof_names), 3).mean(axis=1)
avg_bal_acc = balanced_accuracy_score(y, oof_avg.argmax(1))
print(f"Simple proba average: bal_acc = {avg_bal_acc:.6f}")

# Threshold on proba average
best_threshold_acc = avg_bal_acc
best_class_w = (1.0, 1.0, 1.0)
best_oof_source = "avg"
best_test_source = test_avg

for source_name, oof_src, test_src in [("avg", oof_avg, test_avg), ("meta", oof_ens, test_ens)]:
    if oof_src is None:
        continue
    for w_low in np.arange(0.5, 1.5, 0.05):
        for w_med in np.arange(0.3, 1.3, 0.05):
            for w_high in np.arange(1.0, 6.0, 0.2):
                adjusted = oof_src.copy()
                adjusted[:, 0] *= w_low
                adjusted[:, 1] *= w_med
                adjusted[:, 2] *= w_high
                bal_acc = balanced_accuracy_score(y, adjusted.argmax(1))
                if bal_acc > best_threshold_acc:
                    best_threshold_acc = bal_acc
                    best_class_w = (w_low, w_med, w_high)
                    best_oof_source = source_name
                    best_test_source = test_src

print(f"\nBest threshold: {best_class_w} (source: {best_oof_source})")
print(f"After threshold: {best_threshold_acc:.6f} (was {avg_bal_acc:.6f})")

# Apply
test_adjusted = best_test_source.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

# Also save without threshold
sub_no = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_no["Irrigation_Need"] = [target_inv[p] for p in test_avg.argmax(1)]
sub_no.to_csv(OUT_DIR / "submission_avg.csv", index=False)

results = {
    "trial": "trial_014_ridge_stacking",
    "metric": "balanced_accuracy",
    "trials_used": oof_names,
    "n_stacking_features": X_stack.shape[1],
    "meta_results": {k: round(v["bal_acc"], 6) for k, v in results_meta.items()},
    "best_meta": best_meta,
    "best_meta_bal_acc": round(best_bal_acc, 6),
    "simple_avg_bal_acc": round(avg_bal_acc, 6),
    "threshold_bal_acc": round(best_threshold_acc, 6),
    "threshold_source": best_oof_source,
    "threshold_class_weights": [round(w, 3) for w in best_class_w],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {OUT_DIR}")
print(f"Dist: {pd.Series([target_inv[p] for p in test_adjusted.argmax(1)]).value_counts().to_dict()}")
