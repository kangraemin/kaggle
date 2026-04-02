"""
Trial 001: LightGBM Baseline
- Label encoding for categoricals
- 5-fold Stratified CV
- Default-ish LightGBM params
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Paths
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# Load data
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Target encoding
target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values

# Features
drop_cols = ["id", "Irrigation_Need"]
features = [c for c in train.columns if c not in drop_cols]
cat_cols = train[features].select_dtypes(include=["object", "string"]).columns.tolist()

# Label encode categoricals
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

X = train[features].values
X_test = test[features].values

# LightGBM params
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# CV
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / N_FOLDS

    fold_acc = accuracy_score(y_val, oof_preds[val_idx].argmax(axis=1))
    fold_scores.append(fold_acc)
    print(f"Fold {fold}: accuracy = {fold_acc:.6f}")

# Overall OOF score
oof_labels = oof_preds.argmax(axis=1)
oof_acc = accuracy_score(y, oof_labels)
print(f"\nOOF accuracy: {oof_acc:.6f}")
print(f"Fold std: {np.std(fold_scores):.6f}")

# Save
np.save(OUT_DIR / "oof_preds.npy", oof_preds)
np.save(OUT_DIR / "test_preds.npy", test_preds)

# Submission
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_preds.argmax(axis=1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

# Results
results = {
    "trial": "trial_001_lgbm_baseline",
    "oof_accuracy": round(oof_acc, 6),
    "fold_scores": [round(s, 6) for s in fold_scores],
    "fold_std": round(np.std(fold_scores), 6),
    "params": params,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Files saved to", OUT_DIR)
