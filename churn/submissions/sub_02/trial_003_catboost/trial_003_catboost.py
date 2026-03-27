import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_02/trial_003_catboost"

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# ── Target ────────────────────────────────────────────────────────────────────
train["target"] = (train["Churn"] == "Yes").astype(int)

# ── Fix TotalCharges ──────────────────────────────────────────────────────────
for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ── Feature Engineering ───────────────────────────────────────────────────────
for df in [train, test]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

# ── Features ──────────────────────────────────────────────────────────────────
cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

drop_cols    = ["id", "Churn", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols]

# CatBoost requires cat_features as indices
cat_indices = [feature_cols.index(c) for c in cat_cols]

X      = train[feature_cols].copy()
y      = train["target"]
X_test = test[feature_cols].copy()

# fill cat cols as string
for col in cat_cols:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# ── CV ────────────────────────────────────────────────────────────────────────
skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": SEED,
    "verbose": 200,
    "early_stopping_rounds": 50,
    "cat_features": cat_indices,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr,  X_val = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val = y.iloc[tr_idx],  y.iloc[val_idx]

    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds        += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.Series(model.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

# ── Submission ────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_003_catboost.csv", index=False)
print("\nSubmission saved.")

# ── results.json ──────────────────────────────────────────────────────────────
results = {
    "id": "003",
    "status": "done",
    "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "CatBoost native categorical + charge interaction features",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("results.json saved.")
