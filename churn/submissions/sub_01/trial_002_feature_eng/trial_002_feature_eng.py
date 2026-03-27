import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_01/trial_002_feature_eng"

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

train["target"] = (train["Churn"] == "Yes").astype(int)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["TenureGroup"]      = pd.cut(df["tenure"], bins=[0,12,24,48,72,99], labels=[1,2,3,4,5]).astype(float)
    return df

train = add_features(train)
test  = add_features(test)

new_num_cols = ["AvgMonthlyCharge", "ChargeGap", "ChargeRatio", "TenureGroup"]

# ── Features ──────────────────────────────────────────────────────────────────
drop_cols = ["id", "Churn", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols].copy()
y = train["target"]
X_test = test[feature_cols].copy()

# ── CV with target encoding inside fold ──────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importances = np.zeros(len(feature_cols))

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": SEED,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
    X_te = X_test.copy()
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # target encoding (fit on train fold only)
    global_mean = y_tr.mean()
    for col in cat_cols:
        te_map = y_tr.groupby(X_tr[col]).mean()
        X_tr[col]  = X_tr[col].map(te_map).fillna(global_mean)
        X_val[col] = X_val[col].map(te_map).fillna(global_mean)
        X_te[col]  = X_te[col].map(te_map).fillna(global_mean)

    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_te)[:, 1] / N_FOLDS
    feature_importances += model.feature_importances_ / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.Series(feature_importances, index=feature_cols).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

# ── Submission ────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_002_feature_eng.csv", index=False)
print("\nSubmission saved.")

# ── results.json ──────────────────────────────────────────────────────────────
results = {
    "id": "002",
    "status": "done",
    "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("results.json saved.")
