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

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# ── Target ────────────────────────────────────────────────────────────────────
train["target"] = (train["Churn"] == "Yes").astype(int)

# ── Fix TotalCharges (공백 → NaN → median) ────────────────────────────────────
for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ── Feature Engineering ───────────────────────────────────────────────────────
for df in [train, test]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

# ── Categorical cols ──────────────────────────────────────────────────────────
cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

# ── Features ──────────────────────────────────────────────────────────────────
drop_cols = ["id", "Churn", "target"]
num_cols  = [c for c in train.columns if c not in drop_cols + cat_cols]

# ── CV with Target Encoding ───────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

X_base   = train[num_cols].copy()
X_test_base = test[num_cols].copy()
y = train["target"]

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

# accumulate feature importances over folds
all_feature_names = None
feature_importances = None

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

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y)):
    # ── Target Encoding (fit on train fold only) ──────────────────────────────
    tr_df  = train.iloc[tr_idx].copy()
    val_df = train.iloc[val_idx].copy()

    te_maps = {}
    global_mean = tr_df["target"].mean()

    te_train_cols = {}
    te_val_cols   = {}
    te_test_cols  = {}

    for col in cat_cols:
        mean_map = tr_df.groupby(col)["target"].mean()
        te_maps[col] = mean_map

        te_train_cols[f"te_{col}"] = tr_df[col].map(mean_map).fillna(global_mean).values
        te_val_cols[f"te_{col}"]   = val_df[col].map(mean_map).fillna(global_mean).values
        te_test_cols[f"te_{col}"]  = test[col].map(mean_map).fillna(global_mean).values

    te_train = pd.DataFrame(te_train_cols, index=tr_idx)
    te_val   = pd.DataFrame(te_val_cols,   index=val_idx)
    te_test  = pd.DataFrame(te_test_cols)

    X_tr  = pd.concat([X_base.iloc[tr_idx].reset_index(drop=True),  te_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_base.iloc[val_idx].reset_index(drop=True), te_val.reset_index(drop=True)],   axis=1)
    X_te  = pd.concat([X_test_base.reset_index(drop=True),          te_test.reset_index(drop=True)],  axis=1)

    y_tr  = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    if all_feature_names is None:
        all_feature_names = list(X_tr.columns)
        feature_importances = np.zeros(len(all_feature_names))

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
fi = pd.Series(feature_importances, index=all_feature_names).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

# ── Submission ────────────────────────────────────────────────────────────────
OUT_DIR = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_02/trial_002_feature_eng"
sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_002_feature_eng.csv", index=False)
print("\nSubmission saved.")

# ── results.json ──────────────────────────────────────────────────────────────
results = {
    "id": "002",
    "status": "done",
    "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "target encoding (CV-safe) + charge interaction features",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("results.json saved.")
