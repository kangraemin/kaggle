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
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_03/trial_006_advanced_features"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

def engineer(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # charge features
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["ChargePerTenure"]  = df["MonthlyCharges"] * df["tenure"]

    # service count (Yes=1, No/No phone/No internet=0)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]
    for col in service_cols:
        df[f"{col}_bin"] = (df[col] == "Yes").astype(int)
    df["num_services"] = df[[f"{c}_bin" for c in service_cols]].sum(axis=1)
    df["has_streaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["has_security"]  = ((df["OnlineSecurity"] == "Yes") | (df["DeviceProtection"] == "Yes")).astype(int)

    # tenure buckets
    df["tenure_bin"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72, 999],
                               labels=[0, 1, 2, 3, 4]).astype(int)

    # high risk combo: fiber optic + month-to-month
    df["fiber_monthly"] = ((df["InternetService"] == "Fiber optic") &
                           (df["Contract"] == "Month-to-month")).astype(int)

    # no internet service flag
    df["no_internet"] = (df["InternetService"] == "No").astype(int)

    # charge per service
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    return df

train = engineer(train)
test  = engineer(test)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

drop_cols = ["id", "Churn", "target"] + cat_cols + [f"{c}_bin" for c in
             ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
              "StreamingTV","StreamingMovies","MultipleLines"]]
num_cols  = [c for c in train.columns if c not in drop_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_names = None
feature_importances = None

# best params from trial_004
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "random_state": SEED,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[num_cols], y)):
    tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
    global_mean = tr_df["target"].mean()

    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        mean_map = tr_df.groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = tr_df[col].map(mean_map).fillna(global_mean).values
        te_val[f"te_{col}"] = val_df[col].map(mean_map).fillna(global_mean).values
        te_te[f"te_{col}"]  = test[col].map(mean_map).fillna(global_mean).values

    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1)

    if feature_names is None:
        feature_names = list(X_tr.columns)
        feature_importances = np.zeros(len(feature_names))

    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y.iloc[tr_idx],
              eval_set=[(X_val, y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds        += model.predict_proba(X_te)[:, 1] / N_FOLDS
    feature_importances += model.feature_importances_ / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

fi = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

# save oof for stacking
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)

sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_006_advanced_features.csv", index=False)

results = {
    "id": "006", "status": "done", "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "advanced features: num_services, tenure_bin, fiber_monthly, charge_per_service + target encoding",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
