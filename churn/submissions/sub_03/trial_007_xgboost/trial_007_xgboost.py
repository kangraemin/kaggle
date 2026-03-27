import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_03/trial_007_xgboost"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

def engineer(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in service_cols)
    df["tenure_bin"]   = pd.cut(df["tenure"], bins=[0,12,24,48,72,999],
                                labels=[0,1,2,3,4]).astype(int)
    df["fiber_monthly"] = ((df["InternetService"] == "Fiber optic") &
                           (df["Contract"] == "Month-to-month")).astype(int)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    return df

train = engineer(train)
test  = engineer(test)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbosity": 0,
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

    dtrain = xgb.DMatrix(X_tr,  label=y.iloc[tr_idx])
    dval   = xgb.DMatrix(X_val, label=y.iloc[val_idx])
    dte    = xgb.DMatrix(X_te)

    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200,
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds        += model.predict(dte) / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)

sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_007_xgboost.csv", index=False)

results = {
    "id": "007", "status": "done", "val_score": round(oof_auc, 5),
    "notes": "XGBoost + advanced features + target encoding",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
