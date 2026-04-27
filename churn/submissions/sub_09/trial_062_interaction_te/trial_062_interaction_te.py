import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_09/trial_062_interaction_te"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

# Key interaction pairs (domain-driven)
interaction_pairs = [
    ("Contract", "InternetService"),
    ("Contract", "PaymentMethod"),
    ("Contract", "tenure"),          # tenure binned
    ("InternetService", "OnlineSecurity"),
    ("InternetService", "TechSupport"),
    ("InternetService", "MonthlyCharges"),  # MonthlyCharges binned
    ("Contract", "PaperlessBilling"),
    ("Partner", "Dependents"),
    ("InternetService", "StreamingTV"),
    ("InternetService", "StreamingMovies"),
    ("Contract", "OnlineSecurity"),
    ("PaymentMethod", "PaperlessBilling"),
]

# Create interaction columns
def create_interactions(df):
    for c1, c2 in interaction_pairs:
        col1 = df[c1].astype(str)
        # Bin numeric columns
        if c2 == "tenure":
            col2 = pd.cut(df[c2], bins=[0, 12, 24, 48, 72, 999], labels=["0-12","12-24","24-48","48-72","72+"]).astype(str)
        elif c2 == "MonthlyCharges":
            col2 = pd.cut(df[c2], bins=[0, 30, 60, 90, 120], labels=["low","mid","high","vhigh"]).astype(str)
        else:
            col2 = df[c2].astype(str)
        df[f"ix_{c1}_{c2}"] = col1 + "_" + col2
    return df

train = create_interactions(train)
test  = create_interactions(test)

ix_cols = [f"ix_{c1}_{c2}" for c1, c2 in interaction_pairs]
all_cat = cat_cols + ix_cols

y = train["target"]

# XGB best params from trial 051 (best single-model XGB)
best_params = {
    "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0,
    "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "gamma": 0.5,
}

SEEDS = [42, 0, 1, 2, 3, 4, 5]
N_FOLDS = 5

def make_features(tr_df, val_df, test_df):
    gm = tr_df["target"].mean()
    te_tr, te_val, te_te = {}, {}, {}
    for col in all_cat:
        stats = tr_df.groupby(col)["target"].agg(["sum","count"])
        smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
        te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
        te_te[f"te_{col}"]  = test_df[col].map(smooth).fillna(gm).values
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test_df[num_cols].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
    return X_tr, X_val, X_te

all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for tr_idx, val_idx in skf.split(train[num_cols], y):
        X_tr, X_val, X_te = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
        p = {**best_params, "random_state": seed}
        model = xgb.train(p, xgb.DMatrix(X_tr, label=y.iloc[tr_idx]),
                          num_boost_round=1000,
                          evals=[(xgb.DMatrix(X_val, label=y.iloc[val_idx]),"val")],
                          early_stopping_rounds=50, verbose_eval=False)
        seed_oof[val_idx] = model.predict(xgb.DMatrix(X_val))
        seed_test += model.predict(xgb.DMatrix(X_te)) / N_FOLDS
    auc = roc_auc_score(y, seed_oof)
    print(f"SEED {seed}: {auc:.5f}")
    all_oof += seed_oof / len(SEEDS); all_test += seed_test / len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
json.dump({"id": "062", "status": "done", "val_score": round(oof_auc, 5),
           "strategy": "XGB + interaction feature TE + 7 seeds",
           "interaction_pairs": [f"{c1}×{c2}" for c1,c2 in interaction_pairs]},
          open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
