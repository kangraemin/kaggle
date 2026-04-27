"""Two-stage Ridge → XGB with N-gram features (BlamerX strategy)"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack as sparse_hstack, csr_matrix
from itertools import combinations
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5
N_RIDGE_FOLDS = 20
SEEDS = [42, 0, 1, 2, 3, 4, 5]
ALPHA = 10  # smoothing for TE
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_065_ridge_xgb_ngram"

# --- Load data ---
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
orig = pd.read_csv(f"{DATA_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

train["target"] = (train["Churn"] == "Yes").astype(int)
orig["target"] = (orig["Churn"] == "Yes").astype(int)

for df in [train, test, orig]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

y = train["target"]

# --- Categorical columns ---
cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"]

for df in [train, test, orig]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# --- Feature engineering ---
for df in [train, test, orig]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

eng_num_cols = num_cols + ["AvgMonthlyCharge", "ChargeGap", "ChargeRatio"]

# --- N-gram features (bi-gram + tri-gram from top 6 categoricals) ---
top6_cats = ["Contract", "InternetService", "PaymentMethod",
             "OnlineSecurity", "TechSupport", "PaperlessBilling"]

bigram_cols = []
for c1, c2 in combinations(top6_cats, 2):
    col_name = f"bg_{c1[:4]}_{c2[:4]}"
    for df in [train, test, orig]:
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
    bigram_cols.append(col_name)

trigram_cols = []
for c1, c2, c3 in combinations(top6_cats, 3):
    col_name = f"tg_{c1[:3]}_{c2[:3]}_{c3[:3]}"
    for df in [train, test, orig]:
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
    trigram_cols.append(col_name)

all_cat_cols = cat_cols + bigram_cols + trigram_cols
print(f"Cat cols: {len(cat_cols)} base + {len(bigram_cols)} bigram + {len(trigram_cols)} trigram = {len(all_cat_cols)}", flush=True)

# --- Original data churn probability ---
for col in cat_cols:
    orig_stats = orig.groupby(col)["target"].mean()
    col_name = f"ORIG_{col}"
    for df in [train, test]:
        df[col_name] = df[col].map(orig_stats).fillna(orig["target"].mean())
    eng_num_cols.append(col_name)

# --- Stage 1: Ridge OOF (20-fold, fast) ---
print("\n=== Stage 1: Ridge OOF ===", flush=True)

ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore", min_frequency=5)
ohe.fit(pd.concat([train[all_cat_cols], test[all_cat_cols]], axis=0))

X_train_ohe = ohe.transform(train[all_cat_cols])
X_test_ohe = ohe.transform(test[all_cat_cols])

X_train_ridge = sparse_hstack([X_train_ohe, csr_matrix(train[eng_num_cols].values)])
X_test_ridge = sparse_hstack([X_test_ohe, csr_matrix(test[eng_num_cols].values)])
print(f"Ridge features: {X_train_ridge.shape[1]}", flush=True)

ridge_oof = np.zeros(len(train))
ridge_test = np.zeros(len(test))

skf_ridge = StratifiedKFold(n_splits=N_RIDGE_FOLDS, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf_ridge.split(train, y)):
    ridge = RidgeClassifier(alpha=10.0)
    ridge.fit(X_train_ridge[tr_idx], y.iloc[tr_idx])
    ridge_oof[val_idx] = ridge.decision_function(X_train_ridge[val_idx])
    ridge_test += ridge.decision_function(X_test_ridge) / N_RIDGE_FOLDS

print(f"Ridge OOF AUC: {roc_auc_score(y, ridge_oof):.5f}", flush=True)

# --- Stage 2: XGB with Ridge OOF + simple TE ---
print("\n=== Stage 2: XGB ===", flush=True)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.0063,
    "max_depth": 5,
    "colsample_bytree": 0.32,
    "subsample": 0.8,
    "reg_alpha": 3.5,
    "reg_lambda": 1.3,
    "min_child_weight": 5,
    "tree_method": "hist",
    "verbosity": 0,
}

def simple_te(tr_df, val_df, test_df, col, target_col, alpha=ALPHA):
    """Simple target encoding: use train fold stats for val/test"""
    global_mean = tr_df[target_col].mean()
    stats = tr_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
    return (tr_df[col].map(smooth).fillna(global_mean).values,
            val_df[col].map(smooth).fillna(global_mean).values,
            test_df[col].map(smooth).fillna(global_mean).values)

all_oof = np.zeros(len(train))
all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train))
    seed_test = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        tr_df = train.iloc[tr_idx]
        val_df = train.iloc[val_idx]

        # Target encode all categorical columns
        feat_tr, feat_val, feat_te = {}, {}, {}
        for col in all_cat_cols:
            tr_enc, val_enc, te_enc = simple_te(tr_df, val_df, test, col, "target")
            feat_tr[f"te_{col}"] = tr_enc
            feat_val[f"te_{col}"] = val_enc
            feat_te[f"te_{col}"] = te_enc

        # Build feature matrices
        X_tr = pd.DataFrame(feat_tr)
        X_tr[eng_num_cols] = tr_df[eng_num_cols].values
        X_tr["ridge_oof"] = ridge_oof[tr_idx]

        X_val = pd.DataFrame(feat_val)
        X_val[eng_num_cols] = val_df[eng_num_cols].values
        X_val["ridge_oof"] = ridge_oof[val_idx]

        X_te = pd.DataFrame(feat_te)
        X_te[eng_num_cols] = test[eng_num_cols].values
        X_te["ridge_oof"] = ridge_test

        dtrain = xgb.DMatrix(X_tr, label=y.iloc[tr_idx])
        dval = xgb.DMatrix(X_val, label=y.iloc[val_idx])
        dtest = xgb.DMatrix(X_te)

        model = xgb.train(
            {**xgb_params, "seed": seed},
            dtrain, num_boost_round=5000,
            evals=[(dval, "val")],
            early_stopping_rounds=100, verbose_eval=False,
        )

        seed_oof[val_idx] = model.predict(dval)
        seed_test += model.predict(dtest) / N_FOLDS

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"SEED {seed}: {seed_auc:.5f}", flush=True)
    all_oof += seed_oof / len(SEEDS)
    all_test += seed_test / len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\n=== Final OOF AUC: {oof_auc:.5f} ===", flush=True)

# --- Save ---
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id": test["id"], "Churn": all_test}).to_csv(
    f"{OUT_DIR}/trial_065_ridge_xgb_ngram.csv", index=False)
json.dump({
    "id": "065", "name": "ridge_xgb_ngram",
    "strategy": "Two-stage Ridge→XGB + N-gram features",
    "val_score": round(oof_auc, 5), "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
