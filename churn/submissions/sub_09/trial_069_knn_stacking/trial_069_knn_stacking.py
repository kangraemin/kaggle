"""KNN probability feature + distribution features + XGB multi-seed"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5
SEEDS = [42, 0, 1, 2, 3, 4, 5]
ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_069_knn_stacking"

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

# --- Basic charge features ---
for df in [train, test, orig]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

eng_num_cols = num_cols + ["AvgMonthlyCharge", "ChargeGap", "ChargeRatio"]

# === DISTRIBUTION FEATURES (from original data) ===
print("=== Building distribution features ===", flush=True)

orig_churn = orig[orig["target"] == 1]
orig_no_churn = orig[orig["target"] == 0]

for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
    churn_vals = orig_churn[col].dropna().values
    no_churn_vals = orig_no_churn[col].dropna().values

    churn_mean = churn_vals.mean()
    churn_std = churn_vals.std() + 1e-8
    no_churn_mean = no_churn_vals.mean()
    no_churn_std = no_churn_vals.std() + 1e-8

    for df in [train, test]:
        df[f"pctrank_churn_{col}"] = df[col].apply(
            lambda x: percentileofscore(churn_vals, x, kind="rank") / 100.0)
        df[f"pctrank_nochurn_{col}"] = df[col].apply(
            lambda x: percentileofscore(no_churn_vals, x, kind="rank") / 100.0)
        df[f"zscore_churn_{col}"] = (df[col] - churn_mean) / churn_std
        df[f"zscore_nochurn_{col}"] = (df[col] - no_churn_mean) / no_churn_std
        df[f"churn_gap_{col}"] = df[f"zscore_nochurn_{col}"].abs() - df[f"zscore_churn_{col}"].abs()

    eng_num_cols.extend([
        f"pctrank_churn_{col}", f"pctrank_nochurn_{col}",
        f"zscore_churn_{col}", f"zscore_nochurn_{col}",
        f"churn_gap_{col}",
    ])

# Conditional percentile rank: within InternetService group
for inet_type in orig["InternetService"].unique():
    orig_sub_churn = orig[(orig["InternetService"] == inet_type) & (orig["target"] == 1)]
    orig_sub_no = orig[(orig["InternetService"] == inet_type) & (orig["target"] == 0)]

    if len(orig_sub_churn) < 5 or len(orig_sub_no) < 5:
        continue

    for col in ["MonthlyCharges", "TotalCharges"]:
        ch_vals = orig_sub_churn[col].dropna().values
        no_vals = orig_sub_no[col].dropna().values
        col_name = f"cond_pctrank_{inet_type[:3]}_{col}"

        for df in [train, test]:
            mask = df["InternetService"] == inet_type
            df[col_name] = 0.5
            if mask.sum() > 0:
                df.loc[mask, col_name] = df.loc[mask, col].apply(
                    lambda x: percentileofscore(ch_vals, x, kind="rank") / 100.0
                ) - df.loc[mask, col].apply(
                    lambda x: percentileofscore(no_vals, x, kind="rank") / 100.0
                )
        eng_num_cols.append(col_name)

# MonthlyCharges residual within InternetService group
inet_means = orig.groupby("InternetService")["MonthlyCharges"].mean()
for df in [train, test]:
    df["monthly_residual"] = df["MonthlyCharges"] - df["InternetService"].map(inet_means).fillna(0)
eng_num_cols.append("monthly_residual")

# === QUANTILE DISTANCE FEATURES ===
print("=== Building quantile distance features ===", flush=True)

for col in ["TotalCharges", "MonthlyCharges"]:
    for q in [0.25, 0.5, 0.75]:
        q_churn = orig_churn[col].quantile(q)
        q_no_churn = orig_no_churn[col].quantile(q)

        for df in [train, test]:
            df[f"qdist_churn_q{int(q*100)}_{col}"] = (df[col] - q_churn).abs()
            df[f"qdist_nochurn_q{int(q*100)}_{col}"] = (df[col] - q_no_churn).abs()
            df[f"qgap_q{int(q*100)}_{col}"] = (
                df[f"qdist_nochurn_q{int(q*100)}_{col}"] - df[f"qdist_churn_q{int(q*100)}_{col}"]
            )

        eng_num_cols.extend([
            f"qdist_churn_q{int(q*100)}_{col}",
            f"qdist_nochurn_q{int(q*100)}_{col}",
            f"qgap_q{int(q*100)}_{col}",
        ])

# === DIGIT FEATURES ===
print("=== Building digit features ===", flush=True)

for df in [train, test]:
    df["tenure_first_digit"] = df["tenure"].clip(lower=1).astype(int).astype(str).str[0].astype(int)
    df["tenure_last_digit"] = (df["tenure"].astype(int) % 10)
    df["tenure_mod10"] = df["tenure"].astype(int) % 10
    df["tenure_mod12"] = df["tenure"].astype(int) % 12
    df["tenure_is_mult10"] = (df["tenure"].astype(int) % 10 == 0).astype(int)
    df["tenure_dev_round10"] = df["tenure"].astype(int) % 10

    df["monthly_first_digit"] = df["MonthlyCharges"].clip(lower=0.1).astype(int).clip(lower=1).astype(str).str[0].astype(int)
    df["monthly_last_digit"] = (df["MonthlyCharges"].astype(int) % 10)
    df["monthly_mod10"] = df["MonthlyCharges"].astype(int) % 10
    df["monthly_frac"] = df["MonthlyCharges"] - df["MonthlyCharges"].astype(int)

    df["total_first_digit"] = df["TotalCharges"].clip(lower=0.1).astype(int).clip(lower=1).astype(str).str[0].astype(int)
    df["total_last_digit"] = (df["TotalCharges"].astype(int) % 10)
    df["total_mod100"] = df["TotalCharges"].astype(int) % 100
    df["total_frac"] = df["TotalCharges"] - df["TotalCharges"].astype(int)

digit_cols = [
    "tenure_first_digit", "tenure_last_digit", "tenure_mod10", "tenure_mod12",
    "tenure_is_mult10", "tenure_dev_round10",
    "monthly_first_digit", "monthly_last_digit", "monthly_mod10", "monthly_frac",
    "total_first_digit", "total_last_digit", "total_mod100", "total_frac",
]
eng_num_cols.extend(digit_cols)

# --- Original data stats ---
for col in cat_cols:
    stats = orig.groupby(col)["target"].agg(["mean", "count"])
    col_mean = f"ORIG_mean_{col}"
    for df in [train, test]:
        df[col_mean] = df[col].map(stats["mean"]).fillna(orig["target"].mean())
    eng_num_cols.append(col_mean)

# Remove duplicates
eng_num_cols = list(dict.fromkeys(eng_num_cols))

# === KNN PROBABILITY FEATURE (OOF) ===
print("=== Building KNN probability feature (OOF) ===", flush=True)

# Use numerical features for KNN
knn_features = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "ChargeGap", "ChargeRatio"]

# Prepare scaled data
scaler = StandardScaler()
train_knn_X = scaler.fit_transform(train[knn_features].values)
test_knn_X = scaler.transform(test[knn_features].values)

# OOF KNN predictions
knn_oof = np.zeros(len(train))
knn_test = np.zeros(len(test))
knn_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(knn_skf.split(train, y)):
    knn = KNeighborsClassifier(n_neighbors=100, metric="minkowski", p=2, n_jobs=-1)
    knn.fit(train_knn_X[tr_idx], y.iloc[tr_idx])
    knn_oof[val_idx] = knn.predict_proba(train_knn_X[val_idx])[:, 1]
    knn_test += knn.predict_proba(test_knn_X)[:, 1] / N_FOLDS

knn_auc = roc_auc_score(y, knn_oof)
print(f"KNN OOF AUC: {knn_auc:.5f}", flush=True)

# Add KNN features
train["knn_prob"] = knn_oof
test["knn_prob"] = knn_test
eng_num_cols.append("knn_prob")

# Also add KNN with different K values for diversity
for k in [20, 50, 200]:
    knn_oof_k = np.zeros(len(train))
    knn_test_k = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(knn_skf.split(train, y)):
        knn_k = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2, n_jobs=-1)
        knn_k.fit(train_knn_X[tr_idx], y.iloc[tr_idx])
        knn_oof_k[val_idx] = knn_k.predict_proba(train_knn_X[val_idx])[:, 1]
        knn_test_k += knn_k.predict_proba(test_knn_X)[:, 1] / N_FOLDS

    print(f"KNN(K={k}) OOF AUC: {roc_auc_score(y, knn_oof_k):.5f}", flush=True)
    train[f"knn_prob_{k}"] = knn_oof_k
    test[f"knn_prob_{k}"] = knn_test_k
    eng_num_cols.append(f"knn_prob_{k}")

print(f"Total numerical features: {len(eng_num_cols)}", flush=True)
print(f"Categorical features for TE: {len(cat_cols)}", flush=True)


def target_encode(tr_df, val_df, test_df, col, target_col, alpha=ALPHA):
    """Smoothed target encoding"""
    global_mean = tr_df[target_col].mean()
    stats = tr_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
    return (
        tr_df[col].map(smooth_mean).fillna(global_mean).values,
        val_df[col].map(smooth_mean).fillna(global_mean).values,
        test_df[col].map(smooth_mean).fillna(global_mean).values,
    )


xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "verbosity": 0,
}

# === Train with multi-seed averaging ===
print("\n=== Training XGB multi-seed ===", flush=True)

all_oof = np.zeros(len(train))
all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train))
    seed_test = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        tr_df = train.iloc[tr_idx]
        val_df = train.iloc[val_idx]

        feat_tr, feat_val, feat_te = {}, {}, {}
        for col in cat_cols:
            tr_m, val_m, te_m = target_encode(tr_df, val_df, test, col, "target")
            feat_tr[f"te_{col}"] = tr_m
            feat_val[f"te_{col}"] = val_m
            feat_te[f"te_{col}"] = te_m

        X_tr = pd.DataFrame(feat_tr)
        X_tr[eng_num_cols] = tr_df[eng_num_cols].values
        X_val = pd.DataFrame(feat_val)
        X_val[eng_num_cols] = val_df[eng_num_cols].values
        X_te = pd.DataFrame(feat_te)
        X_te[eng_num_cols] = test[eng_num_cols].values

        dtrain = xgb.DMatrix(X_tr, label=y.iloc[tr_idx])
        dval = xgb.DMatrix(X_val, label=y.iloc[val_idx])
        dtest = xgb.DMatrix(X_te)

        model = xgb.train(
            {**xgb_params, "seed": seed}, dtrain, num_boost_round=3000,
            evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False,
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
    f"{OUT_DIR}/trial_069_knn_stacking.csv", index=False)
json.dump({
    "id": "069", "name": "knn_stacking",
    "strategy": "KNN probability feature (K=20,50,100,200 OOF) + distribution features + digit + ORIG stats + XGB 7-seed",
    "val_score": round(oof_auc, 5), "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
