"""Distribution features + All 2-way combos + pseudo labels + 20-fold CV"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import percentileofscore
from itertools import combinations
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 20
TE_FOLDS = 5
SEEDS = [42, 0, 1, 2, 3, 4, 5]
ALPHA = 10
PSEUDO_THRESHOLD = 0.999
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_068_dist_combos_20fold"

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

# === ALL 2-way categorical combinations (C(16,2) = 120) ===
print("=== Building 120 2-way combo columns ===", flush=True)
combo_cols = []
for c1, c2 in combinations(cat_cols, 2):
    col_name = f"combo_{c1[:5]}_{c2[:5]}"
    for df in [train, test, orig]:
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
    combo_cols.append(col_name)

all_cat_cols = cat_cols + combo_cols
print(f"Combo cols: {len(combo_cols)}", flush=True)

# --- Numerical as category (binned for TE) ---
num_as_cat_cols = []
for col in num_cols:
    col_name = f"numcat_{col}"
    train[col_name] = pd.qcut(train[col], q=20, duplicates="drop").astype(str)
    bins = pd.qcut(train[col], q=20, duplicates="drop").cat.categories
    test[col_name] = pd.cut(test[col], bins=bins).astype(str)
    test[col_name] = test[col_name].fillna("nan")
    num_as_cat_cols.append(col_name)

all_cat_cols = all_cat_cols + num_as_cat_cols

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

# Original stats for numerical cols (binned)
for col in num_cols:
    orig[f"{col}_bin"] = pd.qcut(orig[col], q=10, duplicates="drop").astype(str)
    train[f"{col}_bin"] = pd.cut(train[col], bins=pd.qcut(orig[col], q=10, duplicates="drop").cat.categories).astype(str)
    test[f"{col}_bin"] = pd.cut(test[col], bins=pd.qcut(orig[col], q=10, duplicates="drop").cat.categories).astype(str)
    stats = orig.groupby(f"{col}_bin")["target"].mean()
    col_name = f"ORIG_numbin_{col}"
    for df in [train, test]:
        df[col_name] = df[f"{col}_bin"].map(stats).fillna(orig["target"].mean())
    eng_num_cols.append(col_name)

# --- Frequency encoding ---
for col in num_cols:
    freq = train[col].value_counts(normalize=True)
    col_name = f"freq_{col}"
    for df in [train, test]:
        df[col_name] = df[col].map(freq).fillna(0)
    eng_num_cols.append(col_name)

# Remove duplicates
eng_num_cols = list(dict.fromkeys(eng_num_cols))

print(f"Total numerical features: {len(eng_num_cols)}", flush=True)
print(f"Total categorical features for TE: {len(all_cat_cols)}", flush=True)


def target_encode(tr_df, val_df, test_df, col, target_col, alpha=ALPHA):
    """Smoothed target encoding with mean and std"""
    global_mean = tr_df[target_col].mean()
    stats = tr_df.groupby(col)[target_col].agg(["mean", "count", "std"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
    smooth_std = stats["std"].fillna(0)
    return (
        tr_df[col].map(smooth_mean).fillna(global_mean).values,
        val_df[col].map(smooth_mean).fillna(global_mean).values,
        test_df[col].map(smooth_mean).fillna(global_mean).values,
        tr_df[col].map(smooth_std).fillna(0).values,
        val_df[col].map(smooth_std).fillna(0).values,
        test_df[col].map(smooth_std).fillna(0).values,
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

# === Phase 1: Base model for pseudo labels (5-fold, single seed) ===
print("\n=== Phase 1: Base model for pseudo labels ===", flush=True)

base_oof = np.zeros(len(train))
base_test = np.zeros(len(test))

skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf_base.split(train, y)):
    tr_df = train.iloc[tr_idx]
    val_df = train.iloc[val_idx]

    feat_tr, feat_val, feat_te = {}, {}, {}
    for col in all_cat_cols:
        tr_m, val_m, te_m, tr_s, val_s, te_s = target_encode(tr_df, val_df, test, col, "target")
        feat_tr[f"te_{col}"] = tr_m
        feat_val[f"te_{col}"] = val_m
        feat_te[f"te_{col}"] = te_m
        feat_tr[f"te_std_{col}"] = tr_s
        feat_val[f"te_std_{col}"] = val_s
        feat_te[f"te_std_{col}"] = te_s

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
        {**xgb_params, "seed": 42}, dtrain, num_boost_round=3000,
        evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False,
    )
    base_oof[val_idx] = model.predict(dval)
    base_test += model.predict(dtest) / 5

base_auc = roc_auc_score(y, base_oof)
print(f"Base model AUC: {base_auc:.5f}", flush=True)

# === Generate pseudo labels ===
pseudo_pos = base_test >= PSEUDO_THRESHOLD
pseudo_neg = base_test <= (1 - PSEUDO_THRESHOLD)
pseudo_mask = pseudo_pos | pseudo_neg
n_pseudo = pseudo_mask.sum()
print(f"Pseudo labels: {n_pseudo} samples ({pseudo_pos.sum()} pos, {pseudo_neg.sum()} neg)", flush=True)

if n_pseudo > 0:
    pseudo_df = test[pseudo_mask].copy()
    pseudo_df["target"] = (base_test[pseudo_mask] >= 0.5).astype(int)
    train_aug = pd.concat([train, pseudo_df], ignore_index=True)
    y_aug = train_aug["target"]
    print(f"Augmented train: {len(train)} -> {len(train_aug)}", flush=True)
else:
    train_aug = train
    y_aug = y
    print("No pseudo labels generated", flush=True)

# === Phase 2: Final model with 20-fold CV + multi-seed ===
print(f"\n=== Phase 2: Final model {N_FOLDS}-fold x {len(SEEDS)} seeds ===", flush=True)

all_oof = np.zeros(len(train))
all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train))
    seed_test = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        # Training data: train fold + all pseudo labels
        if n_pseudo > 0:
            tr_indices = list(tr_idx) + list(range(len(train), len(train_aug)))
        else:
            tr_indices = list(tr_idx)

        tr_df = train_aug.iloc[tr_indices]
        val_df = train.iloc[val_idx]

        feat_tr, feat_val, feat_te = {}, {}, {}
        for col in all_cat_cols:
            tr_m, val_m, te_m, tr_s, val_s, te_s = target_encode(tr_df, val_df, test, col, "target")
            feat_tr[f"te_{col}"] = tr_m
            feat_val[f"te_{col}"] = val_m
            feat_te[f"te_{col}"] = te_m
            feat_tr[f"te_std_{col}"] = tr_s
            feat_val[f"te_std_{col}"] = val_s
            feat_te[f"te_std_{col}"] = te_s

        X_tr = pd.DataFrame(feat_tr)
        X_tr[eng_num_cols] = tr_df[eng_num_cols].values
        X_val = pd.DataFrame(feat_val)
        X_val[eng_num_cols] = val_df[eng_num_cols].values
        X_te = pd.DataFrame(feat_te)
        X_te[eng_num_cols] = test[eng_num_cols].values

        dtrain = xgb.DMatrix(X_tr, label=train_aug["target"].iloc[tr_indices].values)
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
    f"{OUT_DIR}/trial_068_dist_combos_20fold.csv", index=False)
json.dump({
    "id": "068", "name": "dist_combos_20fold",
    "strategy": "Distribution features + All 2-way combos (120) + pseudo labels (0.999) + digit features + ORIG stats + freq enc + 20-fold CV + XGB 7-seed",
    "val_score": round(oof_auc, 5), "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
