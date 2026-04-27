"""All 2-way categorical combos + conservative pseudo labels (Traiko Dinev strategy)"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from itertools import combinations
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5
SEEDS = [42, 0, 1, 2, 3, 4, 5]
ALPHA = 10  # smoothing for TE
PSEUDO_THRESHOLD = 0.999
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_066_all_combos_pseudo"

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

# --- Feature engineering: charge features ---
for df in [train, test, orig]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

eng_num_cols = num_cols + ["AvgMonthlyCharge", "ChargeGap", "ChargeRatio"]

# --- ALL 2-way categorical combinations (C(16,2) = 120) ---
combo_cols = []
for c1, c2 in combinations(cat_cols, 2):
    col_name = f"combo_{c1[:5]}_{c2[:5]}"
    for df in [train, test, orig]:
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
    combo_cols.append(col_name)

print(f"Combo cols: {len(combo_cols)} (all 2-way from {len(cat_cols)} cats)", flush=True)

all_cat_cols = cat_cols + combo_cols

# --- Original data stats: group-wise churn mean + std ---
for col in cat_cols:
    stats = orig.groupby(col)["target"].agg(["mean", "count"])
    col_mean = f"ORIG_mean_{col}"
    for df in [train, test]:
        df[col_mean] = df[col].map(stats["mean"]).fillna(orig["target"].mean())
    eng_num_cols.append(col_mean)

# Also add original stats for numerical cols (binned)
for col in num_cols:
    orig[f"{col}_bin"] = pd.qcut(orig[col], q=10, duplicates="drop").astype(str)
    train[f"{col}_bin"] = pd.cut(train[col], bins=pd.qcut(orig[col], q=10, duplicates="drop").cat.categories).astype(str)
    test[f"{col}_bin"] = pd.cut(test[col], bins=pd.qcut(orig[col], q=10, duplicates="drop").cat.categories).astype(str)
    stats = orig.groupby(f"{col}_bin")["target"].mean()
    col_name = f"ORIG_numbin_{col}"
    for df in [train, test]:
        df[col_name] = df[f"{col}_bin"].map(stats).fillna(orig["target"].mean())
    eng_num_cols.append(col_name)

# --- Frequency encoding for numerical columns ---
for col in num_cols:
    freq = train[col].value_counts(normalize=True)
    col_name = f"freq_{col}"
    for df in [train, test]:
        df[col_name] = df[col].map(freq).fillna(0)
    eng_num_cols.append(col_name)

# --- Numerical as category (binned → will be TE'd) ---
num_as_cat_cols = []
for col in num_cols:
    col_name = f"numcat_{col}"
    # Use quantile binning into 20 bins
    train[col_name] = pd.qcut(train[col], q=20, duplicates="drop").astype(str)
    # Use same bins for test
    bins = pd.qcut(train[col], q=20, duplicates="drop").cat.categories
    test[col_name] = pd.cut(test[col], bins=bins).astype(str)
    test[col_name] = test[col_name].fillna("nan")
    num_as_cat_cols.append(col_name)

all_cat_cols = all_cat_cols + num_as_cat_cols
print(f"Total cat cols for TE: {len(all_cat_cols)}", flush=True)
print(f"Total num cols: {len(eng_num_cols)}", flush=True)


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

# === Phase 1: Train base model to get pseudo labels ===
print("\n=== Phase 1: Base model for pseudo labels ===", flush=True)

base_oof = np.zeros(len(train))
base_test = np.zeros(len(test))

skf_base = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
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
    base_test += model.predict(dtest) / N_FOLDS

base_auc = roc_auc_score(y, base_oof)
print(f"Base model AUC: {base_auc:.5f}", flush=True)

# === Generate pseudo labels ===
pseudo_pos = base_test >= PSEUDO_THRESHOLD
pseudo_neg = base_test <= (1 - PSEUDO_THRESHOLD)
pseudo_mask = pseudo_pos | pseudo_neg
n_pseudo = pseudo_mask.sum()
print(f"Pseudo labels: {n_pseudo} samples ({pseudo_pos.sum()} pos, {pseudo_neg.sum()} neg) from {len(test)} test", flush=True)

if n_pseudo > 0:
    pseudo_df = test[pseudo_mask].copy()
    pseudo_df["target"] = (base_test[pseudo_mask] >= 0.5).astype(int)
    train_aug = pd.concat([train, pseudo_df], ignore_index=True)
    y_aug = train_aug["target"]
    print(f"Augmented train: {len(train)} → {len(train_aug)}", flush=True)
else:
    train_aug = train
    y_aug = y
    print("No pseudo labels generated (threshold too strict)", flush=True)

# === Phase 2: Final model with pseudo labels + multi-seed ===
print("\n=== Phase 2: Final model with pseudo labels ===", flush=True)

all_oof = np.zeros(len(train))  # OOF only on original train
all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train))
    seed_test = np.zeros(len(test))

    # For OOF: we fold on original train only
    # But training includes pseudo labels from OUTSIDE the fold
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
    f"{OUT_DIR}/trial_066_all_combos_pseudo.csv", index=False)
json.dump({
    "id": "066", "name": "all_combos_pseudo",
    "strategy": "All 2-way categorical combos (120) + conservative pseudo labels (0.999) + 7-seed XGB",
    "val_score": round(oof_auc, 5), "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
