"""TE std enriched + Optuna params (trial_073): low lr + narrow colsample + gamma"""
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
ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_081_te_std_optuna_params"

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

# --- Charge features ---
for df in [train, test, orig]:
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

eng_num_cols = num_cols + ["AvgMonthlyCharge", "ChargeGap", "ChargeRatio"]

# === DIGIT FEATURES (from trial_067, safe for synthetic data) ===
print("=== Building digit features ===", flush=True)

for df in [train, test]:
    df["tenure_first_digit"] = df["tenure"].clip(lower=1).astype(int).astype(str).str[0].astype(int)
    df["tenure_last_digit"] = (df["tenure"].astype(int) % 10)
    df["tenure_mod12"] = df["tenure"].astype(int) % 12
    df["tenure_is_mult10"] = (df["tenure"].astype(int) % 10 == 0).astype(int)

    df["monthly_first_digit"] = df["MonthlyCharges"].clip(lower=0.1).astype(int).clip(lower=1).astype(str).str[0].astype(int)
    df["monthly_last_digit"] = (df["MonthlyCharges"].astype(int) % 10)
    df["monthly_frac"] = df["MonthlyCharges"] - df["MonthlyCharges"].astype(int)

    df["total_first_digit"] = df["TotalCharges"].clip(lower=0.1).astype(int).clip(lower=1).astype(str).str[0].astype(int)
    df["total_last_digit"] = (df["TotalCharges"].astype(int) % 10)
    df["total_frac"] = df["TotalCharges"] - df["TotalCharges"].astype(int)

digit_cols = [
    "tenure_first_digit", "tenure_last_digit", "tenure_mod12", "tenure_is_mult10",
    "monthly_first_digit", "monthly_last_digit", "monthly_frac",
    "total_first_digit", "total_last_digit", "total_frac",
]
eng_num_cols.extend(digit_cols)

# === ALL 2-way categorical combinations (C(16,2) = 120) ===
combo_cols = []
for c1, c2 in combinations(cat_cols, 2):
    col_name = f"combo_{c1[:5]}_{c2[:5]}"
    for df in [train, test, orig]:
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
    combo_cols.append(col_name)

print(f"Combo cols: {len(combo_cols)}", flush=True)

# --- Original data stats (mean + std) ---
for col in cat_cols:
    stats = orig.groupby(col)["target"].agg(["mean", "std"])
    for df in [train, test]:
        df[f"ORIG_mean_{col}"] = df[col].map(stats["mean"]).fillna(orig["target"].mean())
        df[f"ORIG_std_{col}"] = df[col].map(stats["std"]).fillna(0)
    eng_num_cols.extend([f"ORIG_mean_{col}", f"ORIG_std_{col}"])

# --- Frequency encoding for numerical cols ---
for col in num_cols:
    freq = train[col].value_counts(normalize=True)
    col_name = f"freq_{col}"
    for df in [train, test]:
        df[col_name] = df[col].map(freq).fillna(0)
    eng_num_cols.append(col_name)

# --- Numericals as categorical (for TE) ---
num_as_cat_cols = []
for col in num_cols:
    col_name = f"numcat_{col}"
    train[col_name] = pd.qcut(train[col], q=20, duplicates="drop").astype(str)
    bins = pd.qcut(train[col], q=20, duplicates="drop").cat.categories
    test[col_name] = pd.cut(test[col], bins=bins).astype(str)
    test[col_name] = test[col_name].fillna("nan")
    num_as_cat_cols.append(col_name)

all_cat_cols = cat_cols + combo_cols + num_as_cat_cols
eng_num_cols = list(dict.fromkeys(eng_num_cols))

print(f"Total cat cols for TE: {len(all_cat_cols)}", flush=True)
print(f"Total num cols: {len(eng_num_cols)}", flush=True)


def target_encode(tr_df, val_df, test_df, col, target_col, alpha=ALPHA):
    """Smoothed target encoding returning mean + std"""
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


# XGB params from trial_073 Optuna best (low lr + narrow colsample — BlamerX-like)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.0124,
    "max_depth": 6,
    "colsample_bytree": 0.38,
    "subsample": 0.90,
    "reg_alpha": 0.104,
    "reg_lambda": 0.065,
    "min_child_weight": 8,
    "gamma": 0.613,
    "tree_method": "hist",
    "verbosity": 0,
}

# === Main training: multi-seed ===
print("\n=== Training: multi-seed XGB ===", flush=True)

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
            {**xgb_params, "seed": seed}, dtrain, num_boost_round=10000,
            evals=[(dval, "val")], early_stopping_rounds=200, verbose_eval=False,
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
    f"{OUT_DIR}/submission.csv", index=False)
json.dump({
    "id": "081", "name": "te_std_optuna_params",
    "strategy": "trial_074 features + Optuna params (lr=0.012, colsample=0.38, gamma=0.61) + 10k rounds + 7-seed",
    "val_score": round(oof_auc, 5),
    "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
