"""Optuna XGB on clean feature set (no distribution features) — LB-safe optimization"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 5
SEEDS = [42, 0, 1, 2, 3, 4, 5]
ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_073_optuna_clean"

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

# === DIGIT FEATURES (safe — synthetic data pattern, not distribution-based) ===
print("=== Building digit features ===", flush=True)

for df in [train, test]:
    df["tenure_first_digit"] = df["tenure"].clip(lower=1).astype(int).astype(str).str[0].astype(int)
    df["tenure_last_digit"] = (df["tenure"].astype(int) % 10)
    df["tenure_mod10"] = df["tenure"].astype(int) % 10
    df["tenure_mod12"] = df["tenure"].astype(int) % 12
    df["tenure_is_mult10"] = (df["tenure"].astype(int) % 10 == 0).astype(int)

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
    "tenure_is_mult10",
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

eng_num_cols = list(dict.fromkeys(eng_num_cols))

print(f"Total numerical features: {len(eng_num_cols)}", flush=True)
print(f"Categorical features for TE: {len(cat_cols)}", flush=True)


def target_encode(tr_df, val_df, test_df, col, target_col, alpha=ALPHA):
    global_mean = tr_df[target_col].mean()
    stats = tr_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
    return (
        tr_df[col].map(smooth_mean).fillna(global_mean).values,
        val_df[col].map(smooth_mean).fillna(global_mean).values,
        test_df[col].map(smooth_mean).fillna(global_mean).values,
    )


# =============================================
# Phase 1: Optuna 30 trials (3-fold for speed)
# =============================================
print("\n=== Phase 1: Optuna Hyperparameter Search (30 trials, 3-fold) ===", flush=True)

skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_data = []

for fold, (tr_idx, val_idx) in enumerate(skf_search.split(train, y)):
    tr_df = train.iloc[tr_idx]
    val_df = train.iloc[val_idx]

    feat_tr, feat_val = {}, {}
    for col in cat_cols:
        tr_m, val_m, _ = target_encode(tr_df, val_df, test, col, "target")
        feat_tr[f"te_{col}"] = tr_m
        feat_val[f"te_{col}"] = val_m

    X_tr = pd.DataFrame(feat_tr)
    X_tr[eng_num_cols] = tr_df[eng_num_cols].values
    X_val = pd.DataFrame(feat_val)
    X_val[eng_num_cols] = val_df[eng_num_cols].values

    dtrain = xgb.DMatrix(X_tr, label=y.iloc[tr_idx])
    dval = xgb.DMatrix(X_val, label=y.iloc[val_idx])

    fold_data.append((tr_idx, val_idx, dtrain, dval))

print("Fold data pre-computed.", flush=True)


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }

    oof_preds = np.zeros(len(train))

    for fold, (tr_idx, val_idx, dtrain, dval) in enumerate(fold_data):
        model = xgb.train(
            params, dtrain, num_boost_round=3000,
            evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False,
        )
        oof_preds[val_idx] = model.predict(dval)

    auc = roc_auc_score(y, oof_preds)
    return auc


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
best_optuna_score = study.best_value
print(f"\nOptuna best AUC (3-fold): {best_optuna_score:.5f}", flush=True)
print(f"Best params: {best_params}", flush=True)

json.dump({"best_params": best_params, "best_score_3fold": best_optuna_score},
          open(f"{OUT_DIR}/optuna_best_params.json", "w"), indent=2)

# =============================================
# Phase 2: Multi-seed training with best params (5-fold)
# =============================================
print("\n=== Phase 2: Multi-seed training with best params (5-fold × 7 seeds) ===", flush=True)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "verbosity": 0,
    **best_params,
}

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
            {**xgb_params, "seed": seed}, dtrain, num_boost_round=5000,
            evals=[(dval, "val")], early_stopping_rounds=300, verbose_eval=False,
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
    f"{OUT_DIR}/trial_073_optuna_clean.csv", index=False)
json.dump({
    "id": "073", "name": "optuna_clean",
    "strategy": "Optuna 30 trials XGB on clean feature set (charge+digit+ORIG+TE, no distribution) + 7-seed 5-fold",
    "val_score": round(oof_auc, 5),
    "optuna_best_score_3fold": round(best_optuna_score, 5),
    "best_params": best_params,
    "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
