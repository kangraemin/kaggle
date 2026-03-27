import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_02/trial_004_lgbm_tuned"

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ── Target Encoding helper ────────────────────────────────────────────────────
def make_features(train_df, test_df, tr_idx, val_idx):
    num_cols  = [c for c in train_df.columns
                 if c not in ["id", "Churn", "target"] + cat_cols]
    global_mean = train_df.iloc[tr_idx]["target"].mean()

    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        mean_map = train_df.iloc[tr_idx].groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = train_df.iloc[tr_idx][col].map(mean_map).fillna(global_mean).values
        te_val[f"te_{col}"] = train_df.iloc[val_idx][col].map(mean_map).fillna(global_mean).values
        te_te[f"te_{col}"]  = test_df[col].map(mean_map).fillna(global_mean).values

    X_tr  = pd.concat([train_df.iloc[tr_idx][num_cols].reset_index(drop=True),
                       pd.DataFrame(te_tr)], axis=1)
    X_val = pd.concat([train_df.iloc[val_idx][num_cols].reset_index(drop=True),
                       pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test_df[num_cols].reset_index(drop=True),
                       pd.DataFrame(te_te)], axis=1)
    return X_tr, X_val, X_te

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "verbose": -1,
        "random_state": SEED,
    }

    aucs = []
    for tr_idx, val_idx in skf.split(train, y):
        X_tr, X_val, _ = make_features(train, test, tr_idx, val_idx)
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=500)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    return np.mean(aucs)

print("Running Optuna (50 trials)...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest params: {best_params}")
print(f"Best CV AUC: {study.best_value:.5f}")

# ── Final training with best params ──────────────────────────────────────────
final_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": SEED,
    **best_params,
}

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_names = None
feature_importances = None

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
    X_tr, X_val, X_te = make_features(train, test, tr_idx, val_idx)
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    if feature_names is None:
        feature_names = list(X_tr.columns)
        feature_importances = np.zeros(len(feature_names))

    model = lgb.LGBMClassifier(**final_params, n_estimators=1000)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(200)])

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds        += model.predict_proba(X_te)[:, 1] / N_FOLDS
    feature_importances += model.feature_importances_ / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

fi = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_004_lgbm_tuned.csv", index=False)
print("\nSubmission saved.")

results = {
    "id": "004",
    "status": "done",
    "val_score": round(oof_auc, 5),
    "best_params": best_params,
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "Optuna 50 trials + target encoding + charge features",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("results.json saved.")
