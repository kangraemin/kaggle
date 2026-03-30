import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_06/trial_033_catboost_optuna"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["is_electronic"]    = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_fiber"]         = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_monthly"]       = (df["Contract"] == "Month-to-month").astype(int)
    df["highest_risk"]     = df["is_monthly"] * df["is_fiber"] * df["is_electronic"]

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
feature_cols = [c for c in train.columns if c not in ["id","Churn","target"]]
cat_indices  = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

X = train[feature_cols].copy(); y = train["target"]
X_test = test[feature_cols].copy()
for col in cat_cols:
    X[col] = X[col].astype(str); X_test[col] = X_test[col].astype(str)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def objective(trial):
    params = {
        "iterations": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "eval_metric":"AUC", "random_seed":SEED, "verbose":False,
        "early_stopping_rounds":30, "cat_features":cat_indices
    }
    aucs = []
    for tr_idx,val_idx in skf.split(X,y):
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx],y.iloc[val_idx]))
        aucs.append(roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:,1]))
    return np.mean(aucs)

print("CatBoost Optuna (20 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20, show_progress_bar=True)
print(f"Best CV: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

best_p = {**study.best_params, "iterations":1000, "eval_metric":"AUC",
          "random_seed":SEED, "verbose":False, "early_stopping_rounds":50,
          "cat_features":cat_indices}

oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))
for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
    model = CatBoostClassifier(**best_p)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx],y.iloc[val_idx]))
    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:,1]
    test_preds += model.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx],oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_033_catboost_optuna.csv",index=False)
json.dump({"id":"033","status":"done","val_score":round(oof_auc,5),"best_params":study.best_params},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
