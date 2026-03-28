import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_03/trial_015_xgb_tuned"

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
    return df

train = engineer(train); test = engineer(test)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"]+cat_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def make_features(tr_df, val_df, test_df):
    gm = tr_df["target"].mean()
    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        mm = tr_df.groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = tr_df[col].map(mm).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(mm).fillna(gm).values
        te_te[f"te_{col}"]  = test_df[col].map(mm).fillna(gm).values
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test_df[num_cols].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
    return X_tr, X_val, X_te

def objective(trial):
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "learning_rate": 0.05, "verbosity": 0, "random_state": SEED,
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
    }
    aucs = []
    for tr_idx, val_idx in skf.split(train[num_cols], y):
        X_tr, X_val, _ = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
        dtrain = xgb.DMatrix(X_tr,  label=y.iloc[tr_idx])
        dval   = xgb.DMatrix(X_val, label=y.iloc[val_idx])
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval,"val")], early_stopping_rounds=30,
                          verbose_eval=False)
        aucs.append(roc_auc_score(y.iloc[val_idx], model.predict(dval)))
    return np.mean(aucs)

print("Running Optuna XGB (30 trials)...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = {**study.best_params,
               "objective":"binary:logistic","eval_metric":"auc",
               "learning_rate":0.05,"verbosity":0,"random_state":SEED}
print(f"Best CV AUC: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(train[num_cols], y)):
    X_tr, X_val, X_te = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
    dtrain = xgb.DMatrix(X_tr,  label=y.iloc[tr_idx])
    dval   = xgb.DMatrix(X_val, label=y.iloc[val_idx])
    dte    = xgb.DMatrix(X_te)
    model = xgb.train(best_params, dtrain, num_boost_round=1000,
                      evals=[(dval,"val")], early_stopping_rounds=50,
                      verbose_eval=200)
    oof_preds[val_idx] = model.predict(dval)
    test_preds += model.predict(dte) / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_015_xgb_tuned.csv",index=False)
json.dump({"id":"015","status":"done","val_score":round(oof_auc,5),
           "best_params":study.best_params,"notes":"Optuna 30 trials XGBoost"},
          open(f"{OUT_DIR}/results.json","w"), indent=2)
print("Done.")
