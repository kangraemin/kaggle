import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42; N_FOLDS = 5; ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_051_xgb_wide_optuna"

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
y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def make_features(tr_df, val_df, test_df):
    gm = tr_df["target"].mean()
    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        stats = tr_df.groupby(col)["target"].agg(["sum","count"])
        smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
        te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
        te_te[f"te_{col}"]  = test_df[col].map(smooth).fillna(gm).values
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test_df[num_cols].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
    return X_tr, X_val, X_te

def objective(trial):
    params = {
        "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "max_delta_step":   trial.suggest_int("max_delta_step", 0, 10),
        "grow_policy":      trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "random_state": SEED,
    }
    aucs = []
    for tr_idx, val_idx in skf.split(train[num_cols], y):
        X_tr, X_val, _ = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
        model = xgb.train(params, xgb.DMatrix(X_tr, label=y.iloc[tr_idx]),
                          num_boost_round=500,
                          evals=[(xgb.DMatrix(X_val, label=y.iloc[val_idx]),"val")],
                          early_stopping_rounds=30, verbose_eval=False)
        aucs.append(roc_auc_score(y.iloc[val_idx], model.predict(xgb.DMatrix(X_val))))
    return np.mean(aucs)

print("XGB Wide Optuna (100 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=100, show_progress_bar=True)
print(f"Best CV: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

# final training with best params + 7 seeds
best_p = {**study.best_params, "objective":"binary:logistic","eval_metric":"auc","verbosity":0}
SEEDS = [42,0,1,2,3,4,5]
all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for tr_idx, val_idx in skf2.split(train[num_cols], y):
        X_tr, X_val, X_te = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
        p = {**best_p, "random_state": seed}
        model = xgb.train(p, xgb.DMatrix(X_tr, label=y.iloc[tr_idx]),
                          num_boost_round=1000,
                          evals=[(xgb.DMatrix(X_val, label=y.iloc[val_idx]),"val")],
                          early_stopping_rounds=50, verbose_eval=False)
        seed_oof[val_idx] = model.predict(xgb.DMatrix(X_val))
        seed_test += model.predict(xgb.DMatrix(X_te)) / N_FOLDS
    print(f"SEED {seed}: {roc_auc_score(y,seed_oof):.5f}")
    all_oof += seed_oof/len(SEEDS); all_test += seed_test/len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_051_xgb_wide_optuna.csv",index=False)
json.dump({"id":"051","status":"done","val_score":round(oof_auc,5),
           "best_params":study.best_params},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
