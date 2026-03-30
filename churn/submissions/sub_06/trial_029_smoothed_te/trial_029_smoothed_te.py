import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

SEED = 42; N_FOLDS = 5; ALPHA = 10  # smoothing strength
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_06/trial_029_smoothed_te"

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
    df["fiber_monthly"]    = df["is_fiber"] * df["is_monthly"]
    df["senior_electronic"]= df["SeniorCitizen"] * df["is_electronic"]

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))
feature_names = None; feature_importances = None

params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":64,
          "min_child_samples":20,"feature_fraction":0.8,"bagging_fraction":0.8,
          "bagging_freq":5,"lambda_l1":0.5,"lambda_l2":0.5,"verbose":-1,"random_state":SEED}

def smoothed_te(tr_df, val_df, test_df, col, target_col, global_mean, alpha=10):
    """Bayesian smoothed target encoding"""
    stats = tr_df.groupby(col)[target_col].agg(["sum","count"])
    smooth = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
    tr_enc  = tr_df[col].map(smooth).fillna(global_mean).values
    val_enc = val_df[col].map(smooth).fillna(global_mean).values
    te_enc  = test_df[col].map(smooth).fillna(global_mean).values
    return tr_enc, val_enc, te_enc

for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
    tr_df,val_df = train.iloc[tr_idx],train.iloc[val_idx]
    gm = tr_df["target"].mean()
    te_tr,te_val,te_te = {},{},{}
    for col in cat_cols:
        tr_enc, val_enc, te_enc = smoothed_te(tr_df, val_df, test, col, "target", gm, ALPHA)
        te_tr[f"te_{col}"]  = tr_enc
        te_val[f"te_{col}"] = val_enc
        te_te[f"te_{col}"]  = te_enc
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1)
    if feature_names is None:
        feature_names = list(X_tr.columns); feature_importances = np.zeros(len(feature_names))
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_val,y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(200)])
    oof_preds[val_idx] = model.predict_proba(X_val)[:,1]
    test_preds += model.predict_proba(X_te)[:,1] / N_FOLDS
    feature_importances += model.feature_importances_ / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx],oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_029_smoothed_te.csv",index=False)
json.dump({"id":"029","status":"done","val_score":round(oof_auc,5),
           "notes":f"Bayesian smoothed TE (alpha={ALPHA}) + high-risk features"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
