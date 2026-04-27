"""WoE (Weight of Evidence) encoding + LGBM multi-seed"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5; SEEDS = [42,0,1,2,3,4,5]; ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_055_woe"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]
y = train["target"]

def woe_encode(tr_df, val_df, test_df, col, target_col, eps=1e-6):
    """Weight of Evidence encoding"""
    stats = tr_df.groupby(col)[target_col].agg(["sum","count"])
    stats["non_event"] = stats["count"] - stats["sum"]
    total_event = stats["sum"].sum()
    total_non_event = stats["non_event"].sum()
    stats["pct_event"] = stats["sum"] / (total_event + eps)
    stats["pct_non_event"] = stats["non_event"] / (total_non_event + eps)
    stats["woe"] = np.log((stats["pct_non_event"] + eps) / (stats["pct_event"] + eps))
    woe_map = stats["woe"]
    return (tr_df[col].map(woe_map).fillna(0).values,
            val_df[col].map(woe_map).fillna(0).values,
            test_df[col].map(woe_map).fillna(0).values)

params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":31,
          "min_child_samples":50,"feature_fraction":0.7,"bagging_fraction":0.7,
          "bagging_freq":5,"lambda_l1":2.0,"lambda_l2":2.0,"verbose":-1}

all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
        tr_df,val_df = train.iloc[tr_idx],train.iloc[val_idx]
        te_tr,te_val,te_te = {},{},{}
        for col in cat_cols:
            tr_enc,val_enc,te_enc = woe_encode(tr_df, val_df, test, col, "target")
            te_tr[f"woe_{col}"]  = tr_enc
            te_val[f"woe_{col}"] = val_enc
            te_te[f"woe_{col}"]  = te_enc
        X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
        X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
        X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1)
        model = lgb.LGBMClassifier(**params, n_estimators=1000, random_state=seed)
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_val,y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        seed_oof[val_idx] = model.predict_proba(X_val)[:,1]
        seed_test += model.predict_proba(X_te)[:,1] / N_FOLDS
    print(f"SEED {seed}: {roc_auc_score(y,seed_oof):.5f}")
    all_oof += seed_oof/len(SEEDS); all_test += seed_test/len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_055_woe.csv",index=False)
json.dump({"id":"055","status":"done","val_score":round(oof_auc,5)},open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
