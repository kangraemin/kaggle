import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

SEED = 42; N_FOLDS = 5; THRESHOLD = 0.05  # 확신도: <0.05 or >0.95
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_06/trial_034_pseudo_label"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

# 현재 best test 예측값 (030 mega blend)
best_test_preds_path = f"{SUB_DIR}/sub_06/trial_030_mega_blend/trial_030_mega_blend.csv"
pseudo = pd.read_csv(best_test_preds_path)
high_conf = pseudo[(pseudo["Churn"] < THRESHOLD) | (pseudo["Churn"] > 1-THRESHOLD)].copy()
high_conf["pseudo_label"] = (high_conf["Churn"] > 0.5).astype(int)
print(f"Pseudo-label samples: {len(high_conf)} / {len(pseudo)} ({len(high_conf)/len(pseudo)*100:.1f}%)")
print(f"  Churn=1: {high_conf['pseudo_label'].sum()}, Churn=0: {(high_conf['pseudo_label']==0).sum()}")

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

# pseudo label 데이터 준비
test_pseudo = test[test["id"].isin(high_conf["id"])].copy()
test_pseudo["target"] = high_conf.set_index("id")["pseudo_label"].reindex(test_pseudo["id"]).values

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))

params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":64,
          "min_child_samples":20,"feature_fraction":0.8,"bagging_fraction":0.8,
          "bagging_freq":5,"lambda_l1":0.5,"lambda_l2":0.5,"verbose":-1,"random_state":SEED}

for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
    tr_df = pd.concat([train.iloc[tr_idx], test_pseudo], ignore_index=True)
    val_df = train.iloc[val_idx]
    y_tr = tr_df["target"]
    gm = train.iloc[tr_idx]["target"].mean()

    te_tr,te_val,te_te = {},{},{}
    for col in cat_cols:
        mm = train.iloc[tr_idx].groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = tr_df[col].map(mm).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(mm).fillna(gm).values
        te_te[f"te_{col}"]  = test[col].map(mm).fillna(gm).values
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1)
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(200)])
    oof_preds[val_idx] = model.predict_proba(X_val)[:,1]
    test_preds += model.predict_proba(X_te)[:,1] / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx],oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_034_pseudo_label.csv",index=False)
json.dump({"id":"034","status":"done","val_score":round(oof_auc,5),
           "pseudo_samples":len(high_conf),"threshold":THRESHOLD},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
