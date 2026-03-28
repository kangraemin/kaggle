import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_05/trial_019_orig_as_cols"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
orig  = pd.read_csv(f"{DATA_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

train["target"] = (train["Churn"] == "Yes").astype(int)
orig["target"]  = (orig["Churn"] == "Yes").astype(int)

def engineer(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    return df

train = engineer(train)
test  = engineer(test)
orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)

# ── 원본 데이터를 컬럼으로 추가 (Chris Deotte 방식) ─────────────────────────
# groupby 키별 churn rate → train/test에 merge
merge_keys = [
    ["Contract"],
    ["InternetService"],
    ["PaymentMethod"],
    ["Contract", "InternetService"],
    ["Contract", "PaymentMethod"],
    ["InternetService", "PaymentMethod"],
    ["tenure"],
    ["SeniorCitizen"],
]

for keys in merge_keys:
    col_name = "orig_churn_" + "_".join(keys)
    grp = orig.groupby(keys)["target"].mean().reset_index()
    grp.columns = keys + [col_name]
    train = train.merge(grp, on=keys, how="left")
    test  = test.merge(grp, on=keys, how="left")

orig_feat_cols = [c for c in train.columns if c.startswith("orig_churn_")]
print(f"원본 데이터 컬럼 {len(orig_feat_cols)}개 추가: {orig_feat_cols}")

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

for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
    tr_df,val_df = train.iloc[tr_idx],train.iloc[val_idx]
    gm = tr_df["target"].mean()
    te_tr,te_val,te_te = {},{},{}
    for col in cat_cols:
        mm = tr_df.groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = tr_df[col].map(mm).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(mm).fillna(gm).values
        te_te[f"te_{col}"]  = test[col].map(mm).fillna(gm).values
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

fi = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
print("\nTop 15 features:")
for feat,imp in fi.head(15).items():
    print(f"  {feat}: {imp:.1f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_019_orig_as_cols.csv",index=False)
json.dump({"id":"019","status":"done","val_score":round(oof_auc,5),
           "notes":"원본 데이터 groupby mean → 컬럼으로 merge (Chris Deotte 방식)"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
