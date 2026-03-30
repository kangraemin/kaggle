import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_06/trial_026_high_risk_features"

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

    # EDA 기반 고위험 조합 피처
    df["is_electronic"]      = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_fiber"]           = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_monthly"]         = (df["Contract"] == "Month-to-month").astype(int)
    df["is_senior"]          = df["SeniorCitizen"].astype(int)
    df["is_no_security"]     = (df["OnlineSecurity"] == "No").astype(int)
    df["is_no_support"]      = (df["TechSupport"] == "No").astype(int)

    # 조합
    df["senior_electronic"]  = df["is_senior"] * df["is_electronic"]
    df["fiber_monthly"]      = df["is_fiber"]  * df["is_monthly"]
    df["highest_risk"]       = df["is_monthly"] * df["is_fiber"] * df["is_electronic"]
    df["no_protect_monthly"] = df["is_monthly"] * df["is_no_security"] * df["is_no_support"]

    return df

train = engineer(train); test = engineer(test)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))
feature_names = None; feature_importances = None

# 정규화 강화
params = {"objective":"binary","metric":"auc","learning_rate":0.05,
          "num_leaves":31,           # 64 → 31
          "min_child_samples":50,    # 20 → 50
          "feature_fraction":0.7,
          "bagging_fraction":0.7,
          "bagging_freq":5,
          "lambda_l1":2.0,           # 0.5 → 2.0
          "lambda_l2":2.0,           # 0.5 → 2.0
          "verbose":-1,"random_state":SEED}

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
print("\nTop 10 features:")
for feat,imp in fi.head(10).items():
    print(f"  {feat}: {imp:.1f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_026_high_risk_features.csv",index=False)
json.dump({"id":"026","status":"done","val_score":round(oof_auc,5),
           "notes":"EDA 기반 고위험 조합 피처 + 강한 정규화 (lambda=2.0, num_leaves=31)"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
