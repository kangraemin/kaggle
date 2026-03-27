import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_03/trial_012_catboost_oof"

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
    for col in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]:
        df[col] = df[col].replace("No internet service", "No")
    df["Contract_ord"]    = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})
    df["contract_tenure"] = df["Contract_ord"] * df["tenure"]
    return df

train = engineer(train); test = engineer(test)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
feature_cols = [c for c in train.columns if c not in ["id","Churn","target"]]
cat_indices  = [feature_cols.index(c) for c in cat_cols]

X = train[feature_cols].copy(); y = train["target"]
X_test = test[feature_cols].copy()
for col in cat_cols:
    X[col] = X[col].astype(str); X_test[col] = X_test[col].astype(str)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))

params = {"iterations":1000,"learning_rate":0.05,"depth":6,"eval_metric":"AUC",
          "random_seed":SEED,"verbose":200,"early_stopping_rounds":50,
          "cat_features":cat_indices}

for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
    model = CatBoostClassifier(**params)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:,1]
    test_preds += model.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx],oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_012_catboost_oof.csv",index=False)
json.dump({"id":"012","status":"done","val_score":round(oof_auc,5),
           "notes":"CatBoost + No internet→No + Contract_ord + OOF saved"},
          open(f"{OUT_DIR}/results.json","w"), indent=2)
print("Done.")
