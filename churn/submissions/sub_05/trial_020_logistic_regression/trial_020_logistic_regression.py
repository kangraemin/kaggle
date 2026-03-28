import pandas as pd
import numpy as np
import psutil, os, sys, threading
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

# ── 메모리 가드 (RAM 85% 초과 시 자동 종료) ──────────────────────────────────
RAM_LIMIT = 0.85
def memory_guard():
    while True:
        usage = psutil.virtual_memory().percent / 100
        if usage > RAM_LIMIT:
            print(f"\n⚠️ RAM {usage*100:.1f}% 초과 — 프로세스 종료", flush=True)
            os.kill(os.getpid(), 9)
        import time; time.sleep(3)

threading.Thread(target=memory_guard, daemon=True).start()

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_05/trial_020_logistic_regression"

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
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

y = train["target"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))

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

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y.iloc[tr_idx])

    oof_preds[val_idx] = clf.predict_proba(X_val)[:,1]
    test_preds += clf.predict_proba(X_te)[:,1] / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx],oof_preds[val_idx]):.5f}  RAM: {psutil.virtual_memory().percent:.1f}%")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_020_logistic_regression.csv",index=False)
json.dump({"id":"020","status":"done","val_score":round(oof_auc,5),
           "notes":"Logistic Regression + target encoding + charge features"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
