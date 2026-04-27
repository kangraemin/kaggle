"""RepeatedStratifiedKFold (3x5=15 folds) LGBM"""
import pandas as pd, numpy as np, lightgbm as lgb, json, warnings, psutil, os, threading
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

def memory_guard():
    import time
    while True:
        if psutil.virtual_memory().percent > 85:
            print(f"\n⚠️ RAM {psutil.virtual_memory().percent:.1f}% — 종료"); os.kill(os.getpid(), 9)
        time.sleep(5)
threading.Thread(target=memory_guard, daemon=True).start()

ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_056_repeated_kfold"

train = pd.read_csv(f"{DATA_DIR}/train.csv"); test = pd.read_csv(f"{DATA_DIR}/test.csv")
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

params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":31,
          "min_child_samples":50,"feature_fraction":0.7,"bagging_fraction":0.7,
          "bagging_freq":5,"lambda_l1":2.0,"lambda_l2":2.0,"verbose":-1,"random_state":42}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
oof_sum = np.zeros(len(train)); oof_count = np.zeros(len(train))
test_preds = np.zeros(len(test)); n_folds = 0

for fold, (tr_idx, val_idx) in enumerate(rskf.split(train[num_cols], y)):
    tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
    gm = tr_df["target"].mean()
    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        stats = tr_df.groupby(col)["target"].agg(["sum","count"])
        smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
        te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
        te_te[f"te_{col}"]  = test[col].map(smooth).fillna(gm).values
    X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True), pd.DataFrame(te_tr)], axis=1)
    X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test[num_cols].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_val,y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    preds = model.predict_proba(X_val)[:,1]
    oof_sum[val_idx] += preds; oof_count[val_idx] += 1
    test_preds += model.predict_proba(X_te)[:,1]
    n_folds += 1
    if fold % 5 == 4:
        print(f"Fold {fold+1}/15  RAM: {psutil.virtual_memory().percent:.1f}%")

oof_preds = oof_sum / oof_count
test_preds /= n_folds
oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id":test["id"],"Churn":test_preds}).to_csv(f"{OUT_DIR}/trial_056_repeated_kfold.csv",index=False)
json.dump({"id":"056","status":"done","val_score":round(oof_auc,5)},open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
