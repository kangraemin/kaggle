import pandas as pd
import numpy as np
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import psutil, os, threading, json, warnings
warnings.filterwarnings("ignore")

# 메모리 가드
def memory_guard():
    import time
    while True:
        if psutil.virtual_memory().percent > 85:
            print(f"\n⚠️ RAM {psutil.virtual_memory().percent:.1f}% — 종료")
            os.kill(os.getpid(), 9)
        time.sleep(5)
threading.Thread(target=memory_guard, daemon=True).start()

SEED = 42; N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_047_realmlp"

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

feature_cols = [c for c in train.columns if c not in ["id","Churn","target"]]
X = train[feature_cols].copy()
y = train["target"]
X_test = test[feature_cols].copy()

# cat cols to string
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train)); test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}...")

    model = RealMLP_TD_Classifier(
        random_state=SEED,
        verbosity=2,
        val_metric_name='1-auc_ovr',
        n_epochs=3,
        batch_size=256,
        hidden_sizes=[512, 256, 128],
        n_cv=1,
        n_refit=0,
    )

    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold+1} AUC: {fold_auc:.5f}  RAM: {psutil.virtual_memory().percent:.1f}%")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)
pd.DataFrame({"id": test["id"], "Churn": test_preds}).to_csv(f"{OUT_DIR}/trial_047_realmlp.csv", index=False)
json.dump({"id": "047", "status": "done", "val_score": round(oof_auc, 5),
           "notes": "RealMLP (PyTabKit) 5-fold"},
          open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
