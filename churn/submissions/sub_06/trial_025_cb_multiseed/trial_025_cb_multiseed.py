import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5
SEEDS = [42, 0, 1, 2, 3, 4, 5]
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_06/trial_025_cb_multiseed"

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
cat_indices  = [feature_cols.index(c) for c in cat_cols]

X = train[feature_cols].copy(); y = train["target"]
X_test = test[feature_cols].copy()
for col in cat_cols:
    X[col] = X[col].astype(str); X_test[col] = X_test[col].astype(str)

all_oof = np.zeros(len(train))
all_test = np.zeros(len(test))

for seed in SEEDS:
    print(f"\n── SEED {seed} ──────────────────")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))

    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=6,
            eval_metric="AUC", random_seed=seed,
            verbose=False, early_stopping_rounds=50,
            cat_features=cat_indices
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        seed_oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:,1]
        seed_test += model.predict_proba(X_test)[:,1] / N_FOLDS

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"  OOF AUC: {seed_auc:.5f}")
    all_oof  += seed_oof  / len(SEEDS)
    all_test += seed_test / len(SEEDS)

final_auc = roc_auc_score(y, all_oof)
print(f"\nCatBoost Multi-seed OOF AUC: {final_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_025_cb_multiseed.csv",index=False)
json.dump({"id":"025","status":"done","val_score":round(final_auc,5),"seeds":SEEDS},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
