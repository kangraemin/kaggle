import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5; SEEDS = [42,0,1,2,3,4,5]
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_039_cb_reg"

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
    df["senior_electronic"]= df["SeniorCitizen"] * df["is_electronic"]

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
feature_cols = [c for c in train.columns if c not in ["id","Churn","target"]]
cat_indices  = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

X = train[feature_cols].copy(); y = train["target"]
X_test = test[feature_cols].copy()
for col in cat_cols:
    X[col] = X[col].astype(str); X_test[col] = X_test[col].astype(str)

all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y)):
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=5,
            l2_leaf_reg=10.0,          # 강한 정규화
            min_data_in_leaf=50,
            bagging_temperature=0.3,
            eval_metric="AUC", random_seed=seed,
            verbose=False, early_stopping_rounds=50,
            cat_features=cat_indices
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        seed_oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:,1]
        seed_test += model.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"SEED {seed}: {roc_auc_score(y,seed_oof):.5f}")
    all_oof += seed_oof/len(SEEDS); all_test += seed_test/len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_039_cb_reg.csv",index=False)
json.dump({"id":"039","status":"done","val_score":round(oof_auc,5),
           "notes":"CatBoost l2_leaf_reg=10.0 + depth=5 + min_data_in_leaf=50 + 7seeds"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
