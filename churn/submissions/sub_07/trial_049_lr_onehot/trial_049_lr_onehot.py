import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5; SEEDS = [42,0,1,2,3,4,5]
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_049_lr_onehot"

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
num_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges",
            "AvgMonthlyCharge","ChargeGap","ChargeRatio"]

y = train["target"]

# one-hot encoding (train+test 합쳐서 fit)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(pd.concat([train[cat_cols], test[cat_cols]], ignore_index=True))

X_cat_train = ohe.transform(train[cat_cols])
X_cat_test  = ohe.transform(test[cat_cols])

X_num_train = train[num_cols].values
X_num_test  = test[num_cols].values

X_full_train = np.hstack([X_num_train, X_cat_train])
X_full_test  = np.hstack([X_num_test,  X_cat_test])

print(f"Features: {X_full_train.shape[1]} ({len(num_cols)} num + {X_cat_train.shape[1]} one-hot)")

# alpha 탐색
print("\nAlpha search...")
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_c, best_c_auc = 1.0, 0
for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    aucs = []
    for tr_idx, val_idx in skf3.split(X_full_train, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_full_train[tr_idx])
        X_val = scaler.transform(X_full_train[val_idx])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y.iloc[tr_idx])
        aucs.append(roc_auc_score(y.iloc[val_idx], clf.predict_proba(X_val)[:,1]))
    mean_auc = np.mean(aucs)
    print(f"  C={C}: {mean_auc:.5f}")
    if mean_auc > best_c_auc:
        best_c_auc, best_c = mean_auc, C

print(f"Best C: {best_c} (AUC: {best_c_auc:.5f})")

# multi-seed with best C
all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for fold,(tr_idx,val_idx) in enumerate(skf.split(X_full_train,y)):
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_full_train[tr_idx])
        X_val = scaler.transform(X_full_train[val_idx])
        X_te  = scaler.transform(X_full_test)
        clf = LogisticRegression(C=best_c, max_iter=1000, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y.iloc[tr_idx])
        seed_oof[val_idx] = clf.predict_proba(X_val)[:,1]
        seed_test += clf.predict_proba(X_te)[:,1] / N_FOLDS
    print(f"SEED {seed}: {roc_auc_score(y,seed_oof):.5f}")
    all_oof += seed_oof/len(SEEDS); all_test += seed_test/len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_049_lr_onehot.csv",index=False)
json.dump({"id":"049","status":"done","val_score":round(oof_auc,5),"best_C":best_c,
           "notes":"LR + one-hot encoding + StandardScaler + 7seeds"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
