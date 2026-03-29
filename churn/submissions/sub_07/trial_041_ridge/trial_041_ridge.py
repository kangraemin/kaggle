import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

N_FOLDS = 5; SEEDS = [42,0,1,2,3,4,5]; ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_041_ridge"

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
    df["fiber_monthly"]    = df["is_fiber"] * df["is_monthly"]

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]
y = train["target"]

# alpha 탐색 먼저
print("Alpha search (seed=42, 3-fold)...")
best_alpha, best_alpha_auc = 1.0, 0
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    aucs = []
    for tr_idx, val_idx in skf3.split(train[num_cols], y):
        tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
        gm = tr_df["target"].mean()
        te_tr, te_val = {}, {}
        for col in cat_cols:
            stats = tr_df.groupby(col)["target"].agg(["sum","count"])
            smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
            te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
            te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
        X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True), pd.DataFrame(te_tr)], axis=1).values
        X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1).values
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_tr, y.iloc[tr_idx])
        # Ridge decision_function → AUC
        scores = clf.decision_function(X_val)
        aucs.append(roc_auc_score(y.iloc[val_idx], scores))
    mean_auc = np.mean(aucs)
    print(f"  alpha={alpha}: {mean_auc:.5f}")
    if mean_auc > best_alpha_auc:
        best_alpha_auc, best_alpha = mean_auc, alpha

print(f"\nBest alpha: {best_alpha} (AUC: {best_alpha_auc:.5f})")

# 최적 alpha로 multi-seed
all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
        tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
        gm = tr_df["target"].mean()
        te_tr, te_val, te_te = {}, {}, {}
        for col in cat_cols:
            stats = tr_df.groupby(col)["target"].agg(["sum","count"])
            smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
            te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
            te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
            te_te[f"te_{col}"]  = test[col].map(smooth).fillna(gm).values
        X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1).values
        X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1).values
        X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1).values
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te  = scaler.transform(X_te)
        clf = RidgeClassifier(alpha=best_alpha)
        clf.fit(X_tr, y.iloc[tr_idx])
        seed_oof[val_idx] = clf.decision_function(X_val)
        seed_test += clf.decision_function(X_te) / N_FOLDS
    # normalize to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    seed_oof_norm  = MinMaxScaler().fit_transform(seed_oof.reshape(-1,1)).ravel()
    seed_test_norm = MinMaxScaler().fit_transform(seed_test.reshape(-1,1)).ravel()
    print(f"SEED {seed}: {roc_auc_score(y, seed_oof_norm):.5f}")
    all_oof  += seed_oof_norm  / len(SEEDS)
    all_test += seed_test_norm / len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_041_ridge.csv",index=False)
json.dump({"id":"041","status":"done","val_score":round(oof_auc,5),
           "best_alpha":best_alpha,"notes":"RidgeClassifier + smoothed TE + high-risk + 7seeds"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
