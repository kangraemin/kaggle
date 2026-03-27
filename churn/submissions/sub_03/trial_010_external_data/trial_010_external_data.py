import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_03/trial_010_external_data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
orig  = pd.read_csv(f"{DATA_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── 원본 데이터 전처리 → train 형식으로 맞추기 ────────────────────────────────
orig = orig.rename(columns={"customerID": "id"})
orig["id"] = -1  # 원본 데이터 id는 dummy

train["target"] = (train["Churn"] == "Yes").astype(int)
orig["target"]  = (orig["Churn"] == "Yes").astype(int)

# train + 원본 합치기
train_aug = pd.concat([train, orig], ignore_index=True)
print(f"augmented train shape: {train_aug.shape}")
print(f"Churn rate — original train: {train['target'].mean():.4f}, orig: {orig['target'].mean():.4f}, combined: {train_aug['target'].mean():.4f}")

def engineer(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    # "No internet service" → "No" 통합
    inet_services = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                     "TechSupport","StreamingTV","StreamingMovies"]
    for col in inet_services:
        df[col] = df[col].replace("No internet service", "No")
    return df

train     = engineer(train)
train_aug = engineer(train_aug)
test      = engineer(test)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]
num_cols = [c for c in train_aug.columns if c not in ["id","Churn","target"] + cat_cols]

# CV는 원본 train만 (원본 데이터 포함 train_aug로 학습, val은 train만)
train_idx = train_aug.index[train_aug.index < len(train)]
y_aug = train_aug["target"]
y     = train["target"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "random_state": SEED,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[num_cols], y)):
    # train fold + 원본 전체를 학습에 사용
    orig_rows = train_aug.iloc[len(train):]  # 원본 데이터 rows
    tr_rows   = train_aug.iloc[tr_idx]
    tr_combined = pd.concat([tr_rows, orig_rows], ignore_index=True)
    val_rows  = train_aug.iloc[val_idx]

    global_mean = tr_rows["target"].mean()

    te_tr, te_val, te_te = {}, {}, {}
    for col in cat_cols:
        mean_map = tr_rows.groupby(col)["target"].mean()
        te_tr[f"te_{col}"]  = tr_combined[col].map(mean_map).fillna(global_mean).values
        te_val[f"te_{col}"] = val_rows[col].map(mean_map).fillna(global_mean).values
        te_te[f"te_{col}"]  = test[col].map(mean_map).fillna(global_mean).values

    X_tr  = pd.concat([tr_combined[num_cols].reset_index(drop=True), pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_rows[num_cols].reset_index(drop=True),    pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test[num_cols].reset_index(drop=True),        pd.DataFrame(te_te)],  axis=1)
    y_tr  = tr_combined["target"].values

    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds        += model.predict_proba(X_te)[:, 1] / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", oof_preds)
np.save(f"{OUT_DIR}/test_preds.npy", test_preds)

sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv(f"{OUT_DIR}/trial_010_external_data.csv", index=False)

results = {
    "id": "010", "status": "done", "val_score": round(oof_auc, 5),
    "notes": "원본 Telco 데이터(7043행) 추가 + No internet service → No 통합",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
