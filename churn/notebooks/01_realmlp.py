"""
Kaggle Notebook #1: RealMLP (PyTabKit)
- GPU T4 환경에서 실행
- n_ens=8, 기본 epochs
- OOF + test predictions CSV 저장
"""
import pandas as pd
import numpy as np
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

SEED = 42; N_FOLDS = 5
DATA_DIR = "/kaggle/input/playground-series-s6e3"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

# 피처 엔지니어링 (최소한만)
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

feature_cols = [c for c in train.columns if c not in ["id","Churn","target"]]
X = train[feature_cols].copy()
y = train["target"]
X_test = test[feature_cols].copy()

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Churn rate: {y.mean():.4f}")

# ── RealMLP 5-Fold ────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

start = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

    model = RealMLP_TD_Classifier(
        random_state=SEED + fold,
        verbosity=2,
        val_metric_name='1-auc_ovr',
        n_cv=1,
        n_refit=0,
        device='cuda',
    )

    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    elapsed = (time.time() - start) / 60
    print(f"  Fold {fold+1} AUC: {fold_auc:.5f}  ({elapsed:.1f}min)")

oof_auc = roc_auc_score(y, oof_preds)
total_time = (time.time() - start) / 60
print(f"\n{'='*50}")
print(f"RealMLP OOF AUC: {oof_auc:.5f}  (총 {total_time:.1f}분)")

# ── 저장 ──────────────────────────────────────────────────────────────────────
np.save("/kaggle/working/realmlp_oof.npy", oof_preds)
np.save("/kaggle/working/realmlp_test.npy", test_preds)

sub = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub.to_csv("/kaggle/working/realmlp_submission.csv", index=False)

print(f"\nFiles saved:")
print(f"  realmlp_oof.npy")
print(f"  realmlp_test.npy")
print(f"  realmlp_submission.csv")
print(f"\nDone!")
