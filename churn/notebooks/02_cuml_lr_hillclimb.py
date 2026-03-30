"""
Kaggle Notebook #2: cuML LR + Hill Climbing Ensemble
- cuML GPU Logistic Regression
- OOF 로드 (Notebook #1 output + 로컬 CSV 업로드)
- Hill Climbing으로 최적 앙상블
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time

SEED = 42; N_FOLDS = 5
DATA_DIR = "/kaggle/input/playground-series-s6e3"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
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
num_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges",
            "AvgMonthlyCharge","ChargeGap","ChargeRatio"]

y = train["target"]

# ── cuML LR ───────────────────────────────────────────────────────────────────
try:
    from cuml.linear_model import LogisticRegression as cuLR
    USE_CUML = True
    print("Using cuML LogisticRegression (GPU)")
except ImportError:
    from sklearn.linear_model import LogisticRegression as cuLR
    USE_CUML = False
    print("cuML not available, using sklearn LogisticRegression")

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(pd.concat([train[cat_cols], test[cat_cols]], ignore_index=True))

X_cat_train = ohe.transform(train[cat_cols])
X_cat_test  = ohe.transform(test[cat_cols])
X_full_train = np.hstack([train[num_cols].values, X_cat_train]).astype(np.float32)
X_full_test  = np.hstack([test[num_cols].values, X_cat_test]).astype(np.float32)

print(f"Features: {X_full_train.shape[1]}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
lr_oof = np.zeros(len(train)); lr_test = np.zeros(len(test))

start = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_full_train[tr_idx])
    X_val = scaler.transform(X_full_train[val_idx])
    X_te  = scaler.transform(X_full_test)

    if USE_CUML:
        clf = cuLR(C=1.0, max_iter=1000)
    else:
        clf = cuLR(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y.iloc[tr_idx].values)

    lr_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    lr_test += clf.predict_proba(X_te)[:, 1] / N_FOLDS

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], lr_oof[val_idx]):.5f}")

lr_auc = roc_auc_score(y, lr_oof)
print(f"\ncuML LR OOF AUC: {lr_auc:.5f}  ({(time.time()-start)/60:.1f}min)")

np.save("/kaggle/working/cuml_lr_oof.npy", lr_oof)
np.save("/kaggle/working/cuml_lr_test.npy", lr_test)

# ── Hill Climbing Ensemble ────────────────────────────────────────────────────
print("\n=== Hill Climbing Ensemble ===")

# 사용 가능한 OOF 로드
# Notebook #1 output (같은 세션이면 /kaggle/working에 있음)
# 또는 dataset으로 업로드한 로컬 OOF
import glob, os

oofs = {"cuml_lr": lr_oof}
tests = {"cuml_lr": lr_test}

# Notebook #1 결과
if os.path.exists("/kaggle/working/realmlp_oof.npy"):
    oofs["realmlp"] = np.load("/kaggle/working/realmlp_oof.npy")
    tests["realmlp"] = np.load("/kaggle/working/realmlp_test.npy")
    print(f"  realmlp loaded: {roc_auc_score(y, oofs['realmlp']):.5f}")

# 로컬에서 업로드한 OOF (dataset으로 추가)
local_dir = "/kaggle/input/churn-local-oofs"
if os.path.exists(local_dir):
    for f in glob.glob(f"{local_dir}/*_oof.npy"):
        name = os.path.basename(f).replace("_oof.npy", "")
        oofs[name] = np.load(f)
        test_f = f.replace("_oof.npy", "_test.npy")
        if os.path.exists(test_f):
            tests[name] = np.load(test_f)
        print(f"  {name} loaded: {roc_auc_score(y, oofs[name]):.5f}")

print(f"\nTotal models: {len(oofs)}")

# Hill Climbing
def hill_climbing(oofs_dict, tests_dict, y, max_models=20):
    keys = list(oofs_dict.keys())
    oof_list = [oofs_dict[k] for k in keys]
    test_list = [tests_dict[k] for k in keys]

    # 가장 좋은 단일 모델로 시작
    best_single = max(range(len(keys)), key=lambda i: roc_auc_score(y, oof_list[i]))
    selected = [best_single]
    blend_oof = oof_list[best_single].copy()
    blend_test = test_list[best_single].copy()
    best_auc = roc_auc_score(y, blend_oof)
    print(f"Start: {keys[best_single]} AUC={best_auc:.5f}")

    for step in range(max_models - 1):
        best_add, best_add_auc = -1, best_auc
        n = len(selected)
        for i in range(len(keys)):
            candidate = (blend_oof * n + oof_list[i]) / (n + 1)
            auc = roc_auc_score(y, candidate)
            if auc > best_add_auc:
                best_add_auc = auc
                best_add = i
        if best_add == -1:
            break
        selected.append(best_add)
        blend_oof = (blend_oof * n + oof_list[best_add]) / (n + 1)
        blend_test = (blend_test * n + test_list[best_add]) / (n + 1)
        best_auc = best_add_auc
        print(f"  +{keys[best_add]} → AUC={best_auc:.5f} (n={len(selected)})")

    return blend_oof, blend_test, best_auc, [keys[i] for i in selected]

blend_oof, blend_test, final_auc, selected = hill_climbing(oofs, tests, y)
print(f"\nFinal ensemble AUC: {final_auc:.5f}")
print(f"Selected models: {selected}")

# 제출 파일
sub = pd.DataFrame({"id": test["id"], "Churn": blend_test})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("\nsubmission.csv saved!")
