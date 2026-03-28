import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_03/trial_016_meta_stacking"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = (train["Churn"] == "Yes").astype(int)

# ── Load all OOFs & test preds ────────────────────────────────────────────────
configs = {
    "lgbm_adv":  (f"{SUB_DIR}/sub_03/trial_006_advanced_features/oof_preds.npy",
                  f"{SUB_DIR}/sub_03/trial_006_advanced_features/test_preds.npy"),
    "xgb":       (f"{SUB_DIR}/sub_03/trial_007_xgboost/oof_preds.npy",
                  f"{SUB_DIR}/sub_03/trial_007_xgboost/test_preds.npy"),
    "cb":        (f"{SUB_DIR}/sub_03/trial_012_catboost_oof/oof_preds.npy",
                  f"{SUB_DIR}/sub_03/trial_012_catboost_oof/test_preds.npy"),
    "lgbm_10f":  (f"{SUB_DIR}/sub_03/trial_013_10fold/oof_preds.npy",
                  f"{SUB_DIR}/sub_03/trial_013_10fold/test_preds.npy"),
}

oof_mat  = np.column_stack([np.load(v[0]) for v in configs.values()])
test_mat = np.column_stack([np.load(v[1]) for v in configs.values()])

print("OOF matrix shape:", oof_mat.shape)
for i, k in enumerate(configs.keys()):
    print(f"  {k}: {roc_auc_score(y, oof_mat[:,i]):.5f}")

# ── Meta-learner: Logistic Regression ────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof  = np.zeros(len(train))
meta_test = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(oof_mat, y)):
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(oof_mat[tr_idx])
    X_val = scaler.transform(oof_mat[val_idx])
    X_te  = scaler.transform(test_mat)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y.iloc[tr_idx])

    meta_oof[val_idx] = clf.predict_proba(X_val)[:,1]
    meta_test += clf.predict_proba(X_te)[:,1] / 5

    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], meta_oof[val_idx]):.5f}")

oof_auc = roc_auc_score(y, meta_oof)
print(f"\nMeta OOF AUC: {oof_auc:.5f}")

pd.DataFrame({"id":test["id"],"Churn":meta_test}).to_csv(f"{OUT_DIR}/trial_016_meta_stacking.csv",index=False)
json.dump({"id":"016","status":"done","val_score":round(oof_auc,5),
           "notes":"LR meta-learner on OOF (lgbm_adv + xgb + catboost + lgbm_10fold)"},
          open(f"{OUT_DIR}/results.json","w"), indent=2)
print("Done.")
