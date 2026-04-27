"""Calibrated ensemble — Platt scaling on best blend"""
import pandas as pd, numpy as np, json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_07/trial_057_calibrated"

train = pd.read_csv(f"{DATA_DIR}/train.csv"); test = pd.read_csv(f"{DATA_DIR}/test.csv")
y = (train["Churn"] == "Yes").astype(int)

# best blend (024×0.6 + 037×0.4) OOF
oof_024 = np.load(f"{SUB_DIR}/sub_05/trial_024_xgb_multiseed/oof_preds.npy")
oof_037 = np.load(f"{SUB_DIR}/sub_07/trial_037_lgbm_reg5/oof_preds.npy")
test_024 = np.load(f"{SUB_DIR}/sub_05/trial_024_xgb_multiseed/test_preds.npy")
test_037 = np.load(f"{SUB_DIR}/sub_07/trial_037_lgbm_reg5/test_preds.npy")

blend_oof = 0.6 * oof_024 + 0.4 * oof_037
blend_test = 0.6 * test_024 + 0.4 * test_037

print(f"Raw blend OOF AUC: {roc_auc_score(y, blend_oof):.5f}")

# Platt scaling (isotonic 대신 sigmoid — 더 안정적)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cal_oof = np.zeros(len(train)); cal_test = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(blend_oof.reshape(-1,1), y)):
    lr = LogisticRegression(C=1.0)
    lr.fit(blend_oof[tr_idx].reshape(-1,1), y.iloc[tr_idx])
    cal_oof[val_idx] = lr.predict_proba(blend_oof[val_idx].reshape(-1,1))[:,1]
    cal_test += lr.predict_proba(blend_test.reshape(-1,1))[:,1] / 5

cal_auc = roc_auc_score(y, cal_oof)
print(f"Calibrated OOF AUC: {cal_auc:.5f}")

np.save(f"{OUT_DIR}/oof_preds.npy", cal_oof)
np.save(f"{OUT_DIR}/test_preds.npy", cal_test)
pd.DataFrame({"id":test["id"],"Churn":cal_test}).to_csv(f"{OUT_DIR}/trial_057_calibrated.csv",index=False)
json.dump({"id":"057","status":"done","val_score":round(cal_auc,5)},open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
