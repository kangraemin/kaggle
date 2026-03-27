import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_03/trial_017_final_blend"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = (train["Churn"] == "Yes").astype(int)

oof = {
    "006_lgbm_adv": np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/oof_preds.npy"),
    "007_xgb":      np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/oof_preds.npy"),
    "012_cb":       np.load(f"{SUB_DIR}/sub_03/trial_012_catboost_oof/oof_preds.npy"),
    "013_10fold":   np.load(f"{SUB_DIR}/sub_03/trial_013_10fold/oof_preds.npy"),
    "015_xgb_opt":  np.load(f"{SUB_DIR}/sub_03/trial_015_xgb_tuned/oof_preds.npy"),
}
test_p = {
    "006_lgbm_adv": np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/test_preds.npy"),
    "007_xgb":      np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/test_preds.npy"),
    "012_cb":       np.load(f"{SUB_DIR}/sub_03/trial_012_catboost_oof/test_preds.npy"),
    "013_10fold":   np.load(f"{SUB_DIR}/sub_03/trial_013_10fold/test_preds.npy"),
    "015_xgb_opt":  np.load(f"{SUB_DIR}/sub_03/trial_015_xgb_tuned/test_preds.npy"),
}

keys = list(oof.keys())
for k in keys:
    print(f"  {k}: {roc_auc_score(y, oof[k]):.5f}")

# grid search
best_auc, best_w = 0, None
steps = np.arange(0, 1.01, 0.1)
from itertools import product
for w in product(steps, repeat=len(keys)):
    if abs(sum(w) - 1.0) > 0.01:
        continue
    auc = roc_auc_score(y, sum(w[i]*oof[keys[i]] for i in range(len(keys))))
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"\nBest OOF AUC: {best_auc:.5f}")
for k, w in zip(keys, best_w):
    if w > 0:
        print(f"  {k}: {w:.1f}")

final_test = sum(best_w[i]*test_p[keys[i]] for i in range(len(keys)))
pd.DataFrame({"id":test["id"],"Churn":final_test}).to_csv(f"{OUT_DIR}/trial_017_final_blend.csv",index=False)
json.dump({"id":"017","status":"done","val_score":round(best_auc,5),
           "weights":{k:float(w) for k,w in zip(keys,best_w)},"notes":"6모델 OOF grid search blend"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
