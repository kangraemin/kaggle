import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import product
import json, os

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_06/trial_030_mega_blend"
os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = (train["Churn"] == "Yes").astype(int)

configs = {
    "022_lgbm_multi":  (f"{SUB_DIR}/sub_05/trial_022_multi_seed/oof_preds.npy",
                        f"{SUB_DIR}/sub_05/trial_022_multi_seed/test_preds.npy"),
    "024_xgb_multi":   (f"{SUB_DIR}/sub_05/trial_024_xgb_multiseed/oof_preds.npy",
                        f"{SUB_DIR}/sub_05/trial_024_xgb_multiseed/test_preds.npy"),
    "025_cb_multi":    (f"{SUB_DIR}/sub_06/trial_025_cb_multiseed/oof_preds.npy",
                        f"{SUB_DIR}/sub_06/trial_025_cb_multiseed/test_preds.npy"),
    "026_high_risk":   (f"{SUB_DIR}/sub_06/trial_026_high_risk_features/oof_preds.npy",
                        f"{SUB_DIR}/sub_06/trial_026_high_risk_features/test_preds.npy"),
    "028_lgbm_reg":    (f"{SUB_DIR}/sub_06/trial_028_lgbm_reg_multiseed/oof_preds.npy",
                        f"{SUB_DIR}/sub_06/trial_028_lgbm_reg_multiseed/test_preds.npy"),
    "029_smoothed":    (f"{SUB_DIR}/sub_06/trial_029_smoothed_te/oof_preds.npy",
                        f"{SUB_DIR}/sub_06/trial_029_smoothed_te/test_preds.npy"),
}

oof    = {k: np.load(v[0]) for k,v in configs.items()}
test_p = {k: np.load(v[1]) for k,v in configs.items()}

keys = list(oof.keys())
print("Individual OOF AUCs:")
for k in keys:
    print(f"  {k}: {roc_auc_score(y, oof[k]):.5f}")

best_auc, best_w = 0, None
steps = np.arange(0, 1.01, 0.1)
for w in product(steps, repeat=len(keys)):
    if abs(sum(w) - 1.0) > 0.01:
        continue
    auc = roc_auc_score(y, sum(w[i]*oof[keys[i]] for i in range(len(keys))))
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"\nBest OOF AUC: {best_auc:.5f}")
for k,w in zip(keys, best_w):
    if w > 0:
        print(f"  {k}: {w:.1f}")

final_test = sum(best_w[i]*test_p[keys[i]] for i in range(len(keys)))
pd.DataFrame({"id":test["id"],"Churn":final_test}).to_csv(f"{OUT_DIR}/trial_030_mega_blend.csv",index=False)
json.dump({"id":"030","status":"done","val_score":round(best_auc,5),
           "weights":{k:float(w) for k,w in zip(keys,best_w)}},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
