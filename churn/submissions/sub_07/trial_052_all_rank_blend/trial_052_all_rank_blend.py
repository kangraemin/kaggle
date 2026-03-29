import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize
import json, os

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_07/trial_052_all_rank_blend"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y = (train["Churn"] == "Yes").astype(int)

# 모든 유효한 OOF
configs = {
    "022_lgbm_multi":  f"{SUB_DIR}/sub_05/trial_022_multi_seed",
    "024_xgb_multi":   f"{SUB_DIR}/sub_05/trial_024_xgb_multiseed",
    "025_cb_multi":    f"{SUB_DIR}/sub_06/trial_025_cb_multiseed",
    "028_lgbm_reg":    f"{SUB_DIR}/sub_06/trial_028_lgbm_reg_multiseed",
    "037_lgbm_reg5":   f"{SUB_DIR}/sub_07/trial_037_lgbm_reg5",
    "038_xgb_reg5":    f"{SUB_DIR}/sub_07/trial_038_xgb_reg5",
    "039_cb_reg":      f"{SUB_DIR}/sub_07/trial_039_cb_reg",
    "042_mlp":         f"{SUB_DIR}/sub_07/trial_042_mlp",
    "049_lr":          f"{SUB_DIR}/sub_07/trial_049_lr_onehot",
}

oof    = {k: np.load(f"{v}/oof_preds.npy") for k,v in configs.items()}
test_p = {k: np.load(f"{v}/test_preds.npy") for k,v in configs.items()}
keys = list(oof.keys())

# rank 변환
oof_rank  = {k: rankdata(v) / len(v) for k,v in oof.items()}
test_rank = {k: rankdata(v) / len(v) for k,v in test_p.items()}

oof_mat  = np.column_stack([oof_rank[k] for k in keys])
test_mat = np.column_stack([test_rank[k] for k in keys])

print("Rank OOF AUCs:")
for k in keys:
    print(f"  {k}: {roc_auc_score(y, oof_rank[k]):.5f}")

def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y, oof_mat @ w)

best_auc, best_w = 0, None
np.random.seed(42)
for _ in range(100):
    w0 = np.random.dirichlet(np.ones(len(keys)))
    res = minimize(neg_auc, w0, method="Nelder-Mead", options={"maxiter":300})
    w = np.abs(res.x) / np.abs(res.x).sum()
    auc = roc_auc_score(y, oof_mat @ w)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"\nBest Rank Blend OOF AUC: {best_auc:.5f}")
for k,w in zip(keys, best_w):
    if w > 0.01: print(f"  {k}: {w:.3f}")

final = test_mat @ best_w
pd.DataFrame({"id":test["id"],"Churn":final}).to_csv(f"{OUT_DIR}/trial_052_all_rank_blend.csv",index=False)
json.dump({"id":"052","status":"done","val_score":round(best_auc,5),
           "weights":{k:float(w) for k,w in zip(keys,best_w)}},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
