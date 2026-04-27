"""RealMLP × trial_074 (TE std XGB) blend — grid search optimal weight"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR = f"{DATA_DIR}/submissions/sub_09/trial_078_realmlp_te_std_blend"

# --- Load data ---
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
y = (train["Churn"] == "Yes").astype(int)

# --- Load OOF predictions ---
realmlp_oof = np.load(f"{DATA_DIR}/submissions/sub_08/trial_060_realmlp_kaggle/oof_preds.npy")
realmlp_test = np.load(f"{DATA_DIR}/submissions/sub_08/trial_060_realmlp_kaggle/test_preds.npy")

xgb074_oof = np.load(f"{DATA_DIR}/submissions/sub_09/trial_074_te_std_enriched/oof_preds.npy")
xgb074_test = np.load(f"{DATA_DIR}/submissions/sub_09/trial_074_te_std_enriched/test_preds.npy")

print(f"RealMLP OOF AUC: {roc_auc_score(y, realmlp_oof):.5f}")
print(f"XGB074 OOF AUC:  {roc_auc_score(y, xgb074_oof):.5f}")

# --- Also load other strong OOFs for potential 3+ model blend ---
other_oofs = {}
other_tests = {}

# trial_024 (xgb_multiseed, CV 0.91690)
try:
    oof = np.load(f"{DATA_DIR}/submissions/sub_05/trial_024_xgb_multiseed/oof_preds.npy")
    tst = np.load(f"{DATA_DIR}/submissions/sub_05/trial_024_xgb_multiseed/test_preds.npy")
    other_oofs["xgb024"] = oof
    other_tests["xgb024"] = tst
    print(f"XGB024 OOF AUC:  {roc_auc_score(y, oof):.5f}")
except:
    pass

# trial_028 (lgbm_reg_multiseed, CV 0.91683)
try:
    oof = np.load(f"{DATA_DIR}/submissions/sub_06/trial_028_lgbm_reg_multiseed/oof_preds.npy")
    tst = np.load(f"{DATA_DIR}/submissions/sub_06/trial_028_lgbm_reg_multiseed/test_preds.npy")
    other_oofs["lgbm028"] = oof
    other_tests["lgbm028"] = tst
    print(f"LGBM028 OOF AUC: {roc_auc_score(y, oof):.5f}")
except:
    pass

# --- Grid search: 2-model blend (RealMLP + XGB074) ---
print("\n=== 2-model blend: RealMLP + XGB074 ===")
best_auc = 0
best_w = 0
for w in np.arange(0.0, 1.01, 0.01):
    blend = realmlp_oof * w + xgb074_oof * (1 - w)
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_w = w
print(f"Best weight: RealMLP×{best_w:.2f} + XGB074×{1-best_w:.2f} = AUC {best_auc:.5f}")

# --- Rank averaging blend ---
print("\n=== Rank averaging: RealMLP + XGB074 ===")
best_rank_auc = 0
best_rank_w = 0
realmlp_rank = rankdata(realmlp_oof)
xgb074_rank = rankdata(xgb074_oof)
for w in np.arange(0.0, 1.01, 0.01):
    blend = realmlp_rank * w + xgb074_rank * (1 - w)
    auc = roc_auc_score(y, blend)
    if auc > best_rank_auc:
        best_rank_auc = auc
        best_rank_w = w
print(f"Best rank weight: RealMLP×{best_rank_w:.2f} + XGB074×{1-best_rank_w:.2f} = AUC {best_rank_auc:.5f}")

# --- 3+ model blend (if OOFs available) ---
if other_oofs:
    print(f"\n=== Multi-model blend ({2 + len(other_oofs)} models) ===")
    all_models = {"realmlp": realmlp_oof, "xgb074": xgb074_oof, **other_oofs}
    all_test_models = {"realmlp": realmlp_test, "xgb074": xgb074_test, **other_tests}
    names = list(all_models.keys())
    oofs = [all_models[n] for n in names]

    # Greedy forward selection with rank averaging
    best_multi_auc = 0
    best_multi_blend = None
    best_multi_weights = None

    # Try all combinations of RealMLP weight with XGB074
    for rw in np.arange(0.5, 0.95, 0.05):
        remaining = 1.0 - rw
        for i, name in enumerate(names):
            if name == "realmlp":
                continue
            for w074 in np.arange(0.0, remaining + 0.01, 0.05):
                w_rest = remaining - w074
                if w_rest < -0.01:
                    continue
                blend = rankdata(realmlp_oof) * rw + rankdata(xgb074_oof) * w074
                for j, n2 in enumerate(names):
                    if n2 in ("realmlp", "xgb074"):
                        continue
                    blend += rankdata(oofs[j]) * w_rest / max(len(other_oofs), 1)
                auc = roc_auc_score(y, blend)
                if auc > best_multi_auc:
                    best_multi_auc = auc

    print(f"Best multi-model rank blend AUC: {best_multi_auc:.5f}")

# --- Select best approach ---
approaches = [
    ("prob_blend", best_auc, best_w),
    ("rank_blend", best_rank_auc, best_rank_w),
]
approaches.sort(key=lambda x: x[1], reverse=True)
best_approach, best_final_auc, best_final_w = approaches[0]

print(f"\n=== BEST: {best_approach} — AUC {best_final_auc:.5f} ===")

# --- Generate final predictions ---
if best_approach == "prob_blend":
    final_test = realmlp_test * best_final_w + xgb074_test * (1 - best_final_w)
    final_oof = realmlp_oof * best_final_w + xgb074_oof * (1 - best_final_w)
else:
    # Rank blend for test
    realmlp_test_rank = rankdata(realmlp_test) / len(realmlp_test)
    xgb074_test_rank = rankdata(xgb074_test) / len(xgb074_test)
    final_test = realmlp_test_rank * best_final_w + xgb074_test_rank * (1 - best_final_w)
    final_oof = realmlp_oof * best_final_w + xgb074_oof * (1 - best_final_w)

# --- Save ---
np.save(f"{OUT_DIR}/oof_preds.npy", final_oof)
np.save(f"{OUT_DIR}/test_preds.npy", final_test)

sub = pd.DataFrame({"id": test["id"], "Churn": final_test})
sub.to_csv(f"{OUT_DIR}/submission.csv", index=False)

json.dump({
    "id": "078", "name": "realmlp_te_std_blend",
    "strategy": f"RealMLP(060) × {best_final_w:.2f} + XGB_TE_std(074) × {1-best_final_w:.2f} ({best_approach})",
    "val_score": round(best_final_auc, 5),
    "weights": {"realmlp": round(best_final_w, 2), "xgb074": round(1-best_final_w, 2)},
    "blend_type": best_approach,
    "status": "done",
}, open(f"{OUT_DIR}/results.json", "w"), indent=2)

print(f"\nSubmission saved: {OUT_DIR}/submission.csv")
print("Done.")
