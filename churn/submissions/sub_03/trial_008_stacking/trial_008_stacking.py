import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_03/trial_008_stacking"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = (train["Churn"] == "Yes").astype(int)

# ── Load OOF & test preds ──────────────────────────────────────────────────────
# trial_004: oof 없음 → test preds만 사용 (weight 낮게)
t004_test = pd.read_csv(f"{SUB_DIR}/sub_02/trial_004_lgbm_tuned/trial_004_lgbm_tuned.csv")["Churn"].values
t006_oof  = np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/oof_preds.npy")
t006_test = np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/test_preds.npy")
t007_oof  = np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/oof_preds.npy")
t007_test = np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/test_preds.npy")

# ── OOF-based weighted blend ───────────────────────────────────────────────────
# trial_004 oof 없으므로 006, 007로만 OOF 앙상블 점수 계산
for w6 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w7 = 1 - w6
    blend_oof = w6 * t006_oof + w7 * t007_oof
    auc = roc_auc_score(y, blend_oof)
    print(f"  w006={w6:.1f} w007={w7:.1f} → OOF AUC: {auc:.5f}")

# best weight search
best_auc, best_w6 = 0, 0.5
for w6 in np.arange(0.0, 1.01, 0.05):
    w7 = 1 - w6
    auc = roc_auc_score(y, w6 * t006_oof + w7 * t007_oof)
    if auc > best_auc:
        best_auc, best_w6 = auc, w6

print(f"\nBest OOF AUC (006+007): {best_auc:.5f}  w006={best_w6:.2f}")

# include trial_004 with equal weight in test blend
best_w7 = 1 - best_w6
ensemble_test = (best_w6 * t006_test + best_w7 * t007_test + t004_test) / 3 * 3
# re-normalize: 004 + 006*w6 + 007*w7, then avg with 004
ensemble_test = (t004_test + best_w6 * t006_test + best_w7 * t007_test) / (1 + best_w6 + best_w7)

# simpler: just equal blend of all 3
ensemble_equal = (t004_test + t006_test + t007_test) / 3
auc_equal_oof  = roc_auc_score(y, (t006_oof + t007_oof) / 2)
print(f"Equal blend 006+007 OOF AUC: {auc_equal_oof:.5f}")

# use best OOF weight for final
final_test = best_w6 * t006_test + best_w7 * t007_test
print(f"\nFinal ensemble (006×{best_w6:.2f} + 007×{best_w7:.2f}) saved.")

sub = pd.DataFrame({"id": test["id"], "Churn": final_test})
sub.to_csv(f"{OUT_DIR}/trial_008_stacking.csv", index=False)

results = {
    "id": "008", "status": "done",
    "val_score": round(best_auc, 5),
    "notes": f"weighted blend 006×{best_w6:.2f} + 007×{best_w7:.2f}",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
