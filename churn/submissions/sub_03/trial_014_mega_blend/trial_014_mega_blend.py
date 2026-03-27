import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import product
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = f"{DATA_DIR}/submissions"
OUT_DIR  = f"{SUB_DIR}/sub_03/trial_014_mega_blend"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = (train["Churn"] == "Yes").astype(int)

# ── Load all OOFs ─────────────────────────────────────────────────────────────
oof = {
    "006": np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/oof_preds.npy"),
    "007": np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/oof_preds.npy"),
    "011": np.load(f"{SUB_DIR}/sub_03/trial_011_no_internet_fix/oof_preds.npy"),
    "012": np.load(f"{SUB_DIR}/sub_03/trial_012_catboost_oof/oof_preds.npy"),
    "013": np.load(f"{SUB_DIR}/sub_03/trial_013_10fold/oof_preds.npy"),
}
test_p = {
    "006": np.load(f"{SUB_DIR}/sub_03/trial_006_advanced_features/test_preds.npy"),
    "007": np.load(f"{SUB_DIR}/sub_03/trial_007_xgboost/test_preds.npy"),
    "011": np.load(f"{SUB_DIR}/sub_03/trial_011_no_internet_fix/test_preds.npy"),
    "012": np.load(f"{SUB_DIR}/sub_03/trial_012_catboost_oof/test_preds.npy"),
    "013": np.load(f"{SUB_DIR}/sub_03/trial_013_10fold/test_preds.npy"),
}

# 단일 OOF AUC 출력
for k, v in oof.items():
    print(f"trial_{k} OOF AUC: {roc_auc_score(y, v):.5f}")

# ── Grid search best weights ──────────────────────────────────────────────────
keys = list(oof.keys())
best_auc, best_weights = 0, None

# 각 모델 weight를 0~1, 합=1로 탐색 (step 0.1)
from itertools import product as iproduct

steps = np.arange(0, 1.01, 0.1)
for w in iproduct(steps, repeat=len(keys)):
    if abs(sum(w) - 1.0) > 0.01:
        continue
    blend = sum(w[i] * oof[keys[i]] for i in range(len(keys)))
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_weights = w

print(f"\nBest OOF AUC: {best_auc:.5f}")
print("Best weights:")
for k, w in zip(keys, best_weights):
    print(f"  trial_{k}: {w:.1f}")

# ── Final test blend ──────────────────────────────────────────────────────────
final_test = sum(best_weights[i] * test_p[keys[i]] for i in range(len(keys)))

sub = pd.DataFrame({"id": test["id"], "Churn": final_test})
sub.to_csv(f"{OUT_DIR}/trial_014_mega_blend.csv", index=False)

results = {
    "id": "014", "status": "done", "val_score": round(best_auc, 5),
    "weights": {f"trial_{k}": float(w) for k, w in zip(keys, best_weights)},
    "notes": "5모델 OOF grid search weighted blend (006+007+011+012+013)",
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
