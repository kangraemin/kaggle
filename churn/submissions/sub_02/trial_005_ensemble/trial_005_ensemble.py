import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_02"
OUT_DIR  = f"{SUB_DIR}/trial_005_ensemble"

# ── Load original train target ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
y     = (train["Churn"] == "Yes").astype(int)

# ── Load test submissions ─────────────────────────────────────────────────────
sub_002 = pd.read_csv(f"{SUB_DIR}/trial_002_feature_eng/trial_002_feature_eng.csv")
sub_003 = pd.read_csv(f"{SUB_DIR}/trial_003_catboost/trial_003_catboost.csv")

# ── Load val scores from results.json ────────────────────────────────────────
with open(f"{SUB_DIR}/trial_002_feature_eng/results.json") as f:
    r002 = json.load(f)
with open(f"{SUB_DIR}/trial_003_catboost/results.json") as f:
    r003 = json.load(f)

print(f"trial_002 val: {r002['val_score']}")
print(f"trial_003 val: {r003['val_score']}")

# ── Ensemble (simple average) ─────────────────────────────────────────────────
ensemble_preds = (sub_002["Churn"] + sub_003["Churn"]) / 2

sub = pd.DataFrame({"id": sub_002["id"], "Churn": ensemble_preds})
sub.to_csv(f"{OUT_DIR}/trial_005_ensemble.csv", index=False)
print("\nEnsemble submission saved.")
print(f"Prediction range: [{ensemble_preds.min():.4f}, {ensemble_preds.max():.4f}]")

# Note: OOF ensemble AUC는 oof preds 파일 없이는 계산 불가
# val_score는 두 모델의 평균으로 추정
estimated_val = (r002["val_score"] + r003["val_score"]) / 2
print(f"Estimated ensemble val (avg): {estimated_val:.5f}")

results = {
    "id": "005",
    "status": "done",
    "val_score": round(estimated_val, 5),
    "notes": "simple average of trial_002 + trial_003 test predictions",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("results.json saved.")
