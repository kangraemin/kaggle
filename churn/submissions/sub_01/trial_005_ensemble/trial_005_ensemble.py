import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
SUB_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_01"
OUT_DIR  = f"{SUB_DIR}/trial_005_ensemble"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)
y = train["target"]

# trial_001 oof / test preds 재생성 없이 CSV 기반 앙상블
# test preds 평균
t001 = pd.read_csv(f"{SUB_DIR}/trial_001_lgbm_baseline/trial_001_lgbm_baseline.csv")
t002 = pd.read_csv(f"{SUB_DIR}/trial_002_feature_eng/trial_002_feature_eng.csv")

# 단순 평균
ensemble = (t001["Churn"] + t002["Churn"]) / 2
sub = pd.DataFrame({"id": t001["id"], "Churn": ensemble})
sub.to_csv(f"{OUT_DIR}/trial_005_ensemble.csv", index=False)
print("Ensemble (001+002 avg) saved.")

# val score는 OOF가 없어서 추정 불가 — 제출로 확인 필요
results = {
    "id": "005", "status": "done",
    "val_score": None,
    "notes": "trial_001 + trial_002 test preds 단순 평균. OOF 없어서 val score 추정 불가.",
    "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
