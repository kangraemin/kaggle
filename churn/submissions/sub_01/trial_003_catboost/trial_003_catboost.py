import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_01/trial_003_catboost"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

for col in cat_cols:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

drop_cols = ["id", "Churn", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols]
cat_idx = [feature_cols.index(c) for c in cat_cols]

X = train[feature_cols]
y = train["target"]
X_test = test[feature_cols]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importances = np.zeros(len(feature_cols))

params = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=SEED,
    early_stopping_rounds=50,
    verbose=200,
    cat_features=cat_idx,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = CatBoostClassifier(**params)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    feature_importances += model.get_feature_importance() / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

fi = pd.Series(feature_importances, index=feature_cols).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.2f}")

pd.DataFrame({"id": test["id"], "Churn": test_preds}).to_csv(f"{OUT_DIR}/trial_003_catboost.csv", index=False)

results = {
    "id": "003", "status": "done",
    "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 2) for k, v in top_features.items()},
    "notes": "", "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
