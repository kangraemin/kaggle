import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = "/Users/ram/programming/vibecoding/kaggle/churn/submissions/sub_01/trial_004_lgbm_tuned"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

drop_cols = ["id", "Churn", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols]
X, y, X_test = train[feature_cols], train["target"], test[feature_cols]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importances = np.zeros(len(feature_cols))

# 더 큰 모델: num_leaves 127, 더 작은 lr
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "random_state": SEED,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params, n_estimators=3000)
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    feature_importances += model.feature_importances_ / N_FOLDS
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}  best_iter={model.best_iteration_}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")

fi = pd.Series(feature_importances, index=feature_cols).sort_values(ascending=False)
top_features = fi.head(10).to_dict()
print("\nTop 10 features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.1f}")

pd.DataFrame({"id": test["id"], "Churn": test_preds}).to_csv(f"{OUT_DIR}/trial_004_lgbm_tuned.csv", index=False)

results = {
    "id": "004", "status": "done",
    "val_score": round(oof_auc, 5),
    "top_features": {k: round(v, 1) for k, v in top_features.items()},
    "notes": "", "conclusion": ""
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done.")
