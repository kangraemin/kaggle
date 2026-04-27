import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

ALPHA = 10
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_09/trial_064_adversarial_val"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]

# ── Step 1: Adversarial Validation ──
# Label-encode categoricals for adversarial model
from sklearn.preprocessing import LabelEncoder
adv_train = train[cat_cols + num_cols].copy()
adv_test  = test[cat_cols + num_cols].copy()
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([adv_train[col], adv_test[col]]))
    adv_train[col] = le.transform(adv_train[col])
    adv_test[col]  = le.transform(adv_test[col])

adv_train["is_test"] = 0
adv_test["is_test"]  = 1
adv_full = pd.concat([adv_train, adv_test], axis=0).reset_index(drop=True)
adv_y = adv_full["is_test"]
adv_X = adv_full.drop(columns=["is_test"])

# Train adversarial model
adv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1,
    eval_metric="auc", verbosity=0, random_state=42
)
adv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
adv_scores = []
for tr_idx, val_idx in adv_skf.split(adv_X, adv_y):
    adv_model.fit(adv_X.iloc[tr_idx], adv_y.iloc[tr_idx])
    pred = adv_model.predict_proba(adv_X.iloc[val_idx])[:, 1]
    adv_scores.append(roc_auc_score(adv_y.iloc[val_idx], pred))

adv_auc = np.mean(adv_scores)
print(f"Adversarial AUC: {adv_auc:.5f}")

# Get feature importances and drop features that distinguish train/test
adv_model.fit(adv_X, adv_y)
importances = pd.Series(adv_model.feature_importances_, index=adv_X.columns)
importances = importances.sort_values(ascending=False)
print("\nAdversarial feature importances (top 10):")
print(importances.head(10))

# Remove features with importance > threshold (most discriminative)
# Try removing top features that help distinguish train from test
THRESHOLD = importances.quantile(0.85)  # remove top 15% most discriminative
drop_features = importances[importances > THRESHOLD].index.tolist()
print(f"\nDropping {len(drop_features)} features: {drop_features}")

# Keep remaining features
keep_cat_cols = [c for c in cat_cols if c not in drop_features]
keep_num_cols = [c for c in num_cols if c not in drop_features]
print(f"Keeping {len(keep_cat_cols)} cat + {len(keep_num_cols)} num features")

# ── Step 2: Train XGB with filtered features ──
y = train["target"]

best_params = {
    "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0,
    "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "gamma": 0.5,
}

SEEDS = [42, 0, 1, 2, 3, 4, 5]
N_FOLDS = 5

def make_features(tr_df, val_df, test_df):
    gm = tr_df["target"].mean()
    te_tr, te_val, te_te = {}, {}, {}
    for col in keep_cat_cols:
        stats = tr_df.groupby(col)["target"].agg(["sum","count"])
        smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
        te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
        te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
        te_te[f"te_{col}"]  = test_df[col].map(smooth).fillna(gm).values

    X_tr  = pd.concat([tr_df[keep_num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1)
    X_val = pd.concat([val_df[keep_num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1)
    X_te  = pd.concat([test_df[keep_num_cols].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
    return X_tr, X_val, X_te

all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))
    for tr_idx, val_idx in skf.split(train[keep_num_cols], y):
        X_tr, X_val, X_te = make_features(train.iloc[tr_idx], train.iloc[val_idx], test)
        p = {**best_params, "random_state": seed}
        model = xgb.train(p, xgb.DMatrix(X_tr, label=y.iloc[tr_idx]),
                          num_boost_round=1000,
                          evals=[(xgb.DMatrix(X_val, label=y.iloc[val_idx]),"val")],
                          early_stopping_rounds=50, verbose_eval=False)
        seed_oof[val_idx] = model.predict(xgb.DMatrix(X_val))
        seed_test += model.predict(xgb.DMatrix(X_te)) / N_FOLDS
    auc = roc_auc_score(y, seed_oof)
    print(f"SEED {seed}: {auc:.5f}")
    all_oof += seed_oof / len(SEEDS); all_test += seed_test / len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
json.dump({"id": "064", "status": "done", "val_score": round(oof_auc, 5),
           "strategy": "Adversarial validation feature removal + XGB + TE + 7 seeds"},
          open(f"{OUT_DIR}/results.json", "w"), indent=2)
print("Done.")
