"""
Trial 002: Feature Engineering + CatBoost + LightGBM + XGBoost Ensemble
- Domain-driven FE: ET_proxy, water_balance, interaction terms
- CatBoost native categorical handling
- 3-model soft voting ensemble
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values


def add_features(df):
    # ET proxy (evapotranspiration)
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    # Water balance
    df["water_balance"] = df["Rainfall_mm"] - df["ET_proxy"] * 100
    # Interactions
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
    # Binary indicators
    df["is_active_growth"] = (df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"])).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_low_rain"] = (df["Rainfall_mm"] < 500).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)
    # Ratios
    df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["prev_irr_per_area"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["moisture_deficit"] = 50 - df["Soil_Moisture"]  # deficit from "ideal" 50%
    return df


train = add_features(train)
test = add_features(test)

drop_cols = ["id", "Irrigation_Need"]
features = [c for c in train.columns if c not in drop_cols]
cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

# For LightGBM/XGBoost: label encode
train_enc = train.copy()
test_enc = test.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col])
    test_enc[col] = le.transform(test_enc[col])
    label_encoders[col] = le

X_lgbm = train_enc[features]
X_test_lgbm = test_enc[features]

# For CatBoost: keep strings
X_cat = train[features].copy()
X_test_cat = test[features].copy()
cat_indices = [features.index(c) for c in cat_cols]

# ============ Models ============

lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "enable_categorical": True,
}

cat_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 100,
    "cat_features": cat_indices,
    "auto_class_weights": "Balanced",
}

# ============ CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgbm = np.zeros((len(X_lgbm), 3))
oof_xgb = np.zeros((len(X_lgbm), 3))
oof_cat = np.zeros((len(X_lgbm), 3))
test_lgbm = np.zeros((len(X_test_lgbm), 3))
test_xgb = np.zeros((len(X_test_lgbm), 3))
test_cat = np.zeros((len(X_test_cat), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm, y)):
    print(f"\n{'='*40} Fold {fold} {'='*40}")
    y_tr, y_val = y[train_idx], y[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgbm_params)
    model_lgb.fit(
        X_lgbm.iloc[train_idx], y_tr,
        eval_set=[(X_lgbm.iloc[val_idx], y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    oof_lgbm[val_idx] = model_lgb.predict_proba(X_lgbm.iloc[val_idx])
    test_lgbm += model_lgb.predict_proba(X_test_lgbm) / N_FOLDS
    print(f"  LGBM acc: {accuracy_score(y_val, oof_lgbm[val_idx].argmax(1)):.6f}")

    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_lgbm.iloc[train_idx], y_tr,
        eval_set=[(X_lgbm.iloc[val_idx], y_val)],
        verbose=200,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_lgbm.iloc[val_idx])
    test_xgb += model_xgb.predict_proba(X_test_lgbm) / N_FOLDS
    print(f"  XGB  acc: {accuracy_score(y_val, oof_xgb[val_idx].argmax(1)):.6f}")

    # CatBoost
    model_cb = CatBoostClassifier(**cat_params)
    model_cb.fit(
        X_cat.iloc[train_idx], y_tr,
        eval_set=(X_cat.iloc[val_idx], y_val),
        early_stopping_rounds=100,
    )
    oof_cat[val_idx] = model_cb.predict_proba(X_cat.iloc[val_idx])
    test_cat += model_cb.predict_proba(X_test_cat) / N_FOLDS
    print(f"  CAT  acc: {accuracy_score(y_val, oof_cat[val_idx].argmax(1)):.6f}")

# ============ Ensemble ============

# Individual scores
for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
    acc = accuracy_score(y, oof.argmax(1))
    print(f"\n{name} OOF accuracy: {acc:.6f}")

# Grid search ensemble weights
best_acc = 0
best_w = (1, 1, 1)
for w1 in range(1, 6):
    for w2 in range(1, 6):
        for w3 in range(1, 6):
            total = w1 + w2 + w3
            oof_ens = (w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat) / total
            acc = accuracy_score(y, oof_ens.argmax(1))
            if acc > best_acc:
                best_acc = acc
                best_w = (w1, w2, w3)

print(f"\nBest ensemble weights (lgbm:xgb:cat): {best_w}")
print(f"Best ensemble OOF accuracy: {best_acc:.6f}")

total = sum(best_w)
test_ens = (best_w[0] * test_lgbm + best_w[1] * test_xgb + best_w[2] * test_cat) / total

# Save
np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_lgbm, oof_xgb, oof_cat]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_lgbm, test_xgb, test_cat]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_ens.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_002_fe_catboost",
    "oof_accuracy_lgbm": round(accuracy_score(y, oof_lgbm.argmax(1)), 6),
    "oof_accuracy_xgb": round(accuracy_score(y, oof_xgb.argmax(1)), 6),
    "oof_accuracy_cat": round(accuracy_score(y, oof_cat.argmax(1)), 6),
    "oof_accuracy_ensemble": round(best_acc, 6),
    "ensemble_weights": best_w,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Files saved to", OUT_DIR)
